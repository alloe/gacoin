#!/usr/bin/env python3
"""Phase 19: causal, bounded public-data Event Sniper screening. NEVER places orders.

This is L1 snapshot research, not a fill simulator or funded portfolio backtest.
Decision quantity never uses exit liquidity. Insufficient exit depth remains unresolved.
Phase 17/18 statistics are deliberately NOT merged with this corrected dataset.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import signal
import statistics
import sys
import time
import uuid
import zlib
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VERSION = "19.1-causal"
RUN_ID = uuid.uuid4().hex[:12]
STEP_MS = 100
LOOKBACK_TOL_MS = 150
QUOTE_MAX_AGE_MS = 750
FX_MAX_AGE_MS = 10000
PRE_MS, POST_MS, COOLDOWN_MS = 10000, 30000, 30000
SHOCK_BP, LAG_BP = 30.0, 10.0
ANCHORS = ("BTC", "ETH", "XRP", "SOL", "DOGE", "ADA")
MIN_ANCHORS = 3
MAX_RELATIVE_DIVERGENCE_BP = 500.0  # Quarantine structural / symbol-mapping outliers.
LAGS = (0, 100, 200, 500, 1000)  # Zero is a non-executable upper-bound diagnostic.
HOLDS = (500, 1000, 2000, 5000, 10000, "dynamic")
SIZES = (1000000, 3000000, 5000000, 10000000)
PARTICIPATIONS = (1.0, 0.25)
EXTRA_COST_BP = (0.0, 2.0, 5.0)  # Adverse price cost per execution, not fee discounts.
EDGE_GATES = (0.0, 10.0, 20.0)
UP_FEE, BN_FEE = 0.0005, 0.0005  # Explicit research inputs; no account query.
TP_BP, SL_BP, TIMEOUT_MS = 5.0, 30.0, 10000
SESSION_SEC = min(86400.0, max(60.0, float(os.getenv("SNIPER19_SESSION_SEC", "86400"))))
SPOOL = Path(os.getenv("SNIPER19_SPOOL", "/tmp/sniper19"))
SPOOL_LIMIT = 128 * 1024 * 1024
UP_WS = "wss://api.upbit.com/websocket/v1"
BN_WS = "wss://fstream.binance.com/ws"


def json_text(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def emit(tag: str, **data: Any) -> None:
    print(tag + " " + json_text({"version": VERSION, "run_id": RUN_ID, **data}), flush=True)


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    recv_ms: float
    recv_wall_ns: int
    source_ms: int | None
    transaction_ms: int | None = None
    update_id: int | None = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


def quote_ok(q: Quote | None, at_ms: float, max_age_ms: float = QUOTE_MAX_AGE_MS) -> bool:
    if q is None or not 0 <= at_ms - q.recv_ms <= max_age_ms:
        return False
    return (all(math.isfinite(v) for v in (q.bid, q.ask, q.bid_size, q.ask_size))
            and 0 < q.bid < q.ask and min(q.bid_size, q.ask_size) >= 0)


def make_quote(bid: Any, ask: Any, bs: Any, az: Any, source: Any,
               transaction: Any = None, update: Any = None, mult: float = 1.0) -> Quote:
    q = Quote(float(bid) / mult, float(ask) / mult, float(bs) * mult,
              float(az) * mult, time.monotonic_ns() / 1e6, time.time_ns(),
              int(source) if source is not None else None,
              int(transaction) if transaction is not None else None,
              int(update) if update is not None else None)
    if not quote_ok(q, q.recv_ms):
        raise ValueError("invalid quote")
    return q


def asof(rows: list[dict] | deque, target_ms: float) -> dict | None:
    """Last received sample at/before target. No future interpolation or snapping."""
    for row in reversed(rows):
        if row["t"] <= target_ms:
            return row if target_ms - row["t"] <= LOOKBACK_TOL_MS else None
    return None


def returns(history: deque, row: dict, window_ms: int) -> dict | None:
    old = asof(history, row["t"] - window_ms)
    if old is None:
        return None
    for venue in ("up", "bn"):
        if not quote_ok(row[venue], row["t"]) or not quote_ok(old[venue], row["t"] - window_ms):
            return None
    # Crucially: current MID / previous MID, never current mid / previous ask.
    b = math.log(row["bn"].mid / old["bn"].mid) * 10000
    u = math.log(row["up"].mid / old["up"].mid) * 10000
    return {"window_ms": window_ms, "actual_window_ms": row["t"] - old["t"],
            "bn_bp": b, "up_bp": u, "lag_bp": b - u}


def reference_log_ratio(snap: dict[str, tuple[Quote, Quote]], coin: str) -> tuple[float | None, int]:
    # FX cancels from log(Upbit/Binance)-cross-sectional median; do not gate raw capture on FX.
    values = [math.log(snap[c][0].mid / snap[c][1].mid)
              for c in ANCHORS if c != coin and c in snap]
    return (statistics.median(values), len(values)) if len(values) >= MIN_ANCHORS else (None, len(values))


def diagnostic(row: dict | None, at_ms: float) -> dict | None:
    if row is None or not all(quote_ok(row[k], at_ms) for k in ("up", "bn")):
        return None
    ref = row.get("reference")
    if ref is None:
        return None
    u, b = row["up"], row["bn"]
    residual = (math.log(u.mid / b.mid) - ref) * 10000
    entry_residual = (math.log(u.ask / b.bid) - ref) * 10000
    closing_half_spreads = ((u.ask - u.bid) / u.mid + (b.ask - b.bid) / b.mid) * 5000
    # A full-reversion upper bound, NOT a calibrated expected profit forecast.
    room = -entry_residual - closing_half_spreads - 2 * (UP_FEE + BN_FEE) * 10000
    return {"residual_bp": residual, "entry_residual_bp": entry_residual,
            "reversion_room_net_bp": room, "entry_l1_capacity_krw": min(u.ask_size, b.bid_size) * u.ask,
            "relative_divergence_ok": abs(residual) <= MAX_RELATIVE_DIVERGENCE_BP}


def select_signal(history: deque, row: dict, counters: Counter | None = None) -> dict | None:
    rs = []
    for w in (500, 1000):
        r = returns(history, row, w)
        if counters is not None:
            counters["valid_return_windows" if r else "invalid_return_windows"] += 1
        if r is not None:
            rs.append(r)
    if not rs:
        return None
    r = max(rs, key=lambda x: abs(x["bn_bp"]))
    if abs(r["bn_bp"]) < SHOCK_BP:
        return None
    return {**r, "diagnostic": diagnostic(row, row["t"])}


def entry_quantity(decision: dict, arrival: dict, target_krw: int, fraction: float) -> float:
    # Entry quantities depend ONLY on the decision and arrival snapshots, never exit snapshots.
    planned = min(target_krw / decision["up"].ask,
                  decision["up"].ask_size * fraction, decision["bn"].bid_size * fraction)
    return max(0.0, min(planned, arrival["up"].ask_size * fraction, arrival["bn"].bid_size * fraction))


def pnl_at(entry: dict, exit_row: dict, qty: float, extra_bp: float) -> float:
    """KRW P&L proxy for equal underlying quantities, excludes wallet FX and funding."""
    u, b, ux, bx = entry["up"], entry["bn"], exit_row["up"], exit_row["bn"]
    fx = exit_row["fx"].mid
    adverse = extra_bp / 10000
    buy, sell = u.ask * (1 + adverse), ux.bid * (1 - adverse)
    short, cover = b.bid * (1 - adverse), bx.ask * (1 + adverse)
    return qty * (sell * (1 - UP_FEE) - buy * (1 + UP_FEE)
                  + (short * (1 - BN_FEE) - cover * (1 + BN_FEE)) * fx)


def replay(event: dict, lag: int, hold: int | str, amount: int, participation: float,
           extra_bp: float) -> dict:
    rows, decision = event["rows"], event["decision"]
    arrival_time = decision["t"] + lag
    arrival = asof(rows, arrival_time)
    result = {"lag_ms": lag, "hold": hold, "size_krw": amount, "participation": participation,
              "extra_bp": extra_bp, "status": "entry_data_missing", "entered": False,
              "closed": False, "qty": 0.0, "marked_pnl_krw": None}
    if arrival is None or not all(quote_ok(arrival[k], arrival_time) for k in ("up", "bn")):
        return result
    q = entry_quantity(decision, arrival, amount, participation)
    result.update(qty=q, entry_time_ms=arrival_time, entry_sample_age_ms=arrival_time - arrival["t"])
    if q <= 0:
        result["status"] = "entry_liquidity_zero"
        return result
    result.update(entered=True, entry_notional_krw=q * arrival["up"].ask,
                  entry_fill_ratio=min(1.0, q * arrival["up"].ask / amount))
    # Target/stop use executable close-side P&L including all fees, not entry-side quotes.
    path_ok, reason = True, "fixed"
    if hold == "dynamic":
        deadline = arrival_time + TIMEOUT_MS
        request_time, reason, previous = deadline, "timeout", arrival_time
        for row in rows:
            if row["t"] <= arrival_time:
                continue
            if row["t"] > deadline:
                break
            if row["t"] - previous > 250:
                path_ok = False
            previous = row["t"]
            if not all(quote_ok(row[k], row["t"]) for k in ("up", "bn")):
                path_ok = False
                continue
            if not quote_ok(row["fx"], row["t"], FX_MAX_AGE_MS):
                path_ok = False
                continue
            bp = pnl_at(arrival, row, q, extra_bp) / result["entry_notional_krw"] * 10000
            if bp >= TP_BP or bp <= -SL_BP:
                request_time, reason = row["t"], "target" if bp >= TP_BP else "stop"
                break
    else:
        request_time = arrival_time + hold
    # Exit order latency included as well. 0ms remains diagnostic only.
    exit_time = request_time + lag
    exit_row = asof(rows, exit_time)
    result.update(exit_reason=reason, exit_time_ms=exit_time)
    if exit_row is None or not all(quote_ok(exit_row[k], exit_time) for k in ("up", "bn")):
        result["status"] = "exit_data_unresolved"
        return result
    if not quote_ok(exit_row["fx"], exit_time, FX_MAX_AGE_MS):
        result["status"] = "exit_fx_unresolved"
        return result
    marked = pnl_at(arrival, exit_row, q, extra_bp)
    result.update(marked_pnl_krw=marked, marked_bp=marked / result["entry_notional_krw"] * 10000,
                  exit_sample_age_ms=exit_time - exit_row["t"])
    if not path_ok:
        result["status"] = "dynamic_path_unresolved"
        return result
    exit_capacity = min(exit_row["up"].bid_size, exit_row["bn"].ask_size) * participation
    if exit_capacity + 1e-12 < q:
        # DO NOT reduce entry q retroactively, and DO NOT silently exclude this position.
        result.update(status="exit_liquidity_unresolved", exit_capacity_qty=exit_capacity,
                      unclosed_qty=q)
        return result
    result.update(status="closed_l1_proxy", closed=True, pnl_krw=marked, net_bp=result["marked_bp"])
    return result


def update_stats(st: dict, result: dict) -> None:
    st["attempts"] += 1
    st["entered"] += int(result["entered"])
    st["closed"] += int(result["closed"])
    st["unresolved"] += int(result["entered"] and not result["closed"])
    st["status_counts"][result["status"]] += 1
    if result.get("marked_pnl_krw") is not None:
        st["all_marked_pnl_krw"] += result["marked_pnl_krw"]
    if result["entered"]:
        st["entry_notional_sum"] += result["entry_notional_krw"]
        st["partial_entry"] += int(result["entry_fill_ratio"] < 0.999)
    if result["closed"]:
        pnl = result["pnl_krw"]
        st["closed_pnl_krw"] += pnl
        st["net_bp_sum"] += result["net_bp"]
        st["wins"] += int(pnl > 0)
        st["profit_sum"] += max(0.0, pnl)
        st["loss_sum"] += max(0.0, -pnl)


def new_stats() -> dict:
    return {**{k: 0 for k in ("attempts", "entered", "closed", "unresolved", "wins", "partial_entry")},
            **{k: 0.0 for k in ("all_marked_pnl_krw", "entry_notional_sum", "closed_pnl_krw",
                                "net_bp_sum", "profit_sum", "loss_sum")}, "status_counts": Counter()}


class Engine:
    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.cells: dict = defaultdict(new_stats)
        self.coin_counts: Counter = Counter()
        self.active: dict = {}
        self.cooldown: dict = {}
        self.hist: dict = defaultdict(deque)
        self.up: dict[str, Quote] = {}
        self.bn: dict[str, Quote] = {}
        self.symbol_map: dict = {}
        self.mult: dict = {}
        self.coins: list[str] = []
        self.writer_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.stop = asyncio.Event()
        self.started = time.monotonic()
        self.last_receive = {"upbit": 0.0, "binance": 0.0}
        self.max_tick_ms = 0.0
        self.max_loop_gap_ms = 0.0

    def candidate(self, sig: dict) -> bool:
        d = sig["diagnostic"]
        return bool(sig["bn_bp"] > 0 and sig["lag_bp"] >= LAG_BP and d
                    and d["relative_divergence_ok"] and d["reversion_room_net_bp"] >= 0)

    def evaluate(self, ev: dict) -> list[dict]:
        if ev.get("interrupted"):
            return []
        sig, out = ev["signal"], []
        if not self.candidate(sig):
            return out
        for fraction in PARTICIPATIONS:
            for lag in LAGS:
                for hold in HOLDS:
                    for amount in SIZES:
                        for extra in EXTRA_COST_BP:
                            result = replay(ev, lag, hold, amount, fraction, extra)
                            out.append(result)
        return out

    def queue_event(self, ev: dict) -> None:
        if self.writer_queue.full():
            self.counts["events_dropped_backpressure"] += 1
            emit("SNIPER19_DROP", event_id=ev["id"], reason="writer_backpressure")
        else:
            self.writer_queue.put_nowait(ev)

    def checkpoint(self, tag: str = "SNIPER19_SUMMARY") -> None:
        emit(tag, uptime_sec=round(time.monotonic() - self.started, 3), counts=dict(self.counts),
             coins=len(self.coins), active=len(self.active), max_tick_ms=round(self.max_tick_ms, 3),
             max_loop_gap_ms=round(self.max_loop_gap_ms, 3),
             queue=self.writer_queue.qsize(), coin_events=dict(self.coin_counts), production_pass=False)
        for key, st in sorted(self.cells.items(), key=lambda kv: str(kv[0])):
            gate, fraction, lag, hold, amount, extra = key
            if not st["attempts"]:
                continue
            # Compact primary checkpoint; all configurations are in event bundles.
            if (lag, hold, amount, gate, fraction) != (200, "5000", 1000000, 10.0, 0.25):
                continue
            emit("SNIPER19_PRIMARY", gate_bp=gate, participation=fraction, lag_ms=lag,
                 hold_ms=5000, amount_krw=amount, extra_bp=extra, stats=st,
                 mean_closed_bp=st["net_bp_sum"] / st["closed"] if st["closed"] else None,
                 win_rate_closed=st["wins"] / st["closed"] if st["closed"] else None,
                 pf_closed=st["profit_sum"] / st["loss_sum"] if st["loss_sum"] > 0 else None,
                 production_pass=False)

    async def sampler(self) -> None:
        summary_due = self.started + 60
        previous_tick = time.monotonic_ns() / 1e6
        while not self.stop.is_set():
            tick = time.monotonic()
            t, wall = time.monotonic_ns() / 1e6, time.time_ns()
            gap = t - previous_tick
            self.max_loop_gap_ms = max(self.max_loop_gap_ms, gap)
            self.counts["sampler_gaps_over_250ms"] += int(gap > 250)
            previous_tick = t
            snap = {c: (self.up[c], self.bn[c]) for c in self.coins
                    if quote_ok(self.up.get(c), t) and quote_ok(self.bn.get(c), t)}
            self.counts["ticks"] += 1
            self.counts["aligned_samples"] += len(snap)
            if not quote_ok(self.up.get("USDT"), t, FX_MAX_AGE_MS):
                self.counts["fx_missing_ticks"] += 1
            for c in self.coins:
                h = self.hist[c]
                ref, nr = reference_log_ratio(snap, c)
                row = {"t": t, "wall_ns": wall, "up": self.up.get(c), "bn": self.bn.get(c),
                       "fx": self.up.get("USDT"), "reference": ref, "anchors": nr}
                self.counts["anchor_missing_samples"] += int(nr < MIN_ANCHORS)
                sig = select_signal(h, row, self.counts)
                # Time-based retention, not N valid rows spanning an arbitrary period.
                while h and h[0]["t"] < t - PRE_MS:
                    h.popleft()
                h.append(row)
                if c in self.active:
                    self.active[c]["rows"].append(row)
                if sig and c not in self.active and t >= self.cooldown.get(c, 0):
                    eid = f"{RUN_ID}-{wall // 1000000}-{c}"
                    self.counts["events_detected"] += 1
                    self.coin_counts[c] += 1
                    self.counts["up_shocks" if sig["bn_bp"] > 0 else "down_shocks"] += 1
                    candidate = self.candidate(sig)
                    if candidate:
                        self.counts["edge_candidates"] += 1
                        if sig["diagnostic"]["entry_l1_capacity_krw"] >= 1000000:
                            self.counts["raw_capacity_candidates_1m"] += 1
                        if sig["diagnostic"]["entry_l1_capacity_krw"] * 0.25 >= 1000000:
                            self.counts["capacity_candidates_1m_25pct"] += 1
                    if sig["diagnostic"] and not sig["diagnostic"]["relative_divergence_ok"]:
                        self.counts["quarantined_divergence_events"] += 1
                    if sig["diagnostic"] is None:
                        self.counts["reference_missing_events"] += 1
                    self.active[c] = {"id": eid, "coin": c, "decision": row,
                                      "signal": sig, "rows": list(h)}
                    self.cooldown[c] = t + COOLDOWN_MS
                    emit("SNIPER19_EVENT", event_id=eid, coin=c, signal=sig, edge_candidate=candidate,
                         down_route="inventory_required_not_simulated" if sig["bn_bp"] < 0 else None)
            # Finalize by monotonic deadline EVEN if feeds disappear. Never wait for a fresh quote.
            for c, ev in list(self.active.items()):
                if t >= ev["decision"]["t"] + POST_MS:
                    self.queue_event(ev)
                    del self.active[c]
            self.max_tick_ms = max(self.max_tick_ms, (time.monotonic() - tick) * 1000)
            if time.monotonic() >= summary_due:
                self.checkpoint()
                summary_due = time.monotonic() + 60
            await asyncio.sleep(max(0.001, STEP_MS / 1000 - (time.monotonic() - tick)))

    async def writer(self) -> None:
        SPOOL.mkdir(parents=True, exist_ok=True)
        while True:
            ev = await self.writer_queue.get()
            try:
                results = await asyncio.to_thread(self.evaluate, ev)
                # Only the event-loop thread mutates counters and aggregate tables.
                self.counts["events_finished"] += 1
                self.counts["events_interrupted"] += int(bool(ev.get("interrupted")))
                for result in results:
                    for gate in EDGE_GATES:
                        if ev["signal"]["diagnostic"]["reversion_room_net_bp"] >= gate:
                            key = (gate, result["participation"], result["lag_ms"], str(result["hold"]),
                                   result["size_krw"], result["extra_bp"])
                            update_stats(self.cells[key], result)
                raw = {"version": VERSION, "run_id": RUN_ID, "event": ev, "paper": results,
                       "limitations": "L1 snapshots; paired-entry assumption; no fills, L2, funding, wallet FX, margin or liquidation simulation"}
                data = json.dumps(raw, default=lambda x: asdict(x) if isinstance(x, Quote) else str(x),
                                  separators=(",", ":"), allow_nan=False).encode()
                compressed = zlib.compress(data, 6)
                path = SPOOL / (ev["id"] + ".json.zlib")
                await asyncio.to_thread(path.write_bytes, compressed)
                files = sorted(SPOOL.glob("*.json.zlib"), key=lambda p: p.stat().st_mtime)
                total = sum(p.stat().st_size for p in files)
                for f in files:
                    if total <= SPOOL_LIMIT:
                        break
                    total -= f.stat().st_size
                    f.unlink()
                    self.counts["spool_rotated_files"] += 1
                encoded = base64.b64encode(compressed).decode()
                digest = hashlib.sha256(compressed).hexdigest()
                parts = [encoded[i:i + 2200] for i in range(0, len(encoded), 2200)]
                emit("SNIPER19_BUNDLE", event_id=ev["id"], encoding="zlib+base64", sha256=digest,
                     chunks=len(parts), bytes=len(compressed), rows=len(ev["rows"]), cells=len(results),
                     storage="bounded_ephemeral_spool_and_retained_platform_logs_not_permanent")
                for i, part in enumerate(parts):
                    emit("SNIPER19_CHUNK", event_id=ev["id"], seq=i, data=part)
                    await asyncio.sleep(0.06)  # Pace log bursts; avoid blocking WebSocket receive loop.
                emit("SNIPER19_END", event_id=ev["id"], sha256=digest, chunks=len(parts))
                self.counts["bundles_written"] += 1
            except Exception as exc:
                self.counts["writer_errors"] += 1
                emit("SNIPER19_WRITE_ERROR", event_id=ev["id"], error=repr(exc))
            finally:
                self.writer_queue.task_done()

    async def upbit_stream(self) -> None:
        import websockets
        delay = 1
        while not self.stop.is_set():
            try:
                async with websockets.connect(UP_WS, ping_interval=20, ping_timeout=20,
                                              max_size=2**22, max_queue=1024) as ws:
                    sub = [{"ticket": "sniper19-" + RUN_ID}, {"type": "orderbook",
                           "codes": ["KRW-USDT.1"] + ["KRW-" + c + ".1" for c in self.coins],
                           "level": 0}, {"format": "DEFAULT"}]
                    await ws.send(json_text(sub))
                    emit("SNIPER19_CONNECTED", venue="upbit")
                    delay = 1
                    async for raw in ws:
                        d = json.loads(raw)
                        if "error" in d:
                            raise RuntimeError(str(d["error"]))
                        units = d.get("orderbook_units") or []
                        if not units:
                            continue
                        c = d.get("code", "").split(".")[0].removeprefix("KRW-")
                        if c not in self.coins and c != "USDT":
                            continue
                        u = units[0]
                        try:
                            q = make_quote(u["bid_price"], u["ask_price"], u["bid_size"],
                                           u["ask_size"], d.get("timestamp"))
                        except (ValueError, TypeError, KeyError):
                            self.counts["invalid_up_quotes"] += 1
                            continue
                        old = self.up.get(c)
                        if old and old.source_ms and q.source_ms and q.source_ms < old.source_ms:
                            self.counts["out_of_order_up"] += 1
                            continue
                        self.up[c] = q
                        self.last_receive["upbit"] = time.monotonic()
            except Exception as exc:
                self.counts["up_reconnects"] += 1
                emit("SNIPER19_WS_ERROR", venue="upbit", error=repr(exc), retry_sec=delay)
            finally:
                self.up.clear()  # Quotes from a disconnected session must not remain tradable.
            await asyncio.sleep(delay)
            delay = min(30, delay * 2)

    async def binance_stream(self) -> None:
        import websockets
        reverse = {s: c for c, s in self.symbol_map.items()}
        delay = 1
        while not self.stop.is_set():
            try:
                async with websockets.connect(BN_WS, ping_interval=20, ping_timeout=20,
                                              max_size=2**22, max_queue=1024) as ws:
                    await ws.send(json_text({"method": "SUBSCRIBE", "params": [s.lower() + "@bookTicker"
                                              for s in reverse], "id": 19}))
                    emit("SNIPER19_CONNECTED", venue="binance")
                    delay = 1
                    async for raw in ws:
                        d = json.loads(raw)
                        d = d.get("data", d)
                        if "code" in d and "msg" in d:
                            raise RuntimeError(str(d))
                        c = reverse.get(d.get("s"))
                        if c is None or not all(k in d for k in ("b", "a", "B", "A")):
                            continue
                        try:
                            q = make_quote(d["b"], d["a"], d["B"], d["A"], d.get("E"),
                                           d.get("T"), d.get("u"), self.mult[c])
                        except (ValueError, TypeError, KeyError):
                            self.counts["invalid_bn_quotes"] += 1
                            continue
                        old = self.bn.get(c)
                        if old and old.update_id is not None and q.update_id is not None and q.update_id <= old.update_id:
                            self.counts["out_of_order_bn"] += 1
                            continue
                        self.bn[c] = q
                        self.last_receive["binance"] = time.monotonic()
            except Exception as exc:
                self.counts["bn_reconnects"] += 1
                emit("SNIPER19_WS_ERROR", venue="binance", error=repr(exc), retry_sec=delay)
            finally:
                self.bn.clear()
            await asyncio.sleep(delay)
            delay = min(30, delay * 2)

    async def run(self) -> None:
        # Reuse only the existing public market discovery function; never invoke Phase18 main.
        from phase18_event_sniper_paper import choose_universe
        self.coins, self.symbol_map, self.mult = await asyncio.to_thread(choose_universe)
        emit("SNIPER19_START", coins=self.coins, universe=len(self.coins), session_sec=SESSION_SEC,
             symbol_map=self.symbol_map, contract_multipliers=self.mult,
             lags_ms=LAGS, holds=HOLDS, sizes_krw=SIZES, participations=PARTICIPATIONS,
             fees_bp={"upbit": UP_FEE * 10000, "binance": BN_FEE * 10000}, extra_cost_bp=EXTRA_COST_BP,
             edge_gates=EDGE_GATES, zero_latency="upper_bound_only", lookback_tolerance_ms=LOOKBACK_TOL_MS,
             raw_capture_independent_of_fx=True, min_anchors=MIN_ANCHORS,
             primary={"edge_bp": 10, "participation": 0.25, "lag_ms": 200, "hold_ms": 5000, "size_krw": 1000000},
             no_private_apis=True, production_pass=False)
        loop = asyncio.get_running_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(s, self.stop.set)
            except NotImplementedError:
                pass
        tasks = [asyncio.create_task(f()) for f in (self.upbit_stream, self.binance_stream, self.sampler)]
        writer = asyncio.create_task(self.writer())
        timer = loop.call_later(SESSION_SEC, self.stop.set)
        stopper = asyncio.create_task(self.stop.wait())
        try:
            done, _ = await asyncio.wait([*tasks, writer, stopper], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task is not stopper:
                    error = task.exception()
                    if error is not None:
                        raise error
                    if not self.stop.is_set():
                        raise RuntimeError("critical worker stopped unexpectedly")
        finally:
            timer.cancel()
            self.stop.set()
            stopper.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for ev in self.active.values():
                ev["interrupted"] = True
                self.queue_event(ev)
            self.active.clear()
            try:
                await asyncio.wait_for(self.writer_queue.join(), timeout=20)
            except asyncio.TimeoutError:
                emit("SNIPER19_INCOMPLETE_DRAIN", queued=self.writer_queue.qsize())
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)
            self.checkpoint("SNIPER19_DONE")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        import unittest
        from test_phase19_event_sniper_audit import AuditTests
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    try:
        asyncio.run(Engine().run())
    except Exception as exc:
        emit("SNIPER19_FATAL", error=repr(exc), production_pass=False)
        raise
