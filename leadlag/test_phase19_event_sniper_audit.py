"""Offline regression tests: synthetic prices, not backtest profits."""
import base64
import hashlib
import json
import math
import unittest
import zlib
from collections import deque
from dataclasses import replace

from phase19_event_sniper_audit import (
    Engine, Quote, asof, diagnostic, entry_quantity, make_quote,
    new_stats, pnl_at, quote_ok, reference_log_ratio, replay,
    returns, select_signal, update_stats,
)


def q(bid=99.0, ask=101.0, n=1000.0, t=1000.0):
    return Quote(bid, ask, n, n, t, 0, 0)


def row(t, up=None, bn=None, fx=None):
    return {"t": float(t), "wall_ns": 0,
            "up": up if up is not None else q(t=t),
            "bn": bn if bn is not None else q(.99, 1.01, t=t),
            "fx": fx if fx is not None else q(99.9, 100.1, t=t),
            "reference": math.log(100), "anchors": 3}


def event(rows):
    return {"rows": rows, "decision": rows[0], "signal": {"bn_bp": 40, "lag_bp": 20,
            "diagnostic": {"relative_divergence_ok": True, "reversion_room_net_bp": 20}}}


class AuditTests(unittest.TestCase):
    def test_01_flat_wide_spread_is_zero_return(self):
        old, cur = row(1000), row(1500)
        r = returns(deque([old]), cur, 500)
        self.assertAlmostEqual(r["bn_bp"], 0)
        self.assertAlmostEqual(r["up_bp"], 0)
        self.assertIsNone(select_signal(deque([old]), cur))

    def test_02_multi_second_gap_is_not_500ms(self):
        self.assertIsNone(returns(deque([row(1000)]), row(5000), 500))

    def test_03_asof_never_reads_future(self):
        self.assertIsNone(asof([row(1001)], 1000))
        self.assertEqual(asof([row(1000), row(1100)], 1050)["t"], 1000)

    def test_04_old_sample_rejected(self):
        self.assertIsNone(asof([row(1000)], 1151))

    def test_05_stale_quote_rejected(self):
        self.assertFalse(quote_ok(q(t=1000), 1751))
        self.assertFalse(quote_ok(q(t=1001), 1000))

    def test_06_fx_independent_impulse(self):
        old, cur = row(1000), row(1500, bn=q(1.01, 1.03, t=1500))
        old["fx"] = cur["fx"] = None
        self.assertGreater(select_signal(deque([old]), cur)["bn_bp"], 30)

    def test_07_fx_cancels_from_reference(self):
        r = row(1000)
        a = diagnostic(r, 1000)
        r["fx"] = None
        self.assertEqual(a, diagnostic(r, 1000))

    def test_08_leave_one_out_reference(self):
        snap = {c: (q(), q(.99, 1.01)) for c in ("BTC", "ETH", "XRP", "SOL")}
        ref, n = reference_log_ratio(snap, "BTC")
        self.assertEqual(n, 3)
        self.assertAlmostEqual(ref, math.log(100))
        snap.pop("SOL")
        self.assertIsNone(reference_log_ratio(snap, "BTC")[0])

    def test_09_scaled_contract_normalization(self):
        a = make_quote(.010, .011, 2, 3, 0, mult=1000)
        self.assertAlmostEqual(a.bid, .00001)
        self.assertEqual(a.bid_size, 2000)
        self.assertAlmostEqual(a.ask * a.ask_size, .011 * 3)

    def test_10_exit_liquidity_cannot_change_entry_quantity(self):
        a, x = row(1000), row(2000)
        normal = replay(event([a, x]), 0, 1000, 10100, 1, 0)
        xsmall = row(2000, up=q(n=1, t=2000))
        small = replay(event([a, xsmall]), 0, 1000, 10100, 1, 0)
        self.assertEqual(normal["qty"], small["qty"])
        self.assertEqual(small["qty"], 100)
        self.assertFalse(small["closed"])
        self.assertEqual(small["status"], "exit_liquidity_unresolved")

    def test_11_unresolved_is_kept_in_denominator(self):
        r = replay(event([row(1000), row(2000, up=q(n=1, t=2000))]), 0, 1000, 10100, 1, 0)
        st = new_stats()
        update_stats(st, r)
        self.assertEqual(st["entered"], 1)
        self.assertEqual(st["unresolved"], 1)
        self.assertEqual(st["closed"], 0)
        self.assertNotEqual(st["all_marked_pnl_krw"], 0)

    def test_12_missing_exit_not_dropped(self):
        r = replay(event([row(1000)]), 0, 1000, 10100, 1, 0)
        self.assertTrue(r["entered"])
        self.assertEqual(r["status"], "exit_data_unresolved")

    def test_13_missing_fx_retains_position(self):
        a, x = row(1000), row(2000)
        x["fx"] = None
        r = replay(event([a, x]), 0, 1000, 10100, 1, 0)
        self.assertTrue(r["entered"])
        self.assertEqual(r["status"], "exit_fx_unresolved")

    def test_14_fixed_cost_sensitivity(self):
        ev = event([row(1000), row(2000)])
        pnls = [replay(ev, 0, 1000, 10100, 1, cost)["pnl_krw"] for cost in (0, 2, 5)]
        self.assertGreater(pnls[0], pnls[1])
        self.assertGreater(pnls[1], pnls[2])

    def test_15_fee_identity(self):
        a, x = row(1000), row(2000)
        expected_per_coin = (99 * .9995 - 101 * 1.0005) + (.99 * .9995 - 1.01 * 1.0005) * 100
        self.assertAlmostEqual(pnl_at(a, x, 100, 0), expected_per_coin * 100)

    def test_16_arrival_depth_can_limit_fill(self):
        a = row(1000)
        b = row(1200, bn=q(.99, 1.01, n=7, t=1200))
        self.assertEqual(entry_quantity(a, b, 10100, 1), 7)

    def test_17_participation_haircut(self):
        a = row(1000)
        self.assertEqual(entry_quantity(a, a, 1000000, .25), 250)

    def test_18_future_1200ms_snap_is_not_execution(self):
        r = replay(event([row(1000), row(3000)]), 200, 1000, 10100, 1, 0)
        self.assertFalse(r["entered"])
        self.assertEqual(r["status"], "entry_data_missing")

    def test_19_exit_order_latency_is_included(self):
        rows = [row(t) for t in range(1000, 2601, 100)]
        r = replay(event(rows), 200, 1000, 10100, 1, 0)
        self.assertEqual(r["entry_time_ms"], 1200)
        self.assertEqual(r["exit_time_ms"], 2400)

    def test_20_dynamic_stop_uses_executable_net_pnl(self):
        rows = [row(t) for t in range(1000, 1801, 100)]
        r = replay(event(rows), 200, "dynamic", 10100, 1, 0)
        self.assertEqual(r["exit_reason"], "stop")
        self.assertEqual(r["exit_time_ms"], 1500)

    def test_21_divergence_quarantine(self):
        e = Engine()
        s = event([row(1000)])["signal"]
        self.assertTrue(e.candidate(s))
        s["diagnostic"]["relative_divergence_ok"] = False
        self.assertFalse(e.candidate(s))

    def test_22_negative_direction_requires_inventory(self):
        s = event([row(1000)])["signal"]
        s["bn_bp"] = -50
        s["lag_bp"] = -20
        self.assertFalse(Engine().candidate(s))

    def test_23_pack_digest_round_trip(self):
        data = zlib.compress(json.dumps({"a": [1, 2, 3]}).encode())
        encoded = base64.b64encode(data).decode()
        restored = base64.b64decode(encoded)
        self.assertEqual(hashlib.sha256(data).digest(), hashlib.sha256(restored).digest())
        self.assertEqual(json.loads(zlib.decompress(restored))["a"], [1, 2, 3])

    def test_24_nonfinite_quotes_rejected(self):
        self.assertFalse(quote_ok(replace(q(), bid=float("nan")), 1000))

    def test_25_dynamic_timeout_not_backdated(self):
        def tight(t):
            return row(t, up=q(100, 100.01, t=t), bn=q(1, 1.0001, t=t))
        rows = [tight(t) for t in range(1000, 11401, 100)]
        r = replay(event(rows), 200, "dynamic", 10000, 1, 0)
        self.assertEqual(r["exit_reason"], "timeout")
        self.assertEqual(r["exit_time_ms"], 11400)
        self.assertTrue(r["closed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
