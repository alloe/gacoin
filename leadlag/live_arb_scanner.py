#!/usr/bin/env python3
import asyncio, json, math, os, time
from collections import defaultdict, deque
import websockets

UPBIT_WS="wss://api.upbit.com/websocket/v1"
BITHUMB_WS="wss://ws-api.bithumb.com/websocket/v1"
BINANCE_WS=os.getenv("BINANCE_WS","wss://data-stream.binance.vision/ws/btcusdt@bookTicker")

UPBIT_FEE_BP=float(os.getenv("UPBIT_FEE_BP","5"))
BITHUMB_COUPON_FEE_BP=float(os.getenv("BITHUMB_COUPON_FEE_BP","4"))
BITHUMB_BASE_FEE_BP=float(os.getenv("BITHUMB_BASE_FEE_BP","25"))
BINANCE_FEE_BP=float(os.getenv("BINANCE_FEE_BP","10"))
SUMMARY_SEC=int(os.getenv("SUMMARY_SEC","60"))
MAX_STALE_MS=float(os.getenv("MAX_STALE_MS","1000"))
PAPER_QTY_BTC=float(os.getenv("PAPER_QTY_BTC","0.01"))
LAT_MS=[50,100,200,500]

quotes={}
stats=defaultdict(lambda:{"checks":0,"pos":0,"max_bp":-1e9,"sum_pos_bp":0.0,"episodes":0})
active={}
pending=deque()
start_mono=time.monotonic()
last_summary=start_mono

def now_ns(): return time.time_ns()

def put(key,bid,bidsz,ask,asksz,src_ts=None):
    quotes[key]={"bid":float(bid),"bidsz":float(bidsz),"ask":float(ask),"asksz":float(asksz),
                 "recv_ns":now_ns(),"src_ts":src_ts}

def fresh(key,now):
    q=quotes.get(key)
    return q and (now-q["recv_ns"])/1e6<=MAX_STALE_MS

def net_edge_bp(sell_bid,buy_ask,sell_fee_bp,buy_fee_bp):
    sf=sell_fee_bp/10000;bf=buy_fee_bp/10000
    return math.log((sell_bid*(1-sf))/(buy_ask*(1+bf)))*10000

def calc(now):
    out=[]
    # Upbit <-> Bithumb, two fee scenarios for Bithumb
    if fresh("UP_BTC",now) and fresh("BH_BTC",now):
        u=quotes["UP_BTC"];b=quotes["BH_BTC"]
        for label,bhf in [("coupon",BITHUMB_COUPON_FEE_BP),("base",BITHUMB_BASE_FEE_BP)]:
            e1=net_edge_bp(b["bid"],u["ask"],bhf,UPBIT_FEE_BP)
            e2=net_edge_bp(u["bid"],b["ask"],UPBIT_FEE_BP,bhf)
            out.append((f"UPbuy_BHsell_{label}",e1,min(u["asksz"],b["bidsz"]),b["bid"]*(1-bhf/10000)-u["ask"]*(1+UPBIT_FEE_BP/10000)))
            out.append((f"BHbuy_UPsell_{label}",e2,min(b["asksz"],u["bidsz"]),u["bid"]*(1-UPBIT_FEE_BP/10000)-b["ask"]*(1+bhf/10000)))
    # Binance conversions using Upbit KRW-USDT top of book
    if fresh("UP_USDT",now) and fresh("BN_BTC",now):
        fx=quotes["UP_USDT"];bn=quotes["BN_BTC"]
        if fresh("UP_BTC",now):
            u=quotes["UP_BTC"]
            bn_sell_krw=bn["bid"]*fx["bid"];bn_buy_krw=bn["ask"]*fx["ask"]
            e1=net_edge_bp(bn_sell_krw,u["ask"],BINANCE_FEE_BP,UPBIT_FEE_BP)
            e2=net_edge_bp(u["bid"],bn_buy_krw,UPBIT_FEE_BP,BINANCE_FEE_BP)
            out.append(("UPbuy_BNsell",e1,min(u["asksz"],bn["bidsz"]),bn_sell_krw*(1-BINANCE_FEE_BP/10000)-u["ask"]*(1+UPBIT_FEE_BP/10000)))
            out.append(("BNbuy_UPsell",e2,min(bn["asksz"],u["bidsz"]),u["bid"]*(1-UPBIT_FEE_BP/10000)-bn_buy_krw*(1+BINANCE_FEE_BP/10000)))
        if fresh("BH_BTC",now):
            b=quotes["BH_BTC"];bn_sell_krw=bn["bid"]*fx["bid"];bn_buy_krw=bn["ask"]*fx["ask"]
            for label,bhf in [("coupon",BITHUMB_COUPON_FEE_BP),("base",BITHUMB_BASE_FEE_BP)]:
                e1=net_edge_bp(bn_sell_krw,b["ask"],BINANCE_FEE_BP,bhf)
                e2=net_edge_bp(b["bid"],bn_buy_krw,bhf,BINANCE_FEE_BP)
                out.append((f"BHbuy_BNsell_{label}",e1,min(b["asksz"],bn["bidsz"]),bn_sell_krw*(1-BINANCE_FEE_BP/10000)-b["ask"]*(1+bhf/10000)))
                out.append((f"BNbuy_BHsell_{label}",e2,min(bn["asksz"],b["bidsz"]),b["bid"]*(1-bhf/10000)-bn_buy_krw*(1+BINANCE_FEE_BP/10000)))
    return out

def process_edges():
    global last_summary
    n=now_ns();mono=time.monotonic()
    edges={name:(bp,cap,krw) for name,bp,cap,krw in calc(n)}
    for name,(bp,cap,krw) in edges.items():
        s=stats[name];s["checks"]+=1
        if bp>0:
            s["pos"]+=1;s["sum_pos_bp"]+=bp;s["max_bp"]=max(s["max_bp"],bp)
            if name not in active:
                active[name]={"start_ns":n,"start_bp":bp,"max_bp":bp,"cap":cap}
                s["episodes"]+=1
                print("EDGE_START",json.dumps({"route":name,"bp":bp,"cap_btc":cap,
                    "net_krw_per_btc":krw,"paper_net_krw":krw*min(cap,PAPER_QTY_BTC)},separators=(",",":")),flush=True)
                for lat in LAT_MS: pending.append((n+lat*1_000_000,name,lat))
            else:
                active[name]["max_bp"]=max(active[name]["max_bp"],bp)
        elif name in active:
            a=active.pop(name)
            print("EDGE_END",json.dumps({"route":name,"duration_ms":(n-a["start_ns"])/1e6,
                "start_bp":a["start_bp"],"max_bp":a["max_bp"]},separators=(",",":")),flush=True)
    while pending and pending[0][0]<=n:
        _,name,lat=pending.popleft()
        if name in edges:
            bp,cap,krw=edges[name]
            print("LATENCY",json.dumps({"route":name,"lat_ms":lat,"bp":bp,"positive":bp>0,
                "cap_btc":cap,"paper_net_krw":krw*min(cap,PAPER_QTY_BTC)},separators=(",",":")),flush=True)
    if mono-last_summary>=SUMMARY_SEC:
        payload={}
        for k,s in stats.items():
            payload[k]={"episodes":s["episodes"],"positive_checks":s["pos"],
                "max_bp":None if s["max_bp"]<-1e8 else s["max_bp"],
                "mean_positive_bp":(s["sum_pos_bp"]/s["pos"] if s["pos"] else None)}
        print("SUMMARY",json.dumps({"uptime_sec":mono-start_mono,"routes":payload,
            "ages_ms":{k:(n-v["recv_ns"])/1e6 for k,v in quotes.items()}},separators=(",",":")),flush=True)
        last_summary=mono

async def upbit():
    sub=[{"ticket":"arb-live"},{"type":"orderbook","codes":["KRW-BTC.1","KRW-USDT.1"],"is_only_realtime":True},{"format":"DEFAULT"}]
    while True:
        try:
            async with websockets.connect(UPBIT_WS,ping_interval=20,ping_timeout=20,max_size=2**20) as ws:
                await ws.send(json.dumps(sub))
                print("CONNECTED upbit",flush=True)
                async for raw in ws:
                    d=json.loads(raw)
                    code=d.get("code","").split(".")[0];units=d.get("orderbook_units") or []
                    if not units: continue
                    u=units[0]
                    if code=="KRW-BTC": put("UP_BTC",u["bid_price"],u["bid_size"],u["ask_price"],u["ask_size"],d.get("timestamp"))
                    elif code=="KRW-USDT": put("UP_USDT",u["bid_price"],u["bid_size"],u["ask_price"],u["ask_size"],d.get("timestamp"))
                    process_edges()
        except Exception as e:
            print("WS_ERR upbit",repr(e),flush=True);await asyncio.sleep(1)

async def bithumb():
    sub=[{"ticket":"arb-live"},{"type":"orderbook","codes":["KRW-BTC"],"isOnlyRealtime":True},{"format":"DEFAULT"}]
    while True:
        try:
            async with websockets.connect(BITHUMB_WS,ping_interval=20,ping_timeout=20,max_size=2**20) as ws:
                await ws.send(json.dumps(sub))
                print("CONNECTED bithumb",flush=True)
                async for raw in ws:
                    d=json.loads(raw);units=d.get("orderbook_units") or []
                    if d.get("code")=="KRW-BTC" and units:
                        u=units[0];put("BH_BTC",u["bid_price"],u["bid_size"],u["ask_price"],u["ask_size"],d.get("timestamp"));process_edges()
        except Exception as e:
            print("WS_ERR bithumb",repr(e),flush=True);await asyncio.sleep(1)

async def binance():
    while True:
        try:
            async with websockets.connect(BINANCE_WS,ping_interval=20,ping_timeout=20,max_size=2**20) as ws:
                print("CONNECTED binance",flush=True)
                async for raw in ws:
                    d=json.loads(raw)
                    if d.get("s")=="BTCUSDT" or {"b","B","a","A"}.issubset(d):
                        put("BN_BTC",d["b"],d["B"],d["a"],d["A"],d.get("E"));process_edges()
        except Exception as e:
            print("WS_ERR binance",repr(e),flush=True);await asyncio.sleep(1)

async def heartbeat():
    while True:
        await asyncio.sleep(5);process_edges()

async def main():
    print("LIVE_ARB_START",json.dumps({"upbit_fee_bp":UPBIT_FEE_BP,"bithumb_coupon_bp":BITHUMB_COUPON_FEE_BP,
      "bithumb_base_bp":BITHUMB_BASE_FEE_BP,"binance_fee_bp":BINANCE_FEE_BP,"paper_qty_btc":PAPER_QTY_BTC}),flush=True)
    await asyncio.gather(upbit(),bithumb(),binance(),heartbeat())

if __name__=="__main__":
    asyncio.run(main())
