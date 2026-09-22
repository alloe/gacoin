#!/usr/bin/env python3
import asyncio, json, math, os, time, heapq
from collections import defaultdict
import websockets

UPBIT_WS="wss://api.upbit.com/websocket/v1"
BITHUMB_WS="wss://ws-api.bithumb.com/websocket/v1"
BINANCE_WS=os.getenv("BINANCE_WS","wss://data-stream.binance.vision/ws/btcusdt@bookTicker")

UPBIT_FEE_BP=float(os.getenv("UPBIT_FEE_BP","5"))
BITHUMB_COUPON_FEE_BP=float(os.getenv("BITHUMB_COUPON_FEE_BP","4"))
BITHUMB_BASE_FEE_BP=float(os.getenv("BITHUMB_BASE_FEE_BP","25"))
BITHUMB_FREE_FEE_BP=float(os.getenv("BITHUMB_FREE_FEE_BP","0"))
BINANCE_FEE_BP=float(os.getenv("BINANCE_FEE_BP","10"))
BINANCE_BNB_FEE_BP=float(os.getenv("BINANCE_BNB_FEE_BP","7.5"))
SUMMARY_SEC=int(os.getenv("SUMMARY_SEC","60"))
BUCKET_SUMMARY_SEC=int(os.getenv("BUCKET_SUMMARY_SEC","600"))
MAX_STALE_MS=float(os.getenv("MAX_STALE_MS","1000"))
PAPER_QTY_BTC=float(os.getenv("PAPER_QTY_BTC","0.01"))
LAT_MS=[50,100,200,500]
EDGE_BUCKETS=[(0.0,0.5,"0-0.5"),(0.5,1.0,"0.5-1"),(1.0,1.5,"1-1.5"),(1.5,2.0,"1.5-2"),(2.0,3.0,"2-3"),(3.0,None,"3+")]
EDGE_CUTOFFS=[0.5,1.0,1.5,2.0,3.0]

quotes={}
stats=defaultdict(lambda:{"checks":0,"pos":0,"max_bp":-1e9,"sum_pos_bp":0.0,"episodes":0,
                          "start_bp_sum":0.0,"start_paper_sum":0.0,"max_profit_sum":0.0,
                          "cap_ge_0001":0,"cap_ge_0005":0,"cap_ge_001":0,"cap_ge_005":0})
lat_stats=defaultdict(lambda:{"n":0,"pos":0,"sum_bp":0.0,"sum_paper":0.0,"max_bp":-1e9})
bucket_stats=defaultdict(lambda:{"episodes":0,"start_bp_sum":0.0,"start_paper_sum":0.0,
                                 "cap_ge_0001":0,"cap_ge_0005":0,"cap_ge_001":0,"cap_ge_005":0})
bucket_lat_stats=defaultdict(lambda:{"n":0,"pos":0,"sum_bp":0.0,"sum_paper":0.0,"max_bp":-1e9})
active={}
pending=[]
pending_seq=0
start_mono=time.monotonic()
last_summary=start_mono
last_bucket_summary=start_mono

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

def edge_bucket(bp):
    for lo,hi,label in EDGE_BUCKETS:
        if bp>=lo and (hi is None or bp<hi):
            return label
    return "3+"

def aggregate_bucket_view(route,labels,uptime_h):
    ep=0; start_bp_sum=0.0; start_paper_sum=0.0
    caps={"cap_ge_0001":0,"cap_ge_0005":0,"cap_ge_001":0,"cap_ge_005":0}
    for label in labels:
        b=bucket_stats[(route,label)]
        ep+=b["episodes"]; start_bp_sum+=b["start_bp_sum"]; start_paper_sum+=b["start_paper_sum"]
        for k in caps: caps[k]+=b[k]
    lats={}
    for lat in LAT_MS:
        n=pos=0; sum_bp=sum_paper=0.0; max_bp=-1e9
        for label in labels:
            z=bucket_lat_stats[(route,label,lat)]
            n+=z["n"]; pos+=z["pos"]; sum_bp+=z["sum_bp"]; sum_paper+=z["sum_paper"]; max_bp=max(max_bp,z["max_bp"])
        lats[str(lat)]={"n":n,"pos_rate":(pos/n if n else None),
            "mean_bp":(sum_bp/n if n else None),"mean_paper_krw":(sum_paper/n if n else None),
            "sum_paper_krw":sum_paper,"paper_krw_per_hour":(sum_paper/uptime_h if uptime_h>0 else None),
            "max_bp":None if max_bp<-1e8 else max_bp}
    return {"episodes":ep,"events_per_hour":(ep/uptime_h if uptime_h>0 else None),
        "mean_start_bp":(start_bp_sum/ep if ep else None),
        "mean_start_paper_krw":(start_paper_sum/ep if ep else None),
        "cap_ge_0001_rate":(caps["cap_ge_0001"]/ep if ep else None),
        "cap_ge_0005_rate":(caps["cap_ge_0005"]/ep if ep else None),
        "cap_ge_001_rate":(caps["cap_ge_001"]/ep if ep else None),
        "cap_ge_005_rate":(caps["cap_ge_005"]/ep if ep else None),
        "latency":lats}

def make_bucket_summary(uptime_h):
    routes={}
    route_names=sorted({route for route,_ in bucket_stats.keys()})
    all_labels=[label for _,_,label in EDGE_BUCKETS]
    for route in route_names:
        if sum(bucket_stats[(route,label)]["episodes"] for label in all_labels)==0:
            continue
        bands={}
        for _,_,label in EDGE_BUCKETS:
            if bucket_stats[(route,label)]["episodes"]:
                bands[label]=aggregate_bucket_view(route,[label],uptime_h)
        cutoffs={}
        for cutoff in EDGE_CUTOFFS:
            labels=[label for lo,_,label in EDGE_BUCKETS if lo>=cutoff]
            cutoffs[f"ge_{cutoff:g}"]=aggregate_bucket_view(route,labels,uptime_h)
        routes[route]={"bands":bands,"cutoffs":cutoffs}
    return routes

def calc(now):
    out=[]
    # Upbit <-> Bithumb, two fee scenarios for Bithumb
    if fresh("UP_BTC",now) and fresh("BH_BTC",now):
        u=quotes["UP_BTC"];b=quotes["BH_BTC"]
        for label,bhf in [("coupon",BITHUMB_COUPON_FEE_BP),("base",BITHUMB_BASE_FEE_BP),("free",BITHUMB_FREE_FEE_BP)]:
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
            for suffix,bnf in [("",BINANCE_FEE_BP),("_bnb",BINANCE_BNB_FEE_BP)]:
                e1=net_edge_bp(bn_sell_krw,u["ask"],bnf,UPBIT_FEE_BP)
                e2=net_edge_bp(u["bid"],bn_buy_krw,UPBIT_FEE_BP,bnf)
                out.append((f"UPbuy_BNsell{suffix}",e1,min(u["asksz"],bn["bidsz"]),bn_sell_krw*(1-bnf/10000)-u["ask"]*(1+UPBIT_FEE_BP/10000)))
                out.append((f"BNbuy_UPsell{suffix}",e2,min(bn["asksz"],u["bidsz"]),u["bid"]*(1-UPBIT_FEE_BP/10000)-bn_buy_krw*(1+bnf/10000)))
        if fresh("BH_BTC",now):
            b=quotes["BH_BTC"];bn_sell_krw=bn["bid"]*fx["bid"];bn_buy_krw=bn["ask"]*fx["ask"]
            for label,bhf in [("coupon",BITHUMB_COUPON_FEE_BP),("base",BITHUMB_BASE_FEE_BP),("free",BITHUMB_FREE_FEE_BP)]:
                e1=net_edge_bp(bn_sell_krw,b["ask"],BINANCE_FEE_BP,bhf)
                e2=net_edge_bp(b["bid"],bn_buy_krw,bhf,BINANCE_FEE_BP)
                out.append((f"BHbuy_BNsell_{label}",e1,min(b["asksz"],bn["bidsz"]),bn_sell_krw*(1-BINANCE_FEE_BP/10000)-b["ask"]*(1+bhf/10000)))
                out.append((f"BNbuy_BHsell_{label}",e2,min(bn["asksz"],b["bidsz"]),b["bid"]*(1-bhf/10000)-bn_buy_krw*(1+BINANCE_FEE_BP/10000)))
            for label,bhf in [("coupon",BITHUMB_COUPON_FEE_BP),("free",BITHUMB_FREE_FEE_BP)]:
                bnf=BINANCE_BNB_FEE_BP
                e1=net_edge_bp(bn_sell_krw,b["ask"],bnf,bhf)
                e2=net_edge_bp(b["bid"],bn_buy_krw,bhf,bnf)
                out.append((f"BHbuy_BNsell_{label}_bnb",e1,min(b["asksz"],bn["bidsz"]),bn_sell_krw*(1-bnf/10000)-b["ask"]*(1+bhf/10000)))
                out.append((f"BNbuy_BHsell_{label}_bnb",e2,min(bn["asksz"],b["bidsz"]),b["bid"]*(1-bhf/10000)-bn_buy_krw*(1+bnf/10000)))
    return out

def process_edges():
    global last_summary, last_bucket_summary, pending_seq
    n=now_ns();mono=time.monotonic()
    edges={name:(bp,cap,krw) for name,bp,cap,krw in calc(n)}
    for name,(bp,cap,krw) in edges.items():
        s=stats[name];s["checks"]+=1
        if bp>0:
            s["pos"]+=1;s["sum_pos_bp"]+=bp;s["max_bp"]=max(s["max_bp"],bp)
            if name not in active:
                bucket=edge_bucket(bp)
                active[name]={"start_ns":n,"start_bp":bp,"max_bp":bp,"cap":cap,"bucket":bucket}
                s["episodes"]+=1
                paper=krw*min(cap,PAPER_QTY_BTC)
                maxprofit=krw*cap
                s["start_bp_sum"]+=bp; s["start_paper_sum"]+=paper; s["max_profit_sum"]+=maxprofit
                if cap>=0.001: s["cap_ge_0001"]+=1
                if cap>=0.005: s["cap_ge_0005"]+=1
                if cap>=0.01: s["cap_ge_001"]+=1
                if cap>=0.05: s["cap_ge_005"]+=1
                bs=bucket_stats[(name,bucket)]
                bs["episodes"]+=1; bs["start_bp_sum"]+=bp; bs["start_paper_sum"]+=paper
                if cap>=0.001: bs["cap_ge_0001"]+=1
                if cap>=0.005: bs["cap_ge_0005"]+=1
                if cap>=0.01: bs["cap_ge_001"]+=1
                if cap>=0.05: bs["cap_ge_005"]+=1
                print("EDGE_START",json.dumps({"route":name,"bp":bp,"bucket":bucket,"cap_btc":cap,
                    "net_krw_per_btc":krw,"paper_net_krw":paper,"max_top_level_profit_krw":maxprofit},separators=(",",":")),flush=True)
                for lat in LAT_MS:
                    pending_seq+=1
                    heapq.heappush(pending,(n+lat*1_000_000,pending_seq,n,name,lat,bucket))
            else:
                active[name]["max_bp"]=max(active[name]["max_bp"],bp)
        elif name in active:
            a=active.pop(name)
            print("EDGE_END",json.dumps({"route":name,"duration_ms":(n-a["start_ns"])/1e6,
                "start_bp":a["start_bp"],"max_bp":a["max_bp"]},separators=(",",":")),flush=True)
    if mono-last_summary>=SUMMARY_SEC:
        payload={}
        for k,s in stats.items():
            ep=s["episodes"]
            lats={}
            for lat in LAT_MS:
                z=lat_stats[(k,lat)]
                lats[str(lat)]={"n":z["n"],"pos_rate":(z["pos"]/z["n"] if z["n"] else None),
                    "mean_bp":(z["sum_bp"]/z["n"] if z["n"] else None),
                    "mean_paper_krw":(z["sum_paper"]/z["n"] if z["n"] else None),
                    "max_bp":None if z["max_bp"]<-1e8 else z["max_bp"]}
            payload[k]={"episodes":ep,"positive_checks":s["pos"],
                "max_bp":None if s["max_bp"]<-1e8 else s["max_bp"],
                "mean_positive_bp":(s["sum_pos_bp"]/s["pos"] if s["pos"] else None),
                "mean_start_bp":(s["start_bp_sum"]/ep if ep else None),
                "mean_start_paper_krw":(s["start_paper_sum"]/ep if ep else None),
                "mean_max_top_level_profit_krw":(s["max_profit_sum"]/ep if ep else None),
                "cap_ge_0001":s["cap_ge_0001"],"cap_ge_0005":s["cap_ge_0005"],
                "cap_ge_001":s["cap_ge_001"],"cap_ge_005":s["cap_ge_005"],
                "latency":lats}
        print("SUMMARY",json.dumps({"uptime_sec":mono-start_mono,"routes":payload,
            "ages_ms":{k:(n-v["recv_ns"])/1e6 for k,v in quotes.items()}},separators=(",",":")),flush=True)
        last_summary=mono
    if mono-last_bucket_summary>=BUCKET_SUMMARY_SEC:
        uptime_h=(mono-start_mono)/3600.0
        print("BUCKET_SUMMARY",json.dumps({"uptime_sec":mono-start_mono,"routes":make_bucket_summary(uptime_h),
            "fee_bp":{"upbit":UPBIT_FEE_BP,"bithumb_coupon":BITHUMB_COUPON_FEE_BP,
                      "bithumb_free":BITHUMB_FREE_FEE_BP,"binance":BINANCE_FEE_BP,
                      "binance_bnb":BINANCE_BNB_FEE_BP}},separators=(",",":")),flush=True)
        last_bucket_summary=mono

async def latency_worker():
    while True:
        await asyncio.sleep(0.005)
        n=now_ns()
        while pending and pending[0][0]<=n:
            due,_,start_ns,name,lat,bucket=heapq.heappop(pending)
            edges={route:(bp,cap,krw) for route,bp,cap,krw in calc(n)}
            if name not in edges:
                continue
            bp,cap,krw=edges[name]
            actual=(n-start_ns)/1e6
            paper=krw*min(cap,PAPER_QTY_BTC)
            z=lat_stats[(name,lat)]
            z["n"]+=1; z["pos"]+=1 if bp>0 else 0; z["sum_bp"]+=bp; z["sum_paper"]+=paper; z["max_bp"]=max(z["max_bp"],bp)
            bz=bucket_lat_stats[(name,bucket,lat)]
            bz["n"]+=1; bz["pos"]+=1 if bp>0 else 0; bz["sum_bp"]+=bp; bz["sum_paper"]+=paper; bz["max_bp"]=max(bz["max_bp"],bp)
            print("LATENCY",json.dumps({"route":name,"lat_ms":lat,"actual_ms":actual,"start_bucket":bucket,"bp":bp,
                "positive":bp>0,"cap_btc":cap,"paper_net_krw":paper},separators=(",",":")),flush=True)

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
    print("LIVE_ARB_START",json.dumps({"mode":"paper_only_no_order_api","upbit_fee_bp":UPBIT_FEE_BP,
      "bithumb_coupon_bp":BITHUMB_COUPON_FEE_BP,"bithumb_base_bp":BITHUMB_BASE_FEE_BP,
      "bithumb_free_bp":BITHUMB_FREE_FEE_BP,"binance_fee_bp":BINANCE_FEE_BP,
      "binance_bnb_fee_bp":BINANCE_BNB_FEE_BP,"paper_qty_btc":PAPER_QTY_BTC,
      "bucket_summary_sec":BUCKET_SUMMARY_SEC}),flush=True)
    await asyncio.gather(upbit(),bithumb(),binance(),heartbeat(),latency_worker())

if __name__=="__main__":
    asyncio.run(main())
