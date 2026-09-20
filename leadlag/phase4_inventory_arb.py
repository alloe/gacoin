#!/usr/bin/env python3
import csv,gzip,io,json,math,os,statistics
from datetime import datetime
import requests,numpy as np

BASE="https://datasets.tardis.dev/v1"
DATES=os.getenv("DATES","2026-01-01").split(",")
STEP_US=100_000
UP_FEE_BP=float(os.getenv("UP_FEE_BP","5"))
BN_FEE_BP=float(os.getenv("BN_FEE_BP","10"))
TIMEOUT=(20,600)
LATS=[0,100,300,500,1000]
FEE_SCENARIOS=[0,5,10,12.5,15,20]

def log(m,**kw): print(m+((" "+json.dumps(kw,ensure_ascii=False,sort_keys=True)) if kw else ""),flush=True)
def ts_us(v):
    s=str(v).strip()
    try:
        x=int(float(s))
        if x<10_000_000_000:return x*1_000_000
        if x<10_000_000_000_000:return x*1_000
        if x>10_000_000_000_000_000:return x//1000
        return x
    except:return int(datetime.fromisoformat(s.replace("Z","+00:00")).timestamp()*1_000_000)
def url(ex,date,sym):
    y,m,d=date.split("-");return f"{BASE}/{ex}/quotes/{y}/{m}/{d}/{sym}.csv.gz"
def load(ex,date,sym):
    u=url(ex,date,sym);r=requests.get(u,stream=True,timeout=TIMEOUT)
    if r.status_code!=200:raise RuntimeError(f"HTTP {r.status_code} {u}: {r.text[:120]}")
    r.raw.decode_content=False
    rd=csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw),encoding="utf-8",newline=""))
    need={"local_timestamp","bid_price","bid_amount","ask_price","ask_amount"}
    if not need.issubset(set(rd.fieldnames or [])):raise RuntimeError(str(rd.fieldnames))
    ts=[];bp=[];ba=[];ap=[];aa=[];cur=None;v=None;rows=0
    for row in rd:
        rows+=1
        try:
            t=ts_us(row["local_timestamp"]);b=float(row["bid_price"]);bs=float(row["bid_amount"]);a=float(row["ask_price"]);az=float(row["ask_amount"])
            if not(b>0 and a>=b and bs>=0 and az>=0):continue
        except:continue
        buck=(t//STEP_US)*STEP_US;nv=(b,bs,a,az)
        if cur is None:cur=buck;v=nv
        elif buck==cur:v=nv
        else:ts.append(cur);bp.append(v[0]);ba.append(v[1]);ap.append(v[2]);aa.append(v[3]);cur=buck;v=nv
    if cur is not None:ts.append(cur);bp.append(v[0]);ba.append(v[1]);ap.append(v[2]);aa.append(v[3])
    log("AGG",exchange=ex,date=date,symbol=sym,rows=rows,buckets=len(ts))
    return np.array(ts,np.int64),np.array(bp,float),np.array(ba,float),np.array(ap,float),np.array(aa,float)
def grid(q,start,end):
    ts,bp,ba,ap,aa=q;n=int((end-start)//STEP_US)+1
    outs=[np.full(n,np.nan) for _ in range(4)]
    idx=((ts-start)//STEP_US).astype(np.int64);ok=(idx>=0)&(idx<n);ii=idx[ok]
    for o,v in zip(outs,[bp,ba,ap,aa]):o[ii]=v[ok]
    valid=np.where(np.isfinite(outs[0])&np.isfinite(outs[2]),np.arange(n),-1)
    np.maximum.accumulate(valid,out=valid);m=valid>=0
    for o in outs:o[m]=o[valid[m]]
    return outs
def episode_stats(edge,cap,netkrw,start_us):
    eps=[];i=0;n=len(edge)
    while i<n:
        if not math.isfinite(edge[i]) or edge[i]<=0:i+=1;continue
        s=i;mx=edge[i];mi=i
        while i+1<n and math.isfinite(edge[i+1]) and edge[i+1]>0:
            i+=1
            if edge[i]>mx:mx=edge[i];mi=i
        e=i;eps.append({
          "start_i":s,"end_i":e,"duration_ms":(e-s+1)*100,
          "start_edge_bp":float(edge[s]),"max_edge_bp":float(mx),
          "start_capacity_btc":float(cap[s]) if math.isfinite(cap[s]) else 0.0,
          "maxedge_capacity_btc":float(cap[mi]) if math.isfinite(cap[mi]) else 0.0,
          "start_net_krw_per_btc":float(netkrw[s]) if math.isfinite(netkrw[s]) else None
        });i+=1
    return eps
def mean(a):return None if not a else statistics.fmean(a)
def med(a):return None if not a else statistics.median(a)
def pctl(a,q):return None if not a else float(np.percentile(a,q))
def summarize_eps(eps):
    if not eps:return {"episodes":0}
    return {"episodes":len(eps),"median_duration_ms":med([x["duration_ms"] for x in eps]),
      "p90_duration_ms":pctl([x["duration_ms"] for x in eps],90),
      "median_start_edge_bp":med([x["start_edge_bp"] for x in eps]),
      "median_max_edge_bp":med([x["max_edge_bp"] for x in eps]),
      "max_edge_bp":max(x["max_edge_bp"] for x in eps),
      "median_start_capacity_btc":med([x["start_capacity_btc"] for x in eps]),
      "median_net_krw_per_btc":med([x["start_net_krw_per_btc"] for x in eps if x["start_net_krw_per_btc"] is not None])}
def process(date):
    ub=load("upbit",date,"KRW-BTC");uu=load("upbit",date,"KRW-USDT");bn=load("binance",date,"BTCUSDT")
    start=max(ub[0][0],uu[0][0],bn[0][0]);end=min(ub[0][-1],uu[0][-1],bn[0][-1])
    ubb,ubsz,uba,uasz=grid(ub,start,end)
    usb,usbsz,usa,usasz=grid(uu,start,end)
    bnb,bnbsz,bna,bnasz=grid(bn,start,end)
    # raw executable cross spreads before trading fees, conservative KRW valuation using USDT bid/ask.
    sell_global_krw=bnb*usb
    buy_global_krw=bna*usa
    raw_A=np.log(sell_global_krw/uba)*10000  # buy Upbit BTC ask, sell Binance BTC bid
    raw_B=np.log(ubb/buy_global_krw)*10000   # buy Binance BTC ask, sell Upbit BTC bid
    capA=np.minimum(uasz,bnbsz);capB=np.minimum(ubsz,bnasz)
    # standard account fees: precise multiplicative approximation
    upf=UP_FEE_BP/10000;bnf=BN_FEE_BP/10000
    netkrwA=sell_global_krw*(1-bnf)-uba*(1+upf)
    netkrwB=ubb*(1-upf)-buy_global_krw*(1+bnf)
    stdA=np.log((sell_global_krw*(1-bnf))/(uba*(1+upf)))*10000
    stdB=np.log((ubb*(1-upf))/(buy_global_krw*(1+bnf)))*10000
    out={"date":date,"seconds":len(raw_A)/10,"scenarios":{},"latency":{}}
    for fee in FEE_SCENARIOS:
        ea=raw_A-fee;eb=raw_B-fee
        out["scenarios"][str(fee)]={"A":summarize_eps(episode_stats(ea,capA,netkrwA,start)),
                                    "B":summarize_eps(episode_stats(eb,capB,netkrwB,start))}
    # standard-fee signal persistence: detect raw edge > standard total at t, inspect executable standard edge after latency
    triggerA=raw_A-(UP_FEE_BP+BN_FEE_BP);triggerB=raw_B-(UP_FEE_BP+BN_FEE_BP)
    for lat in LATS:
        k=lat//100
        valsA=[];valsB=[]
        for tr,std,vals in [(triggerA,stdA,valsA),(triggerB,stdB,valsB)]:
            i=0
            while i<len(tr):
                if math.isfinite(tr[i]) and tr[i]>0:
                    j=i+k
                    if j<len(std) and math.isfinite(std[j]):vals.append(float(std[j]))
                    while i+1<len(tr) and math.isfinite(tr[i+1]) and tr[i+1]>0:i+=1
                i+=1
        out["latency"][str(lat)]={"A_n":len(valsA),"A_mean_bp":mean(valsA),"A_median_bp":med(valsA),"A_pos_rate":(sum(x>0 for x in valsA)/len(valsA) if valsA else None),
                                  "B_n":len(valsB),"B_mean_bp":mean(valsB),"B_median_bp":med(valsB),"B_pos_rate":(sum(x>0 for x in valsB)/len(valsB) if valsB else None)}
    log("DAY_DONE",date=date,stdA=out["scenarios"]["15"]["A"]["episodes"],stdB=out["scenarios"]["15"]["B"]["episodes"])
    return out
def combine(days):
    out={"days":len(days),"scenario_episode_counts":{},"latency":{}}
    for f in FEE_SCENARIOS:
        for d in ["A","B"]:
            vals=[x["scenarios"][str(f)][d]["episodes"] for x in days]
            out["scenario_episode_counts"][f"{f}_{d}"]={"total":sum(vals),"days_with_any":sum(v>0 for v in vals),"mean_per_day":mean(vals)}
    for lat in LATS:
        for d in ["A","B"]:
            ns=sum(x["latency"][str(lat)][f"{d}_n"] for x in days)
            # aggregate means approximately weighted by n from day means
            num=sum((x["latency"][str(lat)][f"{d}_mean_bp"] or 0)*x["latency"][str(lat)][f"{d}_n"] for x in days)
            posnum=sum((x["latency"][str(lat)][f"{d}_pos_rate"] or 0)*x["latency"][str(lat)][f"{d}_n"] for x in days)
            out["latency"][f"{lat}_{d}"]={"n":ns,"mean_bp":(num/ns if ns else None),"pos_rate":(posnum/ns if ns else None)}
    return out
def main():
    log("PHASE4_START",dates=DATES,up_fee_bp=UP_FEE_BP,bn_fee_bp=BN_FEE_BP)
    days=[];fails=[]
    for d in DATES:
        try:days.append(process(d))
        except Exception as e:fails.append({"date":d,"error":repr(e)});log("FAIL",date=d,error=repr(e))
    out={"dates":DATES,"failures":fails,"days":days,"combined":combine(days)}
    print("PHASE4_JSON="+json.dumps(out,separators=(",",":")),flush=True)
    c=out["combined"]
    print("PHASE4_SUMMARY",flush=True)
    for f in FEE_SCENARIOS:
        a=c["scenario_episode_counts"][f"{f}_A"];b=c["scenario_episode_counts"][f"{f}_B"]
        print(f"fee={f:>4}bp A episodes={a['total']} days={a['days_with_any']} perDay={a['mean_per_day']:.2f} | B episodes={b['total']} days={b['days_with_any']} perDay={b['mean_per_day']:.2f}",flush=True)
    for lat in LATS:
        a=c["latency"][f"{lat}_A"];b=c["latency"][f"{lat}_B"]
        print(f"lat={lat}ms A n={a['n']} mean={a['mean_bp']} pos={a['pos_rate']} | B n={b['n']} mean={b['mean_bp']} pos={b['pos_rate']}",flush=True)
    log("PHASE4_DONE",days=len(days),failures=len(fails))
if __name__=="__main__":main()
