#!/usr/bin/env python3
import csv,gzip,io,json,math,os,statistics,sys,time
from datetime import datetime
import requests, numpy as np

DATE=os.getenv("PILOT_DATE","2026-09-01")
COIN=os.getenv("COIN","XRP")
BASE="https://datasets.tardis.dev/v1"
STEP_US=100_000
WINDOWS_MS=[500,1000,2000]
HORIZONS_MS=[500,1000,2000,5000,10000,30000]
LATENCIES_MS=[100,200,500,1000,2000]
MIN_LEADER_BP=5.0
COOLDOWN_MS=5000
FEE_ROUNDTRIP_BP=10.0

def log(msg,**kw):
    print(msg+((" "+json.dumps(kw,ensure_ascii=False,sort_keys=True)) if kw else ""),flush=True)

def url(ex,sym):
    y,m,d=DATE.split("-")
    return f"{BASE}/{ex}/quotes/{y}/{m}/{d}/{sym}.csv.gz"

def ts_us(v):
    s=str(v).strip()
    try:
        x=int(float(s))
        if x<10_000_000_000: return x*1_000_000
        if x<10_000_000_000_000: return x*1_000
        if x>10_000_000_000_000_000: return x//1000
        return x
    except Exception:
        return int(datetime.fromisoformat(s.replace("Z","+00:00")).timestamp()*1_000_000)

def load_quotes(ex,sym):
    u=url(ex,sym)
    log("QDOWNLOAD",exchange=ex,symbol=sym,url=u)
    r=requests.get(u,stream=True,timeout=(20,600))
    if r.status_code!=200:
        raise RuntimeError(f"HTTP {r.status_code} {u}: {r.text[:200]}")
    r.raw.decode_content=False
    gz=gzip.GzipFile(fileobj=r.raw,mode="rb")
    txt=io.TextIOWrapper(gz,encoding="utf-8",newline="")
    rd=csv.DictReader(txt)
    f=rd.fieldnames or []
    need={"local_timestamp","ask_price","bid_price"}
    if not need.issubset(set(f)):
        raise RuntimeError(f"unexpected quote columns {f}")
    ts=[]; bid=[]; ask=[]; cur=None; lb=la=None; rows=0
    for row in rd:
        rows+=1
        try:
            t=ts_us(row["local_timestamp"])
            b=float(row["bid_price"]); a=float(row["ask_price"])
            if not (b>0 and a>0 and a>=b): continue
        except Exception:
            continue
        buck=(t//STEP_US)*STEP_US
        if cur is None:
            cur=buck; lb=b; la=a
        elif buck==cur:
            lb=b; la=a
        else:
            ts.append(cur);bid.append(lb);ask.append(la)
            cur=buck;lb=b;la=a
        if rows%5_000_000==0: log("QPROGRESS",exchange=ex,symbol=sym,rows=rows,buckets=len(ts))
    if cur is not None:
        ts.append(cur);bid.append(lb);ask.append(la)
    log("QAGG",exchange=ex,symbol=sym,rows=rows,buckets=len(ts))
    return np.asarray(ts,np.int64),np.asarray(bid,float),np.asarray(ask,float)

def grid(series,start,end):
    ts,bid,ask=series
    n=int((end-start)//STEP_US)+1
    ob=np.full(n,np.nan); oa=np.full(n,np.nan)
    idx=((ts-start)//STEP_US).astype(np.int64)
    ok=(idx>=0)&(idx<n)
    idx=idx[ok]; bb=bid[ok]; aa=ask[ok]
    ob[idx]=bb; oa[idx]=aa
    valid=np.where(np.isfinite(ob)&np.isfinite(oa),np.arange(n),-1)
    np.maximum.accumulate(valid,out=valid)
    m=valid>=0
    ob[m]=ob[valid[m]]; oa[m]=oa[valid[m]]
    return ob,oa

def lret(x,k):
    out=np.full(len(x),np.nan)
    m=np.isfinite(x[k:])&np.isfinite(x[:-k])&(x[:-k]>0)
    vals=np.full(len(x)-k,np.nan)
    vals[m]=np.log(x[k:][m]/x[:-k][m])
    out[k:]=vals
    return out

def pct(vals,q):
    vals=[v for v in vals if v is not None and math.isfinite(v)]
    return None if not vals else float(np.percentile(vals,q))

def main():
    log("QUOTE_PILOT_START",date=DATE,coin=COIN,step_ms=STEP_US/1000)
    spot=load_quotes("binance",COIN+"USDT")
    fut=load_quotes("binance-futures",COIN+"USDT")
    up=load_quotes("upbit","KRW-"+COIN)
    start=max(spot[0][0],fut[0][0],up[0][0]); end=min(spot[0][-1],fut[0][-1],up[0][-1])
    sb,sa=grid(spot,start,end); fb,fa=grid(fut,start,end); ub,ua=grid(up,start,end)
    sm=(sb+sa)/2; fm=(fb+fa)/2; um=(ub+ua)/2
    results={}
    all_events=[]
    for wms in WINDOWS_MS:
        k=max(1,wms*1000//STEP_US)
        rs=lret(sm,k); rf=lret(fm,k); ru=lret(um,k)
        leader=(rs+rf)/2
        last=-10**12
        events=[]
        cooldown=COOLDOWN_MS*1000//STEP_US
        for i in range(k,len(leader)):
            if i-last<cooldown: continue
            if not (math.isfinite(rs[i]) and math.isfinite(rf[i]) and math.isfinite(ru[i])): continue
            if rs[i]<=0 or rf[i]<=0: continue
            lbp=leader[i]*10000
            if lbp<MIN_LEADER_BP: continue
            residual=leader[i]-ru[i]
            if residual<=0: continue
            e={"i":i,"window_ms":wms,"leader_bp":float(lbp),"residual_bp":float(residual*10000)}
            for lat in LATENCIES_MS:
                ei=i+lat*1000//STEP_US
                if ei>=len(ua): continue
                entry=ua[ei]
                if not math.isfinite(entry) or entry<=0: continue
                for h in HORIZONS_MS:
                    xi=ei+h*1000//STEP_US
                    if xi>=len(ub): continue
                    exitp=ub[xi]
                    if not math.isfinite(exitp) or exitp<=0: continue
                    gross=math.log(exitp/entry)*10000
                    e[f"pnl_{lat}_{h}"]=float(gross-FEE_ROUNDTRIP_BP)
            events.append(e); all_events.append(e); last=i
        results[str(wms)]={"events":len(events)}
        for lat in LATENCIES_MS:
            for h in HORIZONS_MS:
                vals=[e.get(f"pnl_{lat}_{h}") for e in events if e.get(f"pnl_{lat}_{h}") is not None]
                results[str(wms)][f"{lat}ms_{h}ms"]={
                    "n":len(vals),
                    "mean_net_bp":None if not vals else statistics.fmean(vals),
                    "median_net_bp":None if not vals else statistics.median(vals),
                    "win_rate":None if not vals else sum(v>0 for v in vals)/len(vals),
                    "p25_bp":pct(vals,25),"p75_bp":pct(vals,75)
                }
        # residual bins at 1s latency, 5s exit to see where economics changes
        bins=[(5,10),(10,20),(20,40),(40,1e9)]
        rb={}
        for lo,hi in bins:
            es=[e for e in events if lo<=e["residual_bp"]<hi]
            vals=[e.get("pnl_1000_5000") for e in es if e.get("pnl_1000_5000") is not None]
            rb[f"{lo}-{hi if hi<1e8 else 'inf'}bp"]={
                "n":len(vals),
                "mean_net_bp":None if not vals else statistics.fmean(vals),
                "median_net_bp":None if not vals else statistics.median(vals),
                "win_rate":None if not vals else sum(v>0 for v in vals)/len(vals)
            }
        results[str(wms)]["residual_bins_1sLatency_5sExit"]=rb

    out={"date":DATE,"coin":COIN,"fee_roundtrip_bp":FEE_ROUNDTRIP_BP,"results":results}
    print("QUOTE_RESULT_JSON="+json.dumps(out,separators=(",",":")),flush=True)
    # concise readable shortlist: 1s impulse window
    rr=results["1000"]
    print("SUMMARY window=1000ms net bp includes spread+10bp fee",flush=True)
    for lat in LATENCIES_MS:
        row=[]
        for h in [1000,2000,5000,10000]:
            x=rr[f"{lat}ms_{h}ms"]
            row.append(f"h{h//1000}s n{x['n']} mean{x['mean_net_bp']:.2f} med{x['median_net_bp']:.2f} win{x['win_rate']*100:.1f}%")
        print(f"lat={lat}ms "+" | ".join(row),flush=True)
    print("RESIDUAL_BINS",json.dumps(rr["residual_bins_1sLatency_5sExit"],separators=(",",":")),flush=True)
    log("QUOTE_PILOT_DONE")

if __name__=="__main__":
    main()
