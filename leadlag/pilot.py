#!/usr/bin/env python3
import csv, gzip, io, json, math, os, statistics, sys, time
from pathlib import Path
from datetime import datetime, timezone
import requests
import numpy as np

DATE=os.getenv("PILOT_DATE","2026-09-01")
COINS=os.getenv("COINS","BTC,ETH,XRP,SOL,DOGE").split(",")
DATA_DIR=Path(os.getenv("DATA_DIR","/tmp/leadlag-data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
BASE="https://datasets.tardis.dev/v1"
HORIZONS=[1,2,5,10,30]
LAGS=list(range(-30,31))
COOLDOWN_S=10
Z_THRESHOLD=3.0
MIN_IMPULSE=0.0005  # 5 bp floor, not a trading threshold; event-study screen
ROLL_VOL_S=600

def log(msg, **kw):
    if kw:
        print(msg, json.dumps(kw, ensure_ascii=False, sort_keys=True), flush=True)
    else:
        print(msg, flush=True)

def date_parts():
    y,m,d=DATE.split("-"); return y,m,d

def url(exchange,symbol):
    y,m,d=date_parts()
    return f"{BASE}/{exchange}/trades/{y}/{m}/{d}/{symbol}.csv.gz"

def download(exchange,symbol):
    out=DATA_DIR/f"{exchange}_{symbol}_{DATE}.csv.gz"
    if out.exists() and out.stat().st_size>100:
        return out
    u=url(exchange,symbol)
    log("DOWNLOAD", exchange=exchange, symbol=symbol, url=u)
    with requests.get(u, stream=True, timeout=(20,300)) as r:
        if r.status_code!=200:
            raise RuntimeError(f"HTTP {r.status_code} {u}: {r.text[:200]}")
        with open(out,"wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk)
    return out

def ts_to_us(v):
    if v is None or v=="":
        return None
    s=str(v).strip()
    try:
        x=int(float(s))
        # normalize seconds/millis/micros/nanos
        if x<10_000_000_000: return x*1_000_000
        if x<10_000_000_000_000: return x*1_000
        if x>10_000_000_000_000_000: return x//1_000
        return x
    except Exception:
        dt=datetime.fromisoformat(s.replace("Z","+00:00"))
        return int(dt.timestamp()*1_000_000)

def stream_aggregate(path, bucket_us=100_000):
    """Return bucket timestamp(us) and last price, using local_timestamp where available."""
    buckets_t=[]; buckets_p=[]
    cur=None; last=None; rows=0
    with gzip.open(path,"rt",newline="") as f:
        rd=csv.DictReader(f)
        fields=rd.fieldnames or []
        pcol="price" if "price" in fields else ("last_price" if "last_price" in fields else None)
        tcol="local_timestamp" if "local_timestamp" in fields else ("timestamp" if "timestamp" in fields else None)
        if not pcol or not tcol:
            raise RuntimeError(f"unexpected columns {fields}")
        for row in rd:
            rows+=1
            try:
                t=ts_to_us(row[tcol]); p=float(row[pcol])
                if not t or not math.isfinite(p) or p<=0: continue
            except Exception:
                continue
            b=(t//bucket_us)*bucket_us
            if cur is None:
                cur=b; last=p
            elif b==cur:
                last=p
            else:
                buckets_t.append(cur); buckets_p.append(last)
                cur=b; last=p
        if cur is not None:
            buckets_t.append(cur); buckets_p.append(last)
    log("AGG", file=str(path), rows=rows, buckets=len(buckets_t))
    return np.asarray(buckets_t,dtype=np.int64), np.asarray(buckets_p,dtype=np.float64)

def series_to_grid(ts, px, start, end, step_us):
    n=int((end-start)//step_us)+1
    grid=np.arange(n,dtype=np.int64)*step_us+start
    out=np.full(n,np.nan,dtype=np.float64)
    idx=((ts-start)//step_us).astype(np.int64)
    ok=(idx>=0)&(idx<n)
    idx=idx[ok]; vals=px[ok]
    out[idx]=vals
    # forward fill
    valid=np.where(np.isfinite(out),np.arange(n),-1)
    np.maximum.accumulate(valid,out=valid)
    m=valid>=0
    out[m]=out[valid[m]]
    return grid,out

def returns(px, steps):
    out=np.full_like(px,np.nan)
    if steps<=0: return out
    good=np.isfinite(px[steps:])&np.isfinite(px[:-steps])&(px[:-steps]>0)
    vals=np.full(len(px)-steps,np.nan)
    vals[good]=np.log(px[steps:][good]/px[:-steps][good])
    out[steps:]=vals
    return out

def rolling_std(x, window):
    # O(n) rolling std ignoring NaNs via cumulative sums
    v=np.nan_to_num(x,nan=0.0)
    valid=np.isfinite(x).astype(np.int64)
    cs=np.concatenate([[0.0],np.cumsum(v)])
    cs2=np.concatenate([[0.0],np.cumsum(v*v)])
    cc=np.concatenate([[0],np.cumsum(valid)])
    n=len(x); out=np.full(n,np.nan)
    for i in range(window,n):
        a=i-window+1; b=i+1
        cnt=cc[b]-cc[a]
        if cnt<max(30,window//4): continue
        s=cs[b]-cs[a]; s2=cs2[b]-cs2[a]
        var=max(0.0,s2/cnt-(s/cnt)**2)
        out[i]=math.sqrt(var)
    return out

def corr_at_lag(x,y,lag):
    # positive lag => y (Upbit) follows x (leader)
    if lag>0:
        a=x[:-lag]; b=y[lag:]
    elif lag<0:
        a=x[-lag:]; b=y[:lag]
    else:
        a=x; b=y
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<100: return float("nan")
    aa=a[m]; bb=b[m]
    if np.std(aa)==0 or np.std(bb)==0: return float("nan")
    return float(np.corrcoef(aa,bb)[0,1])

def percentile(arr,q):
    a=[x for x in arr if x is not None and math.isfinite(x)]
    return float(np.percentile(a,q)) if a else None

def analyze_coin(coin, spot, fut, up, usdt=None):
    # common 1s grid based on intersection
    starts=[spot[0][0],fut[0][0],up[0][0]]
    ends=[spot[0][-1],fut[0][-1],up[0][-1]]
    if usdt is not None and len(usdt[0]):
        starts.append(usdt[0][0]); ends.append(usdt[0][-1])
    start=max(starts); end=min(ends)
    step=1_000_000
    _,sp=series_to_grid(*spot,start,end,step)
    _,fu=series_to_grid(*fut,start,end,step)
    _,uu=series_to_grid(*up,start,end,step)
    if usdt is not None and len(usdt[0]):
        _,usd=series_to_grid(*usdt,start,end,step)
        # fair-price KRW = Binance USDT quote * Upbit KRW-USDT
        fair_sp=sp*usd
        fair_fu=fu*usd
    else:
        fair_sp=sp; fair_fu=fu

    r_sp=returns(fair_sp,1); r_fu=returns(fair_fu,1); r_up=returns(uu,1)
    rg=np.nanmean(np.vstack([r_sp,r_fu]),axis=0)
    corr={str(l):corr_at_lag(rg,r_up,l) for l in LAGS}
    finite_corr={k:v for k,v in corr.items() if math.isfinite(v)}
    best_lag=int(max(finite_corr,key=finite_corr.get)) if finite_corr else None
    best_corr=finite_corr.get(str(best_lag)) if best_lag is not None else None

    # Multi-window impulse event study. Events are independent by cooldown.
    all_events=[]
    for w in (1,2,5):
        rsp=returns(fair_sp,w); rfu=returns(fair_fu,w)
        leader=np.nanmean(np.vstack([rsp,rfu]),axis=0)
        sigma=rolling_std(rg,ROLL_VOL_S)
        z=np.divide(leader, sigma*np.sqrt(w), out=np.full_like(leader,np.nan), where=(sigma>0))
        last=-10**9
        for i in range(len(leader)):
            if i-last<COOLDOWN_S: continue
            if not math.isfinite(leader[i]) or not math.isfinite(z[i]): continue
            # Both spot and futures must confirm positive direction.
            if rsp[i] <= 0 or rfu[i] <= 0: continue
            if leader[i] < MIN_IMPULSE or z[i] < Z_THRESHOLD: continue
            if i-w<0: continue
            u_pre=uu[i-w]; u_now=uu[i]
            if not (math.isfinite(u_pre) and math.isfinite(u_now) and u_pre>0): continue
            up_during=math.log(u_now/u_pre)
            residual0=leader[i]-up_during
            ev={"i":i,"window":w,"leader":float(leader[i]),"z":float(z[i]),"up_during":float(up_during),"residual0":float(residual0)}
            for h in HORIZONS:
                j=i+h
                if j>=len(uu) or not math.isfinite(uu[j]) or uu[i]<=0:
                    ev[f"fwd_{h}"]=None; ev[f"capture_{h}"]=None
                else:
                    fwd=math.log(uu[j]/uu[i])
                    ev[f"fwd_{h}"]=float(fwd)
                    ev[f"capture_{h}"]=float(fwd/residual0) if residual0>1e-9 else None
            all_events.append(ev); last=i

    by_h={}
    for h in HORIZONS:
        vals=[e[f"fwd_{h}"] for e in all_events if e[f"fwd_{h}"] is not None]
        caps=[e[f"capture_{h}"] for e in all_events if e[f"capture_{h}"] is not None and -5<e[f"capture_{h}"]<5]
        by_h[str(h)]={
            "n":len(vals),
            "follow_rate": (sum(v>0 for v in vals)/len(vals) if vals else None),
            "median_fwd_bp": (statistics.median(vals)*10000 if vals else None),
            "mean_fwd_bp": (statistics.fmean(vals)*10000 if vals else None),
            "p25_fwd_bp": (percentile(vals,25)*10000 if vals else None),
            "p75_fwd_bp": (percentile(vals,75)*10000 if vals else None),
            "median_capture": (statistics.median(caps) if caps else None),
        }
    res0=[e["residual0"] for e in all_events]
    return {
        "coin":coin,
        "seconds":len(uu),
        "best_lag_s":best_lag,
        "best_corr":best_corr,
        "events":len(all_events),
        "median_initial_residual_bp":statistics.median(res0)*10000 if res0 else None,
        "horizons":by_h
    }

def main():
    log("PILOT_START", date=DATE, coins=COINS)
    # Download KRW-USDT once if available.
    usdt=None
    try:
        p=download("upbit","KRW-USDT")
        usdt=stream_aggregate(p)
    except Exception as e:
        log("USDT_OPTIONAL_FAILED", error=str(e))
    results=[]
    for coin in COINS:
        try:
            sp=stream_aggregate(download("binance",f"{coin}USDT"))
            fu=stream_aggregate(download("binance-futures",f"{coin}USDT"))
            up=stream_aggregate(download("upbit",f"KRW-{coin}"))
            r=analyze_coin(coin,sp,fu,up,usdt)
            results.append(r)
            log("COIN_RESULT", result=r)
        except Exception as e:
            log("COIN_FAILED", coin=coin, error=repr(e))
    print("RESULT_JSON="+json.dumps({"date":DATE,"results":results},ensure_ascii=False,separators=(",",":")),flush=True)
    print("\nHUMAN_TABLE",flush=True)
    print("coin bestLag corr events residBP f1s f2s f5s f10s f30s",flush=True)
    for r in results:
        def fr(h):
            x=r["horizons"][str(h)]["follow_rate"]
            return "-" if x is None else f"{x*100:.1f}%"
        print(f"{r['coin']:4} {str(r['best_lag_s']):>7} {r['best_corr'] if r['best_corr'] is not None else float('nan'):.3f} {r['events']:6d} {r['median_initial_residual_bp'] if r['median_initial_residual_bp'] is not None else float('nan'):7.2f} {fr(1):>6} {fr(2):>6} {fr(5):>6} {fr(10):>6} {fr(30):>6}",flush=True)
    log("PILOT_DONE", coins_completed=len(results))

if __name__=="__main__":
    main()
