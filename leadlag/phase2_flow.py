#!/usr/bin/env python3
import csv,gzip,io,json,math,os,statistics
from datetime import datetime
import requests, numpy as np

BASE="https://datasets.tardis.dev/v1"
DATES=os.getenv("DATES","2026-01-01,2026-03-01,2026-05-01,2026-07-01,2026-09-01").split(",")
COINS=os.getenv("COINS","BTC,XRP").split(",")
STEP_US=100_000
WINDOWS_MS=[500,1000,2000]
THRESHOLDS_BP=[10,20,30,50]
LATENCIES_MS=[100,500,1000]
HORIZONS_MS=[1000,2000,5000,10000]
COOLDOWN_MS=5000
FEE_BP=float(os.getenv("FEE_BP","10"))
TIMEOUT=(20,600)

def log(msg,**kw):
    print(msg+((" "+json.dumps(kw,ensure_ascii=False,sort_keys=True)) if kw else ""),flush=True)

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

def durl(ex,typ,date,sym):
    y,m,d=date.split("-")
    return f"{BASE}/{ex}/{typ}/{y}/{m}/{d}/{sym}.csv.gz"

def open_csv(ex,typ,date,sym):
    u=durl(ex,typ,date,sym)
    r=requests.get(u,stream=True,timeout=TIMEOUT)
    if r.status_code!=200:
        raise RuntimeError(f"HTTP {r.status_code} {u}: {r.text[:150]}")
    r.raw.decode_content=False
    gz=gzip.GzipFile(fileobj=r.raw,mode="rb")
    txt=io.TextIOWrapper(gz,encoding="utf-8",newline="")
    return csv.DictReader(txt),u

def load_quotes(ex,date,sym):
    rd,u=open_csv(ex,"quotes",date,sym)
    f=rd.fieldnames or []
    required={"local_timestamp","ask_price","ask_amount","bid_price","bid_amount"}
    if not required.issubset(set(f)):
        raise RuntimeError(f"quote cols {f}")
    ts=[]; bid=[]; ask=[]; bamt=[]; aamt=[]
    cur=None; vals=None; rows=0
    for row in rd:
        rows+=1
        try:
            t=ts_us(row["local_timestamp"]); bp=float(row["bid_price"]); ap=float(row["ask_price"])
            ba=float(row["bid_amount"]); aa=float(row["ask_amount"])
            if not(bp>0 and ap>=bp and ba>=0 and aa>=0): continue
        except: continue
        b=(t//STEP_US)*STEP_US
        v=(bp,ap,ba,aa)
        if cur is None: cur=b; vals=v
        elif b==cur: vals=v
        else:
            ts.append(cur);bid.append(vals[0]);ask.append(vals[1]);bamt.append(vals[2]);aamt.append(vals[3])
            cur=b; vals=v
    if cur is not None:
        ts.append(cur);bid.append(vals[0]);ask.append(vals[1]);bamt.append(vals[2]);aamt.append(vals[3])
    log("QUOTE_AGG",exchange=ex,date=date,symbol=sym,rows=rows,buckets=len(ts))
    return tuple(np.asarray(x,dtype=np.float64 if j else np.int64) for j,x in enumerate([ts,bid,ask,bamt,aamt]))

def load_flow(ex,date,sym):
    rd,u=open_csv(ex,"trades",date,sym)
    f=rd.fieldnames or []
    required={"local_timestamp","side","price","amount"}
    if not required.issubset(set(f)): raise RuntimeError(f"trade cols {f}")
    ts=[]; signed=[]; total=[]
    cur=None; sv=tv=0.0; rows=0
    for row in rd:
        rows+=1
        try:
            t=ts_us(row["local_timestamp"]); p=float(row["price"]); a=float(row["amount"]); side=row["side"].lower()
            if not(p>0 and a>=0): continue
            n=p*a
            s=n if side=="buy" else (-n if side=="sell" else 0.0)
        except: continue
        b=(t//STEP_US)*STEP_US
        if cur is None: cur=b; sv=s; tv=n
        elif b==cur: sv+=s; tv+=n
        else:
            ts.append(cur);signed.append(sv);total.append(tv)
            cur=b;sv=s;tv=n
    if cur is not None:
        ts.append(cur);signed.append(sv);total.append(tv)
    log("FLOW_AGG",exchange=ex,date=date,symbol=sym,rows=rows,buckets=len(ts))
    return np.asarray(ts,np.int64),np.asarray(signed,float),np.asarray(total,float)

def qgrid(q,start,end):
    ts,bid,ask,ba,aa=q
    n=int((end-start)//STEP_US)+1
    outs=[np.full(n,np.nan) for _ in range(4)]
    idx=((ts-start)//STEP_US).astype(np.int64)
    ok=(idx>=0)&(idx<n)
    for out,v in zip(outs,[bid,ask,ba,aa]): out[idx[ok]]=v[ok]
    valid=np.where(np.isfinite(outs[0])&np.isfinite(outs[1]),np.arange(n),-1)
    np.maximum.accumulate(valid,out=valid)
    m=valid>=0
    for out in outs: out[m]=out[valid[m]]
    return outs

def fgrid(f,start,end):
    ts,sg,tot=f
    n=int((end-start)//STEP_US)+1
    s=np.zeros(n); t=np.zeros(n)
    idx=((ts-start)//STEP_US).astype(np.int64)
    ok=(idx>=0)&(idx<n)
    np.add.at(s,idx[ok],sg[ok]); np.add.at(t,idx[ok],tot[ok])
    return s,t

def lret(x,k):
    out=np.full(len(x),np.nan)
    if k<=0:return out
    m=np.isfinite(x[k:])&np.isfinite(x[:-k])&(x[:-k]>0)
    v=np.full(len(x)-k,np.nan)
    v[m]=np.log(x[k:][m]/x[:-k][m])
    out[k:]=v
    return out

def rollsum(x,k):
    c=np.concatenate([[0.0],np.cumsum(x)])
    out=np.zeros(len(x))
    out[k-1:]=c[k:]-c[:-k]
    return out

def safe_median(v):
    v=[x for x in v if x is not None and math.isfinite(x)]
    return None if not v else statistics.median(v)

def mean(v):
    v=[x for x in v if x is not None and math.isfinite(x)]
    return None if not v else statistics.fmean(v)

def rate(v):
    v=[x for x in v if x is not None and math.isfinite(x)]
    return None if not v else sum(x>0 for x in v)/len(v)

def process_day(coin,date):
    spq=load_quotes("binance",date,coin+"USDT")
    fuq=load_quotes("binance-futures",date,coin+"USDT")
    upq=load_quotes("upbit",date,"KRW-"+coin)
    spf=load_flow("binance",date,coin+"USDT")
    fuf=load_flow("binance-futures",date,coin+"USDT")
    start=max(spq[0][0],fuq[0][0],upq[0][0],spf[0][0],fuf[0][0])
    end=min(spq[0][-1],fuq[0][-1],upq[0][-1],spf[0][-1],fuf[0][-1])
    sb,sa,sba,saa=qgrid(spq,start,end); fb,fa,fba,faa=qgrid(fuq,start,end); ub,ua,uba,uaa=qgrid(upq,start,end)
    ssg,stot=fgrid(spf,start,end); fsg,ftot=fgrid(fuf,start,end)
    sm=(sb+sa)/2; fm=(fb+fa)/2; um=(ub+ua)/2
    day_events=[]
    cooldown=COOLDOWN_MS*1000//STEP_US
    for wms in WINDOWS_MS:
        k=max(1,wms*1000//STEP_US)
        rs=lret(sm,k); rf=lret(fm,k); ru=lret(um,k); leader=(rs+rf)/2
        ss=rollsum(ssg,k); st=rollsum(stot,k); fs=rollsum(fsg,k); ft=rollsum(ftot,k)
        sim=np.divide(ss,st,out=np.zeros_like(ss),where=st>0); fim=np.divide(fs,ft,out=np.zeros_like(fs),where=ft>0)
        for th in THRESHOLDS_BP:
            last=-10**12
            for i in range(k,len(leader)):
                if i-last<cooldown: continue
                if not(all(math.isfinite(z) for z in [rs[i],rf[i],ru[i],leader[i],ub[i],ua[i],uba[i],uaa[i]])): continue
                if rs[i]<=0 or rf[i]<=0 or leader[i]*10000<th: continue
                residual=(leader[i]-ru[i])*10000
                if residual<=0: continue
                spread=math.log(ua[i]/ub[i])*10000
                upimb=(ub[i]*uba[i])/((ub[i]*uba[i])+(ua[i]*uaa[i])) if (ub[i]*uba[i]+ua[i]*uaa[i])>0 else None
                e={"date":date,"coin":coin,"window_ms":wms,"threshold_bp":th,"i":i,
                   "leader_bp":leader[i]*10000,"residual_bp":residual,"spread_bp":spread,
                   "spot_flow_imb":float(sim[i]),"fut_flow_imb":float(fim[i]),"up_l1_imb":upimb}
                # Pre-specified filters are stored, not optimized.
                e["flow_pos"]=sim[i]>0 and fim[i]>0
                e["flow_strong"]=sim[i]>=0.20 and fim[i]>=0.20
                e["up_support"]=upimb is not None and upimb>=0.50
                for lat in LATENCIES_MS:
                    ei=i+lat*1000//STEP_US
                    if ei>=len(ua) or not(math.isfinite(ua[ei]) and math.isfinite(ub[ei])): continue
                    entry_ask=ua[ei]; entry_mid=um[ei]; entry_bid=ub[ei]
                    for h in HORIZONS_MS:
                        xi=ei+h*1000//STEP_US
                        if xi>=len(ub) or not(math.isfinite(ub[xi]) and math.isfinite(um[xi])): continue
                        # mid alpha isolates predictive signal; taker net is executable top-of-book approximation.
                        e[f"mid_{lat}_{h}"]=math.log(um[xi]/entry_mid)*10000
                        e[f"net_{lat}_{h}"]=math.log(ub[xi]/entry_ask)*10000-FEE_BP
                        # Optimistic entry-maker upper bound: assumes fill at current best bid, still exits as taker.
                        e[f"makerUB_{lat}_{h}"]=math.log(ub[xi]/entry_bid)*10000-FEE_BP
                day_events.append(e); last=i
    log("DAY_DONE",date=date,coin=coin,events=len(day_events))
    return day_events

def agg(events, filt, th, window=1000, lat=100, h=5000, split=None):
    es=[e for e in events if e["threshold_bp"]==th and e["window_ms"]==window]
    if split=="train": es=[e for e in es if e["date"]<="2026-05-01"]
    if split=="test": es=[e for e in es if e["date"]>="2026-07-01"]
    if filt=="flow_pos": es=[e for e in es if e["flow_pos"]]
    elif filt=="flow_strong": es=[e for e in es if e["flow_strong"]]
    elif filt=="flow_pos_up": es=[e for e in es if e["flow_pos"] and e["up_support"]]
    elif filt=="flow_strong_up": es=[e for e in es if e["flow_strong"] and e["up_support"]]
    mid=[e.get(f"mid_{lat}_{h}") for e in es]; net=[e.get(f"net_{lat}_{h}") for e in es]; mub=[e.get(f"makerUB_{lat}_{h}") for e in es]
    return {"n":len(es),"leader_med_bp":safe_median([e["leader_bp"] for e in es]),
            "resid_med_bp":safe_median([e["residual_bp"] for e in es]),"spread_med_bp":safe_median([e["spread_bp"] for e in es]),
            "spotFlow_med":safe_median([e["spot_flow_imb"] for e in es]),"futFlow_med":safe_median([e["fut_flow_imb"] for e in es]),
            "mid_mean_bp":mean(mid),"mid_med_bp":safe_median(mid),"mid_pos_rate":rate(mid),
            "taker_net_mean_bp":mean(net),"taker_net_med_bp":safe_median(net),"taker_win_rate":rate(net),
            "makerUB_mean_bp":mean(mub),"makerUB_med_bp":safe_median(mub),"makerUB_win_rate":rate(mub)}

def main():
    log("PHASE2_START",dates=DATES,coins=COINS)
    all_events=[]
    failures=[]
    for coin in COINS:
        for date in DATES:
            try: all_events.extend(process_day(coin,date))
            except Exception as e:
                failures.append({"coin":coin,"date":date,"error":repr(e)}); log("DAY_FAIL",coin=coin,date=date,error=repr(e))
    # Fixed comparison: 1s impulse, 100ms latency, 5s exit. Also latency/horizon sensitivity for selected filters.
    summary={}
    for coin in COINS:
        cev=[e for e in all_events if e["coin"]==coin]
        summary[coin]={}
        for th in THRESHOLDS_BP:
            summary[coin][str(th)]={}
            for filt in ["price_only","flow_pos","flow_strong","flow_pos_up","flow_strong_up"]:
                summary[coin][str(th)][filt]={
                    "all":agg(cev,filt,th),
                    "train":agg(cev,filt,th,split="train"),
                    "test":agg(cev,filt,th,split="test")
                }
        # Sensitivity on flow_pos at thresholds
        sens={}
        for th in THRESHOLDS_BP:
            sens[str(th)]={}
            for lat in LATENCIES_MS:
                for h in HORIZONS_MS:
                    sens[str(th)][f"L{lat}_H{h}"]=agg(cev,"flow_pos",th,lat=lat,h=h)
        summary[coin]["sensitivity_flow_pos"]=sens
    out={"dates":DATES,"coins":COINS,"fee_bp":FEE_BP,"failures":failures,"summary":summary}
    print("PHASE2_JSON="+json.dumps(out,separators=(",",":")),flush=True)
    print("\nPHASE2_TABLE fixed: 1s impulse, 100ms latency, 5s exit",flush=True)
    print("coin th filter split n leader resid spread midMean midPos takerMean takerWin makerUB",flush=True)
    for coin in COINS:
        for th in THRESHOLDS_BP:
            for filt in ["price_only","flow_pos","flow_strong"]:
                for split in ["all","train","test"]:
                    x=summary[coin][str(th)][filt][split]
                    def f(v,fmt=".2f"):
                        return "-" if v is None else format(v,fmt)
                    print(f"{coin:4} {th:2d} {filt:11} {split:5} {x['n']:4d} {f(x['leader_med_bp']):>6} {f(x['resid_med_bp']):>6} {f(x['spread_med_bp']):>6} {f(x['mid_mean_bp']):>7} {f(None if x['mid_pos_rate'] is None else x['mid_pos_rate']*100,'.1f'):>5}% {f(x['taker_net_mean_bp']):>8} {f(None if x['taker_win_rate'] is None else x['taker_win_rate']*100,'.1f'):>5}% {f(x['makerUB_mean_bp']):>7}",flush=True)
    log("PHASE2_DONE",events=len(all_events),failures=len(failures))

if __name__=="__main__": main()
