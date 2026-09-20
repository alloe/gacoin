#!/usr/bin/env python3
import csv,gzip,io,json,math,os,statistics
from datetime import datetime, timezone, timedelta
import requests, numpy as np

BASE="https://datasets.tardis.dev/v1"
DATES=os.getenv("DATES","2026-01-01").split(",")
STEP_US=100_000
FEE_BP=float(os.getenv("FEE_BP","10"))
LEADER_MIN_BP=10.0
COOLDOWN_STEPS=50   # 5s on 100ms grid
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
    return csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw,mode="rb"),encoding="utf-8",newline=""))

def load_quotes(ex,date,sym):
    rd=open_csv(ex,"quotes",date,sym)
    f=rd.fieldnames or []
    need={"local_timestamp","ask_price","ask_amount","bid_price","bid_amount"}
    if not need.issubset(set(f)): raise RuntimeError(f"quote cols {f}")
    ts=[]; bid=[]; ask=[]; ba=[]; aa=[]; cur=None; vals=None; rows=0
    for row in rd:
        rows+=1
        try:
            t=ts_us(row["local_timestamp"]); bp=float(row["bid_price"]); ap=float(row["ask_price"])
            b=float(row["bid_amount"]); a=float(row["ask_amount"])
            if not(bp>0 and ap>=bp and b>=0 and a>=0): continue
        except: continue
        buck=(t//STEP_US)*STEP_US; v=(bp,ap,b,a)
        if cur is None: cur=buck;vals=v
        elif buck==cur: vals=v
        else:
            ts.append(cur);bid.append(vals[0]);ask.append(vals[1]);ba.append(vals[2]);aa.append(vals[3]);cur=buck;vals=v
    if cur is not None:
        ts.append(cur);bid.append(vals[0]);ask.append(vals[1]);ba.append(vals[2]);aa.append(vals[3])
    log("QUOTE_AGG",exchange=ex,date=date,symbol=sym,rows=rows,buckets=len(ts))
    return np.asarray(ts,np.int64),np.asarray(bid,float),np.asarray(ask,float),np.asarray(ba,float),np.asarray(aa,float)

def load_flow(ex,date,sym):
    rd=open_csv(ex,"trades",date,sym)
    f=rd.fieldnames or []
    need={"local_timestamp","side","price","amount"}
    if not need.issubset(set(f)): raise RuntimeError(f"trade cols {f}")
    ts=[]; sg=[]; tv=[]; cur=None; ss=tt=0.0;rows=0
    for row in rd:
        rows+=1
        try:
            t=ts_us(row["local_timestamp"]); p=float(row["price"]); a=float(row["amount"]); side=row["side"].lower()
            if not(p>0 and a>=0): continue
            n=p*a; s=n if side=="buy" else (-n if side=="sell" else 0.0)
        except: continue
        buck=(t//STEP_US)*STEP_US
        if cur is None: cur=buck;ss=s;tt=n
        elif buck==cur: ss+=s;tt+=n
        else: ts.append(cur);sg.append(ss);tv.append(tt);cur=buck;ss=s;tt=n
    if cur is not None: ts.append(cur);sg.append(ss);tv.append(tt)
    log("FLOW_AGG",exchange=ex,date=date,symbol=sym,rows=rows,buckets=len(ts))
    return np.asarray(ts,np.int64),np.asarray(sg,float),np.asarray(tv,float)

def qgrid(q,start,end):
    ts,bid,ask,ba,aa=q; n=int((end-start)//STEP_US)+1
    outs=[np.full(n,np.nan) for _ in range(4)]
    upd=np.zeros(n,dtype=np.int8)
    idx=((ts-start)//STEP_US).astype(np.int64); ok=(idx>=0)&(idx<n); ii=idx[ok]
    for out,v in zip(outs,[bid,ask,ba,aa]): out[ii]=v[ok]
    upd[ii]=1
    valid=np.where(np.isfinite(outs[0])&np.isfinite(outs[1]),np.arange(n),-1)
    np.maximum.accumulate(valid,out=valid); m=valid>=0
    for out in outs: out[m]=out[valid[m]]
    return outs,upd

def fgrid(f,start,end):
    ts,sg,tv=f; n=int((end-start)//STEP_US)+1
    s=np.zeros(n);t=np.zeros(n); idx=((ts-start)//STEP_US).astype(np.int64);ok=(idx>=0)&(idx<n)
    np.add.at(s,idx[ok],sg[ok]);np.add.at(t,idx[ok],tv[ok])
    return s,t

def lret(x,k):
    out=np.full(len(x),np.nan);m=np.isfinite(x[k:])&np.isfinite(x[:-k])&(x[:-k]>0)
    v=np.full(len(x)-k,np.nan);v[m]=np.log(x[k:][m]/x[:-k][m]);out[k:]=v
    return out

def rollsum(x,k):
    c=np.concatenate([[0.0],np.cumsum(x)]);out=np.zeros(len(x));out[k-1:]=c[k:]-c[:-k];return out

def median(v):
    a=[x for x in v if x is not None and math.isfinite(x)]
    return None if not a else statistics.median(a)
def mean(v):
    a=[x for x in v if x is not None and math.isfinite(x)]
    return None if not a else statistics.fmean(a)
def pct(v,q):
    a=[x for x in v if x is not None and math.isfinite(x)]
    return None if not a else float(np.percentile(a,q))

def stats(es):
    pnl=[e["net5"] for e in es if e.get("net5") is not None and math.isfinite(e["net5"])]
    mid=[e["mid5"] for e in es if e.get("mid5") is not None and math.isfinite(e["mid5"])]
    if not pnl:
        return {"n":0}
    gains=sum(x for x in pnl if x>0);loss=-sum(x for x in pnl if x<0)
    return {
        "n":len(pnl),
        "net_mean_bp":mean(pnl),"net_median_bp":median(pnl),"win_rate":sum(x>0 for x in pnl)/len(pnl),
        "profit_factor":(gains/loss if loss>0 else None),"p10_bp":pct(pnl,10),"p90_bp":pct(pnl,90),
        "worst_bp":min(pnl),"best_bp":max(pnl),
        "mid_mean_bp":mean(mid),"mid_pos_rate":(sum(x>0 for x in mid)/len(mid) if mid else None),
        "leader_med_bp":median([e["leader_bp"] for e in es]),"beta_med":median([e["beta5"] for e in es]),
        "pred_edge_med_bp":median([e["pred_edge_bp"] for e in es]),"spread_med_bp":median([e["spread_bp"] for e in es]),
        "vol_ratio_med":median([e["vol_ratio"] for e in es]),"activity_ratio_med":median([e["activity_ratio"] for e in es]),
        "makerUB_mean_bp":mean([e["makerUB5"] for e in es])
    }

def beta5_at(i,g1,um):
    # Past-only 30m rolling estimate of positive global 1s shock -> Upbit 5s forward response.
    # Pair j is usable only if j+5s <= i, preventing look-ahead.
    end=i-50
    if end<=20:return None,0
    start=max(10,end-18000)
    js=np.arange(start,end,10,dtype=np.int64) # 1s spaced
    x=g1[js]
    valid=(np.isfinite(x))&(x>=0.0001)&np.isfinite(um[js])&np.isfinite(um[js+50])&(um[js]>0)
    js=js[valid];x=x[valid]
    if len(x)<30:return None,len(x)
    y=np.log(um[js+50]/um[js])
    den=float(np.dot(x,x))
    if den<=0:return None,len(x)
    b=float(np.dot(x,y)/den)
    return max(-0.5,min(2.0,b)),len(x)

def vol_features(i,g1):
    # Current 60s vol vs trailing 30m median 60s vol, using only data up to event.
    js=np.arange(max(10,i-600),i+1,10,dtype=np.int64)
    a=g1[js];a=a[np.isfinite(a)]
    if len(a)<20:return None,None
    cur=float(np.std(a))*10000
    hist=[]
    for j in range(max(610,i-18000),i-600,100): # every 10s
        jj=np.arange(max(10,j-600),j+1,10,dtype=np.int64)
        z=g1[jj];z=z[np.isfinite(z)]
        if len(z)>=20: hist.append(float(np.std(z))*10000)
    med=statistics.median(hist) if hist else None
    return cur,(cur/med if med and med>0 else None)

def activity_features(i,qcum):
    # Actual Upbit quote-update buckets in last 10s, compared with trailing 30m typical 10s activity.
    def cnt(j):
        a=max(0,j-100);return float(qcum[j+1]-qcum[a])
    cur=cnt(i);hist=[]
    for j in range(max(100,i-18000),i,100):
        hist.append(cnt(j))
    med=statistics.median(hist) if hist else None
    return cur,(cur/med if med and med>0 else None)

def process(date):
    spq=load_quotes("binance",date,"BTCUSDT");fuq=load_quotes("binance-futures",date,"BTCUSDT");upq=load_quotes("upbit",date,"KRW-BTC")
    spf=load_flow("binance",date,"BTCUSDT");fuf=load_flow("binance-futures",date,"BTCUSDT")
    start=max(spq[0][0],fuq[0][0],upq[0][0],spf[0][0],fuf[0][0]);end=min(spq[0][-1],fuq[0][-1],upq[0][-1],spf[0][-1],fuf[0][-1])
    (sq,supd)=qgrid(spq,start,end);(fq,fupd)=qgrid(fuq,start,end);(uq,uupd)=qgrid(upq,start,end)
    sb,sa,sba,saa=sq;fb,fa,fba,faa=fq;ub,ua,uba,uaa=uq
    sm=(sb+sa)/2;fm=(fb+fa)/2;um=(ub+ua)/2
    ss,st=fgrid(spf,start,end);fs,ft=fgrid(fuf,start,end)
    rs=lret(sm,10);rf=lret(fm,10);ru=lret(um,10);g1=(rs+rf)/2
    ss1=rollsum(ss,10);st1=rollsum(st,10);fs1=rollsum(fs,10);ft1=rollsum(ft,10)
    sim=np.divide(ss1,st1,out=np.zeros_like(ss1),where=st1>0);fim=np.divide(fs1,ft1,out=np.zeros_like(fs1),where=ft1>0)
    qcum=np.concatenate([[0],np.cumsum(uupd,dtype=np.int64)])
    events=[];last=-10**9
    for i in range(10,len(g1)-60):
        if i-last<COOLDOWN_STEPS:continue
        if not(all(math.isfinite(x) for x in [rs[i],rf[i],ru[i],g1[i],ub[i],ua[i],uba[i],uaa[i],um[i]])):continue
        if rs[i]<=0 or rf[i]<=0 or g1[i]*10000<LEADER_MIN_BP:continue
        beta,n_beta=beta5_at(i,g1,um)
        if beta is None:continue
        vol,vr=vol_features(i,g1);act,ar=activity_features(i,qcum)
        if vr is None or ar is None:continue
        leader=g1[i]*10000;up_during=ru[i]*10000
        spread=math.log(ua[i]/ub[i])*10000
        denom=ub[i]*uba[i]+ua[i]*uaa[i]
        upimb=(ub[i]*uba[i]/denom) if denom>0 else None
        flow_strong=sim[i]>=0.20 and fim[i]>=0.20
        up_support=(upimb is not None and upimb>=0.50)
        pred_follow=beta*leader
        pred_edge=pred_follow-up_during-FEE_BP-spread
        ei=i+1;xi=ei+50
        net=math.log(ub[xi]/ua[ei])*10000-FEE_BP if ua[ei]>0 and ub[xi]>0 else None
        mid5=math.log(um[xi]/um[ei])*10000 if um[ei]>0 and um[xi]>0 else None
        mub=math.log(ub[xi]/ub[ei])*10000-FEE_BP if ub[ei]>0 and ub[xi]>0 else None
        score=sum([
            beta>=0.5,
            spread<=3.0,
            vr>=1.5,
            ar>=1.5,
            flow_strong,
            up_support
        ])
        kst_hour=(datetime.fromtimestamp(start/1_000_000,tz=timezone.utc)+timedelta(microseconds=i*STEP_US,hours=9)).hour
        events.append({
            "date":date,"leader_bp":leader,"up_during_bp":up_during,"beta5":beta,"beta_n":n_beta,
            "spread_bp":spread,"vol60_bp":vol,"vol_ratio":vr,"activity10":act,"activity_ratio":ar,
            "spot_flow":float(sim[i]),"fut_flow":float(fim[i]),"flow_strong":flow_strong,
            "up_imb":upimb,"up_support":up_support,"pred_follow_bp":pred_follow,"pred_edge_bp":pred_edge,
            "score":score,"kst_hour":kst_hour,"net5":net,"mid5":mid5,"makerUB5":mub
        })
        last=i
    log("DAY_DONE",date=date,events=len(events))
    return events

def main():
    log("PHASE3_START",dates=DATES)
    ev=[];fails=[]
    for d in DATES:
        try:ev.extend(process(d))
        except Exception as e:fails.append({"date":d,"error":repr(e)});log("DAY_FAIL",date=d,error=repr(e))
    gates={
      "base":lambda e:True,
      "edge_gt0":lambda e:e["pred_edge_bp"]>0,
      "edge_gt3":lambda e:e["pred_edge_bp"]>3,
      "edge_gt5":lambda e:e["pred_edge_bp"]>5,
      "edge_gt0_flow":lambda e:e["pred_edge_bp"]>0 and e["flow_strong"],
      "edge_gt3_flow":lambda e:e["pred_edge_bp"]>3 and e["flow_strong"],
      "edge_gt5_flow":lambda e:e["pred_edge_bp"]>5 and e["flow_strong"],
      "edge_gt0_flow_up":lambda e:e["pred_edge_bp"]>0 and e["flow_strong"] and e["up_support"],
      "edge_gt3_flow_up":lambda e:e["pred_edge_bp"]>3 and e["flow_strong"] and e["up_support"],
      "edge_gt5_flow_up":lambda e:e["pred_edge_bp"]>5 and e["flow_strong"] and e["up_support"],
      "score_ge4":lambda e:e["score"]>=4,
      "score_ge5":lambda e:e["score"]>=5,
      "score_eq6":lambda e:e["score"]==6,
    }
    out={"dates":DATES,"failures":fails,"events":len(ev),"gates":{},"score_exact":{},"time":{},"beta_buckets":{}}
    for name,fn in gates.items():out["gates"][name]=stats([e for e in ev if fn(e)])
    for s in range(7):out["score_exact"][str(s)]=stats([e for e in ev if e["score"]==s])
    for label,a,b in [("00-06",0,6),("06-12",6,12),("12-18",12,18),("18-24",18,24)]:
        out["time"][label]=stats([e for e in ev if a<=e["kst_hour"]<b])
    for label,a,b in [("beta<0.25",-9,0.25),("0.25-0.5",0.25,0.5),("0.5-0.75",0.5,0.75),("0.75-1.0",0.75,1.0),("beta>=1",1.0,9)]:
        out["beta_buckets"][label]=stats([e for e in ev if a<=e["beta5"]<b])
    print("PHASE3_JSON="+json.dumps(out,separators=(",",":")),flush=True)
    print("PHASE3_TABLE gate n netMean med win PF midMean beta predEdge spread volR actR makerUB",flush=True)
    for name,x in out["gates"].items():
        def f(k,d=2):
            v=x.get(k);return "-" if v is None else f"{v:.{d}f}"
        print(f"{name:18} {x.get('n',0):4d} {f('net_mean_bp'):>7} {f('net_median_bp'):>7} {('-' if x.get('win_rate') is None else f'{x['win_rate']*100:.1f}%'):>6} {f('profit_factor'):>5} {f('mid_mean_bp'):>7} {f('beta_med'):>5} {f('pred_edge_med_bp'):>8} {f('spread_med_bp'):>6} {f('vol_ratio_med'):>5} {f('activity_ratio_med'):>5} {f('makerUB_mean_bp'):>7}",flush=True)
    log("PHASE3_DONE",events=len(ev),failures=len(fails))

if __name__=="__main__":main()
