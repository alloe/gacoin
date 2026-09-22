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
LATS=[0,100,200,300,500,1000]
MARGINS=[0,1,2,3,5,7.5,10]
QTYS=[0.001,0.005,0.01]
STRESS_BPS=[0,2,5]
def log(m,**kw): print(m+(" "+json.dumps(kw,ensure_ascii=False,sort_keys=True) if kw else ""),flush=True)
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
    y,m,d=date.split("-"); return f"{BASE}/{ex}/quotes/{y}/{m}/{d}/{sym}.csv.gz"
def load(ex,date,sym):
    r=requests.get(url(ex,date,sym),stream=True,timeout=TIMEOUT)
    if r.status_code!=200: raise RuntimeError(f"HTTP {r.status_code} {url(ex,date,sym)}")
    r.raw.decode_content=False
    rd=csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw),encoding="utf-8",newline=""))
    ts=[];bp=[];ba=[];ap=[];aa=[];cur=None;v=None;rows=0
    for row in rd:
        rows+=1
        try:
            t=ts_us(row["local_timestamp"]);b=float(row["bid_price"]);bs=float(row["bid_amount"]);a=float(row["ask_price"]);az=float(row["ask_amount"])
            if not(b>0 and a>=b and bs>=0 and az>=0): continue
        except: continue
        buck=(t//STEP_US)*STEP_US; nv=(b,bs,a,az)
        if cur is None: cur=buck;v=nv
        elif buck==cur: v=nv
        else:
            ts.append(cur);bp.append(v[0]);ba.append(v[1]);ap.append(v[2]);aa.append(v[3]);cur=buck;v=nv
    if cur is not None: ts.append(cur);bp.append(v[0]);ba.append(v[1]);ap.append(v[2]);aa.append(v[3])
    log("AGG",exchange=ex,date=date,symbol=sym,rows=rows,buckets=len(ts))
    return tuple(np.array(x,dtype=(np.int64 if i==0 else float)) for i,x in enumerate([ts,bp,ba,ap,aa]))
def grid(q,start,end):
    ts,bp,ba,ap,aa=q;n=int((end-start)//STEP_US)+1
    outs=[np.full(n,np.nan) for _ in range(4)]
    idx=((ts-start)//STEP_US).astype(np.int64);ok=(idx>=0)&(idx<n);ii=idx[ok]
    for o,v in zip(outs,[bp,ba,ap,aa]):o[ii]=v[ok]
    valid=np.where(np.isfinite(outs[0])&np.isfinite(outs[2]),np.arange(n),-1)
    np.maximum.accumulate(valid,out=valid);m=valid>=0
    for o in outs:o[m]=o[valid[m]]
    return outs
def starts_above(edge,margin):
    out=[];inside=False
    for i,x in enumerate(edge):
        hit=math.isfinite(x) and x>=margin
        if hit and not inside: out.append(i)
        inside=hit
    return out
def mean(x): return statistics.fmean(x) if x else None
def med(x): return statistics.median(x) if x else None
def pctl(x,q): return float(np.percentile(x,q)) if x else None
def process(date):
    ub=load("upbit",date,"KRW-BTC"); uu=load("upbit",date,"KRW-USDT"); bn=load("binance",date,"BTCUSDT")
    start=max(ub[0][0],uu[0][0],bn[0][0]); end=min(ub[0][-1],uu[0][-1],bn[0][-1])
    ubb,ubsz,uba,uasz=grid(ub,start,end); usb,usbsz,usa,usasz=grid(uu,start,end); bnb,bnbsz,bna,bnasz=grid(bn,start,end)
    # A: buy Upbit BTC ask, sell Binance BTC bid, valuing received USDT at Upbit KRW-USDT bid.
    # B: buy Binance BTC ask, sell Upbit BTC bid, valuing USDT funding at Upbit KRW-USDT ask.
    sell_global=bnb*usb; buy_global=bna*usa
    upf=UP_FEE_BP/10000; bnf=BN_FEE_BP/10000
    netA=np.log((sell_global*(1-bnf))/(uba*(1+upf)))*10000
    netB=np.log((ubb*(1-upf))/(buy_global*(1+bnf)))*10000
    krwA=sell_global*(1-bnf)-uba*(1+upf)
    krwB=ubb*(1-upf)-buy_global*(1+bnf)
    capA=np.minimum(uasz,bnbsz); capB=np.minimum(ubsz,bnasz)
    out={"date":date,"seconds":len(netA)/10,"directions":{}}
    for name,edge,cap,krw in [("A_UPbuy_BNsell",netA,capA,krwA),("B_BNbuy_UPsell",netB,capB,krwB)]:
        r={}
        for margin in MARGINS:
            sig=starts_above(edge,margin)
            z={"signals":len(sig),"latency":{}}
            for lat in LATS:
                k=lat//100
                valid=[i for i in sig if i+k<len(edge) and math.isfinite(edge[i+k])]
                q={}
                for qty in QTYS:
                    full=[i for i in valid if cap[i+k]>=qty]
                    eb=[float(edge[i+k]) for i in full]
                    pnl=[float(krw[i+k]*qty) for i in full]
                    stress={}
                    # approximate extra all-in lifecycle cost stress in bp, applied to notional.
                    mid=[float((sell_global[i+k]+uba[i+k])/2) if name.startswith("A_") else float((ubb[i+k]+buy_global[i+k])/2) for i in full]
                    for sbp in STRESS_BPS:
                        sp=[p-(m*qty*sbp/10000) for p,m in zip(pnl,mid)]
                        stress[str(sbp)]={"mean_pnl_krw":mean(sp),"median_pnl_krw":med(sp),"sum_pnl_krw":sum(sp),
                                          "positive_rate":(sum(x>0 for x in sp)/len(sp) if sp else None)}
                    q[str(qty)]={"n":len(full),"full_fill_rate":(len(full)/len(valid) if valid else None),
                                 "mean_edge_bp":mean(eb),"median_edge_bp":med(eb),"p10_edge_bp":pctl(eb,10),
                                 "mean_pnl_krw":mean(pnl),"sum_pnl_krw":sum(pnl),
                                 "positive_rate":(sum(x>0 for x in pnl)/len(pnl) if pnl else None),
                                 "stress":stress}
                z["latency"][str(lat)]={"valid_n":len(valid),"qty":q}
            r[str(margin)]=z
        out["directions"][name]=r
    log("DAY_DONE",date=date,A0=len(starts_above(netA,0)),B0=len(starts_above(netB,0)))
    return out
def combine(days):
    out={}
    for direction in ["A_UPbuy_BNsell","B_BNbuy_UPsell"]:
        out[direction]={}
        for margin in MARGINS:
            mz={"signals":sum(d["directions"][direction][str(margin)]["signals"] for d in days),"latency":{}}
            for lat in LATS:
                lz={"qty":{}}
                for qty in QTYS:
                    rows=[d["directions"][direction][str(margin)]["latency"][str(lat)]["qty"][str(qty)] for d in days]
                    n=sum(x["n"] for x in rows)
                    # weighted means
                    def wavg(field):
                        return (sum((x[field] or 0)*x["n"] for x in rows)/n) if n else None
                    all_stress={}
                    for sbp in STRESS_BPS:
                        ss=[x["stress"][str(sbp)] for x in rows]
                        sn=sum(x["n"] for x in rows)
                        all_stress[str(sbp)]={"mean_pnl_krw":(sum((s["mean_pnl_krw"] or 0)*x["n"] for s,x in zip(ss,rows))/sn if sn else None),
                                              "sum_pnl_krw":sum(s["sum_pnl_krw"] for s in ss),
                                              "positive_rate":(sum((s["positive_rate"] or 0)*x["n"] for s,x in zip(ss,rows))/sn if sn else None)}
                    lz["qty"][str(qty)]={"n":n,"mean_edge_bp":wavg("mean_edge_bp"),"mean_pnl_krw":wavg("mean_pnl_krw"),
                                         "sum_pnl_krw":sum(x["sum_pnl_krw"] for x in rows),
                                         "positive_rate":(sum((x["positive_rate"] or 0)*x["n"] for x in rows)/n if n else None),
                                         "stress":all_stress}
                mz["latency"][str(lat)]=lz
            out[direction][str(margin)]=mz
    return out
def main():
    log("PHASE5_STANDARD_START",dates=DATES,up_fee_bp=UP_FEE_BP,bn_fee_bp=BN_FEE_BP,margins=MARGINS,qtys=QTYS,stress_bps=STRESS_BPS)
    days=[];fails=[]
    for d in DATES:
        try: days.append(process(d))
        except Exception as e: fails.append({"date":d,"error":repr(e)}); log("FAIL",date=d,error=repr(e))
    combined=combine(days)
    print("PHASE5_JSON="+json.dumps({"dates":DATES,"failures":fails,"days":days,"combined":combined},separators=(",",":")),flush=True)
    print("PHASE5_SUMMARY",flush=True)
    for direction in combined:
        print(direction,flush=True)
        for margin in MARGINS:
            m=combined[direction][str(margin)]
            parts=[]
            for lat in [100,200,300]:
                q=m["latency"][str(lat)]["qty"]["0.01"]
                s2=q["stress"]["2"]
                parts.append(f"{lat}ms n={q['n']} edge={q['mean_edge_bp']} pnl={q['mean_pnl_krw']} pos={q['positive_rate']} stress2_pnl={s2['mean_pnl_krw']}")
            print(f"margin>={margin}bp signals={m['signals']} | "+" | ".join(parts),flush=True)
    log("PHASE5_DONE",days=len(days),failures=len(fails))
if __name__=="__main__": main()
