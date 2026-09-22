#!/usr/bin/env python3
import json, math, os, statistics, time
from datetime import datetime, timezone
import requests, numpy as np

START=os.getenv("BT_START","2025-10-01T00:00:00Z")
END=os.getenv("BT_END","2026-09-22T00:00:00Z")
INTERVAL_MIN=10
UP_FEE_BP=5.0
BN_FUT_TAKER_BP=5.0
SLIP_RT_BP=float(os.getenv("SLIP_RT_BP","4"))
BASE_NOTIONAL=1_000_000.0
COINS=["DOGE","SHIB","HBAR","SOL","BTC","ETH"]
FUTSYM={c:(("1000SHIBUSDT",1000.0) if c=="SHIB" else (c+"USDT",1.0)) for c in COINS}
S=requests.Session();S.headers.update({"User-Agent":"survivor-validation/1.0"})

def log(m,**kw):print(m+(" "+json.dumps(kw,ensure_ascii=False,sort_keys=True) if kw else ""),flush=True)
def ms(s):return int(datetime.fromisoformat(s.replace("Z","+00:00")).timestamp()*1000)
START_MS=ms(START);END_MS=ms(END);MID_MS=(START_MS+END_MS)//2

def upbit(market):
    rows={};to=datetime.fromtimestamp(END_MS/1000,tz=timezone.utc);startdt=datetime.fromtimestamp(START_MS/1000,tz=timezone.utc);calls=0
    while to>startdt:
        r=S.get(f"https://api.upbit.com/v1/candles/minutes/{INTERVAL_MIN}",
                params={"market":market,"to":to.isoformat().replace("+00:00","Z"),"count":200},timeout=30)
        if r.status_code!=200:raise RuntimeError(f"UPBIT {market} {r.status_code} {r.text[:120]}")
        a=r.json();calls+=1
        if not a:break
        for x in a:
            t=int(datetime.fromisoformat(x["candle_date_time_utc"]).replace(tzinfo=timezone.utc).timestamp()*1000)
            if START_MS<=t<END_MS:rows[t]=(float(x["opening_price"]),float(x["trade_price"]),float(x["candle_acc_trade_price"]))
        oldest=datetime.fromisoformat(a[-1]["candle_date_time_utc"]).replace(tzinfo=timezone.utc)
        if oldest<=startdt:break
        to=oldest;time.sleep(0.105)
    log("UP_DONE",market=market,bars=len(rows),calls=calls);return rows

def bn_fut(sym,mult):
    out={};cur=START_MS;calls=0
    while cur<END_MS:
        r=S.get("https://fapi.binance.com/fapi/v1/klines",params={"symbol":sym,"interval":"10m","startTime":cur,"endTime":END_MS-1,"limit":1500},timeout=30)
        if r.status_code!=200:raise RuntimeError(f"BN {sym} {r.status_code} {r.text[:120]}")
        a=r.json();calls+=1
        if not a:break
        for x in a:
            t=int(x[0]);out[t]=(float(x[1])/mult,float(x[4])/mult,float(x[7]))
        nxt=int(a[-1][0])+INTERVAL_MIN*60*1000
        if nxt<=cur:break
        cur=nxt;time.sleep(.02)
    log("BN_DONE",symbol=sym,bars=len(out),calls=calls);return out

def funding(sym):
    out=[];cur=START_MS
    while cur<END_MS:
        r=S.get("https://fapi.binance.com/fapi/v1/fundingRate",params={"symbol":sym,"startTime":cur,"endTime":END_MS-1,"limit":1000},timeout=30)
        if r.status_code!=200:break
        a=r.json()
        if not a:break
        out += [(int(x["fundingTime"]),float(x["fundingRate"])) for x in a if START_MS<=int(x["fundingTime"])<END_MS]
        nxt=int(a[-1]["fundingTime"])+1
        if nxt<=cur or len(a)<1000:break
        cur=nxt;time.sleep(.03)
    return sorted(set(out))

def align(up,fut,fx):
    ks=sorted(set(up)&set(fut)&set(fx))
    return {"t":np.array(ks,np.int64),
            "uo":np.array([up[k][0] for k in ks],float),"uc":np.array([up[k][1] for k in ks],float),
            "fo":np.array([fut[k][0] for k in ks],float),"fc":np.array([fut[k][1] for k in ks],float),
            "xo":np.array([fx[k][0] for k in ks],float),"xc":np.array([fx[k][1] for k in ks],float)}

def rms(x,w):
    n=len(x);m=np.full(n,np.nan);s=np.full(n,np.nan);cs=np.r_[0,np.cumsum(x)];cs2=np.r_[0,np.cumsum(x*x)]
    for i in range(w-1,n):
        a=cs[i+1]-cs[i+1-w];b=cs2[i+1]-cs2[i+1-w];mu=a/w;m[i]=mu;s[i]=math.sqrt(max(0,b/w-mu*mu))
    return m,s

def fundp(funds,t0,t1,side,notional):
    return sum(-side*notional*r for t,r in funds if t0<t<=t1)

def pnl(direction,e,x,a,funds):
    qty=BASE_NOTIONAL/a["uo"][e]
    p=direction*qty*(a["uo"][x]-a["uo"][e]) + (-direction)*qty*(a["fo"][x]-a["fo"][e])*a["xo"][x]
    p+=fundp(funds,int(a["t"][e]),int(a["t"][x]),-direction,BASE_NOTIONAL)
    p-=BASE_NOTIONAL*((2*UP_FEE_BP+2*BN_FUT_TAKER_BP+SLIP_RT_BP)/10000)
    return p

def backtest(coin,a,funds):
    prem=np.log(a["uc"]/(a["fc"]*a["xc"]))*10000
    w=14*24*(60//INTERVAL_MIN);mu,sd=rms(prem,w)
    cost=2*UP_FEE_BP+2*BN_FUT_TAKER_BP+SLIP_RT_BP
    tr=[];i=w
    while i<len(prem)-3:
        if not math.isfinite(sd[i]) or sd[i]<3:i+=1;continue
        z=(prem[i]-mu[i])/sd[i];dev=abs(prem[i]-mu[i])
        if abs(z)<2 or dev<cost+8:i+=1;continue
        direction=-1 if z>0 else 1;e=i+1;j=e+1;mx=min(len(prem)-1,e+24*(60//INTERVAL_MIN))
        while j<mx:
            zj=(prem[j]-mu[j])/sd[j] if math.isfinite(sd[j]) and sd[j]>0 else z
            if (z>0 and zj<=.5) or (z<0 and zj>=-.5):break
            j+=1
        x=min(j+1,len(prem)-1);p=pnl(direction,e,x,a,funds)
        tr.append({"coin":coin,"entry":int(a["t"][e]),"exit":int(a["t"][x]),"pnl":p,
                   "score":dev-cost,"hold_h":(x-e)*INTERVAL_MIN/60})
        i=x+1
    return tr

def stats(tr):
    if not tr:return {"trades":0}
    ps=[x["pnl"] for x in tr];bps=[p/BASE_NOTIONAL*10000 for p in ps]
    eq=peak=mdd=0
    for p in ps:eq+=p;peak=max(peak,eq);mdd=min(mdd,eq-peak)
    return {"trades":len(tr),"win":sum(p>0 for p in ps)/len(ps),"avg_bp":statistics.fmean(bps),
            "median_bp":statistics.median(bps),"total_bp":sum(bps),"pnl_1m":sum(ps),
            "maxdd_1m":mdd,"avg_hold_h":statistics.fmean([x["hold_h"] for x in tr])}

def monthly(tr):
    d={}
    for x in tr:
        k=datetime.fromtimestamp(x["entry"]/1000,tz=timezone.utc).strftime("%Y-%m")
        d[k]=d.get(k,0)+x["pnl"]
    return d

def portfolio(trades,capital,maxpos=3):
    # Conservative 1x hedge: each pair consumes 2x notional (spot cash + futures margin).
    by={}
    for x in trades:by.setdefault(x["entry"],[]).append(x)
    openp=[];real=0;peak=0;mdd=0;accepted=0;month={}
    for t in sorted(by):
        keep=[]
        for p in openp:
            if p["exit"]<=t:
                real+=p["scaled"];peak=max(peak,real);mdd=min(mdd,real-peak)
                k=datetime.fromtimestamp(p["exit"]/1000,tz=timezone.utc).strftime("%Y-%m");month[k]=month.get(k,0)+p["scaled"]
            else:keep.append(p)
        openp=keep;free=maxpos-len(openp)
        notional=capital/(maxpos*2)
        for x in sorted(by[t],key=lambda z:z["score"],reverse=True)[:free]:
            q=dict(x);q["scaled"]=x["pnl"]*(notional/BASE_NOTIONAL);openp.append(q);accepted+=1
    for p in openp:
        real+=p["scaled"];peak=max(peak,real);mdd=min(mdd,real-peak)
        k=datetime.fromtimestamp(p["exit"]/1000,tz=timezone.utc).strftime("%Y-%m");month[k]=month.get(k,0)+p["scaled"]
    months=(END_MS-START_MS)/(1000*86400*30.4375)
    return {"capital":capital,"maxpos":maxpos,"notional_per_pair":capital/(maxpos*2),"accepted":accepted,
            "pnl":real,"return_pct":real/capital*100,"monthly_pct":real/capital/months*100,
            "maxdd_pct":mdd/capital*100,"monthly_pnl":month}

def main():
    log("PHASE9_START",start=START,end=END,interval=INTERVAL_MIN,coins=COINS)
    fx=upbit("KRW-USDT");alltr=[];res={};fails=[]
    for c in COINS:
        try:
            u=upbit("KRW-"+c);sym,m=FUTSYM[c];f=bn_fut(sym,m);a=align(u,f,fx);fd=funding(sym);tr=backtest(c,a,fd);alltr+=tr
            first=[x for x in tr if x["entry"]<MID_MS];second=[x for x in tr if x["entry"]>=MID_MS]
            res[c]={"full":stats(tr),"first_half":stats(first),"second_half":stats(second),"monthly":monthly(tr)}
            log("COIN9",coin=c,result=res[c])
        except Exception as e:fails.append({"coin":c,"error":repr(e)});log("FAIL9",coin=c,error=repr(e))
    ports={}
    for cap in [10_000_000,20_000_000,50_000_000]:
        for mp in [2,3]:
            ports[f"{cap}_{mp}"]=portfolio(alltr,cap,mp)
    out={"failures":fails,"coins":res,"portfolio":ports}
    print("PHASE9_SUMMARY "+json.dumps(out,separators=(",",":")),flush=True)
    log("PHASE9_DONE",trades=len(alltr),failures=len(fails))
if __name__=="__main__":main()
