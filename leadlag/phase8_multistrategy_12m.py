#!/usr/bin/env python3
import json, math, os, statistics, time
from datetime import datetime, timezone, timedelta
import requests
import numpy as np

START=os.getenv("BT_START","2025-10-01T00:00:00Z")
END=os.getenv("BT_END","2026-09-22T00:00:00Z")
INTERVAL_MIN=30
UP_FEE_BP=5.0
BN_FUT_TAKER_BP=5.0
BN_SPOT_TAKER_BP=10.0
SLIP_RT_BP=float(os.getenv("SLIP_RT_BP","4"))
NOTIONAL_KRW=float(os.getenv("NOTIONAL_KRW","1000000"))
COINS=["BTC","ETH","XRP","SOL","DOGE","ADA","LINK","AVAX","DOT","BCH","ETC","AAVE","NEAR","SUI","TRX","SHIB","XLM","HBAR","UNI","APT"]
FUTSYM={c:(("1000SHIBUSDT",1000.0) if c=="SHIB" else (c+"USDT",1.0)) for c in COINS}
SPOTSYM={c:(c+"USDT",1.0) for c in COINS}
UA="multi-strategy-paper-backtest/1.0"
S=requests.Session();S.headers.update({"User-Agent":UA})

def log(m,**kw): print(m+(" "+json.dumps(kw,ensure_ascii=False,sort_keys=True) if kw else ""),flush=True)
def ts(s):
    return int(datetime.fromisoformat(s.replace("Z","+00:00")).timestamp()*1000)
START_MS=ts(START);END_MS=ts(END)

def upbit_candles(market,unit=30):
    rows=[]; to=datetime.fromtimestamp(END_MS/1000,tz=timezone.utc)
    startdt=datetime.fromtimestamp(START_MS/1000,tz=timezone.utc)
    calls=0
    while to>startdt:
        p={"market":market,"to":to.isoformat().replace("+00:00","Z"),"count":200}
        r=S.get(f"https://api.upbit.com/v1/candles/minutes/{unit}",params=p,timeout=30)
        if r.status_code!=200: raise RuntimeError(f"UPBIT {market} {r.status_code} {r.text[:120]}")
        a=r.json();calls+=1
        if not a: break
        for x in a:
            t=int(datetime.fromisoformat(x["candle_date_time_utc"]).replace(tzinfo=timezone.utc).timestamp()*1000)
            if START_MS<=t<END_MS:
                rows.append((t,float(x["opening_price"]),float(x["trade_price"]),float(x["candle_acc_trade_volume"]),float(x["candle_acc_trade_price"])))
        oldest=datetime.fromisoformat(a[-1]["candle_date_time_utc"]).replace(tzinfo=timezone.utc)
        if oldest<=startdt: break
        to=oldest
        time.sleep(0.105)
    d={}
    for r in rows:d[r[0]]=r[1:]
    log("UPBIT_DONE",market=market,bars=len(d),calls=calls)
    return d

def binance_klines(base,symbol,interval="30m",mult=1.0):
    url=("https://fapi.binance.com/fapi/v1/klines" if base=="fut" else "https://api.binance.com/api/v3/klines")
    out={};cur=START_MS;calls=0
    while cur<END_MS:
        p={"symbol":symbol,"interval":interval,"startTime":cur,"endTime":END_MS-1,"limit":1500 if base=="fut" else 1000}
        r=S.get(url,params=p,timeout=30)
        if r.status_code!=200: raise RuntimeError(f"BN {base} {symbol} {r.status_code} {r.text[:120]}")
        a=r.json();calls+=1
        if not a: break
        for x in a:
            t=int(x[0])
            if START_MS<=t<END_MS:
                out[t]=(float(x[1])/mult,float(x[4])/mult,float(x[5])*mult,float(x[7]))
        nxt=int(a[-1][0])+INTERVAL_MIN*60*1000
        if nxt<=cur: break
        cur=nxt
        time.sleep(0.02)
    log("BINANCE_DONE",base=base,symbol=symbol,bars=len(out),calls=calls)
    return out

def funding(symbol):
    out=[];cur=START_MS;calls=0
    while cur<END_MS:
        p={"symbol":symbol,"startTime":cur,"endTime":END_MS-1,"limit":1000}
        r=S.get("https://fapi.binance.com/fapi/v1/fundingRate",params=p,timeout=30)
        if r.status_code!=200:
            log("FUNDING_FAIL",symbol=symbol,status=r.status_code);break
        a=r.json();calls+=1
        if not a: break
        for x in a:
            t=int(x["fundingTime"])
            if START_MS<=t<END_MS: out.append((t,float(x["fundingRate"])))
        nxt=int(a[-1]["fundingTime"])+1
        if nxt<=cur or len(a)<1000:break
        cur=nxt;time.sleep(0.03)
    out=sorted(set(out));log("FUNDING_DONE",symbol=symbol,n=len(out),calls=calls)
    return out

def align(up,fut,spot,fx):
    ks=sorted(set(up)&set(fut)&set(spot)&set(fx))
    if len(ks)<1000:return None
    arr={}
    arr["t"]=np.array(ks,dtype=np.int64)
    for name,d,idx in [("uo",up,0),("uc",up,1),("uv",up,2),("uq",up,3),
                       ("fo",fut,0),("fc",fut,1),("fv",fut,2),("fq",fut,3),
                       ("so",spot,0),("sc",spot,1),("sv",spot,2),("sq",spot,3),
                       ("xo",fx,0),("xc",fx,1)]:
        arr[name]=np.array([d[k][idx] for k in ks],float)
    return arr

def rolling_mean_std(x,w):
    n=len(x);mean=np.full(n,np.nan);std=np.full(n,np.nan)
    cs=np.concatenate(([0.0],np.cumsum(x)))
    cs2=np.concatenate(([0.0],np.cumsum(x*x)))
    for i in range(w-1,n):
        s=cs[i+1]-cs[i+1-w];s2=cs2[i+1]-cs2[i+1-w]
        m=s/w;v=max(0.0,s2/w-m*m);mean[i]=m;std[i]=math.sqrt(v)
    return mean,std

def funding_between(funds,t0,t1,side,notional):
    # side +1 = long future, -1 = short future
    z=0.0
    for t,r in funds:
        if t0<t<=t1:
            z += -side*notional*r
    return z

def pair_pnl(direction,entry_i,exit_i,a,funds,use_spot=False):
    # direction +1: long Upbit, short global leg. -1: short inventory Upbit, long global leg.
    u0=a["uo"][entry_i];u1=a["uo"][exit_i]
    if use_spot:
        g0=a["so"][entry_i];g1=a["so"][exit_i];fx1=a["xo"][exit_i];fee_g=BN_SPOT_TAKER_BP
    else:
        g0=a["fo"][entry_i];g1=a["fo"][exit_i];fx1=a["xo"][exit_i];fee_g=BN_FUT_TAKER_BP
    qty=NOTIONAL_KRW/u0
    spotleg=direction*qty*(u1-u0)
    globalleg=-direction*qty*(g1-g0)*fx1
    fees=NOTIONAL_KRW*((2*UP_FEE_BP+2*fee_g+SLIP_RT_BP)/10000)
    fp=0.0 if use_spot else funding_between(funds,int(a["t"][entry_i]),int(a["t"][exit_i]),-direction,NOTIONAL_KRW)
    return spotleg+globalleg+fp-fees,fp

def trade_stats(trades):
    if not trades:return {"trades":0}
    ps=[x["pnl"] for x in trades];bps=[p/NOTIONAL_KRW*10000 for p in ps]
    eq=0;peak=0;mdd=0
    for p in ps:
        eq+=p;peak=max(peak,eq);mdd=min(mdd,eq-peak)
    return {"trades":len(ps),"win_rate":sum(p>0 for p in ps)/len(ps),
            "avg_bp":statistics.fmean(bps),"median_bp":statistics.median(bps),
            "total_bp":sum(bps),"total_krw_per_1m":sum(ps),"max_dd_krw_per_1m":mdd,
            "avg_hold_h":statistics.fmean([x["hold_bars"]*0.5 for x in trades])}

def premium_mr(a,funds,use_spot=False):
    g=a["sc"] if use_spot else a["fc"]
    prem=np.log(a["uc"]/(g*a["xc"]))*10000
    w=14*48 # 14 days
    mu,sd=rolling_mean_std(prem,w)
    trades=[];i=w
    cost=2*UP_FEE_BP+2*(BN_SPOT_TAKER_BP if use_spot else BN_FUT_TAKER_BP)+SLIP_RT_BP
    while i<len(prem)-2:
        if not math.isfinite(sd[i]) or sd[i]<3:i+=1;continue
        z=(prem[i]-mu[i])/sd[i]
        dev=abs(prem[i]-mu[i])
        if abs(z)<2.0 or dev<cost+8:i+=1;continue
        direction=-1 if z>0 else 1
        entry=i+1; j=entry+1; maxj=min(len(prem)-1,entry+48) # <=24h
        while j<maxj:
            zj=(prem[j]-mu[j])/sd[j] if math.isfinite(sd[j]) and sd[j]>0 else z
            if (z>0 and zj<=0.5) or (z<0 and zj>=-0.5):break
            j+=1
        exit_i=min(j+1,len(prem)-1)
        pnl,fp=pair_pnl(direction,entry,exit_i,a,funds,use_spot)
        score=max(0.0,dev-cost)
        trades.append({"entry":int(a["t"][entry]),"exit":int(a["t"][exit_i]),"pnl":pnl,"funding":fp,
                       "hold_bars":exit_i-entry,"score":score,"strategy":"spotspot_mr" if use_spot else "spotperp_mr"})
        i=exit_i+1
    return trades

def shock_leadlag(a,funds,event_mode=False):
    ru=np.diff(np.log(a["uc"]),prepend=np.log(a["uc"][0]))*10000
    rf=np.diff(np.log(a["fc"]),prepend=np.log(a["fc"][0]))*10000
    gap=rf-ru
    # volume z-score on Binance futures quote volume
    v=np.log1p(a["fq"]);vm,vs=rolling_mean_std(v,48*7)
    trades=[];last_exit=-1
    for i in range(48*7,len(ru)-5):
        if i<=last_exit:continue
        vz=(v[i]-vm[i])/vs[i] if math.isfinite(vs[i]) and vs[i]>0 else 0
        shock=120 if event_mode else 55
        gapth=45 if event_mode else 25
        if abs(rf[i])<shock or abs(gap[i])<gapth:continue
        if event_mode and vz<2.0:continue
        if rf[i]*gap[i]<=0:continue # Upbit underreacted in same direction
        direction=1 if rf[i]>0 else -1
        entry=i+1
        # test 30m, 60m, 120m; choose fixed 60m for no hindsight
        exit_i=min(entry+2,len(ru)-1)
        pnl,fp=pair_pnl(direction,entry,exit_i,a,funds,False)
        score=abs(gap[i])-20 + max(0,vz)*5
        trades.append({"entry":int(a["t"][entry]),"exit":int(a["t"][exit_i]),"pnl":pnl,"funding":fp,
                       "hold_bars":exit_i-entry,"score":score,"strategy":"event_shock" if event_mode else "leadlag_30m"})
        last_exit=exit_i
    return trades

def funding_basis(a,funds):
    if len(funds)<10:return []
    # last known funding rate at each bar; 3-settlement trailing average (known only)
    ft=np.array([x[0] for x in funds],np.int64);fr=np.array([x[1] for x in funds],float)
    prem=np.log(a["uc"]/(a["fc"]*a["xc"]))*10000
    trades=[];inpos=False;i=0
    while i<len(a["t"])-50:
        k=np.searchsorted(ft,a["t"][i],side="right")-1
        if k<2:i+=1;continue
        av=float(np.mean(fr[k-2:k+1]))
        # expected 24h funding bp using last 3 average, roughly 3 settlements
        exp_bp=abs(av)*3*10000
        if exp_bp<12:i+=1;continue
        # require basis direction not strongly against carry
        direction=1 if av>0 else -1 # positive funding => long spot / short perp
        if direction==1 and prem[i]<-25:i+=1;continue
        if direction==-1 and prem[i]>25:i+=1;continue
        entry=i+1;exit_i=min(entry+48,len(a["t"])-1)
        pnl,fp=pair_pnl(direction,entry,exit_i,a,funds,False)
        score=exp_bp+max(0,direction*prem[i])
        trades.append({"entry":int(a["t"][entry]),"exit":int(a["t"][exit_i]),"pnl":pnl,"funding":fp,
                       "hold_bars":exit_i-entry,"score":score,"strategy":"funding_basis"})
        i=exit_i+1
    return trades

def portfolio(trades,capital=50_000_000,max_pos=5):
    # equal slot allocation, score-ranked at each timestamp, no look-ahead
    by={}
    for tr in trades:by.setdefault(tr["entry"],[]).append(tr)
    openpos=[];cash=capital;realized=0;curve=[];accepted=0
    times=sorted(by)
    for t in times:
        remain=[]
        for p in openpos:
            if p["exit"]<=t:
                realized += p["scaled_pnl"]; cash += p["slot"]+p["scaled_pnl"]
            else:remain.append(p)
        openpos=remain
        free=max_pos-len(openpos)
        if free>0:
            cand=sorted(by[t],key=lambda x:x["score"],reverse=True)[:free]
            for tr in cand:
                slot=capital/max_pos
                if cash<slot:continue
                scale=slot/NOTIONAL_KRW
                q=dict(tr);q["slot"]=slot;q["scaled_pnl"]=tr["pnl"]*scale
                cash-=slot;openpos.append(q);accepted+=1
        curve.append(realized)
    for p in openpos:realized+=p["scaled_pnl"]
    ret=realized/capital
    months=(END_MS-START_MS)/(1000*86400*30.4375)
    return {"capital":capital,"max_positions":max_pos,"accepted":accepted,"pnl":realized,
            "return_pct":ret*100,"monthly_pct":ret/months*100}

def main():
    log("PHASE8_START",start=START,end=END,interval_min=INTERVAL_MIN,coins=COINS,
        fees={"upbit_bp":UP_FEE_BP,"binance_fut_taker_bp":BN_FUT_TAKER_BP,"binance_spot_taker_bp":BN_SPOT_TAKER_BP,"slip_rt_bp":SLIP_RT_BP})
    fx=upbit_candles("KRW-USDT",INTERVAL_MIN)
    all_trades=[];results={};fails=[]
    for c in COINS:
        try:
            log("COIN_START",coin=c)
            up=upbit_candles("KRW-"+c,INTERVAL_MIN)
            fsym,fmult=FUTSYM[c];ssym,smult=SPOTSYM[c]
            fut=binance_klines("fut",fsym,"30m",fmult)
            spot=binance_klines("spot",ssym,"30m",smult)
            a=align(up,fut,spot,fx)
            if a is None:raise RuntimeError("insufficient aligned bars")
            funds=funding(fsym)
            strat={}
            for name,tr in [
                ("spotperp_mr",premium_mr(a,funds,False)),
                ("leadlag_30m",shock_leadlag(a,funds,False)),
                ("funding_basis",funding_basis(a,funds)),
                ("spotspot_mr",premium_mr(a,funds,True)),
                ("event_shock",shock_leadlag(a,funds,True))]:
                for x in tr:x["coin"]=c
                all_trades.extend(tr);strat[name]=trade_stats(tr)
            results[c]={"bars":len(a["t"]),"strategies":strat}
            log("COIN_DONE",coin=c,bars=len(a["t"]),summary=strat)
        except Exception as e:
            fails.append({"coin":c,"error":repr(e)});log("COIN_FAIL",coin=c,error=repr(e))
    # Aggregate per strategy across coins
    agg={}
    for name in ["spotperp_mr","leadlag_30m","funding_basis","spotspot_mr","event_shock"]:
        tr=[x for x in all_trades if x["strategy"]==name]
        agg[name]=trade_stats(sorted(tr,key=lambda x:x["entry"]))
    # Cross-sectional portfolio on all candidate hedged strategies except spotspot baseline
    candidate=[x for x in all_trades if x["strategy"] in ("spotperp_mr","leadlag_30m","funding_basis","event_shock")]
    ports={str(k):portfolio(candidate,50_000_000,k) for k in (3,5)}
    out={"period":{"start":START,"end":END,"interval_min":INTERVAL_MIN},"fees":{"upbit_bp":UP_FEE_BP,"fut_taker_bp":BN_FUT_TAKER_BP,"spot_taker_bp":BN_SPOT_TAKER_BP,"slip_rt_bp":SLIP_RT_BP},
         "failures":fails,"coins":results,"aggregate":agg,"portfolio":ports}
    print("PHASE8_JSON="+json.dumps(out,separators=(",",":")),flush=True)
    print("PHASE8_SUMMARY "+json.dumps({"failures":fails,"aggregate":agg,"portfolio":ports},separators=(",",":")),flush=True)
    log("PHASE8_DONE",coins=len(results),failures=len(fails),trades=len(all_trades))
if __name__=="__main__":main()
