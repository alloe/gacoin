#!/usr/bin/env python3
import os,time,math,json,statistics
from datetime import datetime,timezone,timedelta
import requests
import numpy as np
import pandas as pd

COINS=os.getenv("COINS","BTC,ETH,XRP,SOL,DOGE,ADA,LINK,AVAX,DOT,BCH,ETC,AAVE,NEAR,SUI,TRX,SHIB,XLM,HBAR,UNI,APT").split(",")
DAYS=int(os.getenv("DAYS","90"))
INTERVAL_MIN=15
UP_FEE_BP=float(os.getenv("UP_FEE_BP","5"))
BN_SPOT_FEE_BP=float(os.getenv("BN_SPOT_FEE_BP","10"))
BN_FUT_FEE_BP=float(os.getenv("BN_FUT_FEE_BP","5"))
SLIP_ROUND_BP=float(os.getenv("SLIP_ROUND_BP","4"))
PREM_Z=float(os.getenv("PREM_Z","2.0"))
PREM_DEV_BP=float(os.getenv("PREM_DEV_BP","40"))
MAX_HOLD_BARS=int(os.getenv("MAX_HOLD_BARS","96")) # 24h
FUNDING_MIN_BP=float(os.getenv("FUNDING_MIN_BP","2"))
BASIS_MIN_BP=float(os.getenv("BASIS_MIN_BP","15"))
S=requests.Session()
S.headers.update({"User-Agent":"daol-paper-research/1.0"})

def log(m,**kw): print(m+(" "+json.dumps(kw,ensure_ascii=False,sort_keys=True) if kw else ""),flush=True)
def bp(x): return x*10000.0
def safe_mean(x): 
    a=[v for v in x if v is not None and math.isfinite(v)]
    return statistics.fmean(a) if a else None
def safe_med(x):
    a=[v for v in x if v is not None and math.isfinite(v)]
    return statistics.median(a) if a else None

def upbit_minutes(market,start_ms,end_ms):
    out=[]
    to=datetime.fromtimestamp(end_ms/1000,tz=timezone.utc)
    start_dt=datetime.fromtimestamp(start_ms/1000,tz=timezone.utc)
    while True:
        params={"market":market,"to":to.isoformat().replace("+00:00","Z"),"count":200}
        r=S.get(f"https://api.upbit.com/v1/candles/minutes/{INTERVAL_MIN}",params=params,timeout=20)
        if r.status_code!=200: raise RuntimeError(f"upbit {market} {r.status_code} {r.text[:150]}")
        rows=r.json()
        if not rows: break
        for x in rows:
            t=int(datetime.fromisoformat(x["candle_date_time_utc"]+"+00:00").timestamp()*1000)
            if start_ms<=t<=end_ms: out.append((t,float(x["trade_price"])))
        oldest=datetime.fromisoformat(rows[-1]["candle_date_time_utc"]+"+00:00")
        if oldest<=start_dt: break
        to=oldest-timedelta(seconds=1)
        time.sleep(0.11)
    out=sorted(set(out))
    return pd.Series([v for _,v in out],index=pd.to_datetime([t for t,_ in out],unit="ms",utc=True),dtype=float)

def binance_klines(symbol,start_ms,end_ms,futures=False):
    url="https://fapi.binance.com/fapi/v1/klines" if futures else "https://api.binance.com/api/v3/klines"
    out=[];cur=start_ms
    while cur<end_ms:
        params={"symbol":symbol,"interval":"15m","startTime":cur,"endTime":end_ms,"limit":1000}
        r=S.get(url,params=params,timeout=20)
        if r.status_code!=200: raise RuntimeError(f"binance {'fut' if futures else 'spot'} {symbol} {r.status_code} {r.text[:150]}")
        rows=r.json()
        if not rows: break
        out.extend((int(x[0]),float(x[4])) for x in rows)
        nxt=int(rows[-1][0])+INTERVAL_MIN*60*1000
        if nxt<=cur: break
        cur=nxt
        time.sleep(0.03)
    out=sorted(set((t,v) for t,v in out if start_ms<=t<=end_ms))
    return pd.Series([v for _,v in out],index=pd.to_datetime([t for t,_ in out],unit="ms",utc=True),dtype=float)

def funding(symbol,start_ms,end_ms):
    out=[];cur=start_ms
    while cur<end_ms:
        r=S.get("https://fapi.binance.com/fapi/v1/fundingRate",params={"symbol":symbol,"startTime":cur,"endTime":end_ms,"limit":1000},timeout=20)
        if r.status_code!=200: raise RuntimeError(f"funding {symbol} {r.status_code} {r.text[:150]}")
        rows=r.json()
        if not rows: break
        out.extend((int(x["fundingTime"]),float(x["fundingRate"])) for x in rows)
        nxt=int(rows[-1]["fundingTime"])+1
        if nxt<=cur: break
        cur=nxt
        if len(rows)<1000: break
    return sorted(set(out))

def rolling_z(s,window=672):
    med=s.rolling(window,min_periods=max(96,window//4)).median()
    std=s.rolling(window,min_periods=max(96,window//4)).std()
    return med,std,(s-med)/std.replace(0,np.nan)

def funding_sum_bp(frows,start_ts,end_ts,short_futures):
    a=int(start_ts.timestamp()*1000);b=int(end_ts.timestamp()*1000)
    s=0.0
    for t,r in frows:
        if a<t<=b:
            # positive funding: longs pay shorts
            s += bp(r) if short_futures else -bp(r)
    return s

def premium_trades(df,frows,kind):
    # kind: perp or spot
    if kind=="perp":
        prem=np.log(df["up"]/(df["fut"]*df["fx"]))*10000
        round_cost=2*UP_FEE_BP+2*BN_FUT_FEE_BP+SLIP_ROUND_BP
    else:
        prem=np.log(df["up"]/(df["spot"]*df["fx"]))*10000
        round_cost=2*UP_FEE_BP+2*BN_SPOT_FEE_BP+SLIP_ROUND_BP
    med,std,z=rolling_z(prem)
    trades=[];i=673;n=len(df)
    while i<n-2:
        if not all(math.isfinite(v) for v in [prem.iloc[i],med.iloc[i],z.iloc[i]]): i+=1;continue
        dev=prem.iloc[i]-med.iloc[i]
        side=0
        if z.iloc[i]>=PREM_Z and dev>=PREM_DEV_BP: side=-1 # high premium: sell UP / buy hedge
        elif z.iloc[i]<=-PREM_Z and dev<=-PREM_DEV_BP: side=1 # low premium: buy UP / short hedge
        if not side: i+=1;continue
        entry=i+1
        exit_i=min(entry+MAX_HOLD_BARS,n-1)
        for j in range(entry+1,exit_i+1):
            if side==-1 and z.iloc[j]<=0: exit_i=j;break
            if side==1 and z.iloc[j]>=0: exit_i=j;break
        gross = side*(prem.iloc[exit_i]-prem.iloc[entry])
        fund=0.0
        if kind=="perp":
            fund=funding_sum_bp(frows,df.index[entry],df.index[exit_i],short_futures=(side==1))
        net=gross+fund-round_cost
        trades.append({"entry":df.index[entry].isoformat(),"exit":df.index[exit_i].isoformat(),"side":side,
                       "gross_bp":gross,"funding_bp":fund,"cost_bp":round_cost,"net_bp":net,
                       "hold_h":(exit_i-entry)*INTERVAL_MIN/60})
        i=exit_i+1
    return trades

def funding_carry(df,frows):
    # Long Upbit spot + short Binance perp. Enter after funding >= threshold,
    # hold 24h; accumulate actual funding and premium movement.
    prem=np.log(df["up"]/(df["fut"]*df["fx"]))*10000
    cost=2*UP_FEE_BP+2*BN_FUT_FEE_BP+SLIP_ROUND_BP
    idx_ms=np.array([int(x.timestamp()*1000) for x in df.index])
    trades=[];last_exit=-1
    for t,r in frows:
        fr=bp(r)
        if fr<FUNDING_MIN_BP: continue
        i=int(np.searchsorted(idx_ms,t))
        if i<=last_exit or i>=len(df)-2: continue
        entry=min(i+1,len(df)-1);exit_i=min(entry+96,len(df)-1)
        fund=funding_sum_bp(frows,df.index[entry],df.index[exit_i],True)
        # long Upbit short fut => profit when cross premium rises
        gross=prem.iloc[exit_i]-prem.iloc[entry]
        net=gross+fund-cost
        trades.append({"entry":df.index[entry].isoformat(),"exit":df.index[exit_i].isoformat(),
                       "gross_bp":gross,"funding_bp":fund,"cost_bp":cost,"net_bp":net,"hold_h":(exit_i-entry)*0.25})
        last_exit=exit_i
    return trades

def basis_carry(df,frows):
    basis=np.log(df["fut"]/df["spot"])*10000
    cost=2*BN_SPOT_FEE_BP+2*BN_FUT_FEE_BP+SLIP_ROUND_BP
    # use recent funding value carried forward
    fs=pd.Series(np.nan,index=df.index)
    for t,r in frows:
        ts=pd.to_datetime(t,unit="ms",utc=True).floor("15min")
        if ts in fs.index: fs.loc[ts]=bp(r)
    fs=fs.ffill().fillna(0)
    trades=[];i=1;n=len(df)
    while i<n-2:
        if basis.iloc[i]>=BASIS_MIN_BP and fs.iloc[i]>0:
            entry=i+1;exit_i=min(entry+96,n-1)
            for j in range(entry+1,exit_i+1):
                if basis.iloc[j]<=5:exit_i=j;break
            fund=funding_sum_bp(frows,df.index[entry],df.index[exit_i],True)
            gross=basis.iloc[entry]-basis.iloc[exit_i]
            net=gross+fund-cost
            trades.append({"entry":df.index[entry].isoformat(),"exit":df.index[exit_i].isoformat(),
                           "gross_bp":gross,"funding_bp":fund,"cost_bp":cost,"net_bp":net,"hold_h":(exit_i-entry)*0.25})
            i=exit_i+1
        else:i+=1
    return trades

def stats(trades):
    p=[x["net_bp"] for x in trades if math.isfinite(x["net_bp"])]
    if not p:return {"n":0}
    gains=sum(x for x in p if x>0);loss=-sum(x for x in p if x<0)
    return {"n":len(p),"mean_bp":safe_mean(p),"median_bp":safe_med(p),"win_rate":sum(x>0 for x in p)/len(p),
            "profit_factor":(gains/loss if loss>0 else None),"sum_bp":sum(p),
            "mean_hold_h":safe_mean([x["hold_h"] for x in trades]),
            "funding_mean_bp":safe_mean([x.get("funding_bp",0) for x in trades])}

def portfolio_return(alltrades, max_positions=3):
    # Conservative: each trade uses equal slice of total capital; two-leg 1x collateral -> pnl_bp / 2 on allocated capital.
    ev=sorted(alltrades,key=lambda x:x["entry"])
    active=[];ret=0.0;accepted=0
    for tr in ev:
        t=pd.Timestamp(tr["entry"])
        active=[e for e in active if pd.Timestamp(e["exit"])>t]
        if len(active)>=max_positions:continue
        active.append(tr);accepted+=1
        ret += (tr["net_bp"]/10000.0)/2.0/max_positions
    return {"accepted":accepted,"return_pct":ret*100,"monthly_pct":ret*100/(DAYS/30)}

def main():
    end=datetime.now(timezone.utc).replace(second=0,microsecond=0)
    end=end-timedelta(minutes=end.minute%15)
    start=end-timedelta(days=DAYS)
    sm=int(start.timestamp()*1000);em=int(end.timestamp()*1000)
    log("PHASE8_START",start=start.isoformat(),end=end.isoformat(),days=DAYS,coins=COINS,
        fees={"upbit_bp":UP_FEE_BP,"bn_spot_bp":BN_SPOT_FEE_BP,"bn_fut_bp":BN_FUT_FEE_BP,"slip_round_bp":SLIP_ROUND_BP})
    fx=upbit_minutes("KRW-USDT",sm,em)
    result={};portfolio={"premium_perp":[],"premium_spot":[],"funding_carry":[],"basis_carry":[]};fails=[]
    for coin in COINS:
        try:
            up=upbit_minutes("KRW-"+coin,sm,em)
            spot=binance_klines(coin+"USDT",sm,em,False)
            fut=binance_klines(coin+"USDT",sm,em,True)
            fr=funding(coin+"USDT",sm,em)
            df=pd.concat({"up":up,"spot":spot,"fut":fut,"fx":fx},axis=1).dropna()
            if len(df)<1000:raise RuntimeError(f"too few aligned bars {len(df)}")
            pp=premium_trades(df,fr,"perp")
            ps=premium_trades(df,fr,"spot")
            fc=funding_carry(df,fr)
            bc=basis_carry(df,fr)
            for name,trs in [("premium_perp",pp),("premium_spot",ps),("funding_carry",fc),("basis_carry",bc)]:
                for tr in trs:
                    tr["coin"]=coin
                    portfolio[name].append(tr)
            result[coin]={"bars":len(df),"premium_perp":stats(pp),"premium_spot":stats(ps),
                          "funding_carry":stats(fc),"basis_carry":stats(bc)}
            log("COIN_DONE",coin=coin,bars=len(df),stats=result[coin])
        except Exception as e:
            fails.append({"coin":coin,"error":repr(e)});log("COIN_FAIL",coin=coin,error=repr(e))
    ports={k:portfolio_return(v,3) for k,v in portfolio.items()}
    out={"start":start.isoformat(),"end":end.isoformat(),"days":DAYS,"coins":COINS,"failures":fails,
         "costs":{"upbit_bp":UP_FEE_BP,"bn_spot_bp":BN_SPOT_FEE_BP,"bn_fut_bp":BN_FUT_FEE_BP,"slip_round_bp":SLIP_ROUND_BP},
         "coin_results":result,"portfolio":ports}
    print("PHASE8_JSON="+json.dumps(out,separators=(",",":")),flush=True)
    print("PHASE8_TABLE",flush=True)
    for c,r in result.items():
        print(c, json.dumps(r,separators=(",",":")),flush=True)
    print("PORTFOLIO",json.dumps(ports,separators=(",",":")),flush=True)
    log("PHASE8_DONE",coins=len(result),failures=len(fails))
if __name__=="__main__":main()
