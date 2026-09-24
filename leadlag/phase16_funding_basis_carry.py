#!/usr/bin/env python3
"""
Phase 16: Funding / Basis Carry research.
Public data only. No orders/private APIs.

Compare:
1) BINANCE_CARRY = Binance spot long + Binance USD-M perp short
2) UPBIT_CARRY   = Upbit KRW spot long + Binance USD-M perp short

Signals use ONLY information known before entry:
- trailing 3 realized funding rates
- current completed-hour basis/premium
Execution: next hourly open. Funding settlements are booked only if after entry.
OOS evaluation: 2026-04-01 through 2026-09-22.

This is screening, not production: hourly OHLC opens are execution proxies.
"""
from __future__ import annotations
import json,math,os,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import urlparse
import numpy as np, requests

ROOT=Path('/tmp/funding-basis-phase16');ROOT.mkdir(exist_ok=True)
START='2025-10-01T00:00:00Z';END='2026-09-22T00:00:00Z';OOS='2026-04-01T00:00:00Z'
COINS=os.getenv('CARRY_COINS','BTC,ETH,SOL,XRP,DOGE,ADA,SUI,LINK').split(',')
HOLD_H=[8,24,72,168]
FUND_BP=[0.25,0.5,1.0,2.0]
BASIS_BP=[0.,10.,20.,40.]
PREMIUM_MAX=[None,0.,50.,100.]
SLIP_BP=[0.,2.,5.]
SPOT_FEE_BP=float(os.getenv('BINANCE_SPOT_FEE_BP','10'))
FUT_FEE_BP=float(os.getenv('BINANCE_FUT_FEE_BP','5'))
UP_FEE_BP=float(os.getenv('UP_FEE_BP','5'))
HOUR=3_600_000;DAY=86_400_000
MAX_TRADES_PER_CELL=100000
S=requests.Session();S.headers['User-Agent']='daol-funding-basis-research/16'
ALLOWED={
 'data-api.binance.vision':('/api/v3/klines',),
 'fapi.binance.com':('/fapi/v1/klines','/fapi/v1/fundingRate'),
 'api.upbit.com':('/v1/candles/minutes/60',)
}

def ms(s):return int(datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()*1000)
S0,E0,O0=map(ms,(START,END,OOS))
GRID=np.arange(S0,E0,HOUR,dtype=np.int64)

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,float)):
        v=float(x);return v if math.isfinite(v) else None
    if isinstance(x,np.bool_):return bool(x)
    return x

def log(tag,**kw):print(tag+' '+json.dumps(clean(kw),separators=(',',':'),ensure_ascii=False,allow_nan=False),flush=True)

def get(url,params=None):
    p=urlparse(url)
    if p.scheme!='https' or p.hostname not in ALLOWED or not any(p.path.startswith(x) for x in ALLOWED[p.hostname]):
        raise ValueError('endpoint not allowed '+url)
    for a in range(6):
        try:r=S.get(url,params=params,timeout=(10,45))
        except (requests.Timeout,requests.ConnectionError):
            if a==5:raise
            time.sleep(2**a);continue
        if r.status_code in (429,500,502,503,504):
            wait=max(float(r.headers.get('Retry-After','1')),2**a);r.close();time.sleep(min(wait,60));continue
        r.raise_for_status();return r.json()
    raise RuntimeError('retry budget exhausted')

def iso(t):return datetime.fromtimestamp(t/1000,timezone.utc).isoformat()

def fsym(c):return ('1000SHIBUSDT',1000.) if c=='SHIB' else (c+'USDT',1.)

def binance_klines(kind,c):
    sym,m=fsym(c);rows={};cur=S0
    host='https://data-api.binance.vision/api/v3/klines' if kind=='spot' else 'https://fapi.binance.com/fapi/v1/klines'
    while cur<E0:
        a=get(host,{'symbol':sym if kind=='fut' else c+'USDT','interval':'1h','startTime':cur,'endTime':E0-1,'limit':1000})
        if not a:break
        for x in a:
            t=int(x[0])
            if S0<=t<E0:rows[t]=[float(x[k])/m if kind=='fut' else float(x[k]) for k in (1,2,3,4)]
        nxt=int(a[-1][0])+HOUR
        if nxt<=cur:raise RuntimeError('binance pagination stalled')
        cur=nxt;time.sleep(.03)
    if not rows:raise RuntimeError('empty '+kind+' '+c)
    return rows

def upbit_klines(c):
    rows={};cur=E0
    while cur>S0:
        a=get('https://api.upbit.com/v1/candles/minutes/60',{'market':'KRW-'+c,'to':iso(cur),'count':200})
        if not a:break
        for x in a:
            t=ms(x['candle_date_time_utc']+'Z')
            if S0<=t<E0:rows[t]=[float(x[k]) for k in ('opening_price','high_price','low_price','trade_price')]
        nxt=ms(a[-1]['candle_date_time_utc']+'Z')
        if nxt>=cur:raise RuntimeError('upbit pagination stalled')
        cur=nxt;time.sleep(.13)
    if not rows:raise RuntimeError('empty upbit '+c)
    return rows

def grid(rows):
    out=np.full((len(GRID),4),np.nan)
    for t,v in rows.items():
        i=(t-S0)//HOUR
        if 0<=i<len(out):out[int(i)]=v
    return out

def funding(c):
    sym,m=fsym(c);out=[];cur=S0
    while cur<E0:
        a=get('https://fapi.binance.com/fapi/v1/fundingRate',{'symbol':sym,'startTime':cur,'endTime':E0-1,'limit':1000})
        if not a:break
        for x in a:
            t=int(x['fundingTime']);mark=float(x.get('markPrice') or 0)/m
            if S0<=t<E0:out.append((t,float(x['fundingRate']),mark))
        nxt=int(a[-1]['fundingTime'])+1
        if nxt<=cur:raise RuntimeError('funding pagination stalled')
        cur=nxt;time.sleep(.03)
        if len(a)<1000:break
    out=sorted({(t,r,m) for t,r,m in out})
    if len(out)<10:raise RuntimeError('too little funding '+c)
    return out

def trailing_funding(fr):
    ts=np.array([x[0] for x in fr],np.int64);rv=np.array([x[1] for x in fr],float)*10000
    out=np.full(len(GRID),np.nan)
    for i,t in enumerate(GRID):
        k=np.searchsorted(ts,t,side='right')
        if k>=3:out[i]=float(np.mean(rv[k-3:k]))
    return out

def fund_cash(fr,entry,exit,q):
    s=0.
    for t,r,m in fr:
        if entry<t<=exit and m>0:s+=q*m*r
    return s

def trade_binance(c,spot,fut,fr,e,x,slip):
    if x>=len(GRID):return None
    vals=[spot[e,0],spot[x,0],fut[e,0],fut[x,0]]
    if not all(np.isfinite(v) and v>0 for v in vals):return None
    sf=SPOT_FEE_BP/10000;ff=FUT_FEE_BP/10000;sl=slip/10000
    se=spot[e,0]*(1+sl);sx=spot[x,0]*(1-sl)
    fe=fut[e,0]*(1-sl);fx=fut[x,0]*(1+sl)
    q=1/se
    spotp=q*(sx*(1-sf)-se*(1+sf))
    futp=q*(fe*(1-ff)-fx*(1+ff))
    fd=fund_cash(fr,int(GRID[e]),int(GRID[x]),q)
    return (spotp+futp+fd)*10000,fd*10000

def trade_upbit(c,up,fut,fx,fr,e,x,slip):
    if x>=len(GRID):return None
    vals=[up[e,0],up[x,0],fut[e,0],fut[x,0],fx[e,0],fx[x,0]]
    if not all(np.isfinite(v) and v>0 for v in vals):return None
    uf=UP_FEE_BP/10000;ff=FUT_FEE_BP/10000;sl=slip/10000
    ue=up[e,0]*(1+sl);ux=up[x,0]*(1-sl)
    fe=fut[e,0]*(1-sl);fz=fut[x,0]*(1+sl)
    q=1/ue # 1 KRW entry spot notional
    spotp=q*(ux*(1-uf)-ue*(1+uf))
    fd_usdt=fund_cash(fr,int(GRID[e]),int(GRID[x]),q)
    fut_usdt=q*(fe*(1-ff)-fz*(1+ff))+fd_usdt
    pnl_krw=spotp+fut_usdt*fx[x,0]
    return pnl_krw*10000,fd_usdt*fx[x,0]*10000

def metrics(a):
    x=np.asarray(a,float);x=x[np.isfinite(x)]
    if not len(x):return {'n':0}
    pos=x[x>0].sum();neg=-x[x<0].sum()
    return {'n':len(x),'mean_bp':float(x.mean()),'median_bp':float(np.median(x)),
            'win_rate':float(np.mean(x>0)),'pf':float(pos/neg) if neg>0 else None,
            'sum_bp':float(x.sum()),'p05_bp':float(np.quantile(x,.05)),'p95_bp':float(np.quantile(x,.95))}

def generate(c,spot,fut,up,fx,fr):
    tf=trailing_funding(fr)
    basis=np.log(fut[:,3]/spot[:,3])*10000
    prem=np.log(up[:,3]/(fut[:,3]*fx[:,3]))*10000
    rows=[]
    start=max(1,int((O0-S0)//HOUR))
    for fth in FUND_BP:
      for bth in BASIS_BP:
       for hold in HOLD_H:
        sig=np.isfinite(tf)&np.isfinite(basis)&(tf>=fth)&(basis>=bth)
        ix=np.flatnonzero(sig & (GRID>=O0))
        # Avoid entering same coin repeatedly while prior fixed-horizon trade remains open.
        next_ok=-1
        for s in ix:
            if s<next_ok:continue
            e=int(s+1);x=e+hold
            if x>=len(GRID):continue
            score=float(tf[s]*3+basis[s])
            rows.append({'venue':'BINANCE','coin':c,'e':e,'x':x,'entry':int(GRID[e]),'exit':int(GRID[x]),
                         'fth':fth,'bth':bth,'hold':hold,'pmax':None,'score':score,
                         'fund3_bp':float(tf[s]),'basis_bp':float(basis[s]),'premium_bp':float(prem[s]) if np.isfinite(prem[s]) else None})
            next_ok=x
        for pmax in PREMIUM_MAX:
            sig2=np.isfinite(tf)&np.isfinite(basis)&np.isfinite(prem)&(tf>=fth)&(basis>=bth)
            if pmax is not None:sig2&=(prem<=pmax)
            ix=np.flatnonzero(sig2 & (GRID>=O0));next_ok=-1
            for s in ix:
                if s<next_ok:continue
                e=int(s+1);x=e+hold
                if x>=len(GRID):continue
                score=float(tf[s]*3+basis[s]-max(0,prem[s]))
                rows.append({'venue':'UPBIT','coin':c,'e':e,'x':x,'entry':int(GRID[e]),'exit':int(GRID[x]),
                             'fth':fth,'bth':bth,'hold':hold,'pmax':pmax,'score':score,
                             'fund3_bp':float(tf[s]),'basis_bp':float(basis[s]),'premium_bp':float(prem[s])})
                next_ok=x
    return rows

def main():
    log('CARRY_BT_START',period=[START,END],oos=OOS,coins=COINS,hold_h=HOLD_H,funding_threshold_bp=FUND_BP,
        basis_threshold_bp=BASIS_BP,premium_max_bp=PREMIUM_MAX,slippage_bp_per_execution=SLIP_BP,
        fees_bp={'binance_spot':SPOT_FEE_BP,'binance_futures':FUT_FEE_BP,'upbit':UP_FEE_BP},
        note='Signals use trailing realized funding and completed-hour prices; execution next hourly open.')
    fx=grid(upbit_klines('USDT'))
    data={};alltr=[];fails=[]
    for c in COINS:
        try:
            sp=grid(binance_klines('spot',c));fu=grid(binance_klines('fut',c));up=grid(upbit_klines(c));fr=funding(c)
            data[c]=(sp,fu,up,fr);tr=generate(c,sp,fu,up,fx,fr);alltr.extend(tr)
            log('CARRY_COIN_READY',coin=c,trades=len(tr),funding_events=len(fr),
                valid={'binance_spot':int(np.isfinite(sp[:,0]).sum()),'futures':int(np.isfinite(fu[:,0]).sum()),'upbit':int(np.isfinite(up[:,0]).sum())})
        except Exception as e:
            fails.append({'coin':c,'error':repr(e)});log('CARRY_COIN_FAIL',coin=c,error=repr(e))
    results=[]
    keys=sorted(set((t['venue'],t['fth'],t['bth'],t['hold'],t['pmax']) for t in alltr),key=str)
    for venue,fth,bth,hold,pmax in keys:
      trs=[t for t in alltr if (t['venue'],t['fth'],t['bth'],t['hold'],t['pmax'])==(venue,fth,bth,hold,pmax)]
      for slip in SLIP_BP:
        vals=[];funds=[];bycoin={}
        for t in trs:
            sp,fu,up,fr=data[t['coin']]
            r=trade_binance(t['coin'],sp,fu,fr,t['e'],t['x'],slip) if venue=='BINANCE' else trade_upbit(t['coin'],up,fu,fx,fr,t['e'],t['x'],slip)
            if r:
                vals.append(r[0]);funds.append(r[1]);bycoin.setdefault(t['coin'],[]).append(r[0])
        m=metrics(vals)
        if m['n']:
            m.update({'venue':venue,'funding_threshold_bp':fth,'basis_threshold_bp':bth,'hold_h':hold,'premium_max_bp':pmax,
                      'slippage_bp_per_execution':slip,'mean_funding_component_bp':float(np.mean(funds)) if funds else None,
                      'coins':{c:metrics(v) for c,v in bycoin.items()}})
            results.append(m)
    # Report robust cells first; minimum 20 trades, then 10-trade frontier.
    robust=[r for r in results if r['n']>=20];robust.sort(key=lambda r:r['mean_bp'],reverse=True)
    for r in robust[:60]:log('CARRY_RESULT',**r)
    frontier=[r for r in results if r['n']>=10];frontier.sort(key=lambda r:r['mean_bp'],reverse=True)
    for r in frontier[:30]:log('CARRY_FRONTIER',**r)
    # best per venue/slippage
    for venue in ('BINANCE','UPBIT'):
      for slip in SLIP_BP:
        z=[r for r in results if r['venue']==venue and r['slippage_bp_per_execution']==slip and r['n']>=10]
        z.sort(key=lambda r:r['mean_bp'],reverse=True)
        if z:log('CARRY_BEST',**z[0])
    summary={'coins_ok':sorted(data),'failures':fails,'signal_trades':len(alltr),'result_cells':len(results),
             'production_pass':False,'reason':'Hourly execution proxy; no historical L1 fills, liquidation engine, tax/transfer constraints, or untouched future validation.'}
    (ROOT/'phase16_results.json').write_text(json.dumps(clean(results)))
    log('CARRY_BT_DONE',**summary)

if __name__=='__main__':
  try:main()
  except Exception as e:
    log('CARRY_BT_FATAL',error=repr(e),traceback=traceback.format_exc()[-3000:],production_pass=False);raise
