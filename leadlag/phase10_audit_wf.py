#!/usr/bin/env python3
"""Public-data-only retrospective validation. NO account/order/withdrawal code.
No historical L2 or subsecond execution is inferred from OHLC candles.
All parameter choices are fixed before this run; zero eligible coins is allowed.
"""
from __future__ import annotations
import bisect, hashlib, json, math, os, time, traceback
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np

COINS='BTC ETH XRP SOL DOGE ADA LINK AVAX DOT BCH ETC AAVE NEAR SUI TRX SHIB XLM HBAR UNI APT'.split()
START='2025-10-01T00:00:00Z'; END='2026-09-22T00:00:00Z'
STEP=900000; MINUTE=60000; DAY=86400000
UF=0.0005; FF=0.0005
CAPS=[10000000,20000000,50000000]
SLIPS=[4,8,12,40,80]
MAXPOS=2; DEPLOY_FRACTION=0.8
CACHE=Path('/tmp/phase10-audit-cache'); CACHE.mkdir(exist_ok=True)
SESSION=requests.Session(); SESSION.headers['User-Agent']='daol-public-paper-audit/10.0'
ALLOWED={'api.upbit.com','fapi.binance.com','datasets.tardis.dev'}
MIN_CACHE={}

def ms(s): return int(datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()*1000)
def iso(t): return datetime.fromtimestamp(t/1000,timezone.utc).isoformat()
S0=ms(START); E0=ms(END)
GRID=np.arange(S0,E0,STEP,dtype=np.int64)
OOS=ms('2026-04-01T00:00:00Z')
MONTHS=[ms('2026-%02d-01T00:00:00Z'%m) for m in range(4,10)]+[E0]

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,float)): return float(x) if math.isfinite(float(x)) else None
    if isinstance(x,np.bool_): return bool(x)
    return x

def log(tag,**kw):
    print(tag+' '+json.dumps(clean(kw),separators=(',',':'),ensure_ascii=False,allow_nan=False),flush=True)

def get(url,params=None):
    from urllib.parse import urlparse
    if urlparse(url).hostname not in ALLOWED: raise ValueError('host not allowed')
    for attempt in range(6):
        r=SESSION.get(url,params=params,timeout=(10,40))
        if r.status_code in (429,500,502,503,504):
            time.sleep(max(float(r.headers.get('Retry-After','1')),2**attempt));continue
        r.raise_for_status();return r.json()
    raise RuntimeError('retry budget exhausted '+url)

def fsym(c): return ('1000SHIBUSDT',1000.) if c=='SHIB' else (c+'USDT',1.)

def candles(venue,c,unit=15,start=S0,end=E0):
    p=CACHE/('%s_%s_%s_%s_%s.npz'%(venue,c,unit,start,end))
    if p.exists():
        z=np.load(p);return z['t'],z['v']
    rows={};calls=0
    if venue=='up':
        cur=end
        while cur>start:
            a=get('https://api.upbit.com/v1/candles/minutes/'+str(unit),{'market':'KRW-'+c,'to':iso(cur),'count':200})
            calls+=1
            if not a:break
            for x in a:
                t=ms(x['candle_date_time_utc']+'Z')
                if start<=t<end:
                    rows[t]=[float(x[k]) for k in ('opening_price','high_price','low_price','trade_price','candle_acc_trade_price')]
            nxt=ms(a[-1]['candle_date_time_utc']+'Z')
            if nxt>=cur: raise RuntimeError('pagination not advancing')
            cur=nxt;time.sleep(.13)
    else:
        sym,mult=fsym(c);cur=start
        while cur<end:
            a=get('https://fapi.binance.com/fapi/v1/klines',{'symbol':sym,'interval':str(unit)+'m','startTime':cur,'endTime':end-1,'limit':1000})
            calls+=1
            if not a:break
            for x in a:
                t=int(x[0])
                if start<=t<end:rows[t]=[float(x[k])/mult for k in (1,2,3,4)]+[float(x[7])]
            nxt=int(a[-1][0])+unit*MINUTE
            if nxt<=cur:raise RuntimeError('pagination not advancing')
            cur=nxt;time.sleep(.04)
    tt=np.array(sorted(rows),np.int64);vv=np.array([rows[t] for t in tt],float)
    if len(tt)==0:raise RuntimeError('empty candles '+c)
    if not np.all(vv[:,:4]>0):raise RuntimeError('nonpositive prices')
    np.savez_compressed(p,t=tt,v=vv)
    log('DATA',venue=venue,coin=c,unit=unit,bars=len(tt),calls=calls,sha256=hashlib.sha256(tt.tobytes()+vv.tobytes()).hexdigest())
    return tt,vv

def on_grid(data):
    t,v=data;out=np.full((len(GRID),5),np.nan);ii=np.searchsorted(GRID,t)
    ok=(ii<len(GRID))&(t>=S0)&(t<E0);ii=ii[ok];out[ii]=v[ok];return out

def funds(c):
    sym,mult=fsym(c);out=[];cur=S0
    while cur<E0:
        a=get('https://fapi.binance.com/fapi/v1/fundingRate',{'symbol':sym,'startTime':cur,'endTime':E0-1,'limit':1000})
        if not a:break
        for x in a:
            t=int(x['fundingTime']);mark=float(x.get('markPrice') or 0)/mult
            if S0<=t<E0:out.append((t,float(x['fundingRate']),mark))
        nxt=int(a[-1]['fundingTime'])+1
        if nxt<=cur:raise RuntimeError('funding pagination stalled')
        cur=nxt;time.sleep(.04)
        if len(a)<1000:break
    out=sorted(set(out))
    if not out:raise RuntimeError('funding data absent')
    log('FUNDING_DATA',coin=c,n=len(out),missing_mark=sum(x[2]<=0 for x in out),max_gap_h=max(np.diff([x[0] for x in out]),default=0)/3600000)
    return out

def past_stats(x,w=1344):
    finite=np.isfinite(x);v=np.where(finite,x,0)
    a=np.r_[0,np.cumsum(v)];b=np.r_[0,np.cumsum(v*v)];cnt=np.r_[0,np.cumsum(finite)]
    mu=np.full(len(x),np.nan);sd=mu.copy()
    # The baseline at i excludes the current bar. Missing bars never compress time.
    for i in range(w,len(x)):
        n=cnt[i]-cnt[i-w]
        if n<w*.98:continue
        m=(a[i]-a[i-w])/n;mu[i]=m;sd[i]=math.sqrt(max(0,(b[i]-b[i-w])/n-m*m))
    return mu,sd

def all_signals(c,u,f,fx,fr):
    prem=np.log(u[:,3]/(f[:,3]*fx[:,3]))*10000;mu,sd=past_stats(prem)
    z=np.divide(prem-mu,sd,out=np.full(len(prem),np.nan),where=sd>=3)
    frt=np.array([x[0] for x in fr]);frv=np.array([x[1] for x in fr])
    fs=np.searchsorted(frt,GRID+STEP-1,side='right')-1
    known_fund=np.full(len(GRID),np.nan)
    for i,k in enumerate(fs):
        if k>=2:
            gaps=np.diff(frt[max(0,k-2):k+1]);hh=np.median(gaps)/3600000 if len(gaps) else 8
            known_fund[i]=float(np.mean(frv[k-2:k+1]))*10000*24/max(hh,1)
    out={}
    for strategy in ('premium','leadlag15m_proxy','funding','shock_proxy'):
        trades=[];i=1344
        while i<len(GRID)-100:
            if not all(math.isfinite(v) for v in (prem[i],z[i],u[i,3],f[i,3],fx[i,3])):i+=1;continue
            d=0;score=0.;maxh=96
            if strategy=='premium':
                dev=prem[i]-mu[i]
                if abs(z[i])>=2 and abs(dev)>=32:d=-1 if dev>0 else 1;score=abs(dev)-24
            elif strategy=='funding':
                if known_fund[i]>=32:d=1;score=known_fund[i]-24
            elif i>0 and np.all(np.isfinite([u[i-1,3],f[i-1,3]])):
                ru=math.log(u[i,3]/u[i-1,3])*10000;rf=math.log(f[i,3]/f[i-1,3])*10000
                th=120 if strategy=='shock_proxy' else 55;gapth=45 if strategy=='shock_proxy' else 25
                if abs(rf)>=th and abs(rf-ru)>=gapth and rf*(rf-ru)>0:
                    d=1 if rf>0 else -1;score=abs(rf-ru)-24;maxh=4
            if not d:i+=1;continue
            e=i+1;x=min(e+maxh,len(GRID)-2);reason='timeout'
            if strategy=='premium':
                for j in range(e,x):
                    if math.isfinite(z[j]) and ((d==1 and z[j]>=-.5) or (d==-1 and z[j]<=.5)):
                        x=j+1;reason='reversion';break
            if np.all(np.isfinite([u[e,0],u[x,0],f[e,0],f[x,0],fx[e,0],fx[x,0]])):
                trades.append({'coin':c,'strategy':strategy,'side':d,'e':e,'x':x,'entry':int(GRID[e]),'exit':int(GRID[x]),'score':score,'reason':reason})
            i=x+1
        out[strategy]=trades
    return out

def metrics(a):
    a=np.array(a,float)
    if not len(a):return {'n':0}
    pos=a[a>0].sum();neg=-a[a<0].sum()
    return {'n':len(a),'mean_bp':float(a.mean()),'median_bp':float(np.median(a)),'win':float(np.mean(a>0)),
            'pf':float(pos/neg) if neg>0 else None,'sum_bp':float(a.sum()),'p05_bp':float(np.quantile(a,.05))}

def trade_pnl(tr,data,fx,slip=4):
    u,f,fr=data[tr['coin']];e=tr['e'];x=tr['x'];d=tr['side'];q=1/u[e,0]
    gross=d*q*(u[x,0]-u[e,0])-d*q*(f[x,0]-f[e,0])*fx[x,0]
    fee=q*(u[e,0]+u[x,0])*UF+q*(f[e,0]*fx[e,0]+f[x,0]*fx[x,0])*FF
    fund=0.
    for t,r,m in fr:
        if tr['entry']<t<=tr['exit']:
            k=min(np.searchsorted(GRID,t,side='right')-1,len(GRID)-1)
            if m<=0: raise RuntimeError('missing settlement mark '+tr['coin'])
            if not math.isfinite(fx[k,3]):raise RuntimeError('missing funding FX')
            fund+=d*q*m*r*fx[k,3]
    return (gross-fee+fund-slip/10000)*10000

def select_months(trades,data,fx):
    selected={};eligible={};logs=[]
    for start,end in zip(MONTHS[:-1],MONTHS[1:]):
        cutoff=start-DAY;lo=max(S0,cutoff-180*DAY);scores=[]
        for c in COINS:
            ts=[t for t in trades if t['coin']==c and t['side']==1 and lo<=t['entry'] and t['exit']<cutoff]
            vals=np.array([trade_pnl(t,data,fx,8) for t in ts],float)
            if len(vals)<20:continue
            lower=float(vals.mean()-1.645*vals.std(ddof=1)/math.sqrt(len(vals)))
            if lower>0:scores.append((lower,c,len(vals),float(vals.mean())))
        scores.sort(reverse=True)
        selected[start]=[s[1] for s in scores[:5]]
        eligible[start]=[t for t in trades if t['side']==1 and t['coin'] in selected[start] and start<=t['entry']<end and t['exit']<E0]
        row={'start':iso(start),'end':iso(end),'train_start':iso(lo),'train_end_exclusive':iso(cutoff),'selected':selected[start],'scores':scores,'candidates':len(eligible[start])}
        logs.append(row);log('WF_SELECTION',**row)
    return selected,eligible,logs

def ff_prices(a):
    b=a.copy()
    for j in range(4):
        valid=np.where(np.isfinite(b[:,j]),np.arange(len(b)),-1);np.maximum.accumulate(valid,out=valid)
        good=valid>=0;b[good,j]=b[valid[good],j]
    return b

def wallet(trades,data,fx,capital,slip,gate_volume=True):
    # Primary feasible mode: only BUY spot + SHORT perp. No spot short, no borrowing.
    order={}
    for tr in trades:order.setdefault(tr['e'],[]).append(tr)
    k0=int(np.searchsorted(GRID,OOS));FX=ff_prices(fx);marks={c:(ff_prices(d[0]),ff_prices(d[1])) for c,d in data.items()}
    if not math.isfinite(FX[k0,0]):raise RuntimeError('initial FX absent')
    initial_up=capital*.5;initial_bn=capital*.5/FX[k0,0];cashup=initial_up;cashbn=initial_bn
    active=[];accepted=[];blocked={'inventory':0,'slot':0,'volume':0,'data':0};curve=[];dates=[];margin_alerts=0
    fundmap={}
    for c,(_,_,fr) in data.items():
        for t,r,m in fr:
            k=int(np.searchsorted(GRID,t,side='right')-1);fundmap.setdefault(k,[]).append((c,t,r,m))
    h=slip/4/10000
    for i in range(k0,len(GRID)):
        t=int(GRID[i]);fxo=FX[i,0];fxc=FX[i,3]
        # Exits were signalled at a previous close and execute at this bar's open.
        keep=[]
        for p in active:
            if p['x']<=i:
                u,f,_=data[p['coin']]
                if not np.all(np.isfinite([u[i,0],f[i,0]])):keep.append(p);continue
                ux=u[i,0]*(1-h);ff=f[i,0]*(1+h)
                cashup+=p['q']*ux*(1-UF);cashbn+=p['q']*(p['fe']-ff)-p['q']*ff*FF
                tr=dict(p);tr['exit']=t;tr['pnl_krw']=p['q']*ux*(1-UF)-p['paid_up']+(p['q']*(p['fe']-ff)-p['q']*ff*FF-p['entry_fut_fee']+p['fund_usdt'])*fxo
                accepted.append(tr)
            else:keep.append(p)
        active=keep
        for tr in sorted(order.get(i,[]),key=lambda r:(-r['score'],r['coin'])):
            assert tr['side']==1
            if len(active)>=MAXPOS or any(p['coin']==tr['coin'] for p in active):blocked['slot']+=1;continue
            u,f,_=data[tr['coin']]
            if not np.all(np.isfinite([u[i,0],f[i,0],fxo,u[i-1,4],f[i-1,4]])):blocked['data']+=1;continue
            notional=capital*DEPLOY_FRACTION/(2*MAXPOS)
            # This is only a volume participation proxy, NOT an order-book fill model.
            if gate_volume and notional>.005*min(u[i-1,4],f[i-1,4]*fxo):blocked['volume']+=1;continue
            ue=u[i,0]*(1+h);fe=f[i,0]*(1-h);q=notional/ue;paid=q*ue*(1+UF);margin=q*fe;entryfee=q*fe*FF
            used=sum(p['margin'] for p in active)
            pnlopen=sum(p['q']*(p['fe']-marks[p['coin']][1][i,0]) for p in active)
            if cashup<paid or cashbn+min(0,pnlopen)-used<margin+entryfee:blocked['inventory']+=1;continue
            cashup-=paid;cashbn-=entryfee
            p=dict(tr,q=q,ue=ue,fe=fe,paid_up=paid,margin=margin,entry_fut_fee=entryfee,fund_usdt=0.)
            active.append(p)
        # Settlements are booked exactly once. An exact entry timestamp settlement is excluded.
        for c,ft,rate,mark in fundmap.get(i,[]):
            for p in active:
                if p['coin']==c and p['entry']<ft and ft<p['exit']:
                    amount=p['q']*mark*rate;cashbn+=amount;p['fund_usdt']+=amount
        spotval=0.;futureval=0.
        for p in active:
            u,f=marks[p['coin']];spotval+=p['q']*u[i,3];futureval+=p['q']*(p['fe']-f[i,3])
            # High-price stress flag is not an exact liquidation-engine reconstruction.
            stressed=p['margin']+p['q']*(p['fe']-f[i,1])+p['fund_usdt']
            if stressed<=p['q']*f[i,1]*.01:margin_alerts+=1
        equity=cashup+spotval+(cashbn+futureval)*fxc
        benchmark=initial_up+initial_bn*fxc
        curve.append([equity,capital+equity-benchmark]);dates.append(t+STEP)
    eq=np.array(curve);tt=np.array(dates,np.int64)
    dd=eq/np.maximum.accumulate(np.vstack([np.array([[capital,capital]]),eq]),axis=0)[1:]-1
    daily={}
    for t,v in zip(tt,eq):daily[iso(int(t))[:10]]=v
    dv=np.array(list(daily.values()));daily_alpha=np.diff(np.r_[capital,dv[:,1]])
    rng=np.random.default_rng(20260923);boot=[];n=len(daily_alpha)
    for _ in range(1000):
        ind=[]
        while len(ind)<n:
            j=int(rng.integers(n));ind.extend((j+np.arange(7))%n)
        boot.append(float(daily_alpha[np.array(ind[:n])].sum()/capital))
    month={};prev=np.array([capital,capital])
    for t,v in zip(tt,eq):
        key=iso(int(t-1))[:7];month[key]=v
    monthly={}
    for key,v in month.items():monthly[key]={'account_pnl':float(v[0]-prev[0]),'strategy_vs_passive_pnl':float(v[1]-prev[1])};prev=v
    days=(E0-OOS)/DAY
    out={'capital':capital,'slip_round_bp':slip,'max_positions':MAXPOS,'notional_each':capital*DEPLOY_FRACTION/(2*MAXPOS),
         'executed':len(accepted),'still_open':len(active),'blocked':blocked,'account_return_pct':(eq[-1,0]/capital-1)*100,
         'strategy_vs_passive_return_pct':(eq[-1,1]/capital-1)*100,'avg_30d_strategy_pct':(eq[-1,1]/capital-1)*100*30/days,
         'account_mtm_dd_pct':float(dd[:,0].min()*100),'strategy_mtm_dd_pct':float(dd[:,1].min()*100),
         'bootstrap_95_strategy_return_pct':np.quantile(boot,[.025,.975])*100,'margin_stress_flags':margin_alerts,
         'monthly':monthly,'end_balances':{'up_krw':cashup,'bn_usdt':cashbn},
         'limitations':['15m bar execution proxy','no historical depth','no subsecond fills','no mark-price intrabar liquidation','fixed 20-coin research universe']}
    log('WF_WALLET',**out)
    return out,accepted

def minute_at(venue,c,t):
    key=(venue,c,t//(120*MINUTE))
    if key not in MIN_CACHE:
        start=key[2]*120*MINUTE;end=start+120*MINUTE
        ts,vs=candles(venue,c,1,start,end)
        MIN_CACHE[key]={int(k):v for k,v in zip(ts,vs)}
    return MIN_CACHE[key].get(t)

def minute_audit(trades,data,fx):
    # All accepted baseline OOS trades, not just winners. Replay exact minute opens
    # one minute AFTER the original signal boundary, including both entry/exit.
    rows=[];missing=0
    for k,tr in enumerate(trades):
        try:
            c=tr['coin'];te=tr['entry']+MINUTE;tx=tr['exit']+MINUTE
            ue=minute_at('up',c,te);ux=minute_at('up',c,tx);fe=minute_at('fut',c,te);ff=minute_at('fut',c,tx)
            xe=minute_at('up','USDT',te);xx=minute_at('up','USDT',tx)
            if any(v is None for v in (ue,ux,fe,ff,xe,xx)):missing+=1;continue
            q=1/ue[0];gross=q*(ux[0]-ue[0])-q*(ff[0]-fe[0])*xx[0]
            fees=q*(ue[0]+ux[0])*UF+q*(fe[0]*xe[0]+ff[0]*xx[0])*FF
            fund=0.
            for t,r,m in data[c][2]:
                if te<t<=tx:fund+=q*m*r*xx[0]
            row={'coin':c,'entry':te,'exit':tx,'old_bp':trade_pnl(tr,data,fx,4),'minute_bp4':(gross-fees+fund)*10000-4,
                 'last_trade_span_min':(tx-te)/MINUTE}
            rows.append(row)
            if (k+1)%25==0:log('MINUTE_PROGRESS',attempted=k+1,valid=len(rows),missing=missing)
        except Exception as e:missing+=1;log('MINUTE_FAIL',coin=tr['coin'],entry=tr['entry'],error=str(e)[:160])
    out={'attempted':len(trades),'valid':len(rows),'missing':missing,'old_bar':metrics([r['old_bp'] for r in rows]),
         'minute_slip4':metrics([r['minute_bp4'] for r in rows]),'minute_slip8':metrics([r['minute_bp4']-4 for r in rows]),
         'minute_slip12':metrics([r['minute_bp4']-8 for r in rows]),
         'note':'Matched-trade execution sensitivity only, not a rerun of ranking or shared-wallet path; funding valued at exit FX; no historic bid/ask.'}
    log('MINUTE_AUDIT',**out)
    for c in sorted({r['coin'] for r in rows}):log('MINUTE_COIN',coin=c,stats=metrics([r['minute_bp4'] for r in rows if r['coin']==c]))
    return out

def vwap(levels,q):
    left=q;cost=0.
    for p,n in levels:
        take=min(left,n);cost+=take*p;left-=take
        if left<=q*1e-10:return cost/q
    return None

def depth_probe():
    # A current snapshot diagnostic, never labelled historical capacity.
    out=[]
    btc=get('https://api.upbit.com/v1/orderbook',{'markets':'KRW-BTC'})[0]['orderbook_units'][0]['ask_price']
    for c in COINS:
        try:
            a=get('https://api.upbit.com/v1/orderbook',{'markets':'KRW-'+c})[0];sym,m=fsym(c)
            b=get('https://fapi.binance.com/fapi/v1/depth',{'symbol':sym,'limit':100})
            ua=[(x['ask_price'],x['ask_size']) for x in a['orderbook_units']];ub=[(x['bid_price'],x['bid_size']) for x in a['orderbook_units']]
            fa=[(float(p)/m,float(n)*m) for p,n in b['asks']];fb=[(float(p)/m,float(n)*m) for p,n in b['bids']]
            sizes=[]
            for notional in sorted(set([1000000,2000000,4000000,10000000]+[float(btc)*q for q in [.01,.05,.1,.25,.5]])):
                q=notional/ua[0][0];vals=[vwap(ua,q),vwap(ub,q),vwap(fa,q),vwap(fb,q)]
                if any(v is None for v in vals):sizes.append({'notional':notional,'covered':False});continue
                av,bv,af,bf=vals
                rt=((av-bv)/((av+bv)/2)+(af-bf)/((af+bf)/2))*10000
                sizes.append({'notional':notional,'covered':True,'spread_impact_round_bp':rt})
            row={'coin':c,'up_timestamp':a.get('timestamp'),'bn_timestamp':b.get('E'),'up_spread_bp':(ua[0][0]-ub[0][0])/((ua[0][0]+ub[0][0])/2)*10000,
                 'fut_spread_bp':(fa[0][0]-fb[0][0])/((fa[0][0]+fb[0][0])/2)*10000,'sizes':sizes}
            out.append(row);log('DEPTH_SNAPSHOT',**row);time.sleep(.15)
        except Exception as e:log('DEPTH_FAIL',coin=c,error=str(e)[:200])
    for day in ('2026/09/01','2026/09/02'):
        url='https://datasets.tardis.dev/v1/upbit/book_snapshot_25/'+day+'/KRW-DOGE.csv.gz'
        try:
            r=SESSION.head(url,timeout=15,allow_redirects=True)
            log('HISTORICAL_L2_ACCESS',date=day,status=r.status_code,content_length=r.headers.get('content-length'))
        except Exception as e:log('HISTORICAL_L2_ACCESS',date=day,error=str(e)[:160])
    return out

def selftest():
    assert vwap([(100,1),(110,1)],1.5)==(100+55)/1.5
    assert vwap([(100,1)],2) is None
    xx=np.arange(1500,dtype=float);m,s=past_stats(xx)
    assert abs(m[1400]-xx[56:1400].mean())<1e-10
    changed=xx.copy();changed[1400:]=1e9;m2,s2=past_stats(changed)
    assert m2[1400]==m[1400]
    p=np.array([100,110,80,110]);dd=p/np.maximum.accumulate(p)-1
    assert dd.min()<-0.27
    log('SELFTEST_OK',checks=5)

def main():
    selftest();log('AUDIT_START',start=START,end=END,oos=iso(OOS),coins=COINS,paper_only=True,
        primary='buy-funded-Upbit-spot plus short-USDT-perp only',window_days=14,z=2,dev_bp=32,
        selection='monthly 180d past,24h purge,n>=20,8bp slip,normal-approx lower bound >0,top5',
        caveat='retrospective walk-forward after prior exploration, not untouched future OOS')
    depth=depth_probe();fx=on_grid(candles('up','USDT'));data={};alltr={};fail=[]
    for c in COINS:
        try:
            u=on_grid(candles('up',c));f=on_grid(candles('fut',c));fr=funds(c)
            data[c]=(u,f,fr);st=all_signals(c,u,f,fx,fr);alltr[c]=st
            for name,trs in st.items():
                log('COIN_AUDIT',coin=c,strategy=name,both_sides=metrics([trade_pnl(t,data,fx,4) for t in trs]),
                    feasible_long_only=metrics([trade_pnl(t,data,fx,4) for t in trs if t['side']==1]),
                    hypothetical_spot_short=metrics([trade_pnl(t,data,fx,4) for t in trs if t['side']==-1]),
                    missing_up_bars=int(np.sum(~np.isfinite(u[:,0]))))
        except Exception as e:fail.append(c);log('COIN_AUDIT_FAIL',coin=c,error=str(e)[:250])
    primary=[t for st in alltr.values() for t in st['premium']]
    selected,eligible,selectionlog=select_months(primary,data,fx)
    tr=[t for ts in eligible.values() for t in ts];wallets=[];baseline=[]
    for cap in CAPS:
        for slip in SLIPS:
            r,accepted=wallet(tr,data,fx,cap,slip);wallets.append(r)
            if cap==10000000 and slip==4:baseline=accepted
    minute=minute_audit(baseline,data,fx)
    # Missing L2 and genuine subsecond execution ALWAYS block final production pass.
    summary={'data_failures':fail,'selected':{iso(k):v for k,v in selected.items()},'candidate_count':len(tr),
             'baseline_executed':len(baseline),'minute_valid':minute['valid'],'fully_validated':False,
             'blocking_gates':['No continuous historical L2 replay','No true 100/200/500ms fill model','No exchange mark-price liquidation reconstruction','No untouched prospective paper period'],
             'financial_forecast_authorized':False}
    (CACHE/'summary.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2))
    log('AUDIT_DONE',**summary)

if __name__=='__main__':
    try:main()
    except Exception as e:log('AUDIT_FATAL',error=repr(e),traceback=traceback.format_exc()[-4000:]);raise
