#!/usr/bin/env python3
"""Research only: PIT-turnover fixed20/top30/top100 comparison.
Only whitelisted public GET requests; no credentials/order endpoints.
Continuous 15m candles are an explicitly UNVALIDATED screening model.
The monthly quote check is matched-trade diagnostics, not continuous execution.
"""
from __future__ import annotations
import csv, gzip, io, json, math, os, threading, time, hashlib, traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import requests
import phase10_audit_wf as base

ROOT = Path('/tmp/phase13-universe-compare')
ROOT.mkdir(exist_ok=True)
START = base.S0
END = base.E0
OOS = base.OOS
STEP = base.STEP
DAY = base.DAY
GRID = base.GRID
DAYS = np.arange(START, END, DAY, dtype=np.int64)
COINS20 = tuple(base.COINS)
CAPITALS = (10_000_000, 20_000_000, 50_000_000)
EXTRA_RT_BP = (4., 8., 12.)
MAX_POS = 2
STOP_AT = time.monotonic() + 3 * 3600
MIN_NOTIONAL = 10_000.
RATE_LOCK = threading.Lock()
NEXT_REQUEST = {}
TLS = threading.local()
FAILURES = []
MAP = {}
CACHE_LRU = OrderedDict()
STABLE = {'USDT','USDC','DAI','TUSD','USDE','USDS','USD1','FDUSD','BUSD','USDD','USDP','USDCV'}
ALLOWED_PATHS = {
    'api.upbit.com': ('/v1/candles/days','/v1/candles/minutes/','/v1/market/all'),
    'fapi.binance.com': ('/fapi/v1/klines','/fapi/v1/fundingRate','/fapi/v1/exchangeInfo'),
    'api.tardis.dev': ('/v1/exchanges/',),
    'datasets.tardis.dev': ('/v1/upbit/quotes/','/v1/binance-futures/quotes/'),
}

def log(tag, **kw):
    base.log(tag, **kw)

def session():
    if not hasattr(TLS, 'session'):
        TLS.session = requests.Session()
        TLS.session.headers['User-Agent'] = 'daol-public-paper-universe-validation/13'
    return TLS.session

def limited_get(url, params=None, stream=False):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in ALLOWED_PATHS:
        raise ValueError('host not allowlisted')
    if not any(parsed.path.startswith(p) for p in ALLOWED_PATHS[parsed.hostname]):
        raise ValueError('endpoint not allowlisted: ' + parsed.path)
    interval = .18 if parsed.hostname == 'api.upbit.com' else (.25 if parsed.hostname == 'fapi.binance.com' else .08)
    for attempt in range(5):
        if time.monotonic() > STOP_AT:
            raise TimeoutError('three-hour research runtime budget reached')
        with RATE_LOCK:
            now = time.monotonic()
            target = max(now, NEXT_REQUEST.get(parsed.hostname, now))
            NEXT_REQUEST[parsed.hostname] = target + interval
        time.sleep(max(0., target-time.monotonic()))
        try:
            r = session().get(url, params=params, timeout=(10,45), stream=stream)
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 4: raise
            time.sleep(2**attempt)
            continue
        if r.status_code in (429,500,502,503,504):
            wait = max(float(r.headers.get('Retry-After','1')),2**attempt)
            r.close()
            time.sleep(min(wait,90))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError('public GET retry budget exhausted: ' + parsed.path)

def get(url, params=None):
    with limited_get(url, params) as r:
        return r.json()

# Existing loader has no order API. Override its public requester and cache.
base.get = get
base.CACHE = ROOT / 'bars'
base.CACHE.mkdir(exist_ok=True)
base.fsym = lambda c: (MAP[c]['fut_symbol'],MAP[c]['multiplier'])

def since(x):
    return base.ms(x['availableSince'])

def until(x):
    return base.ms(x['availableTo']) if x.get('availableTo') else END + DAY

def universe():
    up = get('https://api.tardis.dev/v1/exchanges/upbit')
    fu = get('https://api.tardis.dev/v1/exchanges/binance-futures')
    if not up.get('availableSymbols') or not fu.get('availableSymbols'):
        raise RuntimeError('historical universe metadata absent; no current-list fallback permitted')
    current = get('https://fapi.binance.com/fapi/v1/exchangeInfo')
    ci = {x['symbol'].upper():x for x in current['symbols']}
    fs = {}
    for f in fu['availableSymbols']:
        name = f['id'].upper()
        if not name.endswith('USDT') or '_' in name or f.get('type') != 'perpetual': continue
        if since(f) >= END or until(f) <= START: continue
        fs[name] = f
    for u in up['availableSymbols']:
        sym = u['id'].upper()
        if not sym.startswith('KRW-') or since(u) >= END or until(u) <= START: continue
        c = sym[4:]
        if c in STABLE: continue
        options = [(c+'USDT',1.)] + [(p+c+'USDT',float(p)) for p in ('1000','1000000')]
        found = [(s,m) for s,m in options if s in fs]
        if len(found) != 1:
            if found: log('AMBIGUOUS_CONTRACT',coin=c,options=found)
            continue
        s,m = found[0]; f=fs[s]
        filters = {x['filterType']:x for x in ci.get(s,{}).get('filters',[])}
        qstep = float(filters.get('LOT_SIZE',{}).get('stepSize','0')) * m
        minimum = float(filters.get('MIN_NOTIONAL',{}).get('notional','5'))
        MAP[c] = {'fut_symbol':s,'multiplier':m,'up_since':since(u),'up_until':until(u),
                  'fut_since':since(f),'fut_until':until(f),'qstep':qstep,'min_usdt':minimum,
                  'current_filters_present':bool(filters)}
    if len(MAP) < 100:
        raise RuntimeError('fewer than 100 metadata-matched contracts: '+str(len(MAP)))
    (ROOT/'universe.json').write_text(json.dumps(MAP,indent=2))
    log('UNIVERSE_READY',matched=len(MAP),basis='historical provider availability metadata',
        current_quantity_filters='proxy only, not historical exchange-rule reconstruction',
        identity='exact ticker or explicitly scaled 1000/1000000 contract only; no ticker migration stitching')

def daily(c):
    rows={};cur=END
    # Request all ranking inputs, not a list selected using the eventual return.
    floor=START-31*DAY
    while cur>floor:
        a=get('https://api.upbit.com/v1/candles/days',{'market':'KRW-'+c,'to':base.iso(cur),'count':200})
        if not a:break
        for x in a:
            t=base.ms(x['candle_date_time_utc']+'Z')
            if floor<=t<END:rows[t]=float(x['candle_acc_trade_price'])
        nxt=base.ms(a[-1]['candle_date_time_utc']+'Z')
        if nxt>=cur:raise RuntimeError('daily pagination stalled')
        cur=nxt
    if not rows:raise RuntimeError('no ranking candles')
    return rows

def ranking():
    daily_data={}
    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks={pool.submit(daily,c):c for c in MAP}
        done=0
        for task in as_completed(tasks):
            c=tasks[task];done+=1
            try:daily_data[c]=task.result()
            except Exception as e:
                FAILURES.append({'phase':'ranking','coin':c,'error':str(e)[:200]})
                log('RANKING_EXCLUSION',coin=c,error=str(e)[:180])
            if done%20==0:log('RANKING_PROGRESS',completed=done,total=len(MAP),usable=len(daily_data))
    ranked={};chosen=set(COINS20)&set(daily_data)
    for day in DAYS:
        candidates=[]
        for c,rows in daily_data.items():
            m=MAP[c]
            if day<max(m['up_since'],m['fut_since'])+30*DAY or day>=min(m['up_until'],m['fut_until']):continue
            history=[rows.get(int(day-j*DAY),0.) for j in range(1,31)]
            if sum(x>0 for x in history)<20 or history[0]<=0:continue
            score=sum(history[:7])/7
            if score>0:candidates.append((score,c))
        candidates.sort(key=lambda p:(-p[0],p[1]))
        ranked[int(day)]=[c for _,c in candidates[:100]]
        if day>=OOS:chosen.update(ranked[int(day)])
        if day>=OOS and base.iso(int(day))[8:10]=='01':
            log('RANK_SNAPSHOT',day=base.iso(int(day)),eligible=len(candidates),top30=ranked[int(day)][:30],
                ranks31_100=ranked[int(day)][30:],lookback='previous completed 7 days; >=20 active days of past30')
    (ROOT/'daily_ranks.json').write_text(json.dumps(ranked))
    counts=[len(ranked[int(d)]) for d in DAYS if d>=OOS]
    log('RANKING_DONE',usable_daily_symbols=len(daily_data),union_to_load=len(chosen),
        min_candidate_count=min(counts),max_candidate_count=max(counts),
        days_below100=sum(n<100 for n in counts),exclusions=len(FAILURES))
    return ranked,sorted(chosen)

def tick(p):
    for low,step in ((2e6,1000),(1e6,1000),(5e5,500),(1e5,100),(5e4,50),(1e4,10),(5000,5),
                     (1000,1),(100,1),(10,.1),(1,.01),(.1,.001),(.01,.0001),(.001,.00001),(.0001,.000001),(.00001,.0000001)):
        if p>=low:return step
    return .00000001

def synthetic_fill(p, buy, extra_bp, spot):
    # Deliberately a MODEL: last-trade OHLC is not midpoint. One-tick halfspread
    # discourages false cheap-coin alpha, but is not an actual bid/ask fill.
    half=tick(p)/2 if spot else 0.
    return (p+half if buy else p-half)*(1+(1 if buy else -1)*extra_bp/4/10000)

def signals(c,u,f,fx):
    prem=np.log(u[:,3]/(f[:,3]*fx[:,3]))*10000
    mu,sd=base.past_stats(prem)
    z=np.divide(prem-mu,sd,out=np.full(len(prem),np.nan),where=sd>=3)
    out=[];i=1344
    while i<len(GRID)-98:
        if not np.isfinite(z[i]) or z[i]>-2:
            i+=1;continue
        dev=mu[i]-prem[i]
        modeled_cost=20+tick(u[i,3])/u[i,3]*10000+4
        if dev<modeled_cost+8 or not np.all(np.isfinite([u[i,4],f[i,4],fx[i,3]])):
            i+=1;continue
        e=i+1;x=e+96;reason='time_limit'
        for j in range(e,x):
            if np.isfinite(z[j]) and z[j]>=-.5:
                x=j+1;reason='mean_reversion';break
        if np.all(np.isfinite([u[e,0],f[e,0],fx[e,0],u[x,0],f[x,0],fx[x,0]])):
            out.append({'coin':c,'e':e,'x':x,'entry':int(GRID[e]),'exit':int(GRID[x]),
                        'score':float(dev-modeled_cost),'signal_cost_bp':float(modeled_cost),
                        'last_volume_krw':float(min(u[i,4],f[i,4]*fx[i,3])), 'reason':reason})
        i=x+1
    return out

def load_coin(c,fx):
    u=base.on_grid(base.candles('up',c))
    f=base.on_grid(base.candles('fut',c))
    fr=base.funds(c)
    if any(m<=0 for _,_,m in fr):raise RuntimeError('missing funding mark; no zero-funding fallback')
    both=np.isfinite(u[:,0])&np.isfinite(f[:,0])&np.isfinite(fx[:,0])
    tr=signals(c,u,f,fx)
    np.savez_compressed(ROOT/(c+'.npz'),u=u,f=f,fr=np.array(fr,float))
    (ROOT/(c+'_signals.json')).write_text(json.dumps(tr))
    return tr,{'coin':c,'valid_bars':int(both.sum()),'signals':len(tr),'funding_n':len(fr)}

def book(c):
    if c in CACHE_LRU:
        CACHE_LRU.move_to_end(c);return CACHE_LRU[c]
    with np.load(ROOT/(c+'.npz')) as a:
        value=(a['u'].copy(),a['f'].copy(),a['fr'].copy())
    CACHE_LRU[c]=value
    if len(CACHE_LRU)>10:CACHE_LRU.popitem(last=False)
    return value

def regime_rank(tr,ranked,k):
    if k=='fixed20':return tr['coin'] in COINS20
    day=tr['entry']//DAY*DAY
    return tr['coin'] in ranked.get(day,[])[:int(k)]

def run_wallet(alltr,ranked,fx,k,cap,extra):
    events={}
    for tr in alltr:
        if tr['entry']>=OOS and tr['exit']<END and regime_rank(tr,ranked,k):
            events.setdefault(tr['e'],[]).append(tr)
    fxmark=base.ff_prices(fx)
    k0=int(np.searchsorted(GRID,OOS));fx0=fxmark[k0,0]
    if not np.isfinite(fx0) or fx0<=0:raise RuntimeError('initial FX unavailable')
    initial_up=cap/2;initial_bn=cap/2/fx0;cash_up=initial_up;cash_bn=initial_bn
    active=[];trades=[];curve=[];skips={'slot':0,'funds':0,'small_order':0,'missing':0,'cost_gate':0}
    flags=0;size_sum=0.;candidate_count=sum(map(len,events.values()))
    for i in range(k0,len(GRID)):
        t=int(GRID[i]);x0=fxmark[i,0];xc=fxmark[i,3]
        if not (np.isfinite(x0) and np.isfinite(xc)):raise RuntimeError('FX mark gap')
        keep=[]
        for p in active:
            if i>=p['x']:
                u,f,fr=book(p['coin'])
                if not np.all(np.isfinite([u[i,0],f[i,0]])):
                    skips['missing']+=1;keep.append(p);continue
                sp=synthetic_fill(u[i,0],False,extra,True)
                fp=synthetic_fill(f[i,0],True,extra,False)
                cash_up+=p['q']*sp*(1-base.UF)
                cash_bn+=p['q']*(p['fe']-fp)-p['q']*fp*base.FF
                r=dict(p)
                r['exit']=t;r['x']=i
                r['pnl_trade_krw']=p['q']*sp*(1-base.UF)-p['paid']+(p['q']*(p['fe']-fp)-p['q']*fp*base.FF-p['fut_fee']+p['fund'])*x0
                trades.append(r)
            else:keep.append(p)
        active=keep
        for tr in sorted(events.get(i,[]),key=lambda p:(-p['score'],p['coin'])):
            if len(active)>=MAX_POS or any(p['coin']==tr['coin'] for p in active):skips['slot']+=1;continue
            if tr['score']<(extra-4)+8:skips['cost_gate']+=1;continue
            u,f,fr=book(tr['coin'])
            if not np.all(np.isfinite([u[i,0],f[i,0]])):skips['missing']+=1;continue
            up=synthetic_fill(u[i,0],True,extra,True);fp=synthetic_fill(f[i,0],False,extra,False)
            used=sum(p['margin'] for p in active)
            floating=0.
            for p in active:
                _,fa,_=book(p['coin']);mark=fa[i,0]
                if not np.isfinite(mark):mark=p['last_fut']
                floating+=p['q']*(p['fe']-mark)
            bn_free=max(0.,cash_bn+min(0.,floating)-used-initial_bn*.2)
            nominal=min(cap*.2,tr['last_volume_krw']*.005,max(0.,cash_up-initial_up*.2)/(1+base.UF),bn_free*x0/1.002)
            if nominal<max(MIN_NOTIONAL,MAP[tr['coin']]['min_usdt']*x0):skips['small_order']+=1;continue
            q=nominal/up;step=MAP[tr['coin']]['qstep']
            if step>0:q=math.floor(q/step)*step
            if q*up<MIN_NOTIONAL or q*fp<MAP[tr['coin']]['min_usdt']:skips['small_order']+=1;continue
            paid=q*up*(1+base.UF);margin=q*fp;fee=q*fp*base.FF
            if paid>cash_up or margin+fee>bn_free:skips['funds']+=1;continue
            cash_up-=paid;cash_bn-=fee
            p=dict(tr,q=q,ue=up,fe=fp,paid=paid,margin=margin,fut_fee=fee,fund=0.,last_up=u[i,0],last_fut=f[i,0])
            p['fund_ix']=int(np.searchsorted(fr[:,0],t,side='right'))
            active.append(p);size_sum+=q*up
        sv=fv=0.
        for p in active:
            u,f,fr=book(p['coin'])
            while p['fund_ix']<len(fr) and fr[p['fund_ix'],0]<t+STEP:
                ft,rate,mark=fr[p['fund_ix']];p['fund_ix']+=1
                if ft>p['entry']:
                    amt=p['q']*mark*rate;cash_bn+=amt;p['fund']+=amt
            if np.isfinite(u[i,3]):p['last_up']=u[i,3]
            if np.isfinite(f[i,3]):p['last_fut']=f[i,3]
            sv+=p['q']*p['last_up'];fv+=p['q']*(p['fe']-p['last_fut'])
            high=f[i,1]
            if np.isfinite(high) and p['margin']+p['q']*(p['fe']-high)+p['fund']<=p['q']*high*.01:flags+=1
        eq=cash_up+sv+(cash_bn+fv)*xc
        passive=initial_up+initial_bn*xc
        curve.append((t+STEP,eq,cap+eq-passive))
    a=np.array(curve,float);eq=a[:,1:];dd=eq/np.maximum.accumulate(np.vstack(([cap,cap],eq)),axis=0)[1:]-1
    monthly_ends={}
    daily_ends={}
    for t,e,alpha in a:
        monthly_ends[base.iso(int(t-1))[:7]]=(e,alpha)
        daily_ends[base.iso(int(t-1))[:10]]=alpha
    prev=np.array([cap,cap],float);months={}
    for m,v in monthly_ends.items():
        change=np.array(v)-prev;prev=np.array(v)
        months[m]={'account_pnl_krw':float(change[0]),'strategy_vs_passive_pnl_krw':float(change[1])}
    dr=np.diff(np.r_[cap,list(daily_ends.values())])/cap
    rng=np.random.default_rng(130023);boots=[]
    for _ in range(500):
        starts=rng.integers(len(dr),size=math.ceil(len(dr)/7))
        ix=np.concatenate([(s+np.arange(7))%len(dr) for s in starts])[:len(dr)]
        boots.append(float(dr[ix].sum()))
    days=(END-OOS)/DAY
    result={'universe':str(k),'capital':cap,'extra_round_bp':extra,'friction_model':'one Upbit tick roundspread proxy + additional round bp; not historical fills',
            'candidates':candidate_count,'trades':len(trades),'still_open':len(active),'skips':skips,
            'mean_entry_notional':size_sum/(len(trades)+len(active)) if trades or active else 0.,
            'account_pnl_krw':float(eq[-1,0]-cap),'strategy_vs_passive_pnl_krw':float(eq[-1,1]-cap),
            'descriptive_30d_strategy_pnl_krw':float((eq[-1,1]-cap)*30/days),
            'strategy_return_pct':float((eq[-1,1]/cap-1)*100),'account_return_pct':float((eq[-1,0]/cap-1)*100),
            'mtm_account_dd_pct':float(dd[:,0].min()*100),'mtm_strategy_dd_pct':float(dd[:,1].min()*100),
            'block_bootstrap_95_strategy_return_pct':np.quantile(boots,[.025,.975])*100,
            'margin_stress_flags':flags,'monthly':months,'production_pass':False}
    file=ROOT/('wallet_%s_%s_%s.json'%(k,cap,int(extra)))
    file.write_text(json.dumps(base.clean({'result':result,'trades':trades}),allow_nan=False))
    log('UNIVERSE_RESULT',**result)
    return result,trades

def quote_array(ex,c,date):
    sym='KRW-'+c if ex=='upbit' else MAP[c]['fut_symbol']
    path=ROOT/('quote_%s_%s_%s.npy'%(ex,c,date))
    if path.exists():return np.load(path,mmap_mode='r')
    url='https://datasets.tardis.dev/v1/%s/quotes/%s/%s.csv.gz'%(ex,date.replace('-','/'),sym)
    out=[];last=-1;mult=1. if ex=='upbit' else MAP[c]['multiplier']
    with limited_get(url,stream=True) as r:
        r.raw.decode_content=False
        with io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw),encoding='utf-8') as f:
            for row in csv.DictReader(f):
                t=int(row['local_timestamp'])
                if t<last:raise RuntimeError('unsorted quote receive time')
                last=t
                v=[float(row[key]) for key in ('bid_price','bid_amount','ask_price','ask_amount')]
                if not all(math.isfinite(x) for x in v) or v[0]<=0 or v[2]<=v[0] or min(v[1],v[3])<0:continue
                out.append([t,v[0]/mult,v[1]*mult,v[2]/mult,v[3]*mult])
    if not out:raise RuntimeError('empty quote sample')
    a=np.asarray(out,float);np.save(path,a)
    log('QUOTE_SAMPLE_READY',coin=c,date=date,exchange=ex,rows=len(a))
    return a

def quote_at(a,t):
    i=int(np.searchsorted(a[:,0],t,side='right')-1)
    if i<0 or t-a[i,0]>2_000_000:return None
    return a[i]

def quote_diagnostic(trades,fx):
    groups={}
    for tr in trades:
        d=base.iso(tr['entry'])[:10]
        if d[8:10]=='01' and d==base.iso(tr['exit'])[:10]:
            groups.setdefault((tr['coin'],d),[]).append(tr)
    results=[];errors=[]
    for (c,d),trs in sorted(groups.items()):
        try:
            u=quote_array('upbit',c,d);f=quote_array('binance-futures',c,d)
            for tr in trs:
                for lat in (100,200,500):
                    ue=quote_at(u,tr['entry']*1000+lat*1000);fe=quote_at(f,tr['entry']*1000+lat*1000)
                    ux=quote_at(u,tr['exit']*1000+lat*1000);ff=quote_at(f,tr['exit']*1000+lat*1000)
                    r={'coin':c,'entry':tr['entry'],'exit':tr['exit'],'lat_ms':lat,'universe':tr['universe']}
                    if any(a is None for a in (ue,fe,ux,ff)):
                        r['status']='stale_or_missing';results.append(r);continue
                    q=tr['q'];capacity=min(ue[4],fe[2],ux[2],ff[4]);r['capacity']=float(capacity)
                    if capacity<q:r['status']='insufficient_L1_quantity';results.append(r);continue
                    x0=fx[tr['e'],0];x1=fx[tr['x'],0]
                    p=q*ux[1]*(1-base.UF)-q*ue[3]*(1+base.UF)+(q*(fe[1]-ff[3])-q*fe[1]*base.FF-q*ff[3]*base.FF+tr['fund'])*x1
                    r.update(status='quote_only_full_L1',pnl_krw=float(p),net_bp=float(p/(q*ue[3])*10000))
                    results.append(r)
        except Exception as e:errors.append({'coin':c,'date':d,'error':str(e)[:200]})
    for lat in (100,200,500):
        for k in ('fixed20','30','100'):
            rr=[x for x in results if x['lat_ms']==lat and x['universe']==k]
            good=[x['net_bp'] for x in rr if x['status']=='quote_only_full_L1']
            log('MATCHED_QUOTE_CHECK',universe=k,lat_ms=lat,attempted=len(rr),full_L1=len(good),
                stats=base.metrics(good),nonfilled=len(rr)-len(good),
                note='Sparse month-start matched-trade audit only. Not true order fills or an independently rerun shared-wallet portfolio.')
    (ROOT/'matched_quotes.json').write_text(json.dumps(base.clean({'rows':results,'errors':errors})))
    return len(results),errors

def selftest():
    assert tick(129)==1 and tick(1001)==1 and tick(999)==1
    assert tick(100000)==100 and tick(.01)==.0001
    assert synthetic_fill(129,True,4,True)>129
    assert synthetic_fill(129,False,4,True)<129
    assert synthetic_fill(1,True,4,False)>1
    a=np.array([[1000,1,1,2,1],[3000,1,1,2,1]],float)
    assert quote_at(a,999) is None and quote_at(a,2500)[0]==1000
    assert quote_at(a,2004000) is None
    tr={'entry':START,'coin':'X'};r={START:['Y','X']}
    assert regime_rank(tr,r,1) is False and regime_rank(tr,r,2) is True
    log('UNIVERSE_SELFTEST_OK',checks=11)

def main():
    selftest()
    log('UNIVERSE_BT_START',period_start=base.iso(START),period_end_exclusive=base.iso(END),
        evaluation_start=base.iso(OOS),bar_minutes=15,universes=['fixed20','30','100'],capital=CAPITALS,
        fees_per_execution_bp={'upbit':5,'binance_futures':5},extra_round_bp=EXTRA_RT_BP,
        signal='past14d excluding current,z<=-2,net expected reversion>=8bp',
        order_size='adaptive min(20pct total capital,0.5pct preceding 15m turnover,funded cash/margin)',
        screening_only=True,no_order_code=True,max_runtime_hours=3)
    universe()
    ranked,coins=ranking()
    log('BAR_DOWNLOAD_START',coins=len(coins),total_15m_grid_bars=len(GRID),workers=6)
    fx=base.on_grid(base.candles('up','USDT'))
    alltr=[];ok=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs={pool.submit(load_coin,c,fx):c for c in coins}
        for i,task in enumerate(as_completed(jobs),1):
            c=jobs[task]
            try:
                tr,info=task.result();alltr.extend(tr);ok.append(c)
                log('COIN_SCREEN_DONE',**info,finished=i,total=len(coins))
            except Exception as e:
                FAILURES.append({'phase':'intraday','coin':c,'error':str(e)[:200]})
                log('COIN_SCREEN_FAIL',coin=c,error=str(e)[:200],finished=i,total=len(coins))
    if not ok:raise RuntimeError('no intraday coins loaded')
    log('BAR_DOWNLOAD_DONE',coins_ok=len(ok),coins_requested=len(coins),independent_episodes=len(alltr),failures=len(FAILURES))
    results=[];baseline=[]
    for k in ('fixed20','30','100'):
        for cap in CAPITALS:
            for extra in EXTRA_RT_BP:
                result,tr=run_wallet(alltr,ranked,fx,k,cap,extra);results.append(result)
                if cap==10_000_000 and extra==4:
                    baseline.extend([dict(p,universe=k) for p in tr])
    n,errors=quote_diagnostic(baseline,fx)
    summary={'period':[base.iso(START),base.iso(END)],'oos':[base.iso(OOS),base.iso(END)],
             'loaded_coins':len(ok),'failures':FAILURES,'quote_rows':n,'quote_errors':errors,
             'production_pass':False,'reason':'Continuous L2, genuine fills, liquidation and untouched future validation missing; candles remain screening only.',
             'forecast_monthly_income':None}
    (ROOT/'summary.json').write_text(json.dumps(base.clean({'summary':summary,'results':results}),indent=2,allow_nan=False))
    log('UNIVERSE_BT_DONE',**summary)

if __name__=='__main__':
    try:main()
    except Exception as e:
        log('UNIVERSE_BT_FATAL',error=repr(e),traceback=traceback.format_exc()[-3000:],production_pass=False)
        raise
