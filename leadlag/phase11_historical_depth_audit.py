#!/usr/bin/env python3
"""Historical market-data audit ONLY. GET/HEAD public endpoints. No trading keys.
Monthly sample days do not represent a continuous investment return.
"""
import csv,gzip,io,json,math,time
from itertools import groupby
from datetime import datetime,timezone
import numpy as np
import requests
BASE='https://datasets.tardis.dev/v1'
DATES=['2025-10-01','2026-06-01','2026-09-01']
SIZES=[1000000,2000000,4000000,10000000]
S=requests.Session();S.headers['User-Agent']='daol-paper-historical-depth-audit/1'

def log(tag,**kw):print(tag+' '+json.dumps(kw,separators=(',',':'),allow_nan=False),flush=True)
def day_us(d):return int(datetime.fromisoformat(d+'T00:00:00+00:00').timestamp()*1000000)
def url(ex,typ,d,sym):return BASE+'/'+ex+'/'+typ+'/'+d.replace('-','/')+'/'+sym+'.csv.gz'
def quant(x):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    if not len(a):return {'n':0}
    return {'n':len(a),'mean':float(a.mean()),'p10':float(np.quantile(a,.1)),'p50':float(np.median(a)),'p90':float(np.quantile(a,.9)),'max':float(a.max())}
def rows(ex,typ,d,sym):
    u=url(ex,typ,d,sym)
    with S.get(u,stream=True,timeout=(15,90)) as r:
        log('CSV_ACCESS',exchange=ex,type=typ,date=d,symbol=sym,status=r.status_code)
        r.raise_for_status();r.raw.decode_content=False
        with io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw),encoding='utf-8',newline='') as txt:
            yield from csv.DictReader(txt)
def quotes(ex,d,sym):
    out=[];last=0
    for r in rows(ex,'quotes',d,sym):
        t=int(r['local_timestamp']);v=[float(r[k]) for k in ['bid_price','bid_amount','ask_price','ask_amount']]
        if t<last:raise ValueError('unsorted receive timestamps')
        last=t
        if not all(math.isfinite(x) for x in v) or v[0]<=0 or v[2]<=v[0] or min(v[1],v[3])<0:continue
        out.append([t]+v)
    return np.array(out,float)
def asof(a,t,max_age_us=5000000):
    ix=np.searchsorted(a[:,0],t,side='right')-1;ok=ix>=0;ix=np.maximum(ix,0)
    ok &= (t-a[ix,0])<=max_age_us
    return a[ix,1:],ok

def quote_day(c,d):
    start=day_us(d);times=np.arange(start,start+86400000000,1000000,dtype=np.int64)
    uq=quotes('upbit',d,'KRW-'+c);fq=quotes('binance-futures',d,c+'USDT')
    u,uo=asof(uq,times);f,fo=asof(fq,times)
    valid=uo&fo;us=(u[:,2]-u[:,0])/((u[:,2]+u[:,0])/2)*10000
    fs=(f[:,2]-f[:,0])/((f[:,2]+f[:,0])/2)*10000
    spread=us+fs;rr={'coin':c,'date':d,'sample_seconds':int(valid.sum()),'coverage':float(valid.mean()),
        'upbit_spread_bp':quant(us[valid]),'futures_spread_bp':quant(fs[valid]),'combined_round_spread_bp':quant(spread[valid]),
        'share_combined_le4bp':float(np.mean(spread[valid]<=4)) if valid.any() else None,
        'note':'Time-weighted 1s as-of quotes, receive age <=5s; simultaneous static round spread excludes fees, impact, future price movement.'}
    depth={}
    for n in SIZES:
        q=n/u[:,2];depth[str(n)]={'both_venues_both_sides_L1_capacity_share':float(np.mean((np.minimum.reduce([u[:,1],u[:,3],f[:,1],f[:,3]])[valid]>=q[valid])))}
    rr['capacity']=depth
    log('HISTORICAL_SPREAD_DAY',**rr)
    # Measured quote changes over nominal delays, NOT two-leg order fill rates.
    # Uniform minute boundaries are non-overlapping, selected without future prices.
    st=np.arange(start+60000000,start+86400000000-1000000,60000000,dtype=np.int64)
    u0,vu0=asof(uq,st);f0,vf0=asof(fq,st)
    for lat in [100,200,500]:
        ul,vul=asof(uq,st+lat*1000);fl,vfl=asof(fq,st+lat*1000);ok=vu0&vf0&vul&vfl
        # buy spot / sell future adverse quote drift in bp relative to observed price.
        drift=np.log(ul[:,2]/u0[:,2])*10000-np.log(fl[:,0]/f0[:,0])*10000
        log('QUOTE_LATENCY_DIAGNOSTIC',coin=c,date=d,latency_ms=lat,drift_bp=quant(drift[ok]),
            unchanged_share=float(np.mean(drift[ok]==0)) if ok.any() else None,
            note='quote-level receive-time sensitivity only, not order or arbitrage survival')
    return rr

def vw(levels,q):
    cost=0.;left=q
    for p,n in levels:
        take=min(n,left);cost+=p*take;left-=take
        if left<=q*1e-10:return cost/q
    return None

def l2_sample(c,d):
    # Use supported incremental_book_L2, NOT an assumed book_snapshot_25 dataset.
    start=day_us(d);end=start+3600000000
    bids={};asks={};ready=False;prevsnap=False;count=0;seen=0;lastsample=start
    spreads={str(n):[] for n in SIZES};covers={str(n):0 for n in SIZES};bad=0
    stream=rows('upbit','incremental_book_L2',d,'KRW-'+c)
    try:
        for key,group in groupby(stream,key=lambda r:int(r['local_timestamp'])):
            if key>end:break
            for r in group:
                snap=r['is_snapshot'].lower()=='true'
                if snap and not prevsnap:bids.clear();asks.clear();ready=True
                prevsnap=snap
                if not ready:continue
                p=float(r['price']);n=float(r['amount']);book=bids if r['side']=='bid' else asks
                if n==0:book.pop(p,None)
                else:book[p]=n
                seen+=1
            if not ready or not bids or not asks or key-lastsample<1000000:continue
            lastsample=key;bs=sorted(bids.items(),reverse=True)[:20];aa=sorted(asks.items())[:20]
            if bs[0][0]>=aa[0][0]:bad+=1;continue
            count+=1
            for n in SIZES:
                q=n/aa[0][0];buy=vw(aa,q);sell=vw(bs,q)
                if buy is None or sell is None:continue
                covers[str(n)]+=1;spreads[str(n)].append((buy-sell)/((buy+sell)/2)*10000)
    finally:stream.close()
    log('HISTORICAL_L2_SAMPLE',coin=c,date=d,hours=1,rows=seen,snapshots=count,crossed_skipped=bad,
        size_metrics={str(n):{'covered':covers[str(n)],'total':count,'upbit_round_spread_impact_bp':quant(spreads[str(n)])} for n in SIZES},
        note='First UTC hour, at most one complete reconstructed update per second, top20 available levels only; NOT full-period trade replay.')

def main():
    assert vw([(100,1),(110,1)],1.5)==155/1.5
    log('DEPTH_AUDIT_START',dates=DATES,coins=['DOGE','HBAR','BTC'],paper_only=True)
    # Access test on the correct type: paid periods are never purchased or bypassed.
    for d in ['2026-09-01','2026-09-02']:
        u=url('upbit','incremental_book_L2',d,'KRW-DOGE')
        try:
            r=S.head(u,timeout=20,allow_redirects=True)
            log('CORRECT_L2_ACCESS',date=d,status=r.status_code,bytes=r.headers.get('content-length'))
        except Exception as e:log('CORRECT_L2_ACCESS',date=d,error=repr(e))
    fails=[]
    for c in ['DOGE','HBAR','BTC']:
        for d in DATES:
            try:quote_day(c,d)
            except Exception as e:fails.append([c,d,repr(e)]);log('HIST_FAIL',coin=c,date=d,error=repr(e))
    for c in ['DOGE','HBAR']:
        try:l2_sample(c,'2026-09-01')
        except Exception as e:fails.append([c,'L2',repr(e)]);log('HIST_L2_FAIL',coin=c,error=repr(e))
    log('DEPTH_AUDIT_DONE',failures=fails,continuous_l2_validated=False)
if __name__=='__main__':main()
