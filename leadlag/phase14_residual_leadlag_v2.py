#!/usr/bin/env python3
"""
Residual Lead-Lag V2 — public historical quote research only.

Purpose:
- Test V2-A taker/taker residual lead-lag
- Test V2-B stale-quote sniper
- Test V2-C BTC-leader filter
- Test V2-D top-of-book imbalance filter
- Test positive/negative directions separately
- Test latency and exit horizons with actual historical L1 quotes

Important limitations:
- Quotes are top-of-book only, not full depth or genuine order fills.
- Maker mode is a conservative/diagnostic quote-touch proxy, not queue simulation.
- Negative direction requires spot shorting and is research-only for Upbit.
- This is screening research, never production trading code.
"""
from __future__ import annotations
import csv, gzip, io, json, math, os, time, traceback
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone, timedelta
import numpy as np
import requests

ROOT=Path('/tmp/residual-leadlag-v2'); ROOT.mkdir(exist_ok=True)
STEP_MS=int(os.getenv('V2_STEP_MS','100'))
MAX_STALE_MS=int(os.getenv('V2_MAX_STALE_MS','1000'))
COINS=os.getenv('V2_COINS','BTC,ETH,XRP,SOL,DOGE,ADA,SUI,LINK').split(',')
DATES=os.getenv('V2_DATES','2026-04-10,2026-05-10,2026-06-10,2026-07-10,2026-08-10,2026-09-10').split(',')
UF=float(os.getenv('UP_FEE_BP','5'))/10000
FF=float(os.getenv('BN_FEE_BP','5'))/10000
LAT_MS=[int(x) for x in os.getenv('V2_LAT_MS','100,200,500').split(',')]
HORIZON_MS=[int(x) for x in os.getenv('V2_HORIZON_MS','1000,2000,5000,10000').split(',')]
PROFILES=[
    ('loose',15.,8.,5.),
    ('base',25.,12.,8.),
    ('strict',40.,20.,12.),
]
TLS=requests.Session(); TLS.headers['User-Agent']='daol-residual-leadlag-v2/14'
ALLOWED={'datasets.tardis.dev'}
DAY_US=86_400_000_000

def log(tag,**kw):
    def clean(x):
        if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        if isinstance(x,np.ndarray): return clean(x.tolist())
        if isinstance(x,(np.integer,)): return int(x)
        if isinstance(x,(np.floating,float)):
            v=float(x); return v if math.isfinite(v) else None
        if isinstance(x,np.bool_): return bool(x)
        return x
    print(tag+' '+json.dumps(clean(kw),separators=(',',':'),ensure_ascii=False,allow_nan=False),flush=True)

def dt_us(date):
    return int(datetime.fromisoformat(date+'T00:00:00+00:00').timestamp()*1_000_000)

def qurl(ex,date,sym):
    return f"https://datasets.tardis.dev/v1/{ex}/quotes/{date.replace('-','/')}/{sym}.csv.gz"

def get_stream(url):
    p=urlparse(url)
    if p.scheme!='https' or p.hostname not in ALLOWED: raise ValueError('host not allowed')
    for a in range(5):
        try:
            r=TLS.get(url,timeout=(15,90),stream=True)
        except (requests.Timeout,requests.ConnectionError):
            if a==4: raise
            time.sleep(2**a); continue
        if r.status_code in (429,500,502,503,504):
            r.close(); time.sleep(min(30,2**a)); continue
        r.raise_for_status(); return r
    raise RuntimeError('download retries exhausted')

def resample_quotes(ex,date,sym,mult=1.0):
    start=dt_us(date); n=86_400_000//STEP_MS
    out=np.full((n,4),np.nan,np.float32)
    last_bucket=-1; last=None; rows=0
    url=qurl(ex,date,sym)
    with get_stream(url) as r:
        r.raw.decode_content=False
        with io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw),encoding='utf-8') as f:
            for row in csv.DictReader(f):
                t=int(row['local_timestamp'])
                if t<start or t>=start+DAY_US: continue
                try:
                    bp=float(row['bid_price'])/mult; bq=float(row['bid_amount'])*mult
                    ap=float(row['ask_price'])/mult; aq=float(row['ask_amount'])*mult
                except Exception: continue
                if not (bp>0 and ap>bp and bq>=0 and aq>=0): continue
                b=(t-start)//(STEP_MS*1000)
                if b<0 or b>=n: continue
                out[int(b)]=[bp,bq,ap,aq]; rows+=1; last_bucket=int(b); last=(bp,bq,ap,aq)
    if rows==0: raise RuntimeError(f'empty quotes {ex} {date} {sym}')
    # forward fill, but later invalidate quotes older than MAX_STALE_MS
    valid=np.where(np.isfinite(out[:,0]))[0]
    ix=np.full(n,-1,np.int32); ix[valid]=valid; np.maximum.accumulate(ix,out=ix)
    good=ix>=0; age=np.arange(n)-ix
    out[good]=out[ix[good]]
    out[age*STEP_MS>MAX_STALE_MS]=np.nan
    log('V2_DATA',exchange=ex,date=date,symbol=sym,raw_rows=rows,grid_rows=n,valid=int(np.isfinite(out[:,0]).sum()))
    return out

def mids(a):
    return (a[:,0].astype(np.float64)+a[:,2].astype(np.float64))/2

def ret_bp(x,w):
    out=np.full(len(x),np.nan)
    ok=np.isfinite(x[w:])&np.isfinite(x[:-w])&(x[:-w]>0)
    out[w:][ok]=np.log(x[w:][ok]/x[:-w][ok])*10000
    return out

def metrics(vals):
    a=np.asarray(vals,float)
    if len(a)==0:return {'n':0}
    pos=a[a>0].sum(); neg=-a[a<0].sum()
    return {'n':int(len(a)),'mean_bp':float(a.mean()),'median_bp':float(np.median(a)),
            'win_rate':float(np.mean(a>0)),'pf':float(pos/neg) if neg>0 else None,
            'sum_bp':float(a.sum()),'p05_bp':float(np.quantile(a,.05)),
            'p95_bp':float(np.quantile(a,.95))}

def collapse(cond,cooldown_ms=2000):
    idx=np.flatnonzero(cond)
    if len(idx)==0:return idx
    keep=[]; next_ok=-1; cool=max(1,cooldown_ms//STEP_MS)
    for i in idx:
        if i>=next_ok:
            keep.append(int(i)); next_ok=int(i)+cool
    return np.asarray(keep,np.int64)

def pnl_taker(up,fu,fx,i,j,d):
    # d=+1: buy spot ask + short future bid; feasible on Upbit.
    # d=-1: short spot bid + long future ask; research-only.
    fxe=float(fx[i]); fxx=float(fx[j])
    if d==1:
        ue=float(up[i,2]); ux=float(up[j,0]); fe=float(fu[i,0]); fz=float(fu[j,2])
        if min(ue,ux,fe,fz,fxe,fxx)<=0:return None
        q=1.0/ue
        pnl=q*(ux*(1-UF)-ue*(1+UF))+q*(fe*(1-FF)-fz*(1+FF))*fxx
        cap=min(float(up[i,3])*ue,float(fu[i,1])*fe*fxe)
    else:
        ue=float(up[i,0]); ux=float(up[j,2]); fe=float(fu[i,2]); fz=float(fu[j,0])
        if min(ue,ux,fe,fz,fxe,fxx)<=0:return None
        q=1.0/ue
        pnl=q*(ue*(1-UF)-ux*(1+UF))+q*(fz*(1-FF)-fe*(1+FF))*fxx
        cap=min(float(up[i,1])*ue,float(fu[i,3])*fe*fxe)
    return pnl*10000,cap

def pnl_maker_touch(up,fu,fx,i,h,d,wait_ms=500):
    # Quote-touch proxy only. No queue position or actual fill assertion.
    wait=max(1,wait_ms//STEP_MS); n=len(up)
    if d==1:
        maker=float(up[i,0])
        fill=None
        for k in range(i+1,min(i+wait+1,n)):
            if np.isfinite(up[k,2]) and float(up[k,2])<=maker:
                fill=k;break
        if fill is None:return None
        j=fill+h
        if j>=n:return None
        fe=float(fu[fill,0]); fz=float(fu[j,2]); ux=float(up[j,0])
        if not all(np.isfinite([maker,fe,fz,ux,fx[fill],fx[j]])):return None
        q=1.0/maker
        # maker fee conservatively treated as same standard fee; no VIP/discount assumptions.
        pnl=q*(ux*(1-UF)-maker*(1+UF))+q*(fe*(1-FF)-fz*(1+FF))*float(fx[j])
        cap=min(float(up[i,1])*maker,float(fu[fill,1])*fe*float(fx[fill]))
    else:
        maker=float(up[i,2])
        fill=None
        for k in range(i+1,min(i+wait+1,n)):
            if np.isfinite(up[k,0]) and float(up[k,0])>=maker:
                fill=k;break
        if fill is None:return None
        j=fill+h
        if j>=n:return None
        fe=float(fu[fill,2]); fz=float(fu[j,0]); ux=float(up[j,2])
        if not all(np.isfinite([maker,fe,fz,ux,fx[fill],fx[j]])):return None
        q=1.0/maker
        pnl=q*(maker*(1-UF)-ux*(1+UF))+q*(fz*(1-FF)-fe*(1+FF))*float(fx[j])
        cap=min(float(up[i,3])*maker,float(fu[fill,3])*fe*float(fx[fill]))
    return pnl*10000,cap,fill

def run_date(date):
    log('V2_DATE_START',date=date,coins=COINS,step_ms=STEP_MS)
    fxq=resample_quotes('upbit',date,'KRW-USDT')
    fx=mids(fxq)
    work=ROOT/date; work.mkdir(exist_ok=True)
    prems=[]; usable=[]
    # pass 1: build per-coin aligned quote arrays and premiums
    for c in COINS:
        try:
            up=resample_quotes('upbit',date,'KRW-'+c)
            # historical symbol scaling for common liquid set; SHIB not in default list
            fs=('1000SHIBUSDT',1000.) if c=='SHIB' else (c+'USDT',1.)
            fu=resample_quotes('binance-futures',date,fs[0],fs[1])
            um=mids(up); fm=mids(fu)
            prem=np.log(um/(fm*fx))*10000
            np.savez_compressed(work/(c+'.npz'),up=up,fu=fu,um=um.astype(np.float32),fm=fm.astype(np.float32),prem=prem.astype(np.float32))
            prems.append(prem); usable.append(c)
        except Exception as e:
            log('V2_COIN_FAIL',date=date,coin=c,error=str(e)[:240])
    if len(usable)<4: raise RuntimeError('fewer than 4 usable coins '+date)
    P=np.vstack(prems)
    common=np.nanmedian(P,axis=0)
    del P,prems
    # BTC leadership arrays
    btc=None
    if 'BTC' in usable:
        z=np.load(work/'BTC.npz'); btc=fm_btc=z['fm'].astype(float)
    rows=[]
    for c in usable:
        z=np.load(work/(c+'.npz')); up=z['up'];fu=z['fu'];um=z['um'].astype(float);fm=z['fm'].astype(float);prem=z['prem'].astype(float)
        residual=prem-common
        upimb=(up[:,1]-up[:,3])/(up[:,1]+up[:,3]+1e-12)
        fuimb=(fu[:,1]-fu[:,3])/(fu[:,1]+fu[:,3]+1e-12)
        for pname,shock_th,lag_th,res_th in PROFILES:
            for win_ms in (200,500,1000):
                w=max(1,win_ms//STEP_MS)
                br=ret_bp(fm,w); ur=ret_bp(um,w); lag=br-ur
                stale_up=(np.r_[np.full(w,np.nan),up[w:,2]]==np.r_[np.full(w,np.nan),up[:-w,2]])
                btclead=np.zeros(len(um),bool)
                if c!='BTC' and btc is not None:
                    bw=max(1,500//STEP_MS)
                    bprev=np.full(len(um),np.nan)
                    if 2*bw<len(um):
                        good=np.isfinite(btc[2*bw:])&np.isfinite(btc[bw:-bw])&(btc[bw:-bw]>0)
                        tmp=np.full(len(um)-2*bw,np.nan)
                        tmp[good]=np.log(btc[2*bw:][good]/btc[bw:-bw][good])*10000
                        bprev[2*bw:]=tmp
                    btclead=np.isfinite(bprev)&(np.sign(bprev)==np.sign(br))&(np.abs(bprev)>=8)
                for d in (1,-1):
                    base=np.isfinite(br)&np.isfinite(lag)&np.isfinite(residual)
                    base &= (d*br>=shock_th)&(d*lag>=lag_th)&(d*residual<=-res_th)
                    variants={
                      'A_taker':base,
                      'B_stale':base&stale_up,
                      'C_btc_leader':base&btclead,
                      'D_imbalance':base&(d*upimb>0)&(d*fuimb>0),
                    }
                    for vname,cond in variants.items():
                        sig=collapse(cond,2000)
                        for latency in LAT_MS:
                            ls=max(1,int(round(latency/STEP_MS)))
                            effective=ls*STEP_MS
                            for horizon in HORIZON_MS:
                                hs=max(1,horizon//STEP_MS)
                                pn=[]; caps=[]
                                for s in sig:
                                    i=int(s+ls); j=int(i+hs)
                                    if j>=len(up):continue
                                    x=pnl_taker(up,fu,fx,i,j,d)
                                    if x is None:continue
                                    pn.append(x[0]);caps.append(x[1])
                                rows.append({'date':date,'coin':c,'variant':vname,'profile':pname,'window_ms':win_ms,
                                  'direction':d,'latency_ms_requested':latency,'latency_ms_effective':effective,'horizon_ms':horizon,
                                  'signals':len(sig),'metrics':metrics(pn),'median_l1_capacity_krw':float(np.median(caps)) if caps else None})
                        # maker touch evaluated separately, no added latency because wait-to-fill is endogenous
                        if vname=='B_stale':
                            for horizon in HORIZON_MS:
                                hs=max(1,horizon//STEP_MS)
                                pn=[];caps=[];fills=[]
                                for s in sig:
                                    x=pnl_maker_touch(up,fu,fx,int(s),hs,d,500)
                                    if x is None:continue
                                    pn.append(x[0]);caps.append(x[1]);fills.append((x[2]-int(s))*STEP_MS)
                                rows.append({'date':date,'coin':c,'variant':'B_maker_touch','profile':pname,'window_ms':win_ms,
                                  'direction':d,'latency_ms_requested':None,'latency_ms_effective':None,'horizon_ms':horizon,
                                  'signals':len(sig),'metrics':metrics(pn),'touch_rate':len(pn)/len(sig) if len(sig) else None,
                                  'median_touch_ms':float(np.median(fills)) if fills else None,
                                  'median_l1_capacity_krw':float(np.median(caps)) if caps else None})
        log('V2_COIN_DONE',date=date,coin=c,rows=sum(1 for x in rows if x['coin']==c))
    # cleanup large files after summarizing date
    for p in work.glob('*.npz'):
        try:p.unlink()
        except:pass
    try:work.rmdir()
    except:pass
    log('V2_DATE_DONE',date=date,usable=usable,result_rows=len(rows))
    return rows

def aggregate(rows):
    groups={}
    for r in rows:
        k=(r['variant'],r['profile'],r['window_ms'],r['direction'],r['latency_ms_effective'],r['horizon_ms'])
        g=groups.setdefault(k,{'pn':[],'caps':[],'signals':0,'touch_num':0,'touch_den':0})
        # aggregate per-cell means weighted by trade count is impossible from metrics alone; save compact approximation
        m=r['metrics']; n=int(m.get('n',0)); g['signals']+=int(r.get('signals',0))
        if n and m.get('mean_bp') is not None:
            g['pn'].extend([float(m['mean_bp'])]*n)
        if r.get('median_l1_capacity_krw') is not None:g['caps'].append(float(r['median_l1_capacity_krw']))
        if r['variant']=='B_maker_touch' and r.get('touch_rate') is not None:
            g['touch_num']+=n;g['touch_den']+=int(r.get('signals',0))
    out=[]
    for k,g in groups.items():
        v,p,w,d,lat,h=k
        out.append({'variant':v,'profile':p,'window_ms':w,'direction':d,'latency_ms_effective':lat,'horizon_ms':h,
                    'signals':g['signals'],'approx_metrics':metrics(g['pn']),
                    'median_of_coin_day_l1_capacity_krw':float(np.median(g['caps'])) if g['caps'] else None,
                    'touch_rate':g['touch_num']/g['touch_den'] if g['touch_den'] else None})
    return out

def main():
    log('V2_BT_START',dates=DATES,coins=COINS,step_ms=STEP_MS,max_stale_ms=MAX_STALE_MS,
        profiles=PROFILES,latency_ms=LAT_MS,horizons_ms=HORIZON_MS,
        fees_bp={'upbit_per_execution':UF*10000,'binance_futures_per_execution':FF*10000},
        variants=['A_taker','B_stale','B_maker_touch','C_btc_leader','D_imbalance'],
        note='Top-of-book historical quote screening. No order endpoints. Maker result is quote-touch proxy only.')
    allrows=[]; failures=[]
    for d in DATES:
        try: allrows.extend(run_date(d))
        except Exception as e:
            failures.append({'date':d,'error':repr(e)})
            log('V2_DATE_FATAL',date=d,error=repr(e),traceback=traceback.format_exc()[-1800:])
    agg=aggregate(allrows)
    # emit best descriptive cells by positive-direction feasible path and minimum 20 approximate trades
    feasible=[x for x in agg if x['direction']==1 and x['approx_metrics'].get('n',0)>=20]
    feasible.sort(key=lambda x:(x['approx_metrics'].get('mean_bp') if x['approx_metrics'].get('mean_bp') is not None else -1e9),reverse=True)
    for x in feasible[:40]: log('V2_RESULT',**x)
    summary={'dates_requested':DATES,'dates_completed':sorted(set(r['date'] for r in allrows)),
             'coins_requested':COINS,'result_rows':len(allrows),'aggregate_cells':len(agg),'failures':failures,
             'production_pass':False,
             'reason':'Historical L1 quote screening only; maker queue, true fills, L2 depth, liquidation, and untouched future validation remain unverified.'}
    (ROOT/'v2_rows.json').write_text(json.dumps(allrows))
    (ROOT/'v2_aggregate.json').write_text(json.dumps(agg))
    log('V2_BT_DONE',**summary)

if __name__=='__main__':
    try: main()
    except Exception as e:
        log('V2_BT_FATAL',error=repr(e),traceback=traceback.format_exc()[-3000:],production_pass=False)
        raise
