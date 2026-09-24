#!/usr/bin/env python3
"""
Phase 15: Extreme Dislocation Sniper (V3) + Maker->Hedge (V4)
Historical public L1 quote research only. No private APIs or orders.

V3:
- 30/50/70/100bp Binance impulse profiles
- leave-one-out common Korean premium
- executable residual at entry
- estimated net edge after 4 standard fees + expected exit half-spreads
- 10/20/30bp minimum net-edge gates
- taker/taker, BTC-leader, order-book-imbalance variants
- fixed exits and convergence/stop/timeout exits

V4:
- Upbit maker-at-bid/ask then Binance taker hedge
- both optimistic "touch" and conservative "cross-through" fill proxies
- same standard fees (no VIP/coupon assumptions)
- queue position is NOT modeled

Limitations:
- L1 only; no L2 depth, true queue fills, liquidation engine, or live network latency.
- Negative direction needs spot shorting on Upbit and is research-only.
"""
from __future__ import annotations
import csv,gzip,io,json,math,os,time,traceback
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime,timezone
import numpy as np, requests

ROOT=Path('/tmp/extreme-dislocation-v3'); ROOT.mkdir(exist_ok=True)
STEP_MS=int(os.getenv('V3_STEP_MS','100'))
MAX_STALE_MS=int(os.getenv('V3_MAX_STALE_MS','1000'))
COINS=os.getenv('V3_COINS','BTC,ETH,XRP,SOL,DOGE,ADA,SUI,LINK').split(',')
DATES=os.getenv('V3_DATES','2026-04-01,2026-05-01,2026-06-01,2026-07-01,2026-08-01,2026-09-01').split(',')
UF=float(os.getenv('UP_FEE_BP','5'))/10000
FF=float(os.getenv('BN_FEE_BP','5'))/10000
LAT_MS=[int(x) for x in os.getenv('V3_LAT_MS','100,200,500').split(',')]
FIXED_HORIZONS=[int(x) for x in os.getenv('V3_HORIZON_MS','1000,2000,5000,10000,30000').split(',')]
NET_GATES=[float(x) for x in os.getenv('V3_NET_GATES_BP','10,20,30').split(',')]
# name, binance shock, lag, executable residual gross-room
PROFILES=[
 ('E30',30.,15.,30.),
 ('E50',50.,20.,40.),
 ('E70',70.,30.,50.),
 ('E100',100.,40.,70.),
]
WINDOWS_MS=(200,500,1000)
COOLDOWN_MS=5000
TARGET_RESID_BP=5.
STOP_WORSEN_BP=50.
MAX_TARGET_MS=30000
MAKER_WAIT_MS=1000
DAY_US=86_400_000_000
S=requests.Session(); S.headers['User-Agent']='daol-extreme-dislocation-v3/15'
ALLOWED={'datasets.tardis.dev'}

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,float)):
        v=float(x); return v if math.isfinite(v) else None
    if isinstance(x,np.bool_): return bool(x)
    return x

def log(tag,**kw): print(tag+' '+json.dumps(clean(kw),separators=(',',':'),ensure_ascii=False,allow_nan=False),flush=True)

def dt_us(date): return int(datetime.fromisoformat(date+'T00:00:00+00:00').timestamp()*1_000_000)

def stream(url):
    p=urlparse(url)
    if p.scheme!='https' or p.hostname not in ALLOWED: raise ValueError('host not allowed')
    for a in range(5):
        try:r=S.get(url,timeout=(15,90),stream=True)
        except (requests.Timeout,requests.ConnectionError):
            if a==4:raise
            time.sleep(2**a);continue
        if r.status_code in (429,500,502,503,504):
            r.close();time.sleep(min(30,2**a));continue
        r.raise_for_status();return r
    raise RuntimeError('download retry exhausted')

def quote_grid(ex,date,symbol,mult=1.):
    start=dt_us(date); n=86_400_000//STEP_MS
    a=np.full((n,4),np.nan,np.float32); rows=0
    url=f"https://datasets.tardis.dev/v1/{ex}/quotes/{date.replace('-','/')}/{symbol}.csv.gz"
    with stream(url) as r:
        r.raw.decode_content=False
        with io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw),encoding='utf-8') as f:
            for row in csv.DictReader(f):
                t=int(row['local_timestamp'])
                if t<start or t>=start+DAY_US:continue
                try:
                    bp=float(row['bid_price'])/mult;bq=float(row['bid_amount'])*mult
                    ap=float(row['ask_price'])/mult;aq=float(row['ask_amount'])*mult
                except:continue
                if not (bp>0 and ap>bp and bq>=0 and aq>=0):continue
                k=(t-start)//(STEP_MS*1000)
                if 0<=k<n:
                    a[int(k)]=[bp,bq,ap,aq];rows+=1
    if rows==0:raise RuntimeError(f'empty {ex} {symbol} {date}')
    valid=np.where(np.isfinite(a[:,0]))[0]
    ix=np.full(n,-1,np.int32);ix[valid]=valid;np.maximum.accumulate(ix,out=ix)
    good=ix>=0;age=np.arange(n)-ix
    a[good]=a[ix[good]]
    a[age*STEP_MS>MAX_STALE_MS]=np.nan
    log('V3_DATA',date=date,exchange=ex,symbol=symbol,raw_rows=rows,valid=int(np.isfinite(a[:,0]).sum()))
    return a

def mid(a): return (a[:,0].astype(float)+a[:,2].astype(float))/2

def ret_bp(x,w):
    o=np.full(len(x),np.nan)
    ok=np.isfinite(x[w:])&np.isfinite(x[:-w])&(x[:-w]>0)
    vals=np.full(len(x)-w,np.nan)
    vals[ok]=np.log(x[w:][ok]/x[:-w][ok])*10000
    o[w:]=vals;return o

def halfspread_bp(a):
    m=mid(a);o=np.full(len(m),np.nan)
    ok=np.isfinite(m)&(m>0)&np.isfinite(a[:,0])&np.isfinite(a[:,2])
    o[ok]=(a[ok,2]-a[ok,0])/m[ok]*5000
    return o

def collapse(mask):
    idx=np.flatnonzero(mask)
    if not len(idx):return idx
    cool=max(1,COOLDOWN_MS//STEP_MS);out=[];nxt=-1
    for i in idx:
        if i>=nxt:out.append(int(i));nxt=int(i)+cool
    return np.asarray(out,np.int64)

def tick(p):
    for low,step in ((2e6,1000),(1e6,1000),(5e5,500),(1e5,100),(5e4,50),(1e4,10),(5000,5),(1000,1),(100,1),(10,.1),(1,.01),(.1,.001),(.01,.0001),(.001,.00001),(.0001,.000001),(.00001,.0000001)):
        if p>=low:return step
    return 1e-8

def executable_residual(up,fu,fx,common,d):
    # residual of executable entry premium vs leave-one-out common premium
    out=np.full(len(fx),np.nan)
    if d==1: # buy spot ask / short future bid
        num=up[:,2].astype(float);den=fu[:,0].astype(float)*fx
    else:    # short spot bid / long future ask (research-only)
        num=up[:,0].astype(float);den=fu[:,2].astype(float)*fx
    ok=np.isfinite(num)&np.isfinite(den)&np.isfinite(common)&(num>0)&(den>0)
    out[ok]=np.log(num[ok]/den[ok])*10000-common[ok]
    return out

def est_net_edge(resid_exec,up_hs,fu_hs,d):
    gross=(-resid_exec if d==1 else resid_exec)
    # entry spread is already embedded in executable residual; estimate only closing half-spreads + all 4 fees.
    return gross-(2*UF+2*FF)*10000-up_hs-fu_hs

def taker_pnl(up,fu,fx,i,j,d):
    if j>=len(up):return None
    fxe=float(fx[i]);fxx=float(fx[j])
    if d==1:
        ue=float(up[i,2]);ux=float(up[j,0]);fe=float(fu[i,0]);fz=float(fu[j,2])
        qtys=(float(up[i,3]),float(fu[i,1]),float(up[j,1]),float(fu[j,3]))
    else:
        ue=float(up[i,0]);ux=float(up[j,2]);fe=float(fu[i,2]);fz=float(fu[j,0])
        qtys=(float(up[i,1]),float(fu[i,3]),float(up[j,3]),float(fu[j,1]))
    if not all(math.isfinite(v) and v>0 for v in (ue,ux,fe,fz,fxe,fxx,*qtys)):return None
    q=1/ue
    if d==1:
        pnl=q*(ux*(1-UF)-ue*(1+UF))+q*(fe*(1-FF)-fz*(1+FF))*fxx
    else:
        pnl=q*(ue*(1-UF)-ux*(1+UF))+q*(fz*(1-FF)-fe*(1+FF))*fxx
    cap=min(qtys[0]*ue,qtys[1]*fe*fxe,qtys[2]*ux,qtys[3]*fz*fxx)
    return pnl*10000,cap

def target_exit(up,fu,fx,common,start,d):
    re=executable_residual(up,fu,fx,common,d)
    if not np.isfinite(re[start]):return None,None
    entry=float(re[start]);maxk=min(len(re)-1,start+MAX_TARGET_MS//STEP_MS)
    for j in range(start+1,maxk+1):
        if not np.isfinite(re[j]):continue
        if d==1:
            if re[j]>=-TARGET_RESID_BP:return j,'target'
            if re[j]<=entry-STOP_WORSEN_BP:return j,'stop'
        else:
            if re[j]<=TARGET_RESID_BP:return j,'target'
            if re[j]>=entry+STOP_WORSEN_BP:return j,'stop'
    return maxk,'timeout'

def maker_fill(up,s,d,mode):
    wait=max(1,MAKER_WAIT_MS//STEP_MS);end=min(len(up)-1,s+wait)
    price=float(up[s,0] if d==1 else up[s,2])
    if not math.isfinite(price) or price<=0:return None
    for k in range(s+1,end+1):
        if d==1:
            ask=float(up[k,2])
            if mode=='touch' and math.isfinite(ask) and ask<=price:return k,price
            if mode=='cross' and math.isfinite(ask) and ask<price:return k,price
        else:
            bid=float(up[k,0])
            if mode=='touch' and math.isfinite(bid) and bid>=price:return k,price
            if mode=='cross' and math.isfinite(bid) and bid>price:return k,price
    return None

def maker_pnl(up,fu,fx,fill,maker,j,d):
    i=fill
    if j>=len(up):return None
    fxe=float(fx[i]);fxx=float(fx[j])
    if d==1:
        ux=float(up[j,0]);fe=float(fu[i,0]);fz=float(fu[j,2])
        qtys=(float(up[i,1]),float(fu[i,1]),float(up[j,1]),float(fu[j,3]))
    else:
        ux=float(up[j,2]);fe=float(fu[i,2]);fz=float(fu[j,0])
        qtys=(float(up[i,3]),float(fu[i,3]),float(up[j,3]),float(fu[j,1]))
    if not all(math.isfinite(v) and v>0 for v in (maker,ux,fe,fz,fxe,fxx,*qtys)):return None
    q=1/maker
    if d==1:
        pnl=q*(ux*(1-UF)-maker*(1+UF))+q*(fe*(1-FF)-fz*(1+FF))*fxx
    else:
        pnl=q*(maker*(1-UF)-ux*(1+UF))+q*(fz*(1-FF)-fe*(1+FF))*fxx
    cap=min(qtys[0]*maker,qtys[1]*fe*fxe,qtys[2]*ux,qtys[3]*fz*fxx)
    return pnl*10000,cap

def add(acc,key,pnl,cap,meta=None):
    g=acc[key];g['n']+=1;g['sum']+=pnl;g['win']+=pnl>0;g['pos']+=max(0,pnl);g['neg']+=max(0,-pnl)
    g['vals'].append(pnl);g['caps'].append(cap)
    if meta:
        for k,v in meta.items():g[k]+=v

def summarize(g):
    if not g['n']:return {'n':0}
    a=np.asarray(g['vals'],float)
    return {'n':g['n'],'mean_bp':g['sum']/g['n'],'median_bp':float(np.median(a)),'win_rate':g['win']/g['n'],
            'pf':g['pos']/g['neg'] if g['neg']>0 else None,'sum_bp':g['sum'],
            'p05_bp':float(np.quantile(a,.05)),'p95_bp':float(np.quantile(a,.95)),
            'median_l1_capacity_krw':float(np.median(g['caps'])) if g['caps'] else None,
            'targets':g.get('targets',0),'stops':g.get('stops',0),'timeouts':g.get('timeouts',0),
            'fills':g.get('fills',0),'signals':g.get('signals',0)}

def run_date(date,acc):
    log('V3_DATE_START',date=date)
    fxq=quote_grid('upbit',date,'KRW-USDT');fx=mid(fxq)
    work=ROOT/date;work.mkdir(exist_ok=True)
    prem=[];usable=[]
    for c in COINS:
        try:
            up=quote_grid('upbit',date,'KRW-'+c)
            fs=('1000SHIBUSDT',1000.) if c=='SHIB' else (c+'USDT',1.)
            fu=quote_grid('binance-futures',date,fs[0],fs[1])
            um=mid(up);fm=mid(fu);p=np.full(len(fx),np.nan)
            ok=np.isfinite(um)&np.isfinite(fm)&np.isfinite(fx)&(um>0)&(fm>0)&(fx>0)
            p[ok]=np.log(um[ok]/(fm[ok]*fx[ok]))*10000
            np.savez_compressed(work/(c+'.npz'),up=up,fu=fu,um=um.astype(np.float32),fm=fm.astype(np.float32),prem=p.astype(np.float32))
            prem.append(p);usable.append(c)
        except Exception as e:log('V3_COIN_FAIL',date=date,coin=c,error=str(e)[:220])
    if len(usable)<4:raise RuntimeError('fewer than 4 usable coins')
    P=np.vstack(prem)
    btc_fm=None
    if 'BTC' in usable:
        z=np.load(work/'BTC.npz');btc_fm=z['fm'].astype(float)
    for ci,c in enumerate(usable):
        z=np.load(work/(c+'.npz'));up=z['up'];fu=z['fu'];um=z['um'].astype(float);fm=z['fm'].astype(float);p=z['prem'].astype(float)
        others=np.delete(P,ci,axis=0);common=np.nanmedian(others,axis=0)
        uhs=halfspread_bp(up);fhs=halfspread_bp(fu)
        uimb=(up[:,1]-up[:,3])/(up[:,1]+up[:,3]+1e-12)
        fimb=(fu[:,1]-fu[:,3])/(fu[:,1]+fu[:,3]+1e-12)
        for win_ms in WINDOWS_MS:
            w=max(1,win_ms//STEP_MS);br=ret_bp(fm,w);ur=ret_bp(um,w);lag=br-ur
            # prior BTC 500ms impulse ending one 500ms block before current time
            btclead=np.zeros(len(fm),bool)
            if c!='BTC' and btc_fm is not None:
                bw=max(1,500//STEP_MS);bprev=np.full(len(fm),np.nan)
                if 2*bw<len(fm):
                    ok=np.isfinite(btc_fm[bw:-bw])&np.isfinite(btc_fm[:-2*bw])&(btc_fm[:-2*bw]>0)
                    tmp=np.full(len(fm)-2*bw,np.nan);tmp[ok]=np.log(btc_fm[bw:-bw][ok]/btc_fm[:-2*bw][ok])*10000
                    bprev[2*bw:]=tmp
            for d in (1,-1):
                rex=executable_residual(up,fu,fx,common,d);net=est_net_edge(rex,uhs,fhs,d)
                gross=(-rex if d==1 else rex)
                for pname,shock_th,lag_th,gross_th in PROFILES:
                    base=np.isfinite(br)&np.isfinite(lag)&np.isfinite(gross)&np.isfinite(net)
                    base&=(d*br>=shock_th)&(d*lag>=lag_th)&(gross>=gross_th)
                    for ng in NET_GATES:
                        gate=base&(net>=ng)
                        variants={
                          'V3_extreme':gate,
                          'V3_btc_leader':gate&(np.isfinite(bprev))&(np.sign(bprev)==d)&(d*bprev>=10),
                          'V3_imbalance':gate&(d*uimb>0)&(d*fimb>0),
                        }
                        for vname,mask in variants.items():
                            sig=collapse(mask)
                            for lat in LAT_MS:
                                ls=max(1,int(round(lat/STEP_MS)))
                                for horizon in FIXED_HORIZONS:
                                    hs=max(1,horizon//STEP_MS)
                                    key=(vname,pname,win_ms,d,ng,lat,'fixed',horizon)
                                    acc[key]['signals']+=len(sig)
                                    for s in sig:
                                        i=int(s+ls);j=i+hs
                                        x=taker_pnl(up,fu,fx,i,j,d)
                                        if x:add(acc,key,x[0],x[1])
                                # dynamic convergence exit
                                key=(vname,pname,win_ms,d,ng,lat,'target',MAX_TARGET_MS)
                                acc[key]['signals']+=len(sig)
                                for s in sig:
                                    i=int(s+ls)
                                    if i>=len(up):continue
                                    j,reason=target_exit(up,fu,fx,common,i,d)
                                    if j is None:continue
                                    x=taker_pnl(up,fu,fx,i,j,d)
                                    if x:add(acc,key,x[0],x[1],{reason+'s':1})
                        # V4 maker variants from same extreme event; no artificial latency before posting.
                        sig=collapse(gate)
                        for mode in ('touch','cross'):
                            for horizon in (5000,10000,30000):
                                key=('V4_maker_'+mode,pname,win_ms,d,ng,None,'fixed',horizon)
                                acc[key]['signals']+=len(sig)
                                for s in sig:
                                    f=maker_fill(up,int(s),d,mode)
                                    if not f:continue
                                    fill,mp=f;j=fill+max(1,horizon//STEP_MS)
                                    x=maker_pnl(up,fu,fx,fill,mp,j,d)
                                    if x:add(acc,key,x[0],x[1],{'fills':1})
                            key=('V4_maker_'+mode,pname,win_ms,d,ng,None,'target',MAX_TARGET_MS)
                            acc[key]['signals']+=len(sig)
                            for s in sig:
                                f=maker_fill(up,int(s),d,mode)
                                if not f:continue
                                fill,mp=f;j,reason=target_exit(up,fu,fx,common,fill,d)
                                if j is None:continue
                                x=maker_pnl(up,fu,fx,fill,mp,j,d)
                                if x:add(acc,key,x[0],x[1],{'fills':1,reason+'s':1})
        log('V3_COIN_DONE',date=date,coin=c)
    for pth in work.glob('*.npz'):
        try:pth.unlink()
        except:pass
    try:work.rmdir()
    except:pass
    log('V3_DATE_DONE',date=date,usable=usable)

def main():
    log('V3_BT_START',dates=DATES,coins=COINS,step_ms=STEP_MS,profiles=PROFILES,net_gates_bp=NET_GATES,
        latency_ms=LAT_MS,horizons_ms=FIXED_HORIZONS,fees_bp={'upbit':UF*10000,'binance_futures':FF*10000},
        target_resid_bp=TARGET_RESID_BP,stop_worsen_bp=STOP_WORSEN_BP,max_target_ms=MAX_TARGET_MS,
        variants=['V3_extreme','V3_btc_leader','V3_imbalance','V4_maker_touch','V4_maker_cross'])
    acc=defaultdict(lambda:{'n':0,'sum':0.,'win':0,'pos':0.,'neg':0.,'vals':[],'caps':[],
                            'signals':0,'fills':0,'targets':0,'stops':0,'timeouts':0})
    failures=[]
    for d in DATES:
        try:run_date(d,acc)
        except Exception as e:
            failures.append({'date':d,'error':repr(e)})
            log('V3_DATE_FATAL',date=d,error=repr(e),traceback=traceback.format_exc()[-1800:])
    rows=[]
    for k,g in acc.items():
        v,p,w,d,ng,lat,exitmode,h=k;s=summarize(g)
        rows.append({'variant':v,'profile':p,'window_ms':w,'direction':d,'net_gate_bp':ng,'latency_ms':lat,
                     'exit_mode':exitmode,'horizon_ms':h,**s})
    # feasible direction only, minimum 15 actual trades; show top by mean, then stricter >=30 separately
    feasible=[r for r in rows if r['direction']==1 and r['n']>=15]
    feasible.sort(key=lambda r:r['mean_bp'],reverse=True)
    for r in feasible[:60]:log('V3_RESULT',**r)
    robust=[r for r in rows if r['direction']==1 and r['n']>=30]
    robust.sort(key=lambda r:r['mean_bp'],reverse=True)
    for r in robust[:30]:log('V3_ROBUST',**r)
    research=[r for r in rows if r['direction']==-1 and r['n']>=15]
    research.sort(key=lambda r:r['mean_bp'],reverse=True)
    for r in research[:20]:log('V3_RESEARCH_SHORT',**r)
    summary={'dates_completed':DATES if not failures else [d for d in DATES if d not in {x['date'] for x in failures}],
             'result_cells':len(rows),'failures':failures,'production_pass':False,
             'reason':'Historical L1 screening only; V4 maker fills are touch/cross proxies without queue position. No L2, true fills, liquidation, or untouched future validation.'}
    (ROOT/'phase15_results.json').write_text(json.dumps(clean(rows)))
    log('V3_BT_DONE',**summary)

if __name__=='__main__':
    try:main()
    except Exception as e:
        log('V3_BT_FATAL',error=repr(e),traceback=traceback.format_exc()[-3000:],production_pass=False)
        raise
