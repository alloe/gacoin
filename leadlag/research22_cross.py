"""Historical-cohort cross-sectional study. Hourly futures execution proxy."""
from __future__ import annotations
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,pandas as pd
from research_core import *
START='2024-12-01';END='2026-09-01';EVAL='2025-01-01';FINAL='2026-01-01';H=3600000

def catalog():
 pre='data/futures/um/monthly/klines/';out=[];marker=None
 for _ in range(5):
  q={'delimiter':'/','prefix':pre,'max-keys':1000}
  if marker:q['marker']=marker
  root=ET.fromstring(fetch('https://s3-ap-northeast-1.amazonaws.com/data.binance.vision',q))
  tags=lambda name:[e.text for e in root.iter() if e.tag.split('}')[-1]==name]
  names=[x for x in tags('Prefix') if x!=pre];out+=names
  if tags('IsTruncated')==['false']:break
  nxt=tags('NextMarker');marker=nxt[0] if nxt else (names[-1] if names else None)
  if not marker:raise ValueError('catalog pagination lacks marker')
 else:raise ValueError('catalog pagination budget exhausted')
 return sorted({x.rstrip('/').split('/')[-1] for x in out if x.rstrip('/').endswith('USDT') and '_' not in x.rstrip('/').split('/')[-1]})

def formation(symbol):
 try:
  d=klines_month(symbol,'1d','2024-12',optional=True)
  if d is None or len(d)<28:return None
  if symbol in {'BTCDOMUSDT','USDCUSDT','PAXGUSDT','XAUTUSDT','DEFIUSDT','BLUEBIRDUSDT','FOOTBALLUSDT'}:return None
  return symbol,float(d.quote_volume.sum())
 except Exception as e:return ('ERROR:'+symbol,str(e))

def neutral_weights(score,beta,k):
 ids=np.flatnonzero(np.isfinite(score)&np.isfinite(beta));w=np.zeros(len(score))
 if len(ids)<2*k+2:return w
 ix=ids[np.argsort(score[ids],kind='stable')];sel=np.r_[ix[:k],ix[-k:]];v=np.r_[-np.ones(k),np.ones(k)]/k
 A=np.vstack([np.ones(2*k),beta[sel]]);v=v-A.T@np.linalg.pinv(A@A.T)@(A@v);g=np.abs(v).sum()
 if g<1e-9:return w
 w[sel]=.5*v/g
 return w

def simulate(op,cl,fu,valid,features,liquidity,beta,grid,top,lookback,reb,k,kind):
 n,m=op.shape;qty=np.zeros(m);marks=np.zeros(m);pnl=0.;turn=0.;fund=0.;fees=0.;equity=[];unresolved=0;fund_missing=0;missing_bars=0;signals=0;trade_count=0;beta_ex=[];gross_ex=[];net_ex=[];coin_pnl=np.zeros(m)
 start=max(int(np.searchsorted(grid,stamp(EVAL))),338)
 for e in range(start,n-1):
  px=op[e];good=np.isfinite(px)&(px>0);held=np.abs(qty)>1e-14
  if np.any(held&~good):
   missing_bars+=int(np.sum(held&~good));unresolved+=1;break
  d=np.where(held,qty*(np.where(good,px,marks)-marks),0.);pnl+=d.sum();coin_pnl+=d
  fp=fu[e]
  if np.any(held&~np.isfinite(fp)):fund_missing+=1;break
  fc=-np.nansum(qty*px*fp);fund+=fc;pnl+=fc;marks=np.where(good,px,marks)
  if (grid[e]//H)%reb==0:
   s=e-2;eligible=valid[s]&np.isfinite(liquidity[s])&np.isfinite(features[lookback][s])&np.isfinite(beta[s])&good
   cand=np.flatnonzero(eligible);rank=cand[np.argsort(liquidity[s,cand])[-top:]] if len(cand) else cand
   score=np.full(m,np.nan);score[rank]=features[lookback][s,rank]*(1 if kind=='momentum' else -1)
   w=neutral_weights(score,beta[s],k);newq=np.divide(w,px,out=np.zeros(m),where=good)
   delta=newq-qty;nt=np.sum(np.abs(delta[good])*px[good]);turn+=nt;fees+=nt*.0005
   trade_count+=int(np.sum(np.abs(delta)>1e-12));signals+=1;qty=newq
   beta_ex.append(float(np.nansum(w*beta[s])));net_ex.append(float(w.sum()));gross_ex.append(float(np.abs(w).sum()))
  equity.append((int(grid[e]),1+pnl-fees,turn,fund))
  if 1+pnl-fees<=0:unresolved+=1;break
 if len(equity) and not unresolved and not fund_missing:
  e=min(e+1,n-1);px=op[e];held=np.abs(qty)>1e-14
  if np.any(held&~np.isfinite(px)):unresolved+=1
  else:
   dd=np.nansum(qty*(px-marks));pnl+=dd
   if np.any(held&~np.isfinite(fu[e])):fund_missing+=1
   else:fc=-np.nansum(qty*px*fu[e]);pnl+=fc;fund+=fc
   nt=np.nansum(np.abs(qty)*px);turn+=nt;fees+=nt*.0005;equity.append((int(grid[e]),1+pnl-fees,turn,fund))
 rows=[]
 for slip in (0,2,5):
  ts=[stamp(EVAL)];eq=[1.]
  for t,v,tr,fd in equity:ts.append(t);eq.append(v-tr*slip/10000)
  allstat=equity_stats(ts,eq,turn,fund,fees+turn*slip/10000);seg={}
  for name,a,b in [('train','2025-01-01','2025-07-01'),('validation','2025-07-01','2026-01-01'),('final','2026-01-01','2026-09-01')]:
   ii=[j for j,t in enumerate(ts) if stamp(a)<=t<stamp(b)]
   if ii:
    lo=max(0,ii[0]-1);hi=ii[-1]+1;seg[name]=equity_stats(ts[lo:hi],eq[lo:hi])
  rows.append({'top':top,'lookback_h':lookback,'rebalance_h':reb,'tail_n':k,'kind':kind,'slip_bp_per_turnover':slip,'metrics':allstat,'segments':seg,'unresolved_price_events':unresolved,'missing_held_bars':missing_bars,'missing_funding_events':fund_missing,'primary_eligible':not (unresolved or fund_missing),'rebalances':signals,'order_legs':trade_count,'max_abs_rebalance_beta':max(map(abs,beta_ex),default=None),'max_abs_rebalance_net':max(map(abs,net_ex),default=None),'mean_gross_initial_equity':np.mean(gross_ex) if gross_ex else 0,'note':'Initial-equity gross0.5; noncompounding fixed risk budget; hourly opens/marks, no L1 fills or liquidation engine.'})
 return rows

def run():
 log('P22_START',formation_month='2024-12',evaluation=[EVAL,END],final=[FINAL,END],primary={'top':30,'lookback':12,'rebalance':4,'tails':5,'gross_initial_equity':.5,'kind':'momentum','fee_bp':5,'slip_bp':2})
 symbols=catalog();log('P22_CATALOG',symbols=len(symbols));form=[];errors=[]
 with ThreadPoolExecutor(max_workers=5) as pool:
  futures={pool.submit(formation,s):s for s in symbols}
  for i,f in enumerate(as_completed(futures)):
   x=f.result()
   if x and x[0].startswith('ERROR:'):errors.append(x)
   elif x:form.append(x)
   if (i+1)%200==0:log('P22_FORMATION_PROGRESS',done=i+1,total=len(symbols),eligible=len(form))
 form.sort(key=lambda a:(-a[1],a[0]));cohort=[s for s,v in form[:60]]
 if 'BTCUSDT' not in cohort or len(cohort)<50:raise RuntimeError('historical cohort insufficient')
 log('P22_COHORT',symbols=cohort,formation_eligible=len(form),errors=errors[:5])
 grid=np.arange(stamp(START),stamp(END),H);n=len(grid);m=len(cohort);op=np.full((n,m),np.nan);cl=op.copy();liq=op.copy();fu=np.zeros((n,m));missing={}
 def data(s):return s,load_klines(s,'1h',START,END),funding_history(s,START,END)
 with ThreadPoolExecutor(max_workers=4) as pool:
  for j,(sym,(d,km),(fr,fm)) in enumerate(pool.map(data,cohort)):
   c=cohort.index(sym);d=d.reindex(grid);op[:,c]=d.open;cl[:,c]=d.close;liq[:,c]=d.quote_volume
   for mm in fm:
    mask=np.array([iso(t)[:7]==mm for t in grid]);fu[mask,c]=np.nan
   for t,r in fr.rate.items():
    e=int(np.ceil((t-stamp(START))/H))
    if 0<=e<n and np.isfinite(fu[e,c]):fu[e,c]+=r
   missing[sym]={'kline_months':km,'funding_months':fm}
   if (j+1)%10==0:log('P22_DATA_PROGRESS',done=j+1,total=m)
 logs=np.log(cl);returns=pd.DataFrame(logs).diff();btc=cohort.index('BTCUSDT');rb=returns[btc]
 beta=np.column_stack([returns[c].rolling(336,min_periods=300).cov(rb).shift(1)/rb.rolling(336,min_periods=300).var().shift(1) for c in range(m)])
 vol=returns.rolling(168,min_periods=150).std().shift(1).to_numpy();res=returns.to_numpy()-beta*rb.to_numpy()[:,None]
 feats={w:pd.DataFrame(res).rolling(w,min_periods=w).sum().to_numpy()/np.maximum(vol,.0001) for w in (4,12,24)}
 rankliq=pd.DataFrame(liq).rolling(168,min_periods=168).sum().to_numpy();valid=pd.DataFrame(np.isfinite(op)).rolling(336,min_periods=336).mean().to_numpy()>=.98
 configs=[(30,12,4,5,'momentum'),(20,12,4,3,'momentum'),(50,12,4,5,'momentum'),(30,4,4,5,'momentum'),(30,24,8,5,'momentum'),(30,12,8,5,'momentum'),(30,12,4,5,'reversal')];rows=[]
 for cfg in configs:
  z=simulate(op,cl,fu,valid,feats,rankliq,beta,grid,*cfg);rows.extend(z)
  for r in z:log('P22_RESULT',**r)
 report={'strategy':'P22_cross_section','cohort':cohort,'historical_cohort_not_all_future_listings':True,'formation_eligible':len(form),'missing':missing,'results':rows,'production_pass':False,'limitations':['Historical fixed formation cohort; excludes later entrants.','Hourly OHLC execution proxy; funding mark uses hourly open proxy.','All unresolved price/funding events invalidate headline NAV rather than quietly dropping positions.','No historical order rules, L1/L2 fills, exchange margin/liquidation, taxes or KRW conversion.']}
 write_report('phase22',report);emit_bundle('phase22',report);log('P22_DONE',cells=len(rows),symbols=m,production_pass=False)
 return report
