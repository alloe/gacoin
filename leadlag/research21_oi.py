"""Five-minute OI-unit/flow study. NOT observed liquidation tape."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd
from research_core import *
START='2026-01-01';END='2026-09-01';B=300000;COINS=['BTC','ETH','SOL','XRP','DOGE','ADA','SUI','LINK']

def load_oi(symbol):
 days=pd.date_range(START,pd.Timestamp(END)-pd.Timedelta(days=1),freq='D');parts=[];miss=[]
 def one(day):
  date=day.strftime('%Y-%m-%d');p=f'data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{date}.zip';return date,archive(p,optional=True)
 with ThreadPoolExecutor(max_workers=4) as pool:
  for date,d in pool.map(one,days):
   if d is None:miss.append(date);continue
   if not {'create_time','sum_open_interest'}.issubset(d):raise ValueError('OI schema changed')
   t=pd.to_datetime(d.create_time,utc=True).astype('int64')//1000000
   parts.append(pd.DataFrame({'available':t+B,'oi':pd.to_numeric(d.sum_open_interest)}))
 if not parts:raise RuntimeError('no OI archives')
 return pd.concat(parts).drop_duplicates('available').sort_values('available'),miss

def candidate_masks(d):
 r15=(np.log(d.close)-np.log(d.close.shift(3)))*10000
 sigma=np.log(d.close).diff().rolling(288,min_periods=250).std().shift(1)*np.sqrt(3)*10000
 th=np.maximum(50,3*sigma);surge=d.quote_volume/d.quote_volume.rolling(288,min_periods=250).median().shift(1);dc=d.oi/d.oi.shift(3)-1
 flow=d.taker_buy_quote_volume/d.quote_volume.replace(0,np.nan);sign=np.sign(r15);shock=(np.abs(r15)>=th)&(surge>=2);nextret=d.close.shift(-1)/d.close-1
 reversal_confirm=(sign*nextret<0)&((sign>0)&(flow.shift(-1)<flow)|(sign<0)&(flow.shift(-1)>flow))
 continuation_confirm=(sign*nextret>0)&(((sign>0)&(flow.shift(-1)>.60))|((sign<0)&(flow.shift(-1)<.40)))
 return {'reversal_price_flow':shock&reversal_confirm,'reversal_oi_drop':shock&reversal_confirm&(dc<=-.01),'continuation_price_flow':shock&continuation_confirm,'continuation_oi_rise':shock&continuation_confirm&(dc>=.01)},sign,r15,dc

def make_trades(coin,d,fr):
 masks,sign,r15,dc=candidate_masks(d);out=[];ft=fr.index.to_numpy(dtype='int64');rates=fr.rate.to_numpy(float)
 for variant,mask in masks.items():
  for hold in (15,30,60,240):
   next_free=-1
   for j in np.flatnonzero(mask.fillna(False).to_numpy()):
    e=j+3;x=e+hold//5
    if e<next_free or x>=len(d):continue
    next_free=x+1;side=int(-sign.iloc[j] if variant.startswith('reversal') else sign.iloc[j]);entry=d.open.iloc[e];exit=d.open.iloc[x]
    item={'coin':coin,'variant':variant,'hold_min':hold,'signal_time':int(d.index[j]+B),'entry_time':int(d.index[e]),'exit_time':int(d.index[x]),'side':side,'shock_bp':r15.iloc[j],'oi_change':dc.iloc[j],'day':iso(d.index[e])[:10]}
    if not (np.isfinite(entry) and np.isfinite(exit)) or d.open.iloc[e:x+1].isna().any():
     item.update(status='unresolved_price_gap');out.append(item);continue
    ix=np.flatnonzero((ft>d.index[e])&(ft<=d.index[x]));fundbp=0.;fundbad=False
    for a in ix:
     z=np.searchsorted(d.index.to_numpy()+B,ft[a],side='right')-1
     if z<0 or not np.isfinite(d.close.iloc[z]):fundbad=True;break
     fundbp+=rates[a]*d.close.iloc[z]/entry*10000
    if fundbad:item.update(status='unresolved_funding_mark');out.append(item);continue
    item.update(status='evaluated',gross_bp=side*(exit/entry-1)*10000,funding_bp=-side*fundbp,net_bp={str(s):net_trade(side,entry,exit,5,s,fundbp) for s in (0,2,5)})
    out.append(item)
 return out

def run():
 log('P21_START',period=[START,END],coins=COINS,oi_unit='contracts_not_USD_value',oi_publication_lag_ms=B,forced_liquidation_tape=False,primary={'variant':'reversal_oi_drop','hold_min':60,'slip_bp_each':2},execution='shock j, confirmation j+1, entry j+3 open')
 allrows=[];coverage={};fail=[]
 for coin in COINS:
  try:
   d,missing=load_klines(coin+'USDT','5m',START,END);grid=np.arange(stamp(START),stamp(END),B);d=d.reindex(grid)
   oi,omiss=load_oi(coin+'USDT');left=pd.DataFrame({'available':grid+B});merged=pd.merge_asof(left,oi,on='available',direction='backward',tolerance=2*B);d['oi']=merged.oi.to_numpy()
   fr,fm=funding_history(coin+'USDT',START,END)
   coverage[coin]={'bars':len(d),'valid_price_bars':int(d.open.notna().sum()),'valid_oi_bars':int(d.oi.notna().sum()),'missing_oi_days':omiss,'missing_price_months':missing,'missing_funding_months':fm}
   if fm:raise RuntimeError('funding archive missing; refusing to treat as zero')
   rows=make_trades(coin,d,fr);allrows+=rows;log('P21_COIN_DONE',coin=coin,records=len(rows),valid_oi=int(d.oi.notna().sum()),missing_oi_days=len(omiss))
  except Exception as e:fail.append({'coin':coin,'error':repr(e)});log('P21_COIN_FAIL',coin=coin,error=repr(e))
 results=[]
 for variant in ('reversal_price_flow','reversal_oi_drop','continuation_price_flow','continuation_oi_rise'):
  for hold in (15,30,60,240):
   for segment,a,b in [('train','2026-01-01','2026-05-01'),('validation','2026-05-01','2026-07-01'),('final','2026-07-01','2026-09-01')]:
    z=[x for x in allrows if x['variant']==variant and x['hold_min']==hold and stamp(a)<=x['entry_time']<stamp(b)];good=[x for x in z if x['status']=='evaluated']
    for slip in (0,2,5):
     vals=[x['net_bp'][str(slip)] for x in good];r={'variant':variant,'hold_min':hold,'segment':segment,'slip_bp_each':slip,'attempts':len(z),'unresolved':len(z)-len(good),'independent_dates':len({x['day'] for x in good}),**trade_stats(vals),'mean_ci95_day_cluster':cluster_ci(vals,[x['day'] for x in good]),'by_coin':{c:trade_stats([x['net_bp'][str(slip)] for x in good if x['coin']==c]) for c in COINS}}
     results.append(r)
     if segment=='final' and slip==2:log('P21_RESULT',**r)
 report={'strategy':'P21_oi_regimes','liquidation_tape_available':False,'records':allrows,'coverage':coverage,'failures':fail,'results':results,'production_pass':False,'limitations':['OI decrease does not prove forced liquidation; OI increase does not identify new longs versus shorts.','Eight fixed liquid coins, not unrestricted all-market universe.','5m publication delay assumed for OI; actual historical release latency unknown.','OHLC execution and funding mark proxies; no account allocation/liquidation/borrow modeling.']}
 write_report('phase21',report);compact={k:v for k,v in report.items() if k!='records'};compact['record_examples']=allrows[:20]
 emit_bundle('phase21',compact);log('P21_DONE',records=len(allrows),result_cells=len(results),failed_coins=len(fail),production_pass=False)
 return compact
