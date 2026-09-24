"""Precisely timed news supplement; manual frozen judgments, not historical LLM OOS."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np,pandas as pd
from research_core import *
from research20_news import series_for,px

def action(p,threshold):
 if p>=threshold:return 1
 if p<=1-threshold:return -1
 return 0

def funding_component(d,e,x,entry,fr,bar=60000):
 cash=0.
 for t,rate in fr.rate.items():
  if not e<t<=x:continue
  j=np.searchsorted(d.index.to_numpy()+bar,t,side='right')-1
  if j<0 or not np.isfinite(d.close.iloc[j]):return None
  cash+=rate*float(d.close.iloc[j])/entry*10000
 return cash

def run():
 path=Path(__file__).with_name('newswire_labels_frozen.json');raw=path.read_bytes();manifest=json.loads(raw);sha=hashlib.sha256(raw).hexdigest()
 for ev in manifest['events']:
  ts=pd.Timestamp(ev['published_at']);assert ts.utcoffset()==ts.tz_convert('America/New_York').utcoffset()
  assert 0<=ev['p_up_60m']<=1 and 0<=ev['p_up_24h']<=1
 log('P20B_FROZEN',sha256=sha,events=len(manifest['events']),labels=manifest['events'],horizon=60,delay=1)
 rows=[];unresolved=[]
 for ev in manifest['events']:
  t=stamp(ev['published_at']);base=((t+60000-1)//60000)*60000
  for coin in ev['symbols']:
   try:
    sym=coin+'USDT';d=series_for(sym,t,'minute');btc=series_for('BTCUSDT',t,'minute');fr,missing=funding_history(sym,iso(base-86400000),iso(base+2*86400000))
    if missing:raise ValueError('funding data incomplete')
    for delay in (1,5):
     e=base+delay*60000;ep=px(d,e)
     for h in (5,15,60,240,1440):
      x=e+h*60000;xp=px(d,x);r={'id':ev['id'],'issuer':ev['issuer'],'coin':coin,'published_at':ev['published_at'],'entry_time':e,'exit_time':x,'delay_min':delay,'horizon_min':h,'semantic':ev['semantic'],'p_up':ev['p_up_60m'] if h!=1440 else ev['p_up_24h']}
      if ep is None or xp is None:r['status']='unresolved_price';rows.append(r);continue
      z=d.loc[(d.index>=e)&(d.index<=x)]
      if len(z)!=(h+1):r['status']='unresolved_path_gaps';rows.append(r);continue
      fd=funding_component(d,e,x,ep,fr)
      if fd is None:r['status']='unresolved_funding_mark';rows.append(r);continue
      rr=(xp/ep-1)*10000;bp=px(btc,e);bx=px(btc,x);pre=px(d,base-3600000);p0=px(d,base);p=r['p_up']
      r.update(status='evaluated',return_bp=rr,benchmark_bp=((bx/bp-1)*10000) if bp and bx else None,abnormal_bp=((xp/ep-bx/bp)*10000) if bp and bx and coin!='BTC' else None,pre60_bp=((p0/pre-1)*10000) if p0 and pre else None,brier=(p-int(rr>0))**2,probability_direction_hit=(int((p>.5)==(rr>0)) if p!=.5 else None),naive_p=.65 if ev['keyword_direction']==1 else .35)
      r['keyword_brier']=(r['naive_p']-int(rr>0))**2
      for th in (.65,.55):
       a=action(p,th);r['action_'+str(th)]=a
       for slip in (0,2,5):r[f'net_{th}_{slip}']=net_trade(a,ep,xp,5,slip,fd) if a else 0.
      r['keyword_net_2']=net_trade(ev['keyword_direction'],ep,xp,5,2,fd);r['always_long_net_2']=net_trade(1,ep,xp,5,2,fd);rows.append(r)
   except Exception as e:unresolved.append({'id':ev['id'],'coin':coin,'reason':repr(e)})
  log('P20B_EVENT_DONE',id=ev['id'])
 metrics=[]
 for delay in (1,5):
  for h in (5,15,60,240,1440):
   sub=[r for r in rows if r['delay_min']==delay and r['horizon_min']==h];good=[r for r in sub if r['status']=='evaluated'];df=pd.DataFrame(good)
   if len(df)==0:metrics.append({'delay_min':delay,'horizon_min':h,'n':0});continue
   numeric=['return_bp','brier','keyword_brier','probability_direction_hit','keyword_net_2','always_long_net_2']+[f'net_{th}_{s}' for th in (.65,.55) for s in (0,2,5)]
   g=df.groupby('id')[numeric].mean();clusters=[]
   for eid,z in df.groupby('id'):
    clusters.append({'id':eid,'direction_hit':z.probability_direction_hit.mean(),'p':z.p_up.mean(),'raw_bp':z.return_bp.mean(),'trade65':bool((z['action_0.65']!=0).any()),'trade55':bool((z['action_0.55']!=0).any()),'n_assets':len(z)})
   r={'delay_min':delay,'horizon_min':h,'documents':len(g),'asset_rows':len(df),'unresolved':len(sub)-len(good),'brier':g.brier.mean(),'brier_baseline_half':.25,'brier_keyword_65pct':g.keyword_brier.mean(),'direction_accuracy_excluding_half':g.probability_direction_hit.mean(),'directional_documents':int(g.probability_direction_hit.notna().sum()),'always_long_accuracy':df.assign(y=df.return_bp>0).groupby('id').y.mean().mean(),'coverage65':np.mean([c['trade65'] for c in clusters]),'coverage55':np.mean([c['trade55'] for c in clusters]),'net_event_mean_bp':{str(th):{str(s):g[f'net_{th}_{s}'].mean() for s in (0,2,5)} for th in (.65,.55)},'keyword_net_event_mean_bp':g.keyword_net_2.mean(),'always_long_net_event_mean_bp':g.always_long_net_2.mean(),'net55_ci95_event_cluster':cluster_ci(g['net_0.55_2'],g.index),'clusters':clusters}
   metrics.append(r)
   if h in (60,1440):log('P20B_RESULT',**r)
 for r in rows:
  if r['delay_min']==1 and r['horizon_min']==60:log('P20B_CASE',**r)
 report={'manifest':manifest,'manifest_sha256':sha,'rows':rows,'metrics':metrics,'unresolved_sources':unresolved,'production_pass':False,'limitations':['Only 12 issuer-selected releases; not representative full archive.','Exact publisher ET clock is not historical wire receipt latency.','Frozen before minute-price outcomes; latent model memory and updated-body leakage remain possible.','Probability 0.5 is abstention, not a down prediction.','Coin price response does not prove news causation.','No historical L1 or account margin; per-trade basis only.']}
 write_report('phase20b',report);emit_bundle('phase20b',report);log('P20B_DONE',documents=len(manifest['events']),rows=len(rows),source_failures=len(unresolved),production_pass=False)
 return report
if __name__=='__main__':run()
