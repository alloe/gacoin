"""Frozen judgments -> retrospective study; NOT historical point-in-time LLM validation."""
from __future__ import annotations
import hashlib,json,re,html
from pathlib import Path
from html.parser import HTMLParser
from functools import lru_cache
import numpy as np,pandas as pd
from research_core import *
class Scripts(HTMLParser):
 def __init__(self):super().__init__();self.on=False;self.cur='';self.blocks=[]
 def handle_starttag(self,tag,attrs):
  if tag=='script':self.on=True;self.cur=''
 def handle_data(self,data):
  if self.on:self.cur+=data
 def handle_endtag(self,tag):
  if tag=='script' and self.on:self.blocks.append(self.cur);self.on=False

def epoch(v):
 try:
  if isinstance(v,(float,int)) or str(v).isdigit():
   n=int(v);return n*1000 if 1e9<n<1e11 else (n if 1e12<n<1e14 else None)
  if 'T' in str(v) and (str(v).endswith('Z') or re.search(r'[+-]\d\d:\d\d$',str(v))):return stamp(v)
 except Exception:return None
 return None

def resolve_time(ev):
 if ev['time_precision']!='requires_machine_timestamp':return None,{'status':ev['time_precision']}
 try:
  raw=fetch(ev['url']);p=Scripts();p.feed(raw.decode('utf-8','replace'));found=[];target=ev['id'];norm=lambda s:re.sub(r'[^a-z0-9]','',s.lower());expected=norm(ev['title'])
  def walk(o):
   if isinstance(o,dict):
    match=any(str(o.get(k,''))==target for k in ('code','articleCode','id'));title=str(o.get('title',o.get('headline','')))
    match=match or (bool(title) and len(norm(title))>20 and norm(title)==expected)
    if match:
     for key in ('releaseDate','publishTime','publishedAt','datePublished'):
      v=epoch(o.get(key))
      if v is not None and abs(stamp(ev['published_date'])-v)<2*86400000:found.append((v,key))
    for v in o.values():walk(v)
   elif isinstance(o,list):
    for v in o:walk(v)
  for block in p.blocks:
   try:walk(json.loads(html.unescape(block)))
   except (ValueError,TypeError):continue
  unique=sorted({x[0] for x in found});audit={'status':'verified_machine_epoch' if len(unique)==1 else 'no_unique_title_matched_epoch','candidates':found,'html_sha256':hashlib.sha256(raw).hexdigest()}
  return (unique[0] if len(unique)==1 else None),audit
 except Exception as e:return None,{'status':'source_fetch_failed','error':str(e)[:180]}

@lru_cache(maxsize=80)
def minutes_day(symbol,date):
 d=archive(f'data/futures/um/daily/klines/{symbol}/1m/{symbol}-1m-{date}.zip',optional=True)
 if d is None:return None
 if d.shape[1]!=12:raise ValueError('minute schema unexpected')
 d.columns=KCOL
 for c in KCOL:d[c]=pd.to_numeric(d[c],errors='coerce')
 if d.t.median()>1e14:d.t=d.t//1000
 return d.dropna(subset=['t','open']).assign(t=lambda x:x.t.astype('int64')).drop_duplicates('t').set_index('t').sort_index()
@lru_cache(maxsize=60)
def hourly(symbol,month):return klines_month(symbol,'1h',month,optional=True)
def series_for(symbol,t,mode):
 if mode=='minute':
  dates=pd.date_range(pd.Timestamp(t-7200000,unit='ms',tz='UTC').floor('D'),pd.Timestamp(t+26*3600000,unit='ms',tz='UTC').floor('D'),freq='D');parts=[minutes_day(symbol,x.strftime('%Y-%m-%d')) for x in dates]
 else:
  a=iso(t-3*86400000);b=iso(t+8*86400000);parts=[hourly(symbol,m) for m in monthly_range(a,b)]
 parts=[x for x in parts if x is not None]
 if not parts:return None
 z=pd.concat(parts).sort_index();return z[~z.index.duplicated()]
def px(d,t):
 if d is None or t not in d.index:return None
 v=float(d.loc[t,'open']);return v if np.isfinite(v) and v>0 else None

def causal_pullback(d,t):
 start=((int(t)+60000-1)//60000)*60000;grid=np.arange(start,start+91*60000,60000);z=d.reindex(grid)
 if z.iloc[:5].close.isna().any():return {'status':'unresolved_observation'}
 peak=float(z.high.iloc[:5].max());trigger=None;entry=None
 for i in range(5,30):
  if not np.isfinite(z.close.iloc[i]):return {'status':'unresolved_observation'}
  peak=max(peak,float(z.high.iloc[i]))
  if trigger is None and z.close.iloc[i]<=peak*.97:trigger=i
  if trigger is not None and i>trigger and z.close.iloc[i]>z.close.iloc[i-1]:entry=i+1;break
 if entry is None:return {'status':'no_pullback_entry'}
 ep=px(d,int(grid[entry]))
 if ep is None:return {'status':'unresolved_entry'}
 stop=ep*.97;target=ep*1.04
 for j in range(entry,min(entry+60,len(z))):
  row=z.iloc[j]
  if row[['open','high','low']].isna().any():return {'status':'unresolved_after_entry','entry_time':int(grid[entry])}
  if row.open<=stop:xp=float(row.open);why='gap_stop';break
  if row.low<=stop:xp=stop;why='stop_first_if_both';break
  if row.high>=target:xp=target;why='target';break
 else:
  j=min(entry+60,len(z)-1);xp=px(d,int(grid[j]));why='timeout'
 if xp is None:return {'status':'unresolved_exit'}
 return {'status':'evaluated','entry_time':int(grid[entry]),'exit_time':int(grid[j]),'exit_reason':why,'entry':ep,'exit':xp,'net_bp_2':net_trade(1,ep,xp,5,2),'funding_included':False,'note':'Intrabar stop/target path unknown; stop first conservative. No orderbook or queue model.'}

def run():
 path=Path(__file__).with_name('news_labels_frozen.json');raw=path.read_bytes();manifest=json.loads(raw);events=manifest['events'];sha=hashlib.sha256(raw).hexdigest()
 log('P20_LABELS_LOCKED',sha256=sha,events=len(events),labels=events,prices_not_yet_queried_by_this_job=True)
 audit={};resolved={}
 for ev in events:
  t,a=resolve_time(ev);audit[ev['id']]=a;resolved[ev['id']]=t
 log('P20_PUBLICATION_AUDIT',verified=sum(t is not None for t in resolved.values()),total=len(events),audit=audit)
 rows=[];skips=[];pullbacks=[]
 for ev in events:
  t=resolved[ev['id']];precise=t is not None
  if t is None:t=stamp(ev['published_date'])+2*86400000
  probability=ev['p_up_60m'] if precise else ev['p_up_delayed_24h'];action=1 if probability>=.65 else (-1 if probability<=.35 else 0)
  mode='minute' if precise else 'hour';bar=60000 if precise else 3600000;base=((t+bar-1)//bar)*bar
  horizons=[5,15,60,240,1440] if precise else [1440,4320,10080]
  ratable=not ev.get('prior_outcome_memory',False) and not ev.get('revised',False) and ev['time_precision']!='search_index_only'
  for coin in ev['symbols']:
   sym=coin+'USDT'
   try:
    d=series_for(sym,t,mode);btc=series_for('BTCUSDT',t,mode)
    if px(d,base) is None:skips.append({'id':ev['id'],'coin':coin,'reason':'instrument_not_available_at_evaluation_start'});continue
    fstart=iso(base-86400000);fend=iso(base+8*86400000);fr,fmissing=funding_history(sym,fstart,fend)
    for delay in ([1,5] if precise else [0]):
     e=base+delay*60000;ep=px(d,e)
     if ep is None:skips.append({'id':ev['id'],'coin':coin,'reason':'entry_bar_missing','delay':delay});continue
     for horizon in horizons:
      x=e+horizon*60000;xp=px(d,x);bp=px(btc,e);bpx=px(btc,x)
      item={'id':ev['id'],'cluster':ev['cluster'],'coin':coin,'semantic':ev['semantic'],'action':action,'kind':ev['kind'],'forecast':probability,'naive_action':ev['naive_action'],'time_quality':'machine_epoch' if precise else 'date_only_delayed_48h','evaluation_start':iso(e),'horizon_min':horizon,'delay_min':delay,'rater_eligible':ratable,'prior_outcome_memory':ev.get('prior_outcome_memory',False)}
      if xp is None or bp is None or bpx is None:item['status']='unresolved_exit_or_benchmark';rows.append(item);continue
      interval=d.loc[(d.index>=e)&(d.index<=x)];expect=(x-e)//bar+1
      if len(interval)<expect*.98:item['status']='unresolved_price_gaps';rows.append(item);continue
      fundbp=0.;fundbad=bool(fmissing)
      for ft,rate in fr.rate.items():
       if not e<ft<=x:continue
       j=np.searchsorted(d.index.to_numpy()+bar,ft,side='right')-1
       if j<0 or not np.isfinite(d.close.iloc[j]):fundbad=True;break
       fundbp+=rate*d.close.iloc[j]/ep*10000
      if fundbad:item['status']='unresolved_funding';rows.append(item);continue
      r=(xp/ep-1)*10000;br=(bpx/bp-1)*10000;pre=px(d,base-(60 if precise else 1440)*60000);act=action
      item.update(status='evaluated',raw_bp=r,btc_bp=br,abnormal_bp=r-br,pre_drift_bp=(px(d,base)/pre-1)*10000 if pre else None,net_bp={str(s):(net_trade(act,ep,xp,5,s,fundbp) if act else 0.) for s in (0,2,5)},naive_net_bp=net_trade(ev['naive_action'],ep,xp,5,2,fundbp),always_long_net_bp=net_trade(1,ep,xp,5,2,fundbp));rows.append(item)
    if precise and ev['action']==1:
     pb=causal_pullback(d,t);pb.update(id=ev['id'],coin=coin,rater_eligible=ratable);pullbacks.append(pb)
   except Exception as e:skips.append({'id':ev['id'],'coin':coin,'reason':repr(e)[:180]})
  log('P20_EVENT_DONE',id=ev['id'],precise=precise)
 metrics=[]
 for quality,horizon,delay in [('machine_epoch',60,1),('machine_epoch',60,5),('date_only_delayed_48h',1440,0)]:
  sel=[x for x in rows if x['time_quality']==quality and x['horizon_min']==horizon and x['delay_min']==delay and x['rater_eligible']];good=[x for x in sel if x['status']=='evaluated'];x=pd.DataFrame(good)
  if len(x):
   event_groups=[]
   for cluster,z in x.groupby('cluster'):
    y=(z.raw_bp>0).to_numpy();p=z.forecast.to_numpy();g={'cluster':cluster,'brier':np.mean((p-y)**2),'always_long_hit':np.mean(y),'hard_hit':np.mean((p>.5)==y),'abnormal_hit':np.mean((p>.5)==(z.abnormal_bp>0)),'n_assets':len(z),'net_bp_2':np.mean([r['2'] for r in z.net_bp]),'naive_net_bp':z.naive_net_bp.mean(),'trade':bool((z.action!=0).any()),'direction':float(z.forecast.mean()),'return_bp':z.raw_bp.mean()};event_groups.append(g)
   g=pd.DataFrame(event_groups);directional=g[abs(g.direction-.5)>=.15]
   r={'time_quality':quality,'horizon_min':horizon,'delay_min':delay,'attempted_asset_rows':len(sel),'unresolved_asset_rows':len(sel)-len(good),'event_clusters':len(g),'asset_rows':len(x),'brier':g.brier.mean(),'brier_random_half':.25,'hard_direction_accuracy':g.hard_hit.mean(),'abnormal_direction_accuracy':g.abnormal_hit.mean(),'always_long_accuracy':g.always_long_hit.mean(),'trade_coverage':g.trade.mean(),'event_mean_net_bp_2':g.net_bp_2.mean(),'event_mean_keyword_net_bp':g.naive_net_bp.mean(),'mean_net_ci95_event_cluster':cluster_ci(g.net_bp_2,g.cluster),'high_confidence_clusters':len(directional),'high_confidence_direction_accuracy':directional.hard_hit.mean() if len(directional) else None,'groups':event_groups}
  else:r={'time_quality':quality,'horizon_min':horizon,'delay_min':delay,'event_clusters':0,'attempted_asset_rows':len(sel),'unresolved_asset_rows':len(sel),'status':'no_eligible_observations'}
  metrics.append(r);log('P20_RESULT',**r)
 report={'strategy':'P20_news','manifest_sha256':sha,'sampling':manifest['sampling'],'publication_audit':audit,'judgments':events,'rows':rows,'skips':skips,'pullbacks':pullbacks,'metrics':metrics,'production_pass':False,'limitations':['Retrospective interpretation; training-memory leakage cannot be eliminated.','Convenience corpus not complete archive; source publication is not historical receipt time.','Machine timestamps from current page do not prove original body at that time.','Date-only delayed results test post-news drift, NOT intraday announcement effect.','Price co-movement is not proof the news caused the move.','Minute/hourly execution price proxies; no historical L1 capacity.']}
 write_report('phase20',report);emit_bundle('phase20',report);log('P20_DONE',documents=len(events),rows=len(rows),skips=len(skips),verified_publications=sum(t is not None for t in resolved.values()),production_pass=False)
 return report
