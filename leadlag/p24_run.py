"""Bounded public-data execution of frozen Phase24.1 conventional wave models.
No credentials/account/order API. 2022-2026 results are retrospective, not a
pristine forward test. Eight fixed liquid coins are NOT an unbiased all-coin set.
"""
from __future__ import annotations
import base64,gzip,hashlib,io,json,math,time,traceback,zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np,pandas as pd,requests
from p24_wave import C,MODELS,FIVE,DAY,features,run_one,nav_metrics,trade_metrics

ROOT=Path('/tmp/phase24_wave');ROOT.mkdir(exist_ok=True)
COINS=('BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','ADAUSDT','DOGEUSDT','LINKUSDT','LTCUSDT')
DATA_START='2021-11-01';START='2022-01-01';END='2026-09-01'
SPLITS=(('2022_2023','2022-01-01','2024-01-01'),('2024','2024-01-01','2025-01-01'),('2025','2025-01-01','2026-01-01'),('2026_8m','2026-01-01','2026-09-01'),('ALL',START,END))
SPEC={'version':'24.1-frozen','coins':COINS,'period':[START,END],'warmup':DATA_START,
 'primary':'T_fixed: 4h EMA50>EMA200, close>EMA200, EMA50 rising over6 bars; 1h RSI14 crosses45 up, close>previous high; stop2ATR14 target4ATR14 max72h.',
 'T_trail':'Same entry; no fixed target. After highest closed hourly close reaches entry+2 entryATR, trail2 current hourlyATR, max72h.',
 'R_plain':'Hourly close recovers above BB(20,2) lower band after prior close below it; RSI14<45. Stop2ATR and frozen SMA20 target, max24h.',
 'R_regime':'Same as R_plain, additionally last completed4h ADX14<20.',
 'clock':'Signal from completed hour; execute first exact5m open at close+5m; never fill missing bars or future interpolation.',
 'cost':'Standard spot10bp each + slip2bp each; no VIP/coupons. Additional0/2/5bp same-fill stress, not reranked trades.',
 'size':'Per-coin isolated10000USDT sleeve; risk0.5%, exposure25%, prior1h volume0.1%. One position/coin, cooldown4h. Equal initial weights across8 sleeves, no cross-sleeve reuse.',
 'caveat':'No historical L1/depth, exchange rules/lot size, tradeable-capacity, API latency or KRW conversions verified. Survivorship of fixed8 liquid coins remains. No news used.',
 'selection':'No parameter search or post-outcome variant selection. Chronological period reports only; first twoyears not tuned.',
 'pass':'No live permission. Positive net expectancy with date-cluster confidence interval, fee stress, nonconcentrated coins and stability would justify only future paper validation.'}
BEGAN=time.monotonic();MANIFEST=[]
def ms(x):return int(pd.Timestamp(x,tz='UTC').timestamp()*1000)
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,np.integer):return int(x)
 if isinstance(x,np.bool_):return bool(x)
 if isinstance(x,(float,np.floating)):return float(x) if math.isfinite(x) else None
 return x

def log(tag,**kw):print(tag+' '+json.dumps(clean(kw),ensure_ascii=False,separators=(',',':'),allow_nan=False),flush=True)
def emit(name,obj):
 raw=json.dumps(clean(obj),ensure_ascii=False,separators=(',',':'),allow_nan=False).encode();(ROOT/(name+'.json')).write_bytes(raw)
 packed=base64.b64encode(gzip.compress(raw,mtime=0)).decode();sha=hashlib.sha256(raw).hexdigest()
 log('P24_BUNDLE',name=name,sha256=sha,raw_bytes=len(raw),chunks=math.ceil(len(packed)/5500))
 for i in range(0,len(packed),5500):log('P24_CHUNK',name=name,index=i//5500,data=packed[i:i+5500])
 log('P24_BUNDLE_END',name=name,sha256=sha)

def load_month(job):
 coin,month=job;url=f'https://data.binance.vision/data/spot/monthly/klines/{coin}/5m/{coin}-5m-{month}.zip'
 if time.monotonic()-BEGAN>2700:raise RuntimeError('research45minute resource budget')
 raw=None
 for i in range(3):
  try:r=requests.get(url,timeout=(10,40),allow_redirects=False,headers={'User-Agent':'DaolReadOnlyResearch/24.1'})
  except (requests.Timeout,requests.ConnectionError):
   if i==2:raise
   time.sleep(1+i);continue
  if r.status_code in (429,500,502,503,504):time.sleep(2+i);continue
  if r.status_code!=200:raise RuntimeError(f'{coin} {month} HTTP {r.status_code}; data not fabricated')
  raw=r.content;break
 if raw is None:raise RuntimeError('retry exhausted')
 if len(raw)>15000000:raise RuntimeError('archive size bound')
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  if len(z.namelist())!=1:raise ValueError('unexpected ZIP contents')
  content=z.read(z.namelist()[0]);first=content.splitlines()[0].split(b',')[0]
  d=pd.read_csv(io.BytesIO(content),header=None if first.isdigit() else 0)
 if d.shape[1]!=12:raise ValueError('kline schema')
 d.columns=['t','open','high','low','close','volume','ct','qv','count','bv','bq','ignore']
 for c in d:d[c]=pd.to_numeric(d[c],errors='coerce')
 if d.t.median()>1e14:d.t=d.t/1000
 if d.t.isna().any() or ((d.t%FIVE)!=0).any() or d.t.duplicated().any():raise ValueError('timestamp integrity')
 a=d[['open','high','low','close','qv']]
 if not np.isfinite(a.to_numpy()).all() or (a.iloc[:,:4]<=0).any().any() or (d.qv<0).any():raise ValueError('invalid prices/volume')
 if (d.low>d[['open','close']].min(axis=1)).any() or (d.high<d[['open','close']].max(axis=1)).any():raise ValueError('OHLC integrity')
 MANIFEST.append({'url':url,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 return d.set_index(d.t.astype('int64'))[['open','high','low','close','qv']]

def passive(d,start,end):
 z=d.loc[(d.index>=start)&(d.index<end)];ep=float(z.open.iloc[0])*(1+C.slip);qty=C.initial*.25/(ep*(1+C.fee));cash=C.initial-qty*ep*(1+C.fee)
 a=z.close*(1-C.slip)*(1-C.fee)*qty+cash;g=a.groupby(z.index//DAY).last();return {int(k):float(v) for k,v in g.items()}

def main():
 specbytes=json.dumps(SPEC,sort_keys=True).encode();log('P24_START',spec_sha256=hashlib.sha256(specbytes).hexdigest(),spec=SPEC,no_orders=True)
 months=[str(x) for x in pd.period_range(DATA_START[:7],'2026-08',freq='M')]
 grid=np.arange(ms(DATA_START),ms(END),FIVE);results={m:{} for m in MODELS};bench={};coverage=[];failures=[]
 for coin in COINS:
  try:
   with ThreadPoolExecutor(max_workers=4) as pool:frames=list(pool.map(load_month,[(coin,m) for m in months]))
   df=pd.concat(frames).sort_index()
   if df.index.duplicated().any():raise ValueError('cross-month duplicate timestamps')
   df=df.reindex(grid);f=features(df);missing=int(df.open.isna().sum());coverage.append({'coin':coin,'bars':len(df),'missing_bars':missing,'months':len(months)})
   log('P24_DATA',coin=coin,bars=len(df),missing=missing)
   for model in MODELS:
    out=run_one(df,f,model,ms(START),ms(END));results[model][coin]=out
    log('P24_COIN',coin=coin,model=model,signals=out['signals'],counts=out['counts'],incomplete=out['incomplete'],metrics=trade_metrics(out['records']))
   bench[coin]=passive(df,ms(START),ms(END));del df,f,frames
  except Exception as e:
   failures.append({'coin':coin,'error':repr(e)});log('P24_COIN_FATAL',coin=coin,error=repr(e),traceback=traceback.format_exc()[-1500:])
 rows=[];trades=[];audit=[]
 for model,coins in results.items():
  eligible=len(coins)==len(COINS) and not any(x['incomplete'] for x in coins.values())
  days=sorted(set().union(*(set(x['daily']) for x in coins.values()))) if coins else []
  common=[d for d in days if all(d in x['daily'] for x in coins.values())]
  nav={day:sum(x['daily'][day] for x in coins.values())/len(COINS) for day in common}
  records=[]
  for coin,out in coins.items():
   records.extend([dict(r,coin=coin,model=model) for r in out['records']]);audit.append({'coin':coin,'model':model,'signals':out['signals'],'counts':out['counts'],'incomplete':out['incomplete']})
  trades+=records
  for label,start,end in SPLITS:
   a,b=ms(start),ms(end);selected=[r for r in records if a<=r['entry_time'] and r['exit_time']<b]
   dd={d:v for d,v in nav.items() if a//DAY<=d<b//DAY};prev=nav.get(a//DAY-1,C.initial)
   nm=nav_metrics(dd,prev) if eligible else {'status':'incomplete_not_reportable'}
   bycoin={c:trade_metrics([r for r in selected if r['coin']==c]) for c in COINS}
   metric=trade_metrics(selected);row={'model':model,'period':label,'primary':model=='T_fixed','complete':eligible,'account':nm,'trades':metric,'by_coin':bycoin}
   rows.append(row);log('P24_RESULT',model=model,period=label,primary=row['primary'],complete=eligible,account=nm,trades=metric)
  emit('trades_'+model,records)
 if len(bench)==len(COINS):
  days=sorted(set.intersection(*(set(x) for x in bench.values())));bn={d:sum(x[d] for x in bench.values())/len(COINS) for d in days}
  log('P24_BENCHMARK',description='25pct initial capital passive equal-coin sleeves plus75pct nonyield cash; not exposure-matched or continuously rebalanced',metrics=nav_metrics(bn))
 else:bn={}
 report={'spec':SPEC,'spec_sha256':hashlib.sha256(specbytes).hexdigest(),'coverage':coverage,'failures':failures,'results':rows,'audit':audit,'passive_25pct':nav_metrics(bn),'production_pass':False}
 emit('summary',report);emit('source_manifest',MANIFEST)
 log('P24_DONE',coins=len(COINS),successful=len(coverage),failure_count=len(failures),models=len(MODELS),trade_records=len(trades),archive_count=len(MANIFEST),download_bytes=sum(x['bytes'] for x in MANIFEST),production_pass=False)
if __name__=='__main__':
 try:main()
 except Exception as e:log('P24_FATAL',error=repr(e),traceback=traceback.format_exc()[-2500:]);raise
