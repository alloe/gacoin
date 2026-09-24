"""Public-data research utilities. No credentials, orders, or account APIs."""
from __future__ import annotations
import base64,hashlib,io,json,math,os,threading,time,zipfile
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import requests
ROOT=Path(os.getenv('RESEARCH_OUTPUT','/tmp/daol-research20-22'));ROOT.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/'cache';CACHE.mkdir(exist_ok=True)
ALLOW={'data.binance.vision':('/data/',),'s3-ap-northeast-1.amazonaws.com':('/data.binance.vision',),'fapi.binance.com':('/fapi/v1/klines','/fapi/v1/fundingRate'),'api.upbit.com':('/v1/candles/',),'www.binance.com':('/en/support/announcement/detail/',),'www.sec.gov':('/newsroom/press-releases/',),'blog.ethereum.org':('/',)}
LOCK=threading.Lock();PRINT=threading.Lock();TLS=threading.local();LAST=0.;NREQ=0;NBYTES=0;BEGAN=time.monotonic();MANIFEST=[]
KCOL=['t','open','high','low','close','volume','close_t','quote_volume','count','taker_buy_volume','taker_buy_quote_volume','ignore']
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,np.integer):return int(x)
 if isinstance(x,np.bool_):return bool(x)
 if isinstance(x,(float,np.floating)):return float(x) if math.isfinite(x) else None
 return x

def log(tag,**kw):
 with PRINT:print(tag+' '+json.dumps(clean(kw),ensure_ascii=False,separators=(',',':'),allow_nan=False),flush=True)

def stamp(x):return int(pd.Timestamp(x,tz='UTC').timestamp()*1000) if pd.Timestamp(x).tzinfo is None else int(pd.Timestamp(x).timestamp()*1000)
def iso(t):return pd.Timestamp(int(t),unit='ms',tz='UTC').isoformat()
def fetch(url,params=None,missing_ok=False):
 global LAST,NREQ,NBYTES
 u=urlparse(url)
 if u.scheme!='https' or u.hostname not in ALLOW or not any(u.path.startswith(p) for p in ALLOW[u.hostname]):raise ValueError('public endpoint allowlist rejected URL')
 key=hashlib.sha256((url+json.dumps(params,sort_keys=True)).encode()).hexdigest();p=CACHE/key
 if p.exists():return p.read_bytes()
 if (CACHE/(key+'.missing')).exists():return None
 if not hasattr(TLS,'session'):
  TLS.session=requests.Session();TLS.session.headers['User-Agent']='DaolIndependentResearch/20-22'
 for trial in range(4):
  with LOCK:
   if NREQ>=12000 or NBYTES>768000000 or time.monotonic()-BEGAN>7200:raise RuntimeError('research resource budget reached')
   delay=max(0,.12-(time.monotonic()-LAST))
   if delay:time.sleep(delay)
   LAST=time.monotonic();NREQ+=1
  try:r=TLS.session.get(url,params=params,timeout=(10,35),allow_redirects=False)
  except (requests.Timeout,requests.ConnectionError):
   if trial==3:raise
   time.sleep(2**trial);continue
  with LOCK:NBYTES+=len(r.content)
  if r.status_code==404 and missing_ok:
   (CACHE/(key+'.missing')).touch();return None
  if r.status_code in (429,500,502,503,504):
   time.sleep(min(30,max(2**trial,float(r.headers.get('Retry-After',0)))));continue
  r.raise_for_status()
  if r.status_code!=200:raise RuntimeError('redirect/non200 not followed')
  if len(r.content)>50000000:raise RuntimeError('single resource too large')
  tmp=CACHE/(key+'.'+str(threading.get_ident())+'.tmp');tmp.write_bytes(r.content);tmp.replace(p)
  with LOCK:MANIFEST.append({'url':r.url,'sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content),'fetched_utc':pd.Timestamp.now(tz='UTC').isoformat()})
  return r.content
 raise RuntimeError('public-data retry budget exhausted')

def archive(path,header=True,optional=False):
 raw=fetch('https://data.binance.vision/'+path,missing_ok=optional)
 if raw is None:return None
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  names=z.namelist()
  if len(names)!=1:raise ValueError('unexpected archive members')
  body=z.read(names[0]);first=body.splitlines()[0].split(b',')[0].strip(b'"')
  numeric_first=first.replace(b'.',b'',1).isdigit()
  return pd.read_csv(io.BytesIO(body),header=None if (not header or numeric_first) else 0)

def klines_month(symbol,interval,month,optional=False):
 d=archive(f'data/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip',optional=optional)
 if d is None:
  if month!=pd.Timestamp.now(tz='UTC').strftime('%Y-%m'):return None
  start=stamp(month+'-01');stop=int(pd.Timestamp.now(tz='UTC').timestamp()*1000);rows=[];cur=start
  while cur<stop:
   raw=fetch('https://fapi.binance.com/fapi/v1/klines',{'symbol':symbol,'interval':interval,'startTime':cur,'endTime':stop-1,'limit':1500},missing_ok=True)
   if raw is None:break
   aa=json.loads(raw)
   if not isinstance(aa,list):raise ValueError('current month kline error')
   if not aa:break
   rows.extend(aa);nxt=int(aa[-1][6])+1
   if nxt<=cur:raise ValueError('kline cursor stalled')
   cur=nxt
  if not rows:return None
  d=pd.DataFrame(rows)
 if d.shape[1]!=12:raise ValueError('unexpected kline columns')
 d.columns=KCOL
 for col in KCOL:d[col]=pd.to_numeric(d[col],errors='coerce')
 if d.t.median()>1e14:d['t']=d.t//1000
 d=d.dropna(subset=['t','open','close']).drop_duplicates('t').sort_values('t');d['t']=d.t.astype('int64')
 if ((d[['open','high','low','close']]<=0).any(axis=1)).any():raise ValueError('nonpositive market prices')
 return d.set_index('t')

def monthly_range(start,end):
 last=iso(stamp(end)-1)[:7]
 return [p.strftime('%Y-%m') for p in pd.period_range(start[:7],last,freq='M')]
def load_klines(symbol,interval,start,end):
 frames=[];missing=[]
 for m in monthly_range(start,end):
  d=klines_month(symbol,interval,m,optional=True)
  if d is None:missing.append(m)
  else:frames.append(d)
 if not frames:raise ValueError('no historical klines '+symbol)
 d=pd.concat(frames).sort_index();d=d[~d.index.duplicated()]
 return d.loc[(d.index>=stamp(start))&(d.index<stamp(end))],missing

def funding_history(symbol,start,end):
 frames=[];missing=[]
 for m in monthly_range(start,end):
  d=archive(f'data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{m}.zip',optional=True)
  if d is None and m==pd.Timestamp.now(tz='UTC').strftime('%Y-%m'):
   raw=fetch('https://fapi.binance.com/fapi/v1/fundingRate',{'symbol':symbol,'startTime':stamp(m+'-01'),'endTime':stamp(end)-1,'limit':1000},missing_ok=True)
   aa=json.loads(raw) if raw else []
   if aa:d=pd.DataFrame({'calc_time':[x['fundingTime'] for x in aa],'last_funding_rate':[x['fundingRate'] for x in aa]})
  if d is None:missing.append(m);continue
  if not {'calc_time','last_funding_rate'}.issubset(d.columns):raise ValueError('unexpected funding archive schema')
  frames.append(d[['calc_time','last_funding_rate']].rename(columns={'calc_time':'t','last_funding_rate':'rate'}))
 if not frames:return pd.DataFrame(columns=['rate']),missing
 d=pd.concat(frames).drop_duplicates('t').sort_values('t');d['t']=pd.to_numeric(d.t).astype('int64');d['rate']=pd.to_numeric(d.rate)
 return d.set_index('t').loc[lambda z:(z.index>=stamp(start))&(z.index<stamp(end))],missing

def net_trade(side,entry,exit,fee_bp=5,slip_bp=2,funding_bp=0):
 r=exit/entry-1
 return side*r*10000-(fee_bp+slip_bp)*(1+exit/entry)-side*funding_bp

def trade_stats(df):
 if len(df)==0:return {'n':0,'mean_bp':None,'win_rate':None,'pf':None}
 x=np.asarray(df,dtype=float);x=x[np.isfinite(x)]
 if len(x)==0:return {'n':0,'mean_bp':None,'win_rate':None,'pf':None}
 losses=-x[x<0].sum()
 return {'n':len(x),'mean_bp':x.mean(),'median_bp':np.median(x),'win_rate':np.mean(x>0),'pf':x[x>0].sum()/losses if losses else None,'p05_bp':np.quantile(x,.05),'p95_bp':np.quantile(x,.95)}

def cluster_ci(values,clusters,seed=20260925):
 d=pd.DataFrame({'v':values,'c':clusters}).dropna();unique=list(d.c.unique())
 if len(unique)<5:return None
 sums=d.groupby('c').v.sum().reindex(unique).to_numpy();counts=d.groupby('c').v.count().reindex(unique).to_numpy()
 rng=np.random.default_rng(seed);sel=rng.integers(0,len(unique),(1000,len(unique)));means=sums[sel].sum(axis=1)/counts[sel].sum(axis=1)
 return np.quantile(means,[.025,.975]).tolist()

def equity_stats(times,equity,turnover=0.,funding=0.,fees=0.):
 e=np.asarray(equity,dtype=float)
 if len(e)<2:return {'bars':len(e)}
 ix=pd.to_datetime(times,unit='ms',utc=True);s=pd.Series(e,index=ix);ml=s.resample('MS').last();prior=ml.shift(1);prior.iloc[0]=e[0];months=(ml/prior-1).to_dict()
 daily=s.resample('D').last().pct_change(fill_method=None).dropna();dd=e/np.maximum.accumulate(e)-1
 return {'bars':len(e),'return_pct':(e[-1]/e[0]-1)*100,'mdd_pct':float(np.min(dd)*100),'daily_sharpe':float(daily.mean()/daily.std()*np.sqrt(365)) if daily.std()>0 else None,'monthly_pct':{str(k)[:7]:v*100 for k,v in months.items()},'turnover_notional':turnover,'funding_pnl':funding,'fees_paid':fees}

def write_report(name,data):
 p=ROOT/(name+'.json');p.write_text(json.dumps(clean(data),ensure_ascii=False,separators=(',',':'),allow_nan=False));return p

def emit_bundle(name,data):
 import gzip
 raw=json.dumps(clean(data),ensure_ascii=False,separators=(',',':'),allow_nan=False).encode();comp=gzip.compress(raw,mtime=0);s=base64.b64encode(comp).decode();sha=hashlib.sha256(raw).hexdigest()
 log('R2022_BUNDLE',name=name,raw_bytes=len(raw),gzip_bytes=len(comp),sha256=sha,chunks=math.ceil(len(s)/6000))
 for i in range(0,len(s),6000):log('R2022_CHUNK',name=name,index=i//6000,data=s[i:i+6000])
 log('R2022_BUNDLE_END',name=name,sha256=sha)
