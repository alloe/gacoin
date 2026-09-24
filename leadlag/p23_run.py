"""Bounded, read-only Phase23.2 release-calendar backtest, NOT news alpha validation."""
import base64,gzip,hashlib,io,json,math,re,threading,time,traceback,zipfile
from pathlib import Path
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from collections import Counter
from urllib.parse import urlparse
import requests,numpy as np,pandas as pd
from p23_engine import CFG,VARIANTS,MIN,first_signal,evaluate,summarize,portfolio,valid_frame
ROOT=Path('/tmp/phase23_2');ROOT.mkdir(exist_ok=True)
CACHE=ROOT/'cache';CACHE.mkdir(exist_ok=True)
NY=ZoneInfo('America/New_York');UTC=timezone.utc
PRINT=threading.Lock();NET=threading.Lock();TLS=threading.local();MANIFEST=[];NET_COUNT=0;NET_BYTES=0;STARTED=time.monotonic()
ALLOW={'www.bea.gov':('/news/',),'data.binance.vision':('/data/spot/daily/klines/',),'query1.finance.yahoo.com':('/v8/finance/chart/',)}

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,(float,np.floating)):return float(x) if math.isfinite(x) else None
    return x

def log(tag,**data):
    with PRINT:print(tag+' '+json.dumps(clean(data),separators=(',',':'),ensure_ascii=False,allow_nan=False),flush=True)

def ms(x):return int(pd.Timestamp(x).timestamp()*1000)
def iso(t):return pd.Timestamp(t,unit='ms',tz='UTC').isoformat()
def et(date,clock):return datetime.fromisoformat(date+'T'+clock).replace(tzinfo=NY)

def fetch(url,optional=False):
    global NET_COUNT,NET_BYTES
    u=urlparse(url)
    if u.scheme!='https' or u.hostname not in ALLOW or not any(u.path.startswith(p) for p in ALLOW[u.hostname]):raise ValueError('endpoint rejected')
    key=hashlib.sha256(url.encode()).hexdigest();p=CACHE/key
    if p.exists():return p.read_bytes()
    if not hasattr(TLS,'s'):
        TLS.s=requests.Session();TLS.s.headers['User-Agent']='DaolPhase23PublicResearch/23.2'
    for attempt in range(3):
        with NET:
            if NET_COUNT>=2000 or NET_BYTES>300000000 or time.monotonic()-STARTED>3600:raise RuntimeError('bounded resource budget reached')
            NET_COUNT+=1
        try:r=TLS.s.get(url,timeout=(10,35),allow_redirects=False)
        except (requests.Timeout,requests.ConnectionError):
            if attempt==2:raise
            time.sleep(2**attempt);continue
        if r.status_code==404 and optional:return None
        if r.status_code in (429,500,502,503,504):time.sleep(2**attempt+1);continue
        if r.status_code!=200:raise RuntimeError(f'HTTP {r.status_code}; no access workaround attempted')
        raw=r.content
        with NET:
            NET_BYTES+=len(raw);MANIFEST.append({'url':url,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        p.write_bytes(raw);return raw
    raise RuntimeError('public request retries exhausted')

class Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,d):self.parts.append(d)

def parse_bea(raw):
    p=Text();p.feed(raw.decode('utf-8','replace'));text=' '.join(' '.join(p.parts).split())
    pat=r'EMBARGOED UNTIL RELEASE AT\s+(\d{1,2}:\d{2})\s*([ap])\.?m\.?\s+(EST|EDT),?\s*(?:[A-Za-z]+,\s*)?([A-Za-z]+\s+\d{1,2},\s*20\d\d)'
    m=re.search(pat,text,re.I)
    if not m:return None
    clock=datetime.strptime(m[1]+m[2].upper()+'M','%I:%M%p').strftime('%H:%M')
    date=datetime.strptime(m[4],'%B %d, %Y').date().isoformat();dt=et(date,clock)
    if dt.tzname()!=m[3].upper():raise ValueError('embargo timezone mismatch')
    return date,clock,ms(dt)

def make_calendar(spec):
    events=[];audit=[]
    for family in ('CPI','EMPLOYMENT','FOMC'):
        clock='14:00' if family=='FOMC' else '08:30'
        for year,dates in spec[family].items():
            for md in dates:
                date=year+'-'+md;t=ms(et(date,clock));events.append({'event_id':family+'-'+date,'family':family,'date':date,'t':t,'clock':clock,'calendar_source':'official_web_read_frozen','group':'event','anchor_year':year})
    for month in pd.period_range('2023-12','2026-07',freq='M'):
        slug=f'personal-income-and-outlays-{month.strftime("%B").lower()}-{month.year}'
        expected=(month+1).year;found=False
        for year in sorted(set([expected,expected+1])):
            if year>2026:continue
            url=f'https://www.bea.gov/news/{year}/{slug}'
            try:
                raw=fetch(url,True)
                if raw is None:continue
                ans=parse_bea(raw)
                if not ans:audit.append({'url':url,'status':'no_verified_embargo_header'});continue
                date,clock,t=ans
                if not '2024-01-01'<=date<'2026-09-23':continue
                events.append({'event_id':'PCE-'+date,'family':'PCE','date':date,'t':t,'clock':clock,'calendar_source':url,'group':'event','anchor_year':date[:4]});found=True;break
            except Exception as e:audit.append({'url':url,'error':repr(e)})
        if not found:audit.append({'reference_month':str(month),'status':'release_not_recovered'})
    url='https://www.bea.gov/news/2026/personal-income-and-outlays-october-and-november-2025'
    try:
        raw=fetch(url,True);ans=parse_bea(raw) if raw else None
        if ans:
            date,clock,t=ans
            events.append({'event_id':'PCE-'+date,'family':'PCE','date':date,'t':t,'clock':clock,'calendar_source':url,'group':'event','anchor_year':date[:4]})
    except Exception as e:audit.append({'url':url,'error':repr(e)})
    events=sorted({e['event_id']:e for e in events}.values(),key=lambda e:(e['t'],e['family']))
    used=set();controls=[]
    for e in events:
        for offset in (7,14,21,28):
            dt=et(e['date'],e['clock'])-timedelta(days=offset);t=ms(dt)
            if t in used or any(abs(t-z['t'])<=4*3600000 for z in events):continue
            c=dict(e);c.update(event_id='CTRL-'+e['event_id'],parent_id=e['event_id'],date=dt.date().isoformat(),t=t,group='control',offset_days=offset)
            controls.append(c);used.add(t);break
        else:audit.append({'event_id':e['event_id'],'status':'no_matched_control'})
    allcases=events+controls
    for e in allcases:
        clock='14:30' if e['clock']=='14:00' else '10:00';e['blackout']=ms(et(e['date'],clock))
        future=[x['t'] for x in events if e['t']<x['t']<e['blackout']]
        if future:e['blackout']=min(future)
        e['simultaneous_target_families']=[x['family'] for x in events if x['t']==e['t']]
    return sorted(allcases,key=lambda e:e['t']),audit

@lru_cache(maxsize=700)
def spot_day(coin,date):
    url=f'https://data.binance.vision/data/spot/daily/klines/{coin}/1m/{coin}-1m-{date}.zip';raw=fetch(url,True)
    if raw is None:return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=z.namelist()
        if len(names)!=1:raise ValueError('unexpected archive')
        body=z.read(names[0]);first=body.splitlines()[0].split(b',')[0]
        d=pd.read_csv(io.BytesIO(body),header=None if first.isdigit() else 0)
    if d.shape[1]!=12:raise ValueError('unexpected candle schema')
    d.columns=['t','open','high','low','close','volume','ct','quote_volume','n','buy_volume','buy_quote','ignore']
    for c in d.columns:d[c]=pd.to_numeric(d[c],errors='coerce')
    if d.t.median()>1e14:d.t=d.t/1000
    if d.t.isna().any() or ((d.t%MIN)!=0).any():raise ValueError('invalid candle timestamps')
    d.t=d.t.astype('int64')
    if d.t.duplicated().any():raise ValueError('duplicate candles not silently dropped')
    return d.set_index('t').sort_index()

def proxy_data():
    out={};audit=[];start=ms('2026-08-01T00:00:00Z')//1000;end=ms('2026-09-23T00:00:00Z')//1000
    for symbol in ('NQ','ZT'):
        url=f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}%3DF?interval=5m&period1={start}&period2={end}'
        try:
            obj=json.loads(fetch(url));chart=obj['chart'];a=(chart.get('result') or [None])[0]
            if a is None:raise ValueError(str(chart.get('error')))
            ts=np.array(a['timestamp'],dtype='int64')*1000;q=a['indicators']['quote'][0]
            d=pd.DataFrame({'t':ts,'close':q['close'],'volume':q['volume']}).dropna();d=d[(d.close>0)&(d.volume>0)]
            d['available']=d.t+6*MIN;out[symbol]=d.sort_values('available')
            audit.append({'symbol':symbol,'bars':len(d),'first':iso(int(d.t.min())),'last':iso(int(d.t.max())),'interval':'5m','assumed_availability':'bar_close+60s','underlying':a['meta'].get('shortName'),'historical_rolls_unverified':True})
        except Exception as e:audit.append({'symbol':symbol,'error':repr(e)})
    def at(sym,t):
        d=out.get(sym)
        if d is None:return None
        j=np.searchsorted(d.available.to_numpy(),t,side='right')-1
        if j<0:return None
        r=d.iloc[j]
        if t-(r.t+5*MIN)>7*MIN:return None
        return float(r.close),int(r.t)
    def gate(now,t0,variant):
        for sym in ('NQ','ZT'):
            a=at(sym,now);b=at(sym,now-5*MIN)
            if a is None or b is None or a[1]<=b[1]:return False,'proxy_macro_missing_or_stale'
            if a[0]<b[0]:return False,'proxy_'+sym+'_not_stable'
        if variant=='A_pullback_flow':
            a=at('NQ',now);b=at('NQ',t0)
            if a is None or b is None or a[0]<=b[0]:return False,'proxy_equity_below_pre_event'
        return True,'proxy_macro_pass'
    return gate,audit

def emit_bundle(name,obj):
    raw=json.dumps(clean(obj),ensure_ascii=False,separators=(',',':'),allow_nan=False).encode();text=base64.b64encode(gzip.compress(raw,mtime=0)).decode();sha=hashlib.sha256(raw).hexdigest()
    (ROOT/(name+'.json')).write_bytes(raw)
    log('P23_BUNDLE',name=name,sha256=sha,chunks=math.ceil(len(text)/6000),raw_bytes=len(raw))
    for i in range(0,len(text),6000):log('P23_CHUNK',name=name,index=i//6000,data=text[i:i+6000])
    log('P23_BUNDLE_END',name=name,sha256=sha)

def main():
    specfile=Path(__file__).with_name('p23_calendar.json');spec=json.loads(specfile.read_bytes())
    log('P23_START',version='23.2',spec_sha256=hashlib.sha256(specfile.read_bytes()).hexdigest(),config=CFG.__dict__,hypotheses=spec['hypotheses'],no_orders=True)
    cases,audit=make_calendar(spec);log('P23_CALENDAR',counts=dict(Counter((e['group']+'_'+e['family']) for e in cases)),cases=cases,audit=audit,price_paths_not_yet_requested=True)
    gate,paudit=proxy_data();log('P23_MACRO_DATA',audit=paudit,full_model_status='BLOCKED_no_PIT_consensus_body_or_US2Y_yield')
    jobs=sorted({(coin,e['date']) for e in cases for coin in ('BTCUSDT','ETHUSDT')})
    failures=[]
    def load(job):
        try:
            d=spot_day(*job)
            return job,('ok' if d is not None else 'missing_archive')
        except Exception as ex:return job,repr(ex)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,(job,status) in enumerate(pool.map(load,jobs),1):
            if status!='ok':failures.append({'coin':job[0],'date':job[1],'status':status})
            if n%80==0:log('P23_DATA_PROGRESS',done=n,total=len(jobs),failures=len(failures))
    rows=[];extremes=[]
    for i,e in enumerate(cases):
        for coin in ('BTCUSDT','ETHUSDT'):
            try:d=spot_day(coin,e['date'])
            except Exception:d=None
            pre=d.reindex(np.arange(e['t']-60*MIN,e['t'],MIN)) if d is not None else None
            if valid_frame(pre):
                p0=float(pre.close.iloc[-1]);post=d.reindex(np.arange(e['t'],e['t']+60*MIN,MIN))
                if valid_frame(post):
                    low=float(post.low.min());li=int(np.argmin(post.low.to_numpy()));after=post.iloc[li+1:]
                    extremes.append({'event_id':e['event_id'],'date':e['date'],'coin':coin,'group':e['group'],'drop_bp':(1-low/p0)*10000,'hindsight_low_then_bounce_bp':(float(after.high.max())/low-1)*10000 if len(after) else None,'not_tradable_oracle':True})
            for variant in VARIANTS:
                row={k:e[k] for k in ('event_id','date','t','family','group','anchor_year')};row.update(coin=coin,variant=variant)
                if d is None:c=None;reasons={'missing_archive':1};out={'status':'unresolved_data'}
                else:
                    c,reasons=first_signal(d,e['t'],variant,e['blackout'])
                    out=evaluate(d,c,e['blackout']) if c else {'status':'unresolved_signal_data' if reasons.get('data_gap_or_invalid',0) else 'no_signal'}
                row.update(candidate=c,reasons=reasons,outcome=out);rows.append(row)
                if e['group']=='event' and e['date']>='2026-08-01' and variant=='B2_price_flow' and d is not None:
                    pc,pr=first_signal(d,e['t'],variant,e['blackout'],gate)
                    prow=dict(row);prow.update(variant='X_B2_5m_proxy',candidate=pc,reasons=pr,outcome=evaluate(d,pc,e['blackout']) if pc else {'status':'no_signal'});rows.append(prow)
        if (i+1)%40==0:log('P23_EVENT_PROGRESS',done=i+1,total=len(cases))
    groups=[];ports=[]
    for group in ('event','control'):
        for year in ('2024','2025','2026','ALL'):
            for variant in VARIANTS:
                selected=[r for r in rows if r['group']==group and r['variant']==variant and (year=='ALL' or r['anchor_year']==year)]
                for extra in (0,2,5):
                    m=summarize(selected,extra);m.update(group=group,year=year,variant=variant);groups.append(m)
                    if extra==0:log('P23_RESULT',**m)
                if group=='event':
                    p=portfolio(selected);p.update(year=year,variant=variant);ports.append(p)
                    log('P23_ACCOUNT',**{k:v for k,v in p.items() if k!='trades'})
    xm=summarize([r for r in rows if r['variant']=='X_B2_5m_proxy']);log('P23_PROXY_RESULT',**xm,note='short recent 5m futures-price proxy; not exact yield model')
    full={'status':'not_executed_missing_inputs','consensus_snapshot':False,'PIT_news_judgment':False,'historical_minute_US2Y_yields':False}
    compactrows=[]
    for r in rows:
        z=dict(r);z['outcome']={k:v for k,v in r['outcome'].items() if k!='pre_exit_low_marks'};compactrows.append(z)
        if r['group']=='event' and r['outcome']['status']=='modeled_closed' and r['variant']=='B2_price_flow':log('P23_TRADE',**z)
    summary={'version':'23.2','case_count':len(cases),'rows':len(rows),'calendar_counts':dict(Counter(e['family'] for e in cases if e['group']=='event')),'calendar_audit':audit,'download_failures':failures,'macro_data_audit':paudit,'groups':groups,'accounts':ports,'proxy_result':xm,'full_model':full,'production_pass':False,'fund_currency':'USDT','no_upbit_execution_claim':True,'http_requests':NET_COUNT,'bytes':NET_BYTES}
    emit_bundle('summary',summary);emit_bundle('trades',[r for r in compactrows if r['candidate'] or r['outcome']['status'].startswith('unresolved')]);emit_bundle('case_audit',compactrows);emit_bundle('calendar',cases);emit_bundle('sources',MANIFEST)
    for group in ('event','control'):
        z=[r for r in extremes if r['group']==group and r['drop_bp']>=80];bounce=[r for r in z if r['hindsight_low_then_bounce_bp'] is not None]
        log('P23_ORACLE_DIAGNOSTIC',group=group,drop_cases=len(z),bounce_observable=len(bounce),fraction_20bp=float(np.mean([r['hindsight_low_then_bounce_bp']>=20 for r in bounce])) if bounce else None,not_tradable=True)
    log('P23_DONE',version='23.2',calendar_cases=len(cases),rows=len(rows),failed_archives=len(failures),full_model=full,production_pass=False)
if __name__=='__main__':
    try:main()
    except Exception as e:log('P23_FATAL',error=repr(e),traceback=traceback.format_exc()[-2500:]);raise
