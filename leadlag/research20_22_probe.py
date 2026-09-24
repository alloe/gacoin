#!/usr/bin/env python3
"""Read-only, bounded data feasibility probe. No credentials or trading endpoints."""
import requests, json, time, io, zipfile, csv, hashlib, re, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
S=requests.Session(); S.headers['User-Agent']='DaolResearch/20-22-public-probe'
def log(tag, **kw): print(tag+' '+json.dumps(kw,ensure_ascii=False,separators=(',',':')),flush=True)
def get(url,params=None):
    r=S.get(url,params=params,timeout=(10,35),allow_redirects=False)
    if r.status_code!=200: raise RuntimeError(str(r.status_code)+' '+r.text[:100])
    return r

def announcements():
    try:
        allrows=[]
        for cat in ('trade','wallet'):
            rows=[]
            for page in range(1,5):
                d=get('https://api-manager.upbit.com/api/v1/announcements',{'os':'web','page':page,'per_page':100,'category':cat}).json()
                payload=d.get('data',d); aa=payload.get('notices',payload.get('announcements',[]))
                if page==1:log('PROBE_NOTICE_SCHEMA',category=cat,top_keys=list(d),data_keys=list(payload),first=aa[0] if aa else None)
                if not aa:break
                rows.extend(aa); time.sleep(.4)
            sel=[]
            for a in rows:
                t=a.get('first_listed_at') or a.get('listed_at') or ''
                if '2026-04-01'<=t[:10]<='2026-09-22': sel.append(a)
            sel=sorted({str(a['id']):a for a in sel}.values(),key=lambda a:a.get('first_listed_at') or a.get('listed_at'),reverse=True)
            for a in sel[:40 if cat=='trade' else 20]:
                log('PROBE_NOTICE',source='upbit',category=cat,id=a.get('id'),title=a.get('title'),first_listed_at=a.get('first_listed_at'),listed_at=a.get('listed_at'),is_updated=a.get('is_updated'))
            allrows.extend(sel)
            log('PROBE_NOTICE_COUNT',category=cat,downloaded=len(rows),in_range=len(sel))
        for a in allrows[:3]:
            try:
                d=get('https://api-manager.upbit.com/api/v1/announcements/'+str(a['id'])).json(); z=d.get('data',d)
                log('PROBE_NOTICE_DETAIL',id=a['id'],keys=list(z),title=z.get('title'),first=z.get('first_listed_at'),listed=z.get('listed_at'),body=(z.get('body') or '')[:1800])
            except Exception as e:log('PROBE_DETAIL_FAIL',id=a['id'],error=str(e))
    except Exception as e:log('PROBE_ANNOUNCEMENTS_FAIL',error=repr(e))

def archives():
    for path in ['data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2026-08-01.zip','data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-07-01.zip','data/futures/um/monthly/metrics/BTCUSDT/BTCUSDT-metrics-2026-08.zip','data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2026-08.zip','data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2026-08.zip']:
        try:
            r=get('https://data.binance.vision/'+path);z=zipfile.ZipFile(io.BytesIO(r.content));lines=z.read(z.namelist()[0]).decode().splitlines()
            log('PROBE_ARCHIVE',path=path,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),rows=len(lines),sample=lines[:3])
        except Exception as e:log('PROBE_ARCHIVE_FAIL',path=path,error=str(e))
    try:
        url='https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
        r=get(url,{'delimiter':'/','prefix':'data/futures/um/monthly/klines/','max-keys':1000})
        root=ET.fromstring(r.content);prefixes=[e.text for e in root.iter() if e.tag.endswith('Prefix')]
        log('PROBE_CATALOG',url=r.url,prefixes=prefixes,count=len(prefixes))
    except Exception as e:log('PROBE_CATALOG_FAIL',error=str(e))

def news():
    try:
        d=get('https://www.binance.com/bapi/composite/v1/public/cms/article/list',{'type':1,'pageNo':1,'pageSize':30,'catalogId':48}).json()
        p=d.get('data',{});log('PROBE_BINANCE_NEWS',keys=list(d),data=p)
    except Exception as e:log('PROBE_BINANCE_NEWS_FAIL',error=str(e))
    try:
        d=get('https://fapi.binance.com/futures/data/openInterestHist',{'symbol':'BTCUSDT','period':'5m','limit':3}).json()
        log('PROBE_OI_API',rows=d)
    except Exception as e:log('PROBE_OI_API_FAIL',error=str(e))

if __name__=='__main__':
    log('PROBE_START',read_only=True,paid_data=False,labels_before_price_evaluation=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda f:f(),[announcements,archives,news]))
    log('PROBE_DONE')
