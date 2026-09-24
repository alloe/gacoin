#!/usr/bin/env python3
"""
Phase 17: live 100-coin L1 event collector for future Event Sniper replay.
Public market data only; no account/order/private API code.

Storage model:
- Sample aligned Upbit/Binance L1 every 100ms in RAM.
- Keep a rolling pre-event buffer.
- On >=30bp Binance shock over 500ms or 1s, preserve ~10s before + 30s after.
- Emit compact EVENT_META / EVENT_CHUNK JSON to Railway logs so event windows survive container exit.
"""
from __future__ import annotations
import asyncio,json,math,os,time
from collections import defaultdict,deque
import requests, websockets, numpy as np

UP_WS="wss://api.upbit.com/websocket/v1"
BN_WS="wss://fstream.binance.com/ws"
UP_API="https://api.upbit.com"
BN_API="https://fapi.binance.com"
N=int(os.getenv('EVENT_UNIVERSE','100'))
SAMPLE_MS=int(os.getenv('EVENT_SAMPLE_MS','100'))
PRE_MS=int(os.getenv('EVENT_PRE_MS','10000'))
POST_MS=int(os.getenv('EVENT_POST_MS','30000'))
TRIGGER_BP=float(os.getenv('EVENT_TRIGGER_BP','30'))
COOLDOWN_MS=int(os.getenv('EVENT_COOLDOWN_MS','30000'))
MAX_STALE_MS=float(os.getenv('EVENT_MAX_STALE_MS','750'))
FX_MAX_STALE_MS=float(os.getenv('EVENT_FX_MAX_STALE_MS','10000'))
SUMMARY_SEC=int(os.getenv('EVENT_SUMMARY_SEC','60'))
CHUNK_ROWS=int(os.getenv('EVENT_CHUNK_ROWS','40'))
R=requests.Session();R.headers['User-Agent']='daol-event-sniper-collector/17'

quotes_up={};quotes_bn={};fx=None
hist=defaultdict(lambda:deque(maxlen=max(20,int(PRE_MS/SAMPLE_MS)+20)))
active={}
cooldown={}
events_total=0
samples_total=0
start_mono=time.monotonic()
universe=[]
bn_symbol={}

def now_wall_ns():return time.time_ns()
def now_mono_ns():return time.monotonic_ns()

def get_json(url,params=None):
    for a in range(5):
        try:r=R.get(url,params=params,timeout=(5,20))
        except Exception:
            if a==4:raise
            time.sleep(2**a);continue
        if r.status_code in (429,500,502,503,504):
            time.sleep(min(10,2**a));continue
        r.raise_for_status();return r.json()
    raise RuntimeError('http retries exhausted')

def choose_universe():
    markets=get_json(UP_API+'/v1/market/all',{'is_details':'false'})
    krw=[x['market'] for x in markets if x['market'].startswith('KRW-') and x['market']!='KRW-USDT']
    ex=get_json(BN_API+'/fapi/v1/exchangeInfo')
    futures={x['symbol'] for x in ex['symbols'] if x.get('contractType')=='PERPETUAL' and x.get('quoteAsset')=='USDT' and x.get('status')=='TRADING'}
    mp={}
    for m in krw:
        c=m[4:];opts=[c+'USDT','1000'+c+'USDT','1000000'+c+'USDT']
        found=[s for s in opts if s in futures]
        if len(found)==1:mp[c]=found[0]
    # Upbit ticker batches to avoid long URLs
    tick=[]
    codes=['KRW-'+c for c in mp]
    for i in range(0,len(codes),80):
        tick.extend(get_json(UP_API+'/v1/ticker',{'markets':','.join(codes[i:i+80])}))
        time.sleep(.12)
    tick.sort(key=lambda x:float(x.get('acc_trade_price_24h') or 0),reverse=True)
    chosen=[]
    for x in tick:
        c=x['market'][4:]
        if c in mp:chosen.append(c)
        if len(chosen)>=N:break
    if len(chosen)<20:raise RuntimeError('too few matched live markets')
    return chosen,{c:mp[c] for c in chosen}

def qput(store,c,bid,bidsz,ask,asksz,src_ts=None):
    try:b=float(bid);bs=float(bidsz);a=float(ask);az=float(asksz)
    except:return
    if not (b>0 and a>b and bs>=0 and az>=0):return
    store[c]={'bid':b,'bidsz':bs,'ask':a,'asksz':az,'src_ts':src_ts,
              'wall_ns':now_wall_ns(),'mono_ns':now_mono_ns()}

def mid(q):return (q['bid']+q['ask'])/2

def fresh(q,n,max_age_ms=MAX_STALE_MS):
    return q is not None and (n-q['mono_ns'])/1e6<=max_age_ms

def past_row(c,delta_ms):
    h=hist[c]
    if not h:return None
    target=h[-1][0]-delta_ms
    for row in reversed(h):
        if row[0]<=target:return row
    return None

def retbp(a,b):
    if a and b and a>0 and b>0:return math.log(a/b)*10000
    return None

def emit_event(ev):
    global events_total
    events_total+=1
    rows=ev['rows']
    print('EVENT_META '+json.dumps({
      'event_id':ev['id'],'coin':ev['coin'],'trigger_wall_ns':ev['trigger_wall_ns'],
      'trigger_mono_ns':ev['trigger_mono_ns'],'trigger_reason':ev['reason'],
      'trigger_ret500_bp':ev['r500'],'trigger_ret1000_bp':ev['r1000'],
      'pre_ms':PRE_MS,'post_ms':POST_MS,'sample_ms':SAMPLE_MS,'rows':len(rows),
      'binance_symbol':bn_symbol.get(ev['coin']),
      'schema':['dt_ms','up_bid','up_ask','up_bidsz','up_asksz','bn_bid','bn_ask','bn_bidsz','bn_asksz',
                'fx_mid','premium_bp','residual_bp','bn_ret500_bp','bn_ret1000_bp','up_ret500_bp','up_ret1000_bp',
                'up_src_ts','bn_src_ts','up_age_ms','bn_age_ms']},separators=(',',':')),flush=True)
    for i in range(0,len(rows),CHUNK_ROWS):
        print('EVENT_CHUNK '+json.dumps({'event_id':ev['id'],'chunk':i//CHUNK_ROWS,'rows':rows[i:i+CHUNK_ROWS]},
              separators=(',',':'),allow_nan=False),flush=True)
    print('EVENT_END '+json.dumps({'event_id':ev['id'],'chunks':math.ceil(len(rows)/CHUNK_ROWS)},separators=(',',':')),flush=True)

async def sampler():
    global fx,samples_total
    last_summary=time.monotonic()
    while True:
        tick_start=time.monotonic()
        n=now_mono_ns();wall=now_wall_ns()
        fq=quotes_up.get('USDT')
        if fresh(fq,n,FX_MAX_STALE_MS):fx=fq
        fxm=mid(fx) if fresh(fx,n,FX_MAX_STALE_MS) else None
        # common premium from all fresh aligned markets
        prems=[];snap={}
        if fxm:
            for c in universe:
                u=quotes_up.get(c);b=quotes_bn.get(c)
                if fresh(u,n) and fresh(b,n):
                    um=mid(u);bm=mid(b)
                    if um>0 and bm>0:
                        p=math.log(um/(bm*fxm))*10000
                        prems.append(p);snap[c]=(u,b,um,bm,p)
        common=float(np.median(prems)) if prems else None
        for c,(u,b,um,bm,p) in snap.items():
            h=hist[c]
            p500=None;p1000=None
            if h:
                target500=wall//1_000_000-500;target1000=wall//1_000_000-1000
                r500=next((r for r in reversed(h) if r[0]<=target500),None)
                r1000=next((r for r in reversed(h) if r[0]<=target1000),None)
            else:r500=r1000=None
            bn500=retbp(bm,r500[6] if r500 else None);bn1000=retbp(bm,r1000[6] if r1000 else None)
            up500=retbp(um,(r500[1]+r500[2])/2 if r500 else None);up1000=retbp(um,(r1000[1]+r1000[2])/2 if r1000 else None)
            tms=wall//1_000_000
            row=[tms,u['bid'],u['ask'],u['bidsz'],u['asksz'],b['bid'],b['ask'],b['bidsz'],b['asksz'],
                 fxm,p,p-common if common is not None else None,bn500,bn1000,up500,up1000,
                 u.get('src_ts'),b.get('src_ts'),(n-u['mono_ns'])/1e6,(n-b['mono_ns'])/1e6]
            # Ensure log-safe finite numeric values.
            row=[None if isinstance(v,float) and not math.isfinite(v) else v for v in row]
            h.append(row);samples_total+=1
            shock=max(abs(bn500 or 0),abs(bn1000 or 0))
            if shock>=TRIGGER_BP and c not in active and tms>=cooldown.get(c,0):
                reason='500ms' if abs(bn500 or 0)>=abs(bn1000 or 0) else '1000ms'
                ev_id=f"{tms}-{c}"
                active[c]={'id':ev_id,'coin':c,'trigger_wall_ns':wall,'trigger_mono_ns':n,'trigger_ms':tms,
                           'reason':reason,'r500':bn500,'r1000':bn1000,'rows':list(h)}
                cooldown[c]=tms+COOLDOWN_MS
                print('EVENT_TRIGGER '+json.dumps({'event_id':ev_id,'coin':c,'reason':reason,'bn500_bp':bn500,
                    'bn1000_bp':bn1000,'premium_bp':p,'residual_bp':p-common if common is not None else None},separators=(',',':')),flush=True)
            if c in active:
                ev=active[c]
                if not ev['rows'] or ev['rows'][-1][0]!=row[0]:ev['rows'].append(row)
                if tms-ev['trigger_ms']>=POST_MS:
                    emit_event(ev);del active[c]
        if time.monotonic()-last_summary>=SUMMARY_SEC:
            print('COLLECTOR_SUMMARY '+json.dumps({
              'uptime_sec':time.monotonic()-start_mono,'universe':len(universe),'aligned_now':len(snap),
              'events_total':events_total,'active_events':len(active),'samples_total':samples_total,
              'fx_fresh':bool(fxm),'fx_max_stale_ms':FX_MAX_STALE_MS,'trigger_bp':TRIGGER_BP},separators=(',',':')),flush=True)
            last_summary=time.monotonic()
        elapsed=(time.monotonic()-tick_start)*1000
        await asyncio.sleep(max(.001,(SAMPLE_MS-elapsed)/1000))

async def upbit_ws():
    codes=['KRW-USDT.1']+['KRW-'+c+'.1' for c in universe]
    sub=[{'ticket':'event-sniper-17'},{'type':'orderbook','codes':codes,'is_only_realtime':True},{'format':'DEFAULT'}]
    while True:
        try:
            async with websockets.connect(UP_WS,ping_interval=20,ping_timeout=20,max_size=2**22) as ws:
                await ws.send(json.dumps(sub));print('COLLECTOR_CONNECTED upbit',flush=True)
                async for raw in ws:
                    d=json.loads(raw);units=d.get('orderbook_units') or []
                    if not units:continue
                    code=d.get('code','').split('.')[0];c=code[4:] if code.startswith('KRW-') else code
                    q=units[0];qput(quotes_up,c,q['bid_price'],q['bid_size'],q['ask_price'],q['ask_size'],d.get('timestamp'))
        except Exception as e:
            print('COLLECTOR_WS_ERR upbit '+repr(e),flush=True);await asyncio.sleep(1)

async def binance_ws():
    params=[bn_symbol[c].lower()+'@bookTicker' for c in universe]
    sub={'method':'SUBSCRIBE','params':params,'id':17}
    while True:
        try:
            async with websockets.connect(BN_WS,ping_interval=20,ping_timeout=20,max_size=2**22) as ws:
                await ws.send(json.dumps(sub));print('COLLECTOR_CONNECTED binance',flush=True)
                async for raw in ws:
                    d=json.loads(raw)
                    if not all(k in d for k in ('s','b','B','a','A')):continue
                    sym=d['s'];c=next((x for x in universe if bn_symbol[x]==sym),None)
                    if c:qput(quotes_bn,c,d['b'],d['B'],d['a'],d['A'],d.get('E') or d.get('T'))
        except Exception as e:
            print('COLLECTOR_WS_ERR binance '+repr(e),flush=True);await asyncio.sleep(1)

async def main():
    global universe,bn_symbol
    universe,bn_symbol=await asyncio.to_thread(choose_universe)
    print('EVENT_COLLECTOR_START '+json.dumps({'mode':'public_l1_no_orders','n':len(universe),'coins':universe,
      'sample_ms':SAMPLE_MS,'pre_ms':PRE_MS,'post_ms':POST_MS,'trigger_bp':TRIGGER_BP,
      'storage':'Railway logs EVENT_META/EVENT_CHUNK'},separators=(',',':')),flush=True)
    await asyncio.gather(upbit_ws(),binance_ws(),sampler())

if __name__=='__main__':asyncio.run(main())
