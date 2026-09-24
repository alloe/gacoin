#!/usr/bin/env python3
"""
Phase 18: live 100-coin L1 Event Sniper collector + paper replay.
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
ENTRY_LAGS_MS=[int(x) for x in os.getenv('PAPER_ENTRY_LAGS_MS','0,100,200,500,1000').split(',')]
HOLD_MS=[int(x) for x in os.getenv('PAPER_HOLD_MS','500,1000,2000,5000,10000').split(',')]
SIZES_KRW=[int(x) for x in os.getenv('PAPER_SIZES_KRW','1000000,3000000,5000000,10000000').split(',')]
UP_FEE_BP=float(os.getenv('PAPER_UP_FEE_BP','5'))
BN_FEE_BP=float(os.getenv('PAPER_BN_FEE_BP','5'))
LAG_GATE_BP=float(os.getenv('PAPER_LAG_GATE_BP','10'))
EDGE_GATES_BP=[float(x) for x in os.getenv('PAPER_EDGE_GATES_BP','0,10,20').split(',')]
PAPER_CHUNK_ROWS=int(os.getenv('PAPER_CHUNK_ROWS','25'))
MIN_CANDIDATE_KRW=int(os.getenv('PAPER_MIN_CANDIDATE_KRW','1000000'))
TARGET_RESID_BP=float(os.getenv('PAPER_TARGET_RESID_BP','5'))
STOP_WORSEN_BP=float(os.getenv('PAPER_STOP_WORSEN_BP','30'))
DYNAMIC_TIMEOUT_MS=int(os.getenv('PAPER_DYNAMIC_TIMEOUT_MS','10000'))
R=requests.Session();R.headers['User-Agent']='daol-event-sniper-paper/18'

quotes_up={};quotes_bn={};fx=None
hist=defaultdict(lambda:deque(maxlen=max(20,int(PRE_MS/SAMPLE_MS)+20)))
active={}
cooldown={}
events_total=0
samples_total=0
paper_events=0
paper_stats=defaultdict(lambda:{'n':0,'full':0,'wins':0,'sum_bp':0.0,'sum_pnl':0.0,'sum_fill':0.0})
dynamic_stats=defaultdict(lambda:{'n':0,'full':0,'wins':0,'sum_bp':0.0,'sum_pnl':0.0,'sum_fill':0.0,
                                 'targets':0,'stops':0,'timeouts':0})
start_mono=time.monotonic()
universe=[]
bn_symbol={}
bn_mult={}

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
        c=m[4:]
        opts=[(c+'USDT',1.0),('1000'+c+'USDT',1000.0),('1000000'+c+'USDT',1000000.0)]
        found=[x for x in opts if x[0] in futures]
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
    return chosen,{c:mp[c][0] for c in chosen},{c:mp[c][1] for c in chosen}

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


def row_at(rows,target_ms,max_after_ms=1200):
    for r in rows:
        if r[0]>=target_ms and r[0]-target_ms<=max_after_ms:return r
    return None

def selected_shock(ev):
    if ev['reason']=='500ms':
        return ev.get('r500'),14
    return ev.get('r1000'),15

def trigger_lag(ev,rows):
    tr=row_at(rows,ev['trigger_ms'],300)
    if not tr:return None,None,None
    shock,up_ix=selected_shock(ev)
    upret=tr[up_ix] if up_ix<len(tr) else None
    if shock is None or upret is None:return shock,upret,None
    return float(shock),float(upret),float(shock-upret)

def entry_diag(r,side):
    # Common premium = observed mid premium - residual.
    if not r or any(r[i] is None for i in (1,2,5,6,9,10,11)):return None
    common=float(r[10])-float(r[11]);fxv=float(r[9])
    if not (fxv>0):return None
    if side==1:
        num=float(r[2]);den=float(r[5])*fxv
    else:
        num=float(r[1]);den=float(r[6])*fxv
    if not (num>0 and den>0):return None
    er=math.log(num/den)*10000-common
    gross=-er if side==1 else er
    um=(float(r[1])+float(r[2]))/2;bm=(float(r[5])+float(r[6]))/2
    if not (um>0 and bm>0):return None
    uhalf=(float(r[2])-float(r[1]))/um*5000
    bhalf=(float(r[6])-float(r[5]))/bm*5000
    est=gross-(2*UP_FEE_BP+2*BN_FEE_BP)-uhalf-bhalf
    return {'exec_residual_bp':er,'gross_convergence_bp':gross,
            'up_halfspread_bp':uhalf,'bn_halfspread_bp':bhalf,'est_net_bp':est}

def entry_capacity_krw(r,side):
    if not r or any(r[i] is None for i in (1,2,3,4,5,6,7,8,9)):return None
    fxv=float(r[9])
    if fxv<=0:return None
    if side==1:
        return min(float(r[4])*float(r[2]),float(r[7])*float(r[5])*fxv)
    return min(float(r[3])*float(r[1]),float(r[8])*float(r[6])*fxv)

def dynamic_exit(rows,entry_row,side):
    d0=entry_diag(entry_row,side)
    if not d0:return None,None
    er0=d0['exec_residual_bp'];deadline=entry_row[0]+DYNAMIC_TIMEOUT_MS
    last=None
    for r in rows:
        if r[0]<=entry_row[0]:continue
        if r[0]>deadline:break
        last=r;d=entry_diag(r,side)
        if not d:continue
        er=d['exec_residual_bp']
        if side==1:
            if er>=-TARGET_RESID_BP:return r,'target'
            if er<=er0-STOP_WORSEN_BP:return r,'stop'
        else:
            if er<=TARGET_RESID_BP:return r,'target'
            if er>=er0+STOP_WORSEN_BP:return r,'stop'
    return last,'timeout' if last else (None,None)

def replay_cell(er,xr,side,requested_krw):
    if not er or not xr:return None
    need=(1,2,3,4,5,6,7,8,9)
    if any(er[i] is None for i in need) or any(xr[i] is None for i in need):return None
    uf=UP_FEE_BP/10000;bf=BN_FEE_BP/10000
    exfx=float(er[9]);xxfx=float(xr[9])
    if side==1:
        up_e=float(er[2]);up_x=float(xr[1]);bn_e=float(er[5]);bn_x=float(xr[6])
        caps=[float(er[4]),float(er[7]),float(xr[3]),float(xr[8])]
        if not all(v>0 and math.isfinite(v) for v in (up_e,up_x,bn_e,bn_x,exfx,xxfx,*caps)):return None
        tq=requested_krw/up_e;q=min(tq,*caps)
        spot=q*(up_x-up_e)-q*up_e*uf-q*up_x*uf
        fut_usdt=q*(bn_e-bn_x)-q*bn_e*bf-q*bn_x*bf
    else:
        # Inventory-assisted route: sell existing Upbit coin + long Binance perp,
        # later buy spot inventory back. Not a naked Upbit short.
        up_e=float(er[1]);up_x=float(xr[2]);bn_e=float(er[6]);bn_x=float(xr[5])
        caps=[float(er[3]),float(er[8]),float(xr[4]),float(xr[7])]
        if not all(v>0 and math.isfinite(v) for v in (up_e,up_x,bn_e,bn_x,exfx,xxfx,*caps)):return None
        tq=requested_krw/up_e;q=min(tq,*caps)
        spot=q*(up_e-up_x)-q*up_e*uf-q*up_x*uf
        fut_usdt=q*(bn_x-bn_e)-q*bn_e*bf-q*bn_x*bf
    filled=q*up_e
    if filled<=0:return None
    pnl=spot+fut_usdt*xxfx
    return {'filled_krw':filled,'fill_ratio':min(1.0,filled/requested_krw),
            'pnl_krw':pnl,'net_bp':pnl/filled*10000,'qty':q}

def evaluate_event(ev):
    global paper_events
    rows=ev['rows'];shock,upret,lag=trigger_lag(ev,rows)
    if shock is None:return
    side=1 if shock>0 else -1
    route='long_upbit_short_binance' if side==1 else 'inventory_sell_upbit_long_binance'
    cells=[];dynamic=[];eligible_counts={str(int(g)):0 for g in EDGE_GATES_BP}
    candidate_lags=[]
    for entry_lag in ENTRY_LAGS_MS:
        er=row_at(rows,ev['trigger_ms']+entry_lag)
        diag=entry_diag(er,side);entry_cap=entry_capacity_krw(er,side)
        if not er or not diag or entry_cap is None:continue
        lag_ok=lag is not None and ((side==1 and lag>=LAG_GATE_BP) or (side==-1 and lag<=-LAG_GATE_BP))
        if side==1 and lag_ok and diag['est_net_bp']>=0 and entry_cap>=MIN_CANDIDATE_KRW:
            candidate_lags.append({'entry_lag_ms':entry_lag,'est_net_bp':diag['est_net_bp'],
                                   'entry_capacity_krw':entry_cap,'exec_residual_bp':diag['exec_residual_bp']})
        for hold in HOLD_MS:
            xr=row_at(rows,er[0]+hold)
            for amount in SIZES_KRW:
                rr=replay_cell(er,xr,side,amount)
                if not rr:continue
                row=[entry_lag,hold,amount,round(rr['filled_krw'],2),round(rr['fill_ratio'],6),
                     round(rr['pnl_krw'],2),round(rr['net_bp'],4),round(diag['est_net_bp'],4),
                     round(diag['exec_residual_bp'],4),bool(lag_ok)]
                cells.append(row)
                if amount==SIZES_KRW[0] and rr['fill_ratio']>=.95 and side==1 and lag_ok:
                    for gate in EDGE_GATES_BP:
                        if diag['est_net_bp']>=gate:
                            key=(gate,entry_lag,hold);s=paper_stats[key]
                            s['n']+=1;s['full']+=1;s['wins']+=rr['net_bp']>0
                            s['sum_bp']+=rr['net_bp'];s['sum_pnl']+=rr['pnl_krw'];s['sum_fill']+=rr['filled_krw']
                            eligible_counts[str(int(gate))]+=1
        xr_dyn,reason=dynamic_exit(rows,er,side)
        if xr_dyn:
            for amount in SIZES_KRW:
                rr=replay_cell(er,xr_dyn,side,amount)
                if not rr:continue
                dynamic.append([entry_lag,xr_dyn[0]-er[0],amount,round(rr['filled_krw'],2),round(rr['fill_ratio'],6),
                                round(rr['pnl_krw'],2),round(rr['net_bp'],4),round(diag['est_net_bp'],4),
                                round(diag['exec_residual_bp'],4),reason,bool(lag_ok)])
                if amount==SIZES_KRW[0] and rr['fill_ratio']>=.95 and side==1 and lag_ok:
                    for gate in EDGE_GATES_BP:
                        if diag['est_net_bp']>=gate and entry_cap>=MIN_CANDIDATE_KRW:
                            key=(gate,entry_lag);s=dynamic_stats[key]
                            s['n']+=1;s['full']+=1;s['wins']+=rr['net_bp']>0
                            s['sum_bp']+=rr['net_bp'];s['sum_pnl']+=rr['pnl_krw'];s['sum_fill']+=rr['filled_krw']
                            s[reason+'s']+=1
    paper_events+=1
    print('PAPER_EVENT_META '+json.dumps({
      'event_id':ev['id'],'coin':ev['coin'],'route':route,'shock_bp':shock,'up_return_bp':upret,'lag_bp':lag,
      'lag_gate_bp':LAG_GATE_BP,'primary_feasible':side==1,
      'fees_bp_per_execution':{'upbit':UP_FEE_BP,'binance_futures':BN_FEE_BP},
      'entry_lags_ms':ENTRY_LAGS_MS,'hold_ms':HOLD_MS,'sizes_krw':SIZES_KRW,
      'eligible_cell_counts_1m':eligible_counts,'candidate_entry_lags':candidate_lags,
      'min_candidate_krw':MIN_CANDIDATE_KRW,'dynamic_rule':{'target_residual_bp':TARGET_RESID_BP,
        'stop_worsen_bp':STOP_WORSEN_BP,'timeout_ms':DYNAMIC_TIMEOUT_MS},
      'cell_schema':['entry_lag_ms','hold_ms','requested_krw','filled_krw','fill_ratio','pnl_krw','net_bp',
                     'entry_est_net_bp','entry_exec_residual_bp','lag_ok'],
      'cells':len(cells),'dynamic_cells':len(dynamic)},separators=(',',':')),flush=True)
    for i in range(0,len(cells),PAPER_CHUNK_ROWS):
        print('PAPER_EVENT_CHUNK '+json.dumps({'event_id':ev['id'],'chunk':i//PAPER_CHUNK_ROWS,
          'rows':cells[i:i+PAPER_CHUNK_ROWS]},separators=(',',':')),flush=True)
    for i in range(0,len(dynamic),PAPER_CHUNK_ROWS):
        print('PAPER_DYNAMIC_CHUNK '+json.dumps({'event_id':ev['id'],'chunk':i//PAPER_CHUNK_ROWS,
          'schema':['entry_lag_ms','elapsed_ms','requested_krw','filled_krw','fill_ratio','pnl_krw','net_bp',
                    'entry_est_net_bp','entry_exec_residual_bp','exit_reason','lag_ok'],
          'rows':dynamic[i:i+PAPER_CHUNK_ROWS]},separators=(',',':')),flush=True)

def emit_paper_summary():
    rows=[]
    for (gate,lag,hold),s in sorted(paper_stats.items()):
        if not s['n']:continue
        rows.append({'edge_gate_bp':gate,'entry_lag_ms':lag,'hold_ms':hold,'n':s['n'],
          'win_rate':s['wins']/s['n'],'mean_bp':s['sum_bp']/s['n'],
          'sum_pnl_krw_1m':s['sum_pnl'],'mean_fill_krw':s['sum_fill']/s['n']})
    dyn=[]
    for (gate,lag),s in sorted(dynamic_stats.items()):
        if not s['n']:continue
        dyn.append({'edge_gate_bp':gate,'entry_lag_ms':lag,'n':s['n'],'win_rate':s['wins']/s['n'],
          'mean_bp':s['sum_bp']/s['n'],'sum_pnl_krw_1m':s['sum_pnl'],'mean_fill_krw':s['sum_fill']/s['n'],
          'targets':s['targets'],'stops':s['stops'],'timeouts':s['timeouts']})
    print('PAPER_SUMMARY '+json.dumps({'paper_events':paper_events,'fixed_primary_cells':rows,
      'dynamic_primary_cells':dyn},separators=(',',':')),flush=True)

def emit_event(ev):
    global events_total
    events_total+=1
    rows=ev['rows']
    evaluate_event(ev)
    print('EVENT_META '+json.dumps({
      'event_id':ev['id'],'coin':ev['coin'],'trigger_wall_ns':ev['trigger_wall_ns'],
      'trigger_mono_ns':ev['trigger_mono_ns'],'trigger_reason':ev['reason'],
      'trigger_ret500_bp':ev['r500'],'trigger_ret1000_bp':ev['r1000'],
      'pre_ms':PRE_MS,'post_ms':POST_MS,'sample_ms':SAMPLE_MS,'rows':len(rows),
      'binance_symbol':bn_symbol.get(ev['coin']),'binance_multiplier':bn_mult.get(ev['coin'],1.0),
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
                shock_now=bn500 if reason=='500ms' else bn1000
                up_now=up500 if reason=='500ms' else up1000
                side_now=1 if (shock_now or 0)>0 else -1
                lag_now=(shock_now-up_now) if shock_now is not None and up_now is not None else None
                diag_now=entry_diag(row,side_now);cap_now=entry_capacity_krw(row,side_now)
                lag_ok_now=lag_now is not None and ((side_now==1 and lag_now>=LAG_GATE_BP) or (side_now==-1 and lag_now<=-LAG_GATE_BP))
                candidate_now=bool(side_now==1 and lag_ok_now and diag_now and cap_now is not None and
                                   diag_now['est_net_bp']>=0 and cap_now>=MIN_CANDIDATE_KRW)
                active[c]={'id':ev_id,'coin':c,'trigger_wall_ns':wall,'trigger_mono_ns':n,'trigger_ms':tms,
                           'reason':reason,'r500':bn500,'r1000':bn1000,'rows':list(h)}
                cooldown[c]=tms+COOLDOWN_MS
                payload={'event_id':ev_id,'coin':c,'reason':reason,'bn500_bp':bn500,'bn1000_bp':bn1000,
                         'up_return_bp':up_now,'lag_bp':lag_now,'premium_bp':p,
                         'residual_bp':p-common if common is not None else None,
                         'entry_est_net_bp':diag_now['est_net_bp'] if diag_now else None,
                         'entry_capacity_krw':cap_now,'trade_candidate':candidate_now}
                print('EVENT_TRIGGER '+json.dumps(payload,separators=(',',':')),flush=True)
                if candidate_now:
                    print('TRADE_CANDIDATE '+json.dumps(payload,separators=(',',':')),flush=True)
            if c in active:
                ev=active[c]
                if not ev['rows'] or ev['rows'][-1][0]!=row[0]:ev['rows'].append(row)
                if tms-ev['trigger_ms']>=POST_MS:
                    emit_event(ev);del active[c]
        if time.monotonic()-last_summary>=SUMMARY_SEC:
            print('COLLECTOR_SUMMARY '+json.dumps({
              'uptime_sec':time.monotonic()-start_mono,'universe':len(universe),'aligned_now':len(snap),
              'events_total':events_total,'paper_events':paper_events,'active_events':len(active),'samples_total':samples_total,
              'fx_fresh':bool(fxm),'fx_max_stale_ms':FX_MAX_STALE_MS,'trigger_bp':TRIGGER_BP},separators=(',',':')),flush=True)
            emit_paper_summary()
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
                    if c:
                        mult=bn_mult.get(c,1.0)
                        qput(quotes_bn,c,float(d['b'])/mult,float(d['B'])*mult,
                             float(d['a'])/mult,float(d['A'])*mult,d.get('E') or d.get('T'))
        except Exception as e:
            print('COLLECTOR_WS_ERR binance '+repr(e),flush=True);await asyncio.sleep(1)

async def main():
    global universe,bn_symbol,bn_mult
    universe,bn_symbol,bn_mult=await asyncio.to_thread(choose_universe)
    print('EVENT_COLLECTOR_START '+json.dumps({'mode':'public_l1_no_orders','n':len(universe),'coins':universe,
      'sample_ms':SAMPLE_MS,'pre_ms':PRE_MS,'post_ms':POST_MS,'trigger_bp':TRIGGER_BP,
      'storage':'Railway logs EVENT_META/EVENT_CHUNK + PAPER_EVENT_*','paper_entry_lags_ms':ENTRY_LAGS_MS,
      'paper_hold_ms':HOLD_MS,'paper_sizes_krw':SIZES_KRW,'paper_edge_gates_bp':EDGE_GATES_BP,
      'paper_min_candidate_krw':MIN_CANDIDATE_KRW,'paper_dynamic_timeout_ms':DYNAMIC_TIMEOUT_MS},separators=(',',':')),flush=True)
    await asyncio.gather(upbit_ws(),binance_ws(),sampler())

if __name__=='__main__':asyncio.run(main())
