"""Frozen 23.2 calendar-response SPOT research. Pure calculations; no APIs/orders.
No news sentiment is inferred from subsequent market prices. Full news/macro model
must remain blocked until point-in-time consensus/body and macro inputs exist.
"""
from dataclasses import dataclass, asdict
from collections import Counter
import math
import numpy as np
import pandas as pd

MIN=60000
@dataclass(frozen=True)
class Config:
    fee_bp:float=10.0
    slip_bp:float=2.0
    min_drop:float=.008
    min_rise:float=.005
    sigma_multiple:float=2.5
    buy_share:float=.55
    sell_fraction:float=.60
    stable_bars:int=3
    min_stop:float=.0035
    max_stop:float=.015
    rr:float=1.5
    entry_cap_bp:float=5.0
    lag_min:int=1
    expiry_min:int=60
    max_hold:int=60
    risk:float=.002
    max_fraction:float=.20
    turnover_fraction:float=.001

CFG=Config()
VARIANTS=('B0_shock','B1_price','B2_price_flow','A_pullback_flow')
def cost_pnl(entry,exit,fee_bp=10,slip_bp=2):
    paid=entry*(1+slip_bp/10000)
    received=exit*(1-slip_bp/10000)
    f=fee_bp/10000
    return received*(1-f)-paid*(1+f)

def valid_frame(d):
    cols=['open','high','low','close','quote_volume','buy_quote']
    if d is None or len(d)==0 or any(c not in d for c in cols):return False
    a=d[cols].to_numpy(float)
    if not np.isfinite(a).all() or (a[:,:4]<=0).any():return False
    return bool((d.low<=d[['open','close']].min(axis=1)).all() and
        (d.high>=d[['open','close']].max(axis=1)).all() and
        (d.buy_quote>=0).all() and (d.buy_quote<=d.quote_volume+1e-6).all())

def observe(d,t,k,variant,cfg=CFG):
    """Only bars whose exclusive close <= decision t+k minutes are used."""
    now=t+k*MIN
    if k<6 or k>cfg.expiry_min:return None,'outside_window'
    ix=np.arange(t-60*MIN,now,MIN)
    z=d.reindex(ix)
    if not valid_frame(z):return None,'data_gap_or_invalid'
    pre=z.iloc[:60];post=z.iloc[60:];p0=float(pre.close.iloc[-1]);p=float(post.close.iloc[-1])
    sig=float(np.std(np.diff(np.log(pre.close)),ddof=1)*np.sqrt(5))
    tr=np.maximum(pre.high.to_numpy()[1:]-pre.low.to_numpy()[1:],np.maximum(abs(pre.high.to_numpy()[1:]-pre.close.to_numpy()[:-1]),abs(pre.low.to_numpy()[1:]-pre.close.to_numpy()[:-1])))
    atr=float(np.mean(tr));recent=post.iloc[-3:];volume=float(recent.quote_volume.sum())
    buy=float(recent.buy_quote.sum()/volume) if volume>0 else 0.
    sell0=float((post.quote_volume-post.buy_quote).iloc[:3].sum())
    sell=float((recent.quote_volume-recent.buy_quote).sum())
    sr=sell/sell0 if sell0>0 else math.inf
    is_a=variant=='A_pullback_flow';local=post.iloc[5:] if is_a else post
    if len(local)==0:return None,'no_local_bars'
    low=float(local.low.min());drop=(p0-low)/p0
    if is_a:
        peak=float(post.high.iloc[:5].max());impulse=peak/p0-1
        if impulse<max(cfg.min_rise,cfg.sigma_multiple*sig):return None,'rise_too_small'
        retrace=(peak-low)/(peak-p0)
        if not .2<=retrace<=.5:return None,'pullback_range'
    else:
        if drop<max(cfg.min_drop,cfg.sigma_multiple*sig):return None,'drop_too_small'
    if variant!='B0_shock':
        lastlow=int(np.flatnonzero(local.low.to_numpy()==low)[-1])
        if len(local)-1-lastlow<cfg.stable_bars:return None,'low_not_stable'
        if p<=float(post.high.iloc[-4:-1].max()):return None,'no_breakout'
    if variant in ('B2_price_flow','A_pullback_flow'):
        if buy<cfg.buy_share:return None,'buy_flow_weak'
        if not is_a and sr>cfg.sell_fraction:return None,'selling_not_eased'
    limit=p*(1+cfg.entry_cap_bp/10000);stop=low-.25*atr;risk=limit-stop
    if not cfg.min_stop<=risk/limit<=cfg.max_stop:return None,'risk_distance'
    target=limit+2*risk
    if not is_a:target=min(target,p0)
    gain=target*(1-cfg.slip_bp/10000)*(1-cfg.fee_bp/10000)-limit*(1+cfg.fee_bp/10000)
    loss=limit*(1+cfg.fee_bp/10000)-stop*(1-cfg.slip_bp/10000)*(1-cfg.fee_bp/10000)
    if gain<=0 or loss<=0 or gain/loss<cfg.rr:return None,'net_reward_risk'
    return {'decision':int(now),'limit':limit,'stop':stop,'target':target,'p0':p0,'signal_price':p,
        'low':low,'drop_bp':drop*10000,'buy_share':buy,'sell_fraction':sr,'rr':gain/loss,
        'planned_loss_per_unit':loss,'prior_turnover':float(post.quote_volume.iloc[-1])},'candidate'

def first_signal(d,t,variant,blackout=None,macro_gate=None,cfg=CFG):
    reasons=Counter()
    for k in range(6,cfg.expiry_min+1):
        now=t+k*MIN
        if blackout is not None and blackout-now<=10*MIN:
            reasons['next_clock_boundary']+=1;break
        c,reason=observe(d,t,k,variant,cfg);reasons[reason]+=1
        if c is not None:
            if macro_gate is not None:
                ok,why=macro_gate(now,t,variant);reasons[why]+=1
                if not ok:continue
            return c,dict(reasons)
    return None,dict(reasons)

def evaluate(d,c,blackout=None,cfg=CFG):
    """Single IOC-price-cap proxy, entry only from predetermined exact arrival open.
    Bar quantity/depth not observed: fills here are model assumptions, not fills.
    No later quantity or exit data influence requested quantity. Missing bars are
    unresolved, and a bar with both barriers is stop-first.
    """
    e=c['decision']+cfg.lag_min*MIN
    if e not in d.index or not valid_frame(d.loc[[e]]):return {'status':'unresolved_entry','arrival':e}
    ep=float(d.loc[e,'open']);paid=ep*(1+cfg.slip_bp/10000)
    if paid>c['limit']:return {'status':'no_fill_price_cap','arrival':e}
    if paid<=c['stop'] or paid>=c['target']:return {'status':'no_fill_invalidated','arrival':e}
    deadline=e+cfg.max_hold*MIN
    if blackout is not None:deadline=min(deadline,blackout-MIN)
    if deadline<=e:return {'status':'no_fill_clock_boundary','arrival':e}
    mae=0.;mfe=0.;lows=[]
    for ts in range(e,deadline+1,MIN):
        if ts not in d.index or not valid_frame(d.loc[[ts]]):
            return {'status':'unresolved_open_position','entry_time':e,'entry_raw':ep,'entry_paid':paid,'missing_time':ts}
        b=d.loc[ts];amb=False
        if ts==deadline:
            xp=float(b.open);reason='clock_exit' if blackout is not None and deadline<e+cfg.max_hold*MIN else 'timeout'
        elif b.open<=c['stop']:xp=float(b.open);reason='gap_stop'
        else:
            hitstop=bool(b.low<=c['stop']);hittarget=bool(b.high>=c['target'])
            if hitstop:xp=c['stop'];reason='stop';amb=hittarget
            elif hittarget:xp=c['target'];reason='target'
            else:
                mae=min(mae,float(b.low)/paid-1);mfe=max(mfe,float(b.high)/paid-1)
                lows.append((int(ts),float(b.low)));continue
        mae=min(mae,xp/paid-1);mfe=max(mfe,xp/paid-1);lows.append((int(ts),xp))
        received=xp*(1-cfg.slip_bp/10000);pnl=received*(1-cfg.fee_bp/10000)-paid*(1+cfg.fee_bp/10000)
        return {'status':'modeled_closed','entry_time':e,'exit_time_upper':ts if ts==deadline else ts+MIN,
            'entry_raw':ep,'entry_paid':paid,'exit_raw':xp,'exit_received':received,'pnl_per_unit':pnl,
            'net_bp':pnl/paid*10000,'exit_reason':reason,'both_barriers':amb,'mae_bp_observed':mae*10000,
            'mfe_bp_observed':mfe*10000,'pre_exit_low_marks':lows,'execution_verified':False}
    raise AssertionError('unreachable')

def fixed_cost(out,extra_bp):
    if out['status']!='modeled_closed':return dict(out)
    r=dict(out);additional=(r['entry_paid']+r['exit_received'])*extra_bp/10000
    r['pnl_per_unit']-=additional;r['net_bp']=r['pnl_per_unit']/r['entry_paid']*10000
    return r

def summarize(records,extra=0):
    states=Counter(r['outcome']['status'] for r in records);closed=[r for r in records if r['outcome']['status']=='modeled_closed']
    vals=np.array([fixed_cost(r['outcome'],extra)['net_bp'] for r in closed]);days=[r['date'] for r in closed]
    result={'cases':len(records),'states':dict(states),'closed':len(vals),'extra_cost_bp_each':extra,'mean_bp':None,'ci95_date_cluster':None}
    if not len(vals):return result
    losses=-vals[vals<0].sum();result.update(mean_bp=float(vals.mean()),median_bp=float(np.median(vals)),win_rate=float(np.mean(vals>0)),pf=float(vals[vals>0].sum()/losses) if losses else None,
        exits=dict(Counter(r['outcome']['exit_reason'] for r in closed)),p05_bp=float(np.quantile(vals,.05)),p95_bp=float(np.quantile(vals,.95)),unique_dates=len(set(days)))
    groups=pd.DataFrame({'v':vals,'d':days}).groupby('d').v.agg(['sum','count'])
    if len(groups)>=5:
        rng=np.random.default_rng(2302);sel=rng.integers(0,len(groups),(2000,len(groups)))
        mu=groups['sum'].to_numpy()[sel].sum(axis=1)/groups['count'].to_numpy()[sel].sum(axis=1)
        result['ci95_date_cluster']=np.quantile(mu,[.025,.975]).tolist()
    result['target_first_fraction_of_closed']=sum(r['outcome']['exit_reason']=='target' for r in closed)/len(closed)
    return result

def portfolio(records,cfg=CFG):
    """Spot cash ledger in USDT; no leverage. BTC wins exact decision-time ties.
    Candidate quantity depends only on prior equity and pre-decision turnover.
    Failed attempts consume an event. Any unresolved position halts reporting.
    """
    initial=10000.;cash=initial;busy=-1;used=set();days={};taken=[];counts=Counter();curve=[initial];incomplete=any(r['outcome']['status'].startswith('unresolved') and not r.get('candidate') for r in records)
    rr=sorted([r for r in records if r.get('candidate')],key=lambda r:(r['candidate']['decision'],0 if r['coin']=='BTCUSDT' else 1))
    for r in rr:
        c=r['candidate'];o=r['outcome'];date=r['date'];now=c['decision']
        if now<busy:counts['busy']+=1;continue
        if r['event_id'] in used:counts['same_event']+=1;continue
        d0=days.setdefault(date,cash)
        if cash<=d0*(1-.006):counts['daily_loss_gate']+=1;continue
        used.add(r['event_id']);counts['attempts']+=1
        if not o['status'].startswith('modeled_'):
            counts[o['status']]+=1
            if o['status'].startswith('unresolved'):incomplete=True;break
            continue
        qty=min(cash*cfg.risk/c['planned_loss_per_unit'],cash*cfg.max_fraction/(c['limit']*(1+cfg.fee_bp/10000)),c['prior_turnover']*cfg.turnover_fraction/c['limit'])
        if qty*o['entry_paid']<10:counts['too_small']+=1;continue
        for ts,low in o['pre_exit_low_marks']:
            mark=cash+qty*(low*(1-cfg.fee_bp/10000)-o['entry_paid']*(1+cfg.fee_bp/10000));curve.append(mark)
        pnl=qty*o['pnl_per_unit'];cash+=pnl;curve.append(cash);busy=o['exit_time_upper'];counts['closed']+=1
        taken.append({'event_id':r['event_id'],'date':date,'coin':r['coin'],'quantity':qty,'entry_notional_usdt':qty*o['entry_paid'],'pnl_usdt':pnl,'outcome':{k:v for k,v in o.items() if k!='pre_exit_low_marks'}})
        if cash<=0:incomplete=True;counts['insolvent']+=1;break
    a=np.array(curve);dd=np.min(a/np.maximum.accumulate(a)-1)
    stress={str(s):sum(x['quantity']*fixed_cost(x['outcome'],s)['pnl_per_unit'] for x in taken)/initial*100 for s in (0,2,5)}
    return {'initial_usdt':initial,'return_pct':(cash/initial-1)*100 if not incomplete else None,'partial_return_pct':(cash/initial-1)*100,'bar_low_mdd_pct':float(dd*100),'same_fill_extra_cost_returns_pct':stress,'counts':dict(counts),'incomplete':incomplete,'trades':taken,'capital_currency':'USDT_not_KRW','depth_or_execution_verified':False}
