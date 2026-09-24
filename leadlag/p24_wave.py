"""Phase24.1 conventional swing study, pure causal research, no order APIs.
Fixed indicators/parameters before outcome queries. Spot long-only, USDT.
Four-hour regime + hourly signals, first five-minute open AFTER a five-minute
assumed execution delay. Future pivots/ZigZag labels are never used.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import numpy as np
import pandas as pd

FIVE=300000; HOUR=3600000; DAY=86400000
MODELS=('T_fixed','T_trail','R_plain','R_regime')
@dataclass(frozen=True)
class Config:
    fee:float=.001
    slip:float=.0002
    risk:float=.005
    allocation:float=.25
    turnover:float=.001
    initial:float=10000.
    cooldown_h:int=4
    stop_atr:float=2.
    target_atr:float=4.
C=Config()

def rma(s:pd.Series,n:int)->pd.Series:
    """Wilder smoothing with SMA seed and reset on missing data."""
    out=np.full(len(s),np.nan);seed=[];value=np.nan
    for i,x in enumerate(s.to_numpy(float)):
        if not np.isfinite(x):seed=[];value=np.nan;continue
        if not np.isfinite(value):
            seed.append(x)
            if len(seed)==n:value=float(np.mean(seed));out[i]=value
        else:value=(value*(n-1)+x)/n;out[i]=value
    return pd.Series(out,index=s.index)

def indicators(d:pd.DataFrame)->pd.DataFrame:
    a=d.copy();diff=a.close.diff();u=rma(diff.clip(lower=0),14);v=rma(-diff.clip(upper=0),14)
    a['rsi']=100-100/(1+u/v.replace(0,np.nan));a.loc[(v==0)&(u>0),'rsi']=100
    a.loc[(u==0)&(v==0),'rsi']=50
    pc=a.close.shift(1);tr=pd.concat([a.high-a.low,(a.high-pc).abs(),(a.low-pc).abs()],axis=1).max(axis=1)
    tr[a.close.isna()]=np.nan;a['atr']=rma(tr,14)
    up=a.high.diff();dn=-a.low.diff();plus=up.where((up>dn)&(up>0),0.);minus=dn.where((dn>up)&(dn>0),0.)
    plus[a.close.isna()]=np.nan;minus[a.close.isna()]=np.nan
    p=rma(plus,14)/a.atr;mi=rma(minus,14)/a.atr;den=p+mi;dx=100*(p-mi).abs()/den.replace(0,np.nan);dx[den==0]=0
    a['adx']=rma(dx,14)
    for n in (20,50,200):a['ema'+str(n)]=a.close.ewm(span=n,adjust=False,min_periods=n).mean()
    a['mid']=a.close.rolling(20,min_periods=20).mean();sd=a.close.rolling(20,min_periods=20).std(ddof=0)
    a['lower']=a.mid-2*sd;a['upper']=a.mid+2*sd
    return a

def aggregate(d:pd.DataFrame,hours:int)->pd.DataFrame:
    z=d.copy();z.index=pd.to_datetime(z.index,unit='ms',utc=True)
    rule=str(hours)+'h';g=z.resample(rule,label='right',closed='left',origin='epoch')
    a=g.agg({'open':'first','high':'max','low':'min','close':'last','qv':'sum'})
    ct=g.close.count();a.loc[ct!=hours*12,:]=np.nan
    a.index=(a.index.astype('int64')//1000000).astype('int64')
    return a

def features(d:pd.DataFrame)->pd.DataFrame:
    h=indicators(aggregate(d,1));four=indicators(aggregate(d,4))
    four['slope50']=four.ema50-four.ema50.shift(6)
    four['valid220']=four.close.notna().rolling(220,min_periods=220).sum()==220
    fc=four[['close','ema50','ema200','slope50','adx','valid220']].rename(columns=lambda k:'f_'+k)
    f=pd.merge_asof(h.reset_index(names='t'),fc.reset_index(names='ft'),left_on='t',right_on='ft',direction='backward').set_index('t')
    f['valid']=f.close.notna().rolling(50,min_periods=50).sum().eq(50)&f.f_valid220.fillna(False)
    f['trend']=(f.f_close>f.f_ema200)&(f.f_ema50>f.f_ema200)&(f.f_slope50>0)
    f['range']=(f.f_adx<20)
    # Crossing (not level) reduces repeated triggers from the same dip.
    f['trend_signal']=f.valid&f.trend&(f.rsi.shift(1)<=45)&(f.rsi>45)&(f.close>f.high.shift(1))
    f['range_signal']=f.valid&(f.close.shift(1)<f.lower.shift(1))&(f.close>=f.lower)&(f.rsi<45)
    return f

def make_signals(f:pd.DataFrame,model:str,cfg:Config=C)->dict:
    mask=f.trend_signal if model.startswith('T') else f.range_signal
    if model=='R_regime':mask=mask&f['range']
    out={}
    for t,r in f.loc[mask].iterrows():
        atr=float(r.atr);p=float(r.close)
        if not (np.isfinite(atr) and atr>0 and p>0):continue
        stop=p-cfg.stop_atr*atr;target=(p+cfg.target_atr*atr if model=='T_fixed' else (float(r.mid) if model.startswith('R') else np.inf))
        limit=p+.5*atr
        if stop<=0 or target*(1-cfg.slip)*(1-cfg.fee)<=limit*(1+cfg.fee):continue
        out[int(t+FIVE)]={'signal':int(t),'p':p,'atr':atr,'stop':stop,'target':target,'limit':limit,'qv':float(r.qv),
            'rsi':float(r.rsi),'adx4':float(r.f_adx),'regime':'up' if r.trend else ('range' if r['range'] else 'other')}
    return out

def exit_price(o,h,l,stop,target,time_exit=False):
    if o<=stop:return o,'gap_stop',False
    if time_exit:return o,'timeout',False
    if l<=stop:return stop,'stop',bool(h>=target)
    if h>=target:return target,'target',False
    return None,None,False

def trade_metrics(records:list)->dict:
    if not records:return {'n':0,'mean_bp':None,'win_rate':None,'pf':None}
    a=np.array([r['net_bp'] for r in records]);wins=a[a>0];loss=a[a<0]
    daily=pd.DataFrame({'v':a,'d':[r['entry_date'] for r in records]}).groupby('d').v.agg(['sum','count'])
    ci=None
    if len(daily)>=10:
        rng=np.random.default_rng(2401);idx=rng.integers(0,len(daily),(1000,len(daily)))
        av=daily['sum'].to_numpy()[idx].sum(axis=1)/daily['count'].to_numpy()[idx].sum(axis=1);ci=np.quantile(av,[.025,.975]).tolist()
    streak=maxstreak=0
    for r in sorted(records,key=lambda z:z['exit_time']):
        streak=streak+1 if r['net_bp']<0 else 0;maxstreak=max(maxstreak,streak)
    return {'n':len(a),'mean_bp':float(a.mean()),'median_bp':float(np.median(a)),'win_rate':float(np.mean(a>0)),
        'pf':float(wins.sum()/-loss.sum()) if len(loss) else None,'mean_win_bp':float(wins.mean()) if len(wins) else None,
        'mean_loss_bp':float(loss.mean()) if len(loss) else None,'ci95_date_cluster':ci,'unique_entry_dates':len(daily),
        'max_loss_streak_chronological':maxstreak,'mean_hold_h':float(np.mean([r['hold_h'] for r in records])),
        'exits':dict(Counter(r['reason'] for r in records)),'both_barriers':sum(r['both_barriers'] for r in records),
        'same_fill_extra_mean_bp':{str(bp):float(np.mean([r['net_bp']-bp*(1+r['exit_received']/r['entry_paid']) for r in records])) for bp in (0,2,5)}}

def run_one(d:pd.DataFrame,f:pd.DataFrame,model:str,start:int,end:int,cfg:Config=C):
    sig=make_signals(f,model,cfg);ts=d.index.to_numpy(dtype='int64');bars=d[['open','high','low','close','qv']].to_numpy(float)
    valid=np.isfinite(bars).all(axis=1)&(bars[:,:4]>0).all(axis=1)
    lo=int(np.searchsorted(ts,start));hi=int(np.searchsorted(ts,end));cash=cfg.initial;pos=None;next_entry=start
    records=[];daily={};intrabar_min={};counts=Counter();incomplete=False;last_px=None
    hold_h=72 if model.startswith('T') else 24
    feature_at={int(t+FIVE):(float(r.close),float(r.atr),float(r.f_close),float(r.f_ema200)) for t,r in f.iterrows() if bool(r.valid)}
    for i in range(lo,hi):
        t=int(ts[i]);o,h,l,c,qv=bars[i];day=t//DAY
        if not valid[i]:
            counts['missing_flat_bars' if pos is None else 'missing_held_bars']+=1
            if pos is not None:incomplete=True;break
            daily[day]=cash;intrabar_min[day]=min(intrabar_min.get(day,cash),cash);continue
        last_px=c
        if pos is not None and model=='T_trail' and t in feature_at and t>pos['entry_time']:
            close,atr,fc,ema=feature_at[t];pos['maxclose']=max(pos['maxclose'],close)
            if pos['maxclose']>=pos['entry_paid']+cfg.stop_atr*pos['atr']:
                pos['stop']=max(pos['stop'],pos['maxclose']-cfg.stop_atr*atr)
        closed_now=False
        if pos is not None:
            force=t>=pos['deadline'] or i==hi-1
            xp,reason,amb=exit_price(o,h,l,pos['stop'],pos['target'],force)
            if xp is not None:
                received=xp*(1-cfg.slip);cash+=pos['qty']*received*(1-cfg.fee)
                pnlu=received*(1-cfg.fee)-pos['entry_paid']*(1+cfg.fee)
                rec={k:pos[k] for k in ('entry_time','signal','entry_paid','atr','rsi','adx4','regime','qty')}
                rec.update(exit_time=t+FIVE,entry_date=pd.Timestamp(pos['entry_time'],unit='ms',tz='UTC').strftime('%Y-%m-%d'),
                    exit_received=received,pnl_usdt=pos['qty']*pnlu,net_bp=pnlu/pos['entry_paid']*10000,
                    reason='end_of_sample' if force and i==hi-1 else reason,both_barriers=amb,hold_h=(t+FIVE-pos['entry_time'])/HOUR)
                records.append(rec);counts['closed']+=1;pos=None;next_entry=t+FIVE+cfg.cooldown_h*HOUR;closed_now=True
        if pos is None and not closed_now and t>=next_entry and i<hi-1 and t in sig:
            s=sig[t];counts['attempts']+=1;paid=o*(1+cfg.slip)
            if paid>s['limit']:counts['no_fill_cap']+=1
            elif paid<=s['stop'] or paid>=s['target']:counts['no_fill_invalidated']+=1
            else:
                loss=s['limit']*(1+cfg.fee)-s['stop']*(1-cfg.slip)*(1-cfg.fee)
                qty=min(cash*cfg.risk/loss,cash*cfg.allocation/(s['limit']*(1+cfg.fee)),s['qv']*cfg.turnover/s['limit'])
                if qty*paid<10:counts['min_notional']+=1
                else:
                    cash-=qty*paid*(1+cfg.fee);pos=dict(s,entry_paid=paid,entry_time=t,qty=qty,
                        maxclose=paid,deadline=t+hold_h*HOUR);counts['entered']+=1
                    # Existing protective orders can be hit later in the entry bar.
                    xp,reason,amb=exit_price(paid,h,l,pos['stop'],pos['target'],False)
                    if xp is not None:
                        received=xp*(1-cfg.slip);cash+=qty*received*(1-cfg.fee)
                        pnlu=received*(1-cfg.fee)-paid*(1+cfg.fee)
                        rec={k:pos[k] for k in ('entry_time','signal','entry_paid','atr','rsi','adx4','regime','qty')}
                        rec.update(exit_time=t+FIVE,entry_date=pd.Timestamp(t,unit='ms',tz='UTC').strftime('%Y-%m-%d'),exit_received=received,
                            pnl_usdt=qty*pnlu,net_bp=pnlu/paid*10000,reason=reason,both_barriers=amb,hold_h=FIVE/HOUR)
                        records.append(rec);counts['closed']+=1;pos=None;next_entry=t+FIVE+cfg.cooldown_h*HOUR
        equity=cash+(pos['qty']*c*(1-cfg.slip)*(1-cfg.fee) if pos else 0)
        loweq=cash+(pos['qty']*l*(1-cfg.slip)*(1-cfg.fee) if pos else 0)
        daily[day]=equity;intrabar_min[day]=min(intrabar_min.get(day,loweq),loweq)
        if cash<-1e-7 or equity<=0:incomplete=True;counts['cash_invariant_failed']+=1;break
    if pos is not None:incomplete=True;counts['unresolved_position']+=1
    return {'records':records,'daily':daily,'low_marks':intrabar_min,'counts':dict(counts),'incomplete':incomplete,
        'ending_cash':cash if pos is None else None,'signals':sum(start<=t<end for t in sig)}

def nav_metrics(daily:dict,initial:float=10000.):
    if not daily:return {'days':0}
    ix=sorted(daily);a=np.array([daily[i] for i in ix]);ser=pd.Series(a,index=pd.to_datetime(np.array(ix)*DAY,unit='ms',utc=True))
    monthly=ser.resample('ME').last();prev=monthly.shift(1);prev.iloc[0]=initial;mr=monthly/prev-1
    draw=np.r_[initial,a]/np.maximum.accumulate(np.r_[initial,a])-1
    return {'days':len(a),'return_pct':float((a[-1]/initial-1)*100),'daily_mdd_pct':float(draw.min()*100),
        'positive_months':int((mr>0).sum()),'negative_months':int((mr<0).sum()),'flat_months':int((mr==0).sum()),
        'monthly_pct':{str(k)[:7]:float(v*100) for k,v in mr.items()}}
