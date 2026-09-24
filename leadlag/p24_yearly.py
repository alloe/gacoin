"""Accounting-only follow-up. EXACT same Phase24.1 signal/exit parameters.
Independently funded calendar-year sleeves; missing-held-data years remain
unreportable, never merged into a fictional complete 56-month NAV.
Probability estimates are frozen at year start using completed prior-year trades.
"""
import hashlib,json,traceback
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np,pandas as pd
import p24_run as io
from p24_wave import C,MODELS,FIVE,DAY,features,run_one,nav_metrics,trade_metrics
YEARS=(2022,2023,2024,2025,2026)

def combine(outs):
    good=len(outs)==len(io.COINS) and not any(o['incomplete'] for o in outs.values())
    if not outs:return good,{},[]
    days=sorted(set.intersection(*(set(x['daily']) for x in outs.values())))
    nav={day:sum(x['daily'][day] for x in outs.values())/len(io.COINS) for day in days}
    records=[dict(r,coin=coin) for coin,o in outs.items() for r in o['records']]
    return good,nav,records

def posterior(train,test):
    if len(train)<50 or not test:return {'status':'insufficient_history','train_n':len(train),'test_n':len(test)}
    p=(sum(r['net_bp']>0 for r in train)+1)/(len(train)+2)
    y=np.array([r['net_bp']>0 for r in test],float)
    return {'train_n':len(train),'test_n':len(test),'p_profit_frozen':p,'actual_profit_fraction':float(y.mean()),
        'brier':float(np.mean((p-y)**2)),'brier_half':.25,'definition':'probability net trade positive, NOT exact bottom prediction'}

def main():
    spec=dict(io.SPEC)
    spec.update(accounting='Each calendar year starts independently with 8 equal10000USDT sleeves. Any held data gap invalidates that year composite. Signals/stops/targets/costs unchanged.',
        reason='Continuous run encountered historical price gaps. No data patching, parameter optimization or selection of profitable years.',
        probability='For each evaluation year freeze smoothed win fraction using complete coin-year records from preceding2 calendar years; no trade filter based on prediction.')
    io.log('P24Y_START',spec=spec,source_sha256=hashlib.sha256(open(__file__.replace('p24_yearly.py','p24_wave.py'),'rb').read()).hexdigest(),no_orders=True)
    months=[str(x) for x in pd.period_range(io.DATA_START[:7],'2026-08',freq='M')]
    grid=np.arange(io.ms(io.DATA_START),io.ms(io.END),FIVE)
    runs={(m,y):{} for m in MODELS for y in YEARS};bench={y:{} for y in YEARS};fail=[];coverage=[]
    for coin in io.COINS:
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:frames=list(pool.map(io.load_month,[(coin,m) for m in months]))
            d=pd.concat(frames).sort_index()
            if d.index.duplicated().any():raise ValueError('duplicate monthly timestamp')
            d=d.reindex(grid);f=features(d)
            missing=d.index[d.open.isna()].to_numpy()
            coverage.append({'coin':coin,'bars':len(d),'missing_bars':len(missing),'missing_times':[pd.Timestamp(t,unit='ms',tz='UTC').isoformat() for t in missing]})
            for y in YEARS:
                a=io.ms(str(y)+'-01-01');b=min(io.ms(str(y+1)+'-01-01'),io.ms(io.END))
                bench[y][coin]=io.passive(d,a,b)
                for model in MODELS:
                    out=run_one(d,f,model,a,b);runs[model,y][coin]=out
                    io.log('P24Y_COIN',coin=coin,year=y,model=model,signals=out['signals'],counts=out['counts'],incomplete=out['incomplete'],n=len(out['records']))
            io.log('P24Y_DATA',coin=coin,bars=len(d),missing_bars=len(missing));del d,f,frames
        except Exception as e:
            fail.append({'coin':coin,'error':repr(e)});io.log('P24Y_COIN_FATAL',coin=coin,error=repr(e),traceback=traceback.format_exc()[-1500:])
    rows=[];records=[];probs=[];navexport={}
    for model in MODELS:
        for y in YEARS:
            outs=runs[model,y];good,nav,tr=combine(outs)
            for r in tr:r.update(model=model,year=y,coin_year_complete=not outs[r['coin']]['incomplete'])
            allm=trade_metrics(tr);acct=nav_metrics(nav) if good else {'status':'unreportable_held_data_gap_or_missing_coin'}
            bycoin={c:{'complete':not o['incomplete'],'signals':o['signals'],'counts':o['counts'],'nav':nav_metrics(o['daily']) if not o['incomplete'] else None,'metrics':trade_metrics(o['records'])} for c,o in outs.items()}
            row={'model':model,'year':y,'complete':good,'account':acct,'record_statistics_are_partial':not good,'trade_metrics':allm,'by_coin':bycoin}
            rows.append(row);records.extend(tr)
            io.log('P24Y_RESULT',**{k:v for k,v in row.items() if k!='by_coin'})
            if good:navexport[model+'_'+str(y)]=nav
        for y in YEARS[1:]:
            training=[r for r in records if r['model']==model and r['coin_year_complete'] and y-2<=r['year']<y]
            test=[r for r in records if r['model']==model and r['year']==y and r['coin_year_complete']]
            diag=posterior(training,test);diag.update(model=model,year=y);probs.append(diag);io.log('P24Y_PROBABILITY',**diag)
    benchmarks=[]
    for y,coins in bench.items():
        if len(coins)!=len(io.COINS):continue
        days=sorted(set.intersection(*(set(d) for d in coins.values())));nav={t:sum(d[t] for d in coins.values())/len(io.COINS) for t in days}
        benchmarks.append({'year':y,'metrics':nav_metrics(nav),'description':'each Jan1 25pct initial capital passive,75pct cash; not matched actual holding exposure'})
    for b in benchmarks:io.log('P24Y_BENCHMARK',**b)
    report={'spec':spec,'coverage':coverage,'failures':fail,'rows':rows,'probabilities':probs,'benchmarks':benchmarks,'production_pass':False}
    io.emit('yearly_summary',report);io.emit('yearly_nav',navexport)
    for model in MODELS:io.emit('yearly_trades_'+model,[r for r in records if r['model']==model])
    io.emit('yearly_sources',io.MANIFEST)
    io.log('P24Y_DONE',coins=len(coverage),failures=len(fail),cells=len(rows),complete_cells=sum(r['complete'] for r in rows),trade_records=len(records),production_pass=False)
if __name__=='__main__':
    try:main()
    except Exception as e:io.log('P24Y_FATAL',error=repr(e),traceback=traceback.format_exc()[-2000:]);raise
