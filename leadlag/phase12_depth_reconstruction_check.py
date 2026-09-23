#!/usr/bin/env python3
"""Public monthly sample only: verify L2 reconstruction against independent L1.
No account or order APIs. Invalid books never contribute to VWAP results.
"""
from itertools import groupby
import math
import numpy as np
from phase11_historical_depth_audit import rows,day_us,quant,vw,log,SIZES
DATE='2026-09-01'

def load_reference(c):
    end=day_us(DATE)+3600000000;a=[]
    stream=rows('upbit','quotes',DATE,'KRW-'+c)
    try:
        for r in stream:
            t=int(r['local_timestamp'])
            if t>=end:break
            a.append([t,float(r['bid_price']),float(r['ask_price'])])
    finally:stream.close()
    if not a:raise ValueError('empty independent quote reference')
    return np.array(a,float)

def check(c):
    reference=load_reference(c);end=day_us(DATE)+3600000000
    modes={'transition':{'b':{},'a':{},'ready':False,'prev':False},'message_snapshot':{'b':{},'a':{},'ready':False,'prev':False}}
    audit={m:{'evaluated':0,'crossed':0,'matched':0,'mismatch':0,'no_reference':0,'samples':{str(n):[] for n in SIZES},'capacity':{str(n):0 for n in SIZES}} for m in modes}
    count=0;snapgroups=0;mixedgroups=0;lastsample=day_us(DATE)-1000000
    stream=rows('upbit','incremental_book_L2',DATE,'KRW-'+c)
    try:
        for t,g in groupby(stream,key=lambda r:int(r['local_timestamp'])):
            if t>=end:break
            batch=list(g);count+=1;has=any(r['is_snapshot'].lower()=='true' for r in batch)
            snapgroups+=int(has);mixedgroups+=int(has and any(r['is_snapshot'].lower()=='false' for r in batch))
            for mode,book in modes.items():
                if mode=='message_snapshot' and has:
                    book['b'].clear();book['a'].clear();book['ready']=True
                for r in batch:
                    snap=r['is_snapshot'].lower()=='true'
                    if mode=='transition' and snap and not book['prev']:
                        book['b'].clear();book['a'].clear();book['ready']=True
                    book['prev']=snap
                    if not book['ready']:continue
                    dest=book['b'] if r['side']=='bid' else book['a'];p=float(r['price']);q=float(r['amount'])
                    if q==0:dest.pop(p,None)
                    else:dest[p]=q
            if t-lastsample<1000000:continue
            lastsample=t;k=int(np.searchsorted(reference[:,0],t,side='right')-1)
            for mode,book in modes.items():
                z=audit[mode]
                if not book['ready'] or not book['b'] or not book['a']:continue
                z['evaluated']+=1;bids=sorted(book['b'].items(),reverse=True)[:20];asks=sorted(book['a'].items())[:20]
                if bids[0][0]>=asks[0][0]:z['crossed']+=1;continue
                if k<0 or t-reference[k,0]>5000000:z['no_reference']+=1;continue
                if not (math.isclose(bids[0][0],reference[k,1],rel_tol=1e-10) and math.isclose(asks[0][0],reference[k,2],rel_tol=1e-10)):
                    z['mismatch']+=1;continue
                z['matched']+=1
                for n in SIZES:
                    q=n/asks[0][0];pa=vw(asks,q);pb=vw(bids,q)
                    if pa is None or pb is None:continue
                    z['capacity'][str(n)]+=1;z['samples'][str(n)].append((pa-pb)/((pa+pb)/2)*10000)
    finally:stream.close()
    summary={}
    for mode,z in audit.items():
        denominator=z['evaluated']-z['no_reference']
        summary[mode]={'evaluated':z['evaluated'],'crossed':z['crossed'],'matched':z['matched'],'mismatch':z['mismatch'],'no_reference':z['no_reference'],
            'match_rate':z['matched']/denominator if denominator else None,
            'upbit_only_vwap_round_bp':{str(n):quant(z['samples'][str(n)]) for n in SIZES},'capacity':z['capacity']}
    m=summary['message_snapshot'];quality=bool(m['evaluated']>=100 and m['crossed']==0 and m['match_rate'] is not None and m['match_rate']>=.99)
    log('L2_RECONSTRUCTION_CHECK',coin=c,date=DATE,hours=1,groups=count,snapshot_groups=snapgroups,mixed_groups=mixedgroups,
        reference_rows=len(reference),modes=summary,message_snapshot_quality_pass=quality,
        note='Only matched valid reconstructed updates enter size stats. Sample-only L2 quality pass is NOT strategy/continuous-period execution validation.')
    return quality

def main():
    log('L2_CHECK_START',coins=['DOGE','HBAR'],date=DATE,hours=1,paper_only=True)
    quality={}
    for c in ['DOGE','HBAR']:
        try:quality[c]=check(c)
        except Exception as e:quality[c]=False;log('L2_CHECK_FAIL',coin=c,error=repr(e))
    log('L2_CHECK_DONE',quality=quality,continuous_backtest_pass=False)
if __name__=='__main__':main()
