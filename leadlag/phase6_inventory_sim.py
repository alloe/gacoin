#!/usr/bin/env python3
import csv,gzip,io,json,math,os,statistics
from datetime import datetime
import requests,numpy as np

BASE="https://datasets.tardis.dev/v1"
DATES=os.getenv("DATES","2026-01-01").split(",")
STEP_US=100_000
UP_FEE_BP=float(os.getenv("UP_FEE_BP","5"))
BN_FEE_BP=float(os.getenv("BN_FEE_BP","10"))
TIMEOUT=(20,600)
CAPITALS=[5_000_000,10_000_000]
MARGINS=[5.0,7.5,10.0]
LATS=[100,200]
QTYS=[0.001,0.005,0.01]
STRESS=[0,2,5]

def log(m,**kw): print(m+(" "+json.dumps(kw,ensure_ascii=False,sort_keys=True) if kw else ""),flush=True)
def ts_us(v):
    s=str(v).strip()
    try:
        x=int(float(s))
        if x<10_000_000_000:return x*1_000_000
        if x<10_000_000_000_000:return x*1_000
        if x>10_000_000_000_000_000:return x//1000
        return x
    except:return int(datetime.fromisoformat(s.replace("Z","+00:00")).timestamp()*1_000_000)
def url(ex,date,sym):
    y,m,d=date.split("-");return f"{BASE}/{ex}/quotes/{y}/{m}/{d}/{sym}.csv.gz"
def load(ex,date,sym):
    r=requests.get(url(ex,date,sym),stream=True,timeout=TIMEOUT)
    if r.status_code!=200:raise RuntimeError(f"HTTP {r.status_code} {url(ex,date,sym)}")
    r.raw.decode_content=False
    rd=csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw),encoding="utf-8",newline=""))
    ts=[];bp=[];ba=[];ap=[];aa=[];cur=None;v=None
    for row in rd:
        try:
            t=ts_us(row["local_timestamp"]);b=float(row["bid_price"]);bs=float(row["bid_amount"]);a=float(row["ask_price"]);az=float(row["ask_amount"])
            if not(b>0 and a>=b and bs>=0 and az>=0):continue
        except:continue
        buck=(t//STEP_US)*STEP_US;nv=(b,bs,a,az)
        if cur is None:cur=buck;v=nv
        elif buck==cur:v=nv
        else:ts.append(cur);bp.append(v[0]);ba.append(v[1]);ap.append(v[2]);aa.append(v[3]);cur=buck;v=nv
    if cur is not None:ts.append(cur);bp.append(v[0]);ba.append(v[1]);ap.append(v[2]);aa.append(v[3])
    return tuple(np.array(x,dtype=(np.int64 if i==0 else float)) for i,x in enumerate([ts,bp,ba,ap,aa]))
def grid(q,start,end):
    ts,bp,ba,ap,aa=q;n=int((end-start)//STEP_US)+1
    outs=[np.full(n,np.nan) for _ in range(4)]
    idx=((ts-start)//STEP_US).astype(np.int64);ok=(idx>=0)&(idx<n);ii=idx[ok]
    for o,v in zip(outs,[bp,ba,ap,aa]):o[ii]=v[ok]
    valid=np.where(np.isfinite(outs[0])&np.isfinite(outs[2]),np.arange(n),-1)
    np.maximum.accumulate(valid,out=valid);m=valid>=0
    for o in outs:o[m]=o[valid[m]]
    return outs
def starts(edge,margin):
    out=[];inside=False
    for i,x in enumerate(edge):
        hit=math.isfinite(x) and x>=margin
        if hit and not inside:out.append(i)
        inside=hit
    return out
def simulate(date,capital,margin,lat,qty,stress_bp,arr):
    ubb,ubsz,uba,uasz,usb,usa,bnb,bnbsz,bna,bnasz=arr
    upf=UP_FEE_BP/10000;bnf=BN_FEE_BP/10000;k=lat//100
    sell_global=bnb*usb;buy_global=bna*usa
    edgeA=np.log((sell_global*(1-bnf))/(uba*(1+upf)))*10000
    edgeB=np.log((ubb*(1-upf))/(buy_global*(1+bnf)))*10000
    events=[]
    for i in starts(edgeA,margin):
        j=i+k
        if j<len(edgeA):events.append((j,"A",i))
    for i in starts(edgeB,margin):
        j=i+k
        if j<len(edgeB):events.append((j,"B",i))
    events.sort(key=lambda x:(x[0],x[1]))
    # Start positioned for the historically dominant A direction:
    # Upbit all KRW; Binance all BTC (KRW-equivalent capital on each venue).
    fx0=(usb[0]+usa[0])/2;bnmid0=(bnb[0]+bna[0])/2
    up_krw=float(capital);up_btc=0.0
    bn_btc=float(capital/(bnmid0*fx0));bn_usdt=0.0
    init_bn_btc=bn_btc
    gross=0.0;stress_cost=0.0;execA=execB=blocked=stale=0
    first_block=None
    for j,side,i0 in events:
        if side=="A":
            e=float(edgeA[j]);cap=float(min(uasz[j],bnbsz[j]))
            if not math.isfinite(e) or e<=0 or cap<qty:stale+=1;continue
            cost_krw=float(uba[j]*(1+upf)*qty)
            if up_krw+1e-9<cost_krw or bn_btc+1e-12<qty:
                blocked+=1
                if first_block is None:first_block=j
                continue
            recv_usdt=float(bnb[j]*(1-bnf)*qty)
            pnl=float((bnb[j]*usb[j]*(1-bnf)-uba[j]*(1+upf))*qty)
            notional=float(((bnb[j]*usb[j])+uba[j])*0.5*qty)
            up_krw-=cost_krw;up_btc+=qty;bn_btc-=qty;bn_usdt+=recv_usdt
            gross+=pnl;stress_cost+=notional*stress_bp/10000;execA+=1
        else:
            e=float(edgeB[j]);cap=float(min(ubsz[j],bnasz[j]))
            if not math.isfinite(e) or e<=0 or cap<qty:stale+=1;continue
            cost_usdt=float(bna[j]*(1+bnf)*qty)
            if up_btc+1e-12<qty or bn_usdt+1e-12<cost_usdt:
                blocked+=1
                if first_block is None:first_block=j
                continue
            recv_krw=float(ubb[j]*(1-upf)*qty)
            pnl=float((ubb[j]*(1-upf)-bna[j]*usa[j]*(1+bnf))*qty)
            notional=float((ubb[j]+bna[j]*usa[j])*0.5*qty)
            up_btc-=qty;up_krw+=recv_krw;bn_usdt-=cost_usdt;bn_btc+=qty
            gross+=pnl;stress_cost+=notional*stress_bp/10000;execB+=1
    # Forced end-of-day rebalance back toward Upbit KRW / Binance BTC.
    # Net A inventory is closed with a B-direction trade at the final executable quotes.
    imbalance=max(0.0,min(up_btc,init_bn_btc-bn_btc))
    rebalance_pnl=0.0
    if imbalance>1e-12:
        last=len(ubb)-1
        max_by_usdt=bn_usdt/(bna[last]*(1+bnf)) if bna[last]>0 else 0
        rq=min(imbalance,max_by_usdt)
        if rq>1e-12:
            rebalance_pnl=float((ubb[last]*(1-upf)-bna[last]*usa[last]*(1+bnf))*rq)
    net=gross-stress_cost+rebalance_pnl
    return {"capital_each_krw":capital,"margin_bp":margin,"lat_ms":lat,"qty_btc":qty,"stress_bp":stress_bp,
      "signals":len(events),"exec_A":execA,"exec_B":execB,"executed":execA+execB,"blocked_inventory":blocked,
      "skipped_edge_or_depth":stale,"gross_arb_pnl":gross,"stress_cost":stress_cost,
      "end_imbalance_btc":imbalance,"forced_rebalance_pnl":rebalance_pnl,"net_after_rebalance":net,
      "first_inventory_block_ms":(first_block*100 if first_block is not None else None),
      "ending":{"up_krw":up_krw,"up_btc":up_btc,"bn_btc":bn_btc,"bn_usdt":bn_usdt}}
def main():
    log("PHASE6_INVENTORY_START",dates=DATES,capitals=CAPITALS,margins=MARGINS,lats=LATS,qtys=QTYS,stress=STRESS)
    results=[];fails=[]
    for date in DATES:
        try:
            ub=load("upbit",date,"KRW-BTC");uu=load("upbit",date,"KRW-USDT");bn=load("binance",date,"BTCUSDT")
            start=max(ub[0][0],uu[0][0],bn[0][0]);end=min(ub[0][-1],uu[0][-1],bn[0][-1])
            ubb,ubsz,uba,uasz=grid(ub,start,end);usb,usbsz,usa,usasz=grid(uu,start,end);bnb,bnbsz,bna,bnasz=grid(bn,start,end)
            arr=(ubb,ubsz,uba,uasz,usb,usa,bnb,bnbsz,bna,bnasz)
            for c in CAPITALS:
                for m in MARGINS:
                    for lat in LATS:
                        for q in QTYS:
                            for sbp in STRESS:
                                z=simulate(date,c,m,lat,q,sbp,arr);z["date"]=date;results.append(z)
            log("DAY_DONE",date=date)
        except Exception as e:fails.append({"date":date,"error":repr(e)});log("FAIL",date=date,error=repr(e))
    # Compact aggregate by scenario.
    agg={}
    for z in results:
        key=(z["capital_each_krw"],z["margin_bp"],z["lat_ms"],z["qty_btc"],z["stress_bp"])
        a=agg.setdefault(key,{"days":0,"executed":0,"A":0,"B":0,"blocked":0,"gross":0.0,"reb":0.0,"net":0.0,
                              "imbalance":0.0,"positive_days":0,"block_days":0})
        a["days"]+=1;a["executed"]+=z["executed"];a["A"]+=z["exec_A"];a["B"]+=z["exec_B"];a["blocked"]+=z["blocked_inventory"]
        a["gross"]+=z["gross_arb_pnl"];a["reb"]+=z["forced_rebalance_pnl"];a["net"]+=z["net_after_rebalance"];a["imbalance"]+=z["end_imbalance_btc"]
        a["positive_days"]+=1 if z["net_after_rebalance"]>0 else 0;a["block_days"]+=1 if z["blocked_inventory"]>0 else 0
    compact=[]
    for key,a in sorted(agg.items()):
        c,m,lat,q,sbp=key
        compact.append({"capital_each_krw":c,"margin_bp":m,"lat_ms":lat,"qty_btc":q,"stress_bp":sbp,
          "days":a["days"],"avg_trades_day":a["executed"]/a["days"],"A":a["A"],"B":a["B"],
          "avg_blocked_day":a["blocked"]/a["days"],"block_day_rate":a["block_days"]/a["days"],
          "avg_gross_day":a["gross"]/a["days"],"avg_rebalance_day":a["reb"]/a["days"],
          "avg_net_day":a["net"]/a["days"],"monthly_30d_projection":a["net"]/a["days"]*30,
          "positive_day_rate":a["positive_days"]/a["days"],"avg_end_imbalance_btc":a["imbalance"]/a["days"]})
    print("PHASE6_COMPACT="+json.dumps({"failures":fails,"scenarios":compact},separators=(",",":")),flush=True)
    print("PHASE6_SUMMARY",flush=True)
    for z in compact:
        if z["qty_btc"]==0.01 and z["stress_bp"] in (0,2) and z["margin_bp"] in (5.0,7.5):
            print(json.dumps(z,separators=(",",":")),flush=True)
    log("PHASE6_DONE",rows=len(results),failures=len(fails))
if __name__=="__main__":main()
