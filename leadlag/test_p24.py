"""Synthetic invariants, not empirical trading evidence."""
import unittest
import numpy as np,pandas as pd
from dataclasses import replace
from p24_wave import *

def data(n=14000):
 rng=np.random.default_rng(24);c=100*np.exp(np.cumsum(rng.normal(0,.002,n)));o=np.r_[c[0],c[:-1]]
 return pd.DataFrame({'open':o,'high':np.maximum(o,c)*1.001,'low':np.minimum(o,c)*.999,'close':c,'qv':1e7},index=np.arange(n)*FIVE)

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.d=data();cls.f=features(cls.d)
 def test_rma_seed(self):self.assertEqual(rma(pd.Series([1.,2.,3.,4.]),3).iloc[2],2)
 def test_rma_update(self):self.assertAlmostEqual(rma(pd.Series([1.,2.,3.,4.]),3).iloc[3],8/3)
 def test_rma_gap(self):self.assertTrue(np.isnan(rma(pd.Series([1.,2.,3.,np.nan,4.,5.]),3).iloc[-1]))
 def test_rsi_flat(self):
  z=aggregate(self.d,1);z[['open','high','low','close']]=100;self.assertEqual(indicators(z).rsi.iloc[-1],50)
 def test_rsi_up(self):
  z=aggregate(self.d,1);z.close=np.arange(len(z))+100.;self.assertEqual(indicators(z).rsi.iloc[-1],100)
 def test_rsi_down(self):
  z=aggregate(self.d,1);z.close=2000-np.arange(len(z));self.assertEqual(indicators(z).rsi.iloc[-1],0)
 def test_aggregation_time(self):self.assertEqual(aggregate(self.d.iloc[:12],1).index[0],HOUR)
 def test_partial_hour_invalid(self):self.assertTrue(aggregate(self.d.iloc[:11],1).close.isna().all())
 def test_four_hour_not_future(self):self.assertTrue((self.f.ft.dropna()<=self.f.ft.dropna().index).all())
 def test_future_prices_do_not_change_signals(self):
  a=self.d.copy();cut=12000*FIVE;a.loc[a.index>=cut,'close']*=10
  fa=features(a);ix=self.f.index[self.f.index<=cut]
  pd.testing.assert_frame_equal(self.f.loc[ix],fa.loc[ix])
 def test_trend_entry_causal(self):
  sig=make_signals(self.f,'T_fixed');self.assertTrue(all(t==s['signal']+FIVE for t,s in sig.items()))
 def test_no_lookahead_pivots(self):self.assertNotIn('future_low',self.f.columns)
 def test_stop_before_target(self):self.assertEqual(exit_price(100,110,90,95,105), (95,'stop',True))
 def test_gap_stop(self):self.assertEqual(exit_price(90,92,85,95,105),(90,'gap_stop',False))
 def test_timeout_open(self):self.assertEqual(exit_price(100,101,99,95,105,True),(100,'timeout',False))
 def test_no_exit(self):self.assertEqual(exit_price(100,101,99,95,105),(None,None,False))
 def test_fixed_two_R(self):
  sig=make_signals(self.f,'T_fixed');self.assertTrue(all(abs((s['target']-s['p'])/(s['p']-s['stop'])-2)<1e-10 for s in sig.values()))
 def test_range_is_subset(self):self.assertTrue(set(make_signals(self.f,'R_regime'))<=set(make_signals(self.f,'R_plain')))
 def test_time_missing_does_not_fill(self):
  from unittest.mock import patch
  z=data(30);f=features(z);s={FIVE:{'signal':0,'p':100,'atr':1,'stop':98,'target':104,'limit':101,'qv':1e8,'rsi':45,'adx4':20,'regime':'up'}}
  z.loc[FIVE]=np.nan
  with patch('p24_wave.make_signals',return_value=s):out=run_one(z,f,'T_fixed',0,30*FIVE)
  self.assertEqual(len(out['records']),0)
 def test_ledger_cash_reconciles(self):
  out=run_one(self.d,self.f,'R_plain',10000*FIVE,14000*FIVE)
  self.assertFalse(out['incomplete']);self.assertAlmostEqual(out['ending_cash']-C.initial,sum(r['pnl_usdt'] for r in out['records']),places=7)
 def test_one_position_nonoverlap(self):
  out=run_one(self.d,self.f,'R_plain',10000*FIVE,14000*FIVE);r=out['records'];self.assertTrue(all(a['exit_time']<=b['entry_time'] for a,b in zip(r,r[1:])))
 def test_slip_stress_negative(self):
  out=run_one(self.d,self.f,'R_plain',10000*FIVE,14000*FIVE);m=trade_metrics(out['records'])
  self.assertGreater(m['n'],0);self.assertGreater(m['same_fill_extra_mean_bp']['0'],m['same_fill_extra_mean_bp']['5'])
 def test_nav_constant(self):self.assertEqual(nav_metrics({1:10000,2:10000})['return_pct'],0)
 def test_no_sample_not_success(self):self.assertIsNone(trade_metrics([])['mean_bp'])
 def test_no_daily_nav_no_claim(self):self.assertEqual(nav_metrics({}),{'days':0})
 def test_data_gap_flat_account_survives(self):
  z=data(50);z.iloc[4]=np.nan;o=run_one(z,features(z),'R_plain',0,50*FIVE);self.assertFalse(o['incomplete'])
 def test_input_not_mutated(self):
  z=self.d.copy();features(self.d);pd.testing.assert_frame_equal(z,self.d)
 def test_ema_long_warmup(self):self.assertFalse(self.f.trend_signal.iloc[:800].any())
 def test_coin_position_cap(self):
  o=run_one(self.d,self.f,'R_plain',10000*FIVE,14000*FIVE)
  self.assertTrue(all(r['qty']*r['entry_paid']<3000 for r in o['records']))
 def test_trailing_finishes_with_stop_or_timeout(self):
  o=run_one(self.d,self.f,'T_trail',10000*FIVE,14000*FIVE);self.assertTrue(all(r['reason']!='target' for r in o['records']))
if __name__=='__main__':unittest.main()
