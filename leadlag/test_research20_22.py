"""Synthetic regression checks, not trading profitability evidence."""
import unittest,numpy as np,pandas as pd
from research_core import *
from research22_cross import neutral_weights
from research20_news import epoch,causal_pullback
from research21_oi import candidate_masks
class Tests(unittest.TestCase):
 def test_end_month_exclusive(self):self.assertEqual(monthly_range('2026-01-01','2026-03-01'),['2026-01','2026-02'])
 def test_same_month(self):self.assertEqual(monthly_range('2026-01-01','2026-01-20'),['2026-01'])
 def test_fee_identity(self):self.assertAlmostEqual(net_trade(1,100,100,5,0),-10)
 def test_short_fee_identity(self):self.assertAlmostEqual(net_trade(-1,100,100,5,0),-10)
 def test_cost_monotone(self):self.assertGreater(net_trade(1,100,110,5,0),net_trade(1,100,110,5,5))
 def test_short_direction(self):self.assertGreater(net_trade(-1,100,90),0)
 def test_funding_sign(self):self.assertEqual(net_trade(-1,100,100,0,0,20),20)
 def test_neutral_exposure(self):
  b=np.linspace(.5,1.5,30)**2;s=np.sin(np.arange(30));w=neutral_weights(s,b,5)
  self.assertAlmostEqual(w.sum(),0,places=12);self.assertAlmostEqual(w@b,0,places=12);self.assertAlmostEqual(abs(w).sum(),.5)
 def test_insufficient_universe(self):self.assertEqual(abs(neutral_weights(np.arange(5.),np.ones(5),3)).sum(),0)
 def test_missing_scores_not_traded(self):
  s=np.arange(30.);s[0]=np.nan;self.assertEqual(neutral_weights(s,np.linspace(1,2,30),5)[0],0)
 def test_epoch_ms(self):self.assertEqual(epoch(1785542400001),1785542400001)
 def test_epoch_s(self):self.assertEqual(epoch(1785542400),1785542400000)
 def test_no_ambiguous_timezone(self):self.assertIsNone(epoch('2026-08-01 01:00'))
 def test_epoch_iso(self):self.assertEqual(epoch('2026-08-01T09:00:00+09:00'),1785542400000)
 def test_datecluster_minimum(self):self.assertIsNone(cluster_ci([1,2,3],['x','x','x']))
 def test_monthly_return_chaining(self):
  z=equity_stats([stamp('2026-01-01'),stamp('2026-01-31'),stamp('2026-02-01'),stamp('2026-02-28')],[1,1.1,1.15,1.21]);self.assertAlmostEqual(z['monthly_pct']['2026-02'],10)
 def test_flat_has_no_oi_signal(self):
  n=500;d=pd.DataFrame({'close':np.ones(n)*100,'quote_volume':np.ones(n)*100,'taker_buy_quote_volume':np.ones(n)*50,'oi':np.ones(n)*1000})
  m,_,_,_=candidate_masks(d);self.assertFalse(any(x.any() for x in m.values()))
 def test_oi_unit_invariance(self):
  n=500;d=pd.DataFrame({'close':np.ones(n)*100,'quote_volume':np.ones(n)*100,'taker_buy_quote_volume':np.ones(n)*50,'oi':np.ones(n)*1000})
  d['oi_value']=d.oi*d.close;d.loc[490:,'oi_value']*=100;m,_,_,dc=candidate_masks(d);self.assertEqual(dc.iloc[-1],0)
 def test_pullback_cannot_enter_before_observation(self):
  t=stamp('2026-01-01');grid=np.arange(t,t+91*60000,60000);c=np.ones(91)*100;c[7]=96;c[8]=97
  d=pd.DataFrame({'open':c,'high':c+.1,'low':c-.1,'close':c},index=grid);z=causal_pullback(d,t);self.assertEqual(z['entry_time'],t+9*60000)
 def test_pullback_does_not_use_later_peak(self):
  t=stamp('2026-01-01');grid=np.arange(t,t+91*60000,60000);c=np.ones(91)*100;c[50]=200
  d=pd.DataFrame({'open':c,'high':c,'low':c,'close':c},index=grid);self.assertEqual(causal_pullback(d,t)['status'],'no_pullback_entry')
if __name__=='__main__':unittest.main()
