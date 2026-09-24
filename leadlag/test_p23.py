"""Offline synthetic invariants. Passing tests are not profitable market evidence."""
import unittest
from dataclasses import replace
import numpy as np,pandas as pd
from p23_engine import *
from p23_run import parse_bea,et,ms

def frame(n=180,start=0,price=100.):
    a=np.ones(n)*price
    return pd.DataFrame({'open':a,'high':a+.02,'low':a-.02,'close':a,'quote_volume':np.ones(n)*10000,'buy_quote':np.ones(n)*5000},index=np.arange(start,start+n*MIN,MIN))
def cand(t=0):
    return {'decision':t,'limit':100.1,'stop':98.0,'target':104.0,'planned_loss_per_unit':2.3,'prior_turnover':1e9}
class Tests(unittest.TestCase):
    def test_flat_fees(self):self.assertAlmostEqual(cost_pnl(100,100,10,0),-.2)
    def test_cost_is_monotonic(self):self.assertLess(cost_pnl(100,102,10,5),cost_pnl(100,102,10,2))
    def test_jan_et(self):self.assertEqual(et('2025-01-15','08:30').astimezone(timezone.utc).hour,13)
    def test_jul_et(self):self.assertEqual(et('2025-07-15','08:30').astimezone(timezone.utc).hour,12)
    def test_bea_embargo(self):
        x=parse_bea(b'<p>EMBARGOED UNTIL RELEASE AT 8:30 a.m. EST, Friday, January 26, 2024</p>');self.assertEqual(x[:2],('2024-01-26','08:30'))
    def test_bea_dst_mismatch(self):
        with self.assertRaises(ValueError):parse_bea(b'EMBARGOED UNTIL RELEASE AT 8:30 a.m. EST, Wednesday, August 26, 2026')
    def test_no_date_guess(self):self.assertIsNone(parse_bea(b'updated August 26, 2026 at 8:30'))
    def test_bad_ohlc(self):
        d=frame();d.loc[0,'low']=101;self.assertFalse(valid_frame(d))
    def test_bad_flow(self):
        d=frame();d.loc[0,'buy_quote']=10001;self.assertFalse(valid_frame(d))
    def test_flat_no_shock(self):
        c,r=observe(frame(),60*MIN,10,'B2_price_flow');self.assertIsNone(c);self.assertEqual(r,'drop_too_small')
    def test_future_never_used(self):
        d=frame();a=observe(d,60*MIN,10,'B1_price');d.loc[70*MIN:,'low']=1;b=observe(d,60*MIN,10,'B1_price');self.assertEqual(a,b)
    def test_missing_pre_no_trade(self):
        d=frame().drop(20*MIN);c,r=observe(d,60*MIN,10,'B1_price');self.assertIsNone(c);self.assertEqual(r,'data_gap_or_invalid')
    def test_min_observation(self):self.assertEqual(observe(frame(),60*MIN,5,'B0_shock')[1],'outside_window')
    def test_closed_bars_exact(self):
        d=frame();d.loc[69*MIN,'low']=90;c,r=observe(d,60*MIN,9,'B0_shock');self.assertEqual(r,'drop_too_small')
    def test_exact_arrival_no_future_substitution(self):
        d=frame().drop(MIN);self.assertEqual(evaluate(d,cand())['status'],'unresolved_entry')
    def test_ioc_no_fill_gap(self):
        d=frame(price=101);self.assertEqual(evaluate(d,cand())['status'],'no_fill_price_cap')
    def test_entry_below_stop_abstains(self):
        d=frame(price=97);self.assertEqual(evaluate(d,cand())['status'],'no_fill_invalidated')
    def test_both_barriers_loss_first(self):
        d=frame();d.loc[MIN,'low']=97;d.loc[MIN,'high']=105;o=evaluate(d,cand());self.assertEqual(o['exit_reason'],'stop');self.assertTrue(o['both_barriers'])
    def test_gap_stop_worse(self):
        d=frame();d.loc[2*MIN,['open','low','high','close']]=[95,94,96,95];o=evaluate(d,cand());self.assertEqual(o['exit_reason'],'gap_stop');self.assertEqual(o['exit_raw'],95)
    def test_timeout_open(self):
        d=frame();o=evaluate(d,cand());self.assertEqual(o['exit_reason'],'timeout');self.assertEqual(o['exit_time_upper'],61*MIN)
    def test_missing_while_open_not_dropped(self):
        d=frame().drop(3*MIN);o=evaluate(d,cand());self.assertEqual(o['status'],'unresolved_open_position')
    def test_target_not_count_later_data(self):
        d=frame();d.loc[3*MIN,'high']=105;a=evaluate(d,cand());d.loc[4*MIN:,'low']=1;b=evaluate(d,cand());self.assertEqual(a,b)
    def test_clock_exit_before_new_event(self):
        o=evaluate(frame(),cand(),30*MIN);self.assertEqual(o['exit_time_upper'],29*MIN);self.assertEqual(o['exit_reason'],'clock_exit')
    def test_future_size_cannot_change_outcome(self):
        d=frame();a=evaluate(d,cand());d['quote_volume']*=.01;d['buy_quote']*=.01;b=evaluate(d,cand());self.assertEqual(a,b)
    def test_extra_cost_same_fills(self):
        o=evaluate(frame(),cand());z=fixed_cost(o,5);self.assertEqual(o['entry_time'],z['entry_time']);self.assertLess(z['net_bp'],o['net_bp'])
    def test_unresolved_retained(self):
        z=summarize([{'outcome':{'status':'unresolved_open_position'}}]);self.assertEqual(z['cases'],1);self.assertEqual(z['closed'],0)
    def test_zero_profit_not_success(self):
        z=summarize([]);self.assertIsNone(z['mean_bp'])
    def test_portfolio_gap_not_valid_nav(self):
        p=portfolio([{'outcome':{'status':'unresolved_data'},'candidate':None}]);self.assertTrue(p['incomplete']);self.assertIsNone(p['return_pct'])
    def test_portfolio_one_event(self):
        d=frame();o=evaluate(d,cand());r={'event_id':'E','date':'2025-01-01','coin':'ETHUSDT','candidate':cand(),'outcome':o};r2=dict(r,coin='BTCUSDT');p=portfolio([r,r2]);self.assertEqual(len(p['trades']),1);self.assertEqual(p['trades'][0]['coin'],'BTCUSDT')
    def test_entry_size_no_exceeds_cash_cap(self):
        d=frame();r={'event_id':'E','date':'2025-01-01','coin':'BTCUSDT','candidate':cand(),'outcome':evaluate(d,cand())};p=portfolio([r]);self.assertLessEqual(p['trades'][0]['entry_notional_usdt'],2000)
    def test_missing_macro_rejects_candidate(self):
        from unittest.mock import patch
        with patch('p23_engine.observe',return_value=(cand(),'candidate')):
            c,r=first_signal(frame(),0,'B2_price_flow',macro_gate=lambda *args:(False,'macro_missing'))
            self.assertIsNone(c);self.assertGreater(r['macro_missing'],0)
    def test_decision_before_blackout(self):
        c,r=first_signal(frame(),60*MIN,'B2_price_flow',65*MIN);self.assertIsNone(c);self.assertIn('next_clock_boundary',r)

from datetime import timezone
if __name__=='__main__':unittest.main()
