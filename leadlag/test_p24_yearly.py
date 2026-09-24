import unittest
from p24_yearly import combine,posterior
class Tests(unittest.TestCase):
 def test_missing_coin_incomplete(self):self.assertFalse(combine({'X':{'incomplete':False,'daily':{1:10000},'records':[]}})[0])
 def test_no_history_no_probability(self):self.assertEqual(posterior([],[])['status'],'insufficient_history')
 def test_posterior(self):self.assertAlmostEqual(posterior([{'net_bp':1}]*60,[{'net_bp':-1}])['p_profit_frozen'],61/62)
 def test_test_outcome_cannot_change_prediction(self):
  tr=[{'net_bp':1}]*60
  self.assertEqual(posterior(tr,[{'net_bp':1}])['p_profit_frozen'],posterior(tr,[{'net_bp':-1}])['p_profit_frozen'])
if __name__=='__main__':unittest.main()
