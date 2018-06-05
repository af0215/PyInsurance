__author__ = ''

import unittest
import datetime as dt

import numpy as np
import pandas as pd

import Infra.IndexProvider as ip
from Products.InsCreditRate import InsCreditRateFixed, InsCreditRateMutualFunds
from Managers.ScenarioManager import EqBSEngine, ScenarioGenerator
from lib.constants import BDAYS_PER_YEAR
from Managers.MarketDataManager import MARKET_DATA_MANAGER


class TestInsuranceLib(unittest.TestCase):
    def setUp(self):
        pass
Ting
    def test_fixed_credit_rate(self):
        start_date = dt.date(2013, 2, 1)
        credit_rate = 0.03
        step_per_year = 12
        periods = 360
        credit_rider = InsCreditRateFixed(credit_rate)
        inv_index = credit_rider.inv_index(start_date, periods, step_per_year)
        data = np.array(inv_index.data)
        returns = [y/x for x, y in zip(data[:-2], data[1:])]
        self.assertAlmostEqual(np.sum([(x-1-credit_rate/step_per_year)**2 for x in returns]), 0.0)

    def test_variable_credit_rate(self):
        # TODO: bear in mind that the fund info fee is not implemented yet
        init_df = [pd.TimeSeries(data=np.exp(np.random.randn(10)),
                                 index=pd.date_range(start=dt.date(2014, 1, 1), periods=10, freq='D').date,
                                 name='stock A'),
                   pd.TimeSeries(data=np.exp(np.random.randn(10)),
                                 index=pd.date_range(start=dt.date(2014, 1, 1), periods=10, freq='D').date,
                                 name='stock B'),
                   ]
        eq_index = [ip.IndexProvider(init_df[0], 'stock A'), ip.IndexProvider(init_df[1], 'stock B')]
        sim_engine = EqBSEngine(np.array([0.02, 0.02]), np.array([0.2, 0.25]), corr=np.array([[1., 0.3], [0.3, 1.]]))
        simulator = ScenarioGenerator(eq_index, sim_engine, **{'max_time_step': 5. / BDAYS_PER_YEAR})

        MARKET_DATA_MANAGER.reset()
        MARKET_DATA_MANAGER.setup(dt.date(2014, 1, 1))
        MARKET_DATA_MANAGER.index_table['stock A'] = eq_index[0]
        MARKET_DATA_MANAGER.index_table['stock B'] = eq_index[1]
        MARKET_DATA_MANAGER.scen_gen_table['stock A'] = simulator
        MARKET_DATA_MANAGER.scen_gen_table['stock B'] = simulator

        mix_weight = 0.4

        fund_info = {'Fund A':
                         {
                             'Allocations': {
                                 'stock A': 1,
                                 'stock B': 0,
                             },
                             'Management Fee': 0.01,
                             'Description': 'blah blah',
                         },
                     'Fund B':
                         {
                             'Allocations': {
                                 'stock A': 0,
                                 'stock B': 1,
                             },
                             'Management Fee': 0.01,
                             'Description': 'blah blah',
                         },
                     'Fund C':
                         {
                             'Allocations': {
                                 'stock A': mix_weight,
                                 'stock B': 1-mix_weight,
                             },
                             'Management Fee': 0.01,
                             'Description': 'blah blah',
                         },
        }

        ic = InsCreditRateMutualFunds(fund_info=fund_info)
        index = ic.inv_index(dt.date(2014, 1, 10), periods=360, step_per_year=12)

        # since Fund A is 100% stock A, the return should be equal
        self.assertEqual(np.array(abs(eq_index[0].data - index.data['Fund A'])).max(),0.0)

        # since Fund B is 100% stock B, the return should be equal
        self.assertEqual(np.array(abs(eq_index[1].data - index.data['Fund B'])).max(),0.0)

        # Fund C is 50% 50% mix of stock A and B
        self.assertEqual(np.array(abs(mix_weight * eq_index[0].data + (1-mix_weight) * eq_index[1].data)
                                  - index.data['Fund C']).max(), 0.0)

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()