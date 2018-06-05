__author__ = 'Ting'

import unittest
from Securities.AssBond import AssBondFixRate, AssBondFloater
import datetime as dt
from lib.utils import dcf_act_act
import pandas
import numpy as np

from Managers.MarketDataManager import MARKET_DATA_MANAGER
from Managers.ScenarioManager import ScenarioGenerator, FixRateEngine
from lib.constants import BDAYS_PER_YEAR
from Infra.IndexProvider import IndexProvider

TEST_FOLDER_PATH = "./test_case/"
FIXED_MATURED_1 = 'fixed_bond_matured_1.csv'
FIXED_MATURED_2 = 'fixed_bond_matured_2.csv'
FIXED_TOCOME_1 = 'fixed_bond_to_come_1.csv'
FIXED_TOCOME_2 = 'fixed_bond_to_come_2.csv'

FLOATER_MATURED_1 = 'floater_bond_matured_1.csv'
FLOATER_MATURED_2 = 'floater_bond_matured_2.csv'
FLOATER_TOCOME_1 = 'floater_bond_to_come_1.csv'
FLOATER_TOCOME_2 = 'floater_bond_to_come_2.csv'

# errr, better not change these below: since the files are generated using these parameters
COUPON_RATE = 0.05
SPREAD = 0.005
LIBOR_3M = 0.05

class TestAssModel(unittest.TestCase):

    def test_fixed_bond(self):
        # 1/ create a fixed rate bond
        b = AssBondFixRate(face=100,
                           coupon_rate=COUPON_RATE,
                           frequency=4,
                           issue_date=dt.date(2014, 4, 15),
                           expiration_date=dt.date(2016, 4, 15),
                           dcf=dcf_act_act,
                           pricing_model=None,
                           name='Sample Bond 1',
                           )

        self.assertEqual(b.coupon_rate(dt.date(2014, 6, 12)), COUPON_RATE,
                         'Coupon Rate %s not the same as Expected :%s' %
                         (b.coupon_rate(dt.date(2014, 6, 12)), COUPON_RATE))

        it = b.bond_iterator()

        self.loadAndAssertEqual(it.cash_flow_matured, FIXED_MATURED_1)
        self.loadAndAssertEqual(it.cash_flow_schedule_to_come, FIXED_TOCOME_1)

        it.next(_next_date=b.coupon_schedule()[1])

        self.loadAndAssertEqual(it.cash_flow_matured, FIXED_MATURED_2)
        self.loadAndAssertEqual(it.cash_flow_schedule_to_come, FIXED_TOCOME_2)

    def test_floater_bond(self):
        # a LIBOR 3M +50 bps floater
        MARKET_DATA_MANAGER.reset()
        MARKET_DATA_MANAGER.market_data_date = dt.date(2014,4,15)

        libor_3m = IndexProvider(pandas.TimeSeries(index=[dt.date(2014,4,15)],data=[LIBOR_3M]), index_name='LIBOR_3M')
        ir_eng = FixRateEngine(LIBOR_3M)
        scen_gen_libor = ScenarioGenerator([libor_3m], ir_eng, **{'max_time_step': 5. / BDAYS_PER_YEAR})
        MARKET_DATA_MANAGER._index_table['LIBOR_3M'] = libor_3m
        MARKET_DATA_MANAGER._scen_gen_table['LIBOR_3M'] = scen_gen_libor

        b = AssBondFloater(rate_index_name='LIBOR_3M', spread=SPREAD)

        self.assertEqual(b.coupon_rate(dt.date(2014, 6, 12)), SPREAD + LIBOR_3M,
                         'Coupon Rate %s not the same as Expected :%s' %
                         (b.coupon_rate(dt.date(2014, 6, 12)), SPREAD + LIBOR_3M))

        it = b.bond_iterator()

        self.loadAndAssertEqual(it.cash_flow_matured, FLOATER_MATURED_1)
        self.loadAndAssertEqual(it.cash_flow_schedule_to_come, FLOATER_TOCOME_1)

        it.next(_next_date=b.coupon_schedule()[1])

        self.loadAndAssertEqual(it.cash_flow_matured, FLOATER_MATURED_2)
        self.loadAndAssertEqual(it.cash_flow_schedule_to_come, FLOATER_TOCOME_2)

    def loadAndAssertEqual(self, df, filename):
        benchmark = pandas.DataFrame.from_csv(TEST_FOLDER_PATH + filename)
        diff = np.array(abs(df - benchmark))

        if len(diff):
            self.assertAlmostEqual(np.array(abs(diff)).max(), 0.0, places=9)
        else:
            self.assertTrue(True)