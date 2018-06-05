__author__ = ''

import unittest
from Securities.AssBond import AssBondFixRate, AssBondFloater
from Models.AssPortfolioModel import AssPortfolioModelBase
from Infra.CurveAggregator import create_curve_aggregator
import datetime as dt
from Managers.ProjectionManager import ProjectionManager

import pandas
import numpy as np

from Managers.MarketDataManager import MARKET_DATA_MANAGER
from lib.utils import dcf_act_act

TEST_FOLDER_PATH = "./test_case/"
TEST_ASSET_PORTFOLIO = 'test_asset_portfolio.csv'


class TestAssModel(unittest.TestCase):

    def loadAndAssertEqual(self, df, filename):
        benchmark = pandas.DataFrame.from_csv(TEST_FOLDER_PATH + filename)
        diff = np.array(abs(df - benchmark))

        if len(diff):
            self.assertAlmostEqual(np.array(abs(diff)).max(), 0.0, places=9)
        else:
            self.assertTrue(True)Ting

    def test_asset_portfolio(self):
        MARKET_DATA_MANAGER.reset()
        MARKET_DATA_MANAGER.setup(dt.date(2014,4,15))
        # Setup a bond
        asset1 = AssBondFixRate(face=100,
                                   coupon_rate=0.05,
                                   frequency=4,
                                   issue_date=dt.date(2014, 4, 15),
                                   expiration_date=dt.date(2015, 4, 15),
                                   dcf=dcf_act_act,
                                   pricing_model=None,
                                   name='Sample Bond 1',
                                )

        # or a "float rate bond" but provided with a scenario generator gives you flat rate
        # a LIBOR 3M +50 bps floater
        asset2 = AssBondFloater(rate_index_name='LIBOR_3M', spread=0.005)

        pricing_date = dt.date(2014, 4, 15)

        # create a portfolio
        port = AssPortfolioModelBase([asset1,asset2], [1,1])
        port.create_iterator(dt.date(2014, 4, 15))
        print("-----here is my portfolio------------")
        print(port)

        metrics = ['Date', 'CF_In:Interest', 'CF_In:Principal']

        crv_aggregator = create_curve_aggregator(metrics)

        params = {'pricing date': pricing_date, 'periods': 60, 'frequency': 'MS'}
        proj_mgr = ProjectionManager(crv_aggregator, port, **params)
        proj_mgr.run()

        df = crv_aggregator.to_dataframe()

        self.loadAndAssertEqual(df, TEST_ASSET_PORTFOLIO)


