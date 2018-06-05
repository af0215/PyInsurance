__author__ = ''

import unittest
from Securities.AssBond import AssBondBase, AssBondFixRate, AssBondFloater
import datetime as dt
from lib.utils import dcf_act_act
from Models.AssModel import NoDefaultCreditModel, AssModelBase
from Infra.CurveAggregator import create_curve_aggregator
from Managers.ProjectionManager import ProjectionManager
import pandas as pd
import numpy as np

TEST_FOLDER_PATH = "./test_case/"
FIXED_FILE = "test_assmodel_fixed.csv"
FLOATER_FILE = "test_assmodel_floater.csv"

class TestAssModel(unittest.TestCase):
    def setUp(self):
        pass

    def test_fixed_bond(self):
        asset = AssBondFixRate(face=100,
                       coupon_rate=0Ting.05,
                       frequency=4,
                       issue_date=dt.date(2014, 4, 15),
                       expiration_date=dt.date(2016, 4, 15),
                       dcf=dcf_act_act,
                       pricing_model=None,
                       name='Sample Bond 1',
                       )

        df = self.run_bond(asset)
        benchmark = pd.DataFrame.from_csv(TEST_FOLDER_PATH + FIXED_FILE)
        diff = df - benchmark

        self.assertAlmostEqual(np.array(abs(diff)).max(), 0.0, places=9)

    def test_floater_bond(self):
        # by default, MKT data manager generate libor_3m = 0.05
        asset = AssBondFloater(rate_index_name='LIBOR_3M', spread=0.005)

        df = self.run_bond(asset)
        benchmark = pd.DataFrame.from_csv(TEST_FOLDER_PATH + FLOATER_FILE)
        diff = df - benchmark

        self.assertAlmostEqual(np.array(abs(diff)).max(), 0.0, places=9)

    def run_bond(self, asset):
        from Managers.MarketDataManager import MARKET_DATA_MANAGER
        MARKET_DATA_MANAGER.reset()
        MARKET_DATA_MANAGER.setup(dt.date(2014,4,15))

        credit_model = NoDefaultCreditModel()

        # Setup Asset Model
        pricing_date = dt.date(2014, 4, 15)
        model = AssModelBase(asset, credit_model)
        model_iter = model.create_iterator(pricing_date)

        metrics = ['Date', 'CF_In:Interest', 'CF_In:Principal']

        crv_aggregator = create_curve_aggregator(metrics)

        params = {'pricing date': pricing_date, 'periods': 60, 'frequency': 'MS'}
        proj_mgr = ProjectionManager(crv_aggregator, model_iter, **params)
        proj_mgr.run()

        df = crv_aggregator.to_dataframe()
        #df.to_csv('test_1')
        #print(df)
        return df

    def tearDown(self):
        pass
