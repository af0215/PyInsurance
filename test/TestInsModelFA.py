__author__ = 'Ting'

import unittest
import datetime as dt

import pandas as pd
import numpy as np

from Account.InsAcct import InsAcct
import Products.InsCreditRate as isr
import Products.InsFee as mif
import Products.InsRider as mir
from lib.utils import extract_strict
from Models.InsMortModel import InsMortModel
from Models.InsLapseModel import LapseStatic, SurrenderCharge
from lib.insurance import InsStepFunc
from lib.constants import DAYS_PER_YEAR
from Infra.CurveAggregator import create_curve_aggregator
from Models.InsModelFA import InsModelFA
from Products.InsProduct import InsProduct


TEST_FOLDER_PATH = "./test_case/"
FA_ACCT_ITER_FILE = "test_fa_model.csv"


class TestInsModelFA(unittest.TestCase):
    def setUp(self):
        pass

    def test_simple_fa(self):
        df = self.run_fa_model()
        benchmark = pd.DataFrame.from_csv(TEST_FOLDER_PATH + FA_ACCT_ITER_FILE)
        diff = df - benchmark

        self.assertAlmostEqual(np.array(diff).max(), 0.0, places=9)

    def tearDown(self):
        pass

    def run_fa_model(self):
        raw_input = {"Acct Value": 1344581.6,
                     "Attained Age": 52.8,
                     "ID": "000001",
                     "Issue Age": 45.1,
                     "Issue Date": dt.date(2005, 6, 22),
                     "Initial Date": dt.date(2013, 2, 1),
                     "Maturity Age": 90,
                     "Population": 1,
                     "Riders": dict({}),
                     "ROP Amount": 1038872.0,
                     "Gender": "F",
                     "RPB": 1038872.0,
                     "Free Withdrawal Rate": 0.1,
                     "Asset Names": ["Credit Account"],
                     "Asset Values": [1344581.6]}

        # For now, we assume the init_date is month begin
        step_per_year = 12
        credit_rate = 0.03
        periods = 360
        init_date = dt.date(2013, 2, 1)
        pricing_date = init_date
        # Set up the investment index
        credit_rider = isr.InsCreditRateFixed(credit_rate)

        # Set up non-rider fees
        annual_fee_rate = 0.01
        annual_booking_fee = 100
        mgmt_fee = mif.InsFeeProp(annual_fee_rate, fee_name="Mgmt Fee")
        booking_fee = mif.InsFeeConst(annual_booking_fee, fee_name="Booking Fee")
        fees = [mgmt_fee, booking_fee]

        # Set up rider
        db_rider_fee_rate = 0.005
        db_rider = mir.InsRiderDB(extract_strict(raw_input, "ROP Amount"), db_rider_fee_rate, rider_name="UWL")
        riders = [db_rider]

        # Setup investment index
        inv_index = credit_rider.inv_index(init_date, periods, step_per_year)

        # Setup iteration
        product = InsProduct(riders, fees, inv_index)
        acct = InsAcct(raw_input, product)

        # Setup lapse function and lapse model
        xs = [0]
        ys = [0.0, 0.1]
        lapse_model = LapseStatic(InsStepFunc(xs, ys))

        # Setup surrender charge
        xs = [0]
        ys = [100, 100]
        fixed_charge_func = InsStepFunc(xs, ys)
        xs = [0, 1, 2]
        ys = [0.0, 0.3, 0.2, 0.0]
        pct_charge_func = InsStepFunc(xs, ys)
        surrender_charge = SurrenderCharge(fixed_charge_func, pct_charge_func)

        # Setup mortality function and mortality model
        xs = [x for x in range(0, 100)]
        ys = [0.01] * 100
        ys.append(float('inf'))
        mort_model = InsMortModel(InsStepFunc(xs, ys))

        # Setup FA Model
        model = InsModelFA(acct, lapse_model, mort_model, surrender_charge)
        model_iter = model.create_iterator(pricing_date)

        # model iterator to evolve the model_iter to move forward
        metrics = ['Account Value',
                   'Active Population',
                   'Benefit Base.UWL',
                   'Rider Fee.UWL',
                   'Benefit.UWL',
                   'Fee.Mgmt Fee',
                   'Fee.Booking Fee',
                   'Date',
                   'Attained Age',
                   'Anniv Flag',
                   'Death',
                   'Lapse',
                   'Paid Benefit.UWL',
                   'Surrender Charge',
                   ]
        crv_aggregator = create_curve_aggregator(metrics)

        time_line = pd.date_range(init_date, periods=60, freq='MS').date
        prev_date = init_date
        for d in time_line:
            if d != init_date:
                year_frac = (d - prev_date).days / DAYS_PER_YEAR
                model_iter.next(d, year_frac)
            crv_aggregator.collect_element(model_iter)
            prev_date = d
        df = crv_aggregator.to_dataframe()
        return df


if __name__ == '__main__':
    unittest.main()