__author__ = 'Ting'

import unittest
import datetime as dt

import pandas as pd
import numpy as np

import Products.InsCreditRate as isr
import Products.InsFee as mif
import Products.InsRider as mir
from lib.utils import extract_strict
from Account.InsAcct import InsAcct
from Products.InsProduct import InsProduct


TEST_FOLDER_PATH = "./test_case/"
FA_ACCT_ITER_FILE = "test_fa_acct_iter.csv"


class TestInsAcct(unittest.TestCase):
    def setUp(self):
        pass

    def test_simple_calculation(self):
        df = self.run_fa_acct_iter()
        benchmark = pd.DataFrame.from_csv(TEST_FOLDER_PATH + FA_ACCT_ITER_FILE)
        diff = df - benchmark

        self.assertAlmostEqual(np.array(abs(diff)).max(), 0.0, places=9)

    def tearDown(self):
        pass

    def run_fa_acct_iter(self):
        raw_input = {
            "Acct Value": 1344581.6,
            "Attained Age": 52.8,
            "DB Rider Name": "Step-up",
            "WB Rider Name": "PP",
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
            "Asset Values": [1344581.6]
        }

        # For now, we assume the init_date is month begin
        step_per_year = 12
        credit_rate = 0.03
        periods = 360
        init_date = dt.date(2013, 2, 1)
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
        acct_iter = acct.acct_iterator()

        # model iterator to evolve the model_iter to move forward
        time_line = pd.date_range(init_date, periods=60, freq='3MS').date
        rs = {'Account Value': [], 'UWL Benefit Base': [], 'UWL Fee': [], 'UWL Benefit': [],
              'Mgmt Fee': [], 'Booking Fee': [], 'Date': [], 'Age': [], 'Pre Fee Acct': [],
              'Anniv Flag': []}
        for d in time_line:
            if d == init_date:
                rs['Account Value'].append(acct_iter.acct_value)
                rs['UWL Fee'].append(acct_iter.rider_fee("UWL"))
                rs['Date'].append(acct_iter.date)
                rs['Age'].append(acct_iter.attained_age)
                rs['Booking Fee'].append(acct_iter.non_rider_fee("Booking Fee"))
                rs['Mgmt Fee'].append(acct_iter.non_rider_fee('Mgmt Fee'))
                rs['UWL Benefit'].append(acct_iter.rider_benefit('UWL'))
                rs['UWL Benefit Base'].append(acct_iter.rider_benefit_base('UWL'))
                rs['Pre Fee Acct'].append(acct_iter.acct_value_pre_fee)
                rs['Anniv Flag'].append(acct_iter.anniv_flag)
            else:
                acct_iter.next(d)
                rs['Account Value'].append(acct_iter.acct_value)
                rs['UWL Fee'].append(acct_iter.rider_fee("UWL"))
                rs['Date'].append(acct_iter.date)
                rs['Age'].append(acct_iter.attained_age)
                rs['Booking Fee'].append(acct_iter.non_rider_fee("Booking Fee"))
                rs['Mgmt Fee'].append(acct_iter.non_rider_fee('Mgmt Fee'))
                rs['UWL Benefit'].append(acct_iter.rider_benefit('UWL'))
                rs['UWL Benefit Base'].append(acct_iter.rider_benefit_base('UWL'))
                rs['Pre Fee Acct'].append(acct_iter.acct_value_pre_fee)
                rs['Anniv Flag'].append(acct_iter.anniv_flag)
        df = pd.DataFrame.from_dict(rs)
        df.set_index(['Date'], inplace=True)
        df['Anniv Flag'] = df['Anniv Flag'].astype(int)

        return df

if __name__ == '__main__':
    unittest.main()