"""
Objects for dealing with

This module provides a number of objects (mostly functions) useful for
dealing with Laguerre series, including a `Laguerre` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
- `InsAcct`: used to record a UNIT account basic (original) information.
- `InsAcctIter`: InsAcctIter is used to evolve a UNIT account based on account(InsAcct) information and
                 input form the model(e.g., InsModelFA), such as the withdrawal rate. The change of population
                 due to mortality, utilization, and lapse is recorded in model(distribution of states, i.e,
                 active, surrendered, dead, etc).
                 This object stores the logic (e.g., InsCreditRate), states (e.g., account value)
                 and flows(e.g., fees)
See also
--------
`?`

"""


# TODO: 1. move the properties from InsAcct to InsAcctIter, InsAcct only need the initial info

__author__ = 'Ting'

import numpy as np
import pandas as pd

from lib.utils import extract_or_default, extract_strict, check_equal, inv_return
import Products.InsCreditRate as isr
import Products.InsFee as mif
import Products.InsRider as mir
from lib.constants import DAYS_PER_YEAR
import Infra.IndexProvider as ip
from Products.InsProduct import InsProduct


class InsAcct(object):
    def __init__(self,
                 cell,
                 product,
                 ):
        # temporary solution for compatibility
        riders = product.riders
        fees = product.fees
        inv_index = product.inv_index
        product.bind_to_cell(cell)

        self._o_acct_value = extract_strict(cell, "Acct Value")
        self._id = extract_or_default(cell, "ID", None)
        self._issue_date = extract_strict(cell, "Issue Date")
        self._issue_age = extract_strict(cell, "Issue Age")
        self._attained_age = extract_strict(cell, "Attained Age")
        self._gender = extract_or_default(cell, "Gender", "M")
        self._maturity_age = extract_strict(cell, "Maturity Age")
        self._inv_index = inv_index
        self._init_date = extract_strict(cell, "Initial Date")

        assert isinstance(riders, list), "Riders should be of type list"
        assert isinstance(fees, list), "Fees should be of type list"

        self._asset_names = np.array(extract_strict(cell, "Asset Names"))
        self._asset_values = np.array(extract_strict(cell, "Asset Values"))
        if len(self._asset_names) != len(self._asset_values):
            raise ValueError("# of asset values is different from # of asset names")
        if not check_equal(sum(self._asset_values), self._o_acct_value):
            raise ValueError("Account value is not same as total asset value")
        self._fees = fees
        self._riders = riders

        self._rider_name_index = dict()
        self._fee_name_index = {v.fee_name: i for i, v in enumerate(self._fees)}
        self._rider_name_index = {v.rider_name: i for i, v in enumerate(self._riders)}

    def rider_name_index(self, rider_name):
        return self._rider_name_index.get(rider_name)

    def fee_name_index(self, fee_name):
        return self._fee_name_index.get(fee_name)

    def acct_iterator(self):
        return InsAcctIter(self)

    @property
    def id(self):
        return self._id

    @property
    def age(self):
        return self._attained_age

    @property
    def issue_date(self):
        return self._issue_date

    @property
    def o_acct_value(self):
        return self._o_acct_value

    @property
    def riders(self):
        return self._riders

    @property
    def fees(self):
        return self._fees

    @property
    def asset_names(self):
        return self._asset_names

    @property
    def asset_values(self):
        return self._asset_values

    @property
    def attained_age(self):
        return self._attained_age

    @property
    def issue_age(self):
        return self._attained_age

    @property
    def inv_index(self):
        return self._inv_index

    @property
    def init_date(self):
        return self._init_date


class InsAcctIter(object):
    def __init__(self,
                 acct,
                 ):
        self._acct = acct
        self._attained_age = acct.attained_age
        self._issue_age = acct.issue_age
        self._duration = self._attained_age - self._issue_age
        self._issue_date = acct.issue_date
        self._acct_value = acct.o_acct_value
        self._acct_value_pre_fee = self._acct_value
        self._riders = acct.riders
        self._fees = acct.fees
        self._anniv_flag = False
        self._date = acct.init_date
        self._inv_index = acct.inv_index
        self._asset_names = acct.asset_names.copy()
        self._asset_values = acct.asset_values.copy()
        self._rider_fees = [0.0] * len(self._riders)
        self._non_rider_fees = [0.0] * len(self._fees)
        self._rider_benefits = [rider.benefit(self) for rider in self._riders]
        self._year_frac = None

    def next(self, _next_date=None, _year_frac=None):
        # The logic is as follows:
        # 1. As end of previous period (t-1), update the account value and calculate the end of period fees,
        #    such as the management fees.
        #
        # 2. As beginning of the new period, update the benefit base and calculate the beginning of the period fees,
        #    such as the rider/insurance related fees.

        # TODO:
        # 1. date calculation
        # 2. acct_value_update, which updates the account value with indexed investment return

        # First update basic information as date, age, anniversary flag and etc.
        _prev_date = self._date
        _prev_age = self._attained_age
        self._date = _next_date

        self._year_frac = _year_frac or (self._date - _prev_date).days / DAYS_PER_YEAR
        self._attained_age += self._year_frac
        self._duration += self._year_frac
        self.check_anniversary(_prev_age, self._attained_age)

        # End of period calculation, including account value update and management/other fee calculation
        self._asset_values += self.asset_return(_prev_date)
        self._acct_value = sum(self._asset_values)
        self._acct_value_pre_fee = self._acct_value

        # Update rider fees and assume the fee is proportionally subtracted from each asset
        self._non_rider_fees = [fee_item.fee(self) for fee_item in self._fees]
        self._acct_value -= sum(self._non_rider_fees)
        self._asset_values *= self._acct_value/self._acct_value_pre_fee
        self._rider_benefits = [rider.benefit(self) for rider in self._riders]
        # rider benefits is before rider fees but after management fees and fund fees are charged ???

        # Beginning of period calculation, including rider benefit base update and rider fees calculation
        _acct_value_pre_rider_fee = self._acct_value
        for rider in self._riders:
            rider.set_benefit_base(self)
        self._rider_fees = [rider.rider_fee(self) for rider in self._riders]
        self._acct_value -= sum(self._rider_fees)
        self._asset_values *= self._acct_value/_acct_value_pre_rider_fee
        # End of update.

    def withdrawal(self, amount):
        # TODO: Leave here as an interface for future implementation of withdrawal( partial or complete )
        # Note that, this function only handles the accounting given withdrawal amount. The withdrawal
        # amount is projected in InsModel
        pass

    def new_premium(self, amount):
        # TODO: Leave here as an interface for new premium in the future
        # Note that, this function only handles the accounting given new premium amount. The premium amount
        # is projected in InsModel
        pass

    def check_anniversary(self, prev_age, age):
        self._anniv_flag = True if prev_age//1 == (age//1 - 1) else False

    def asset_return(self, _prev_date):

        # reorder the inv index to be consistent with asset names if needed.
        # if the investment index is of a single asset: use that asset for all accounts -> dangerous
        # TODO: more strict asset setting: crediting index must have the same asset name as self._asset_names

        return_index = self._inv_index if self._inv_index.n_assets == 1 else \
            ip.IndexProvider(self._inv_index.data[self._asset_names])
        return inv_return(_prev_date, self._date, self._asset_values, return_index)

    def rider_fee(self, rider_name):
        return self._rider_fees[self._acct.rider_name_index(rider_name)]

    def non_rider_fee(self, fee_name):
        return self._non_rider_fees[self._acct.fee_name_index(fee_name)]

    def rider_benefit_base(self, rider_name):
        return self._riders[self._acct.rider_name_index(rider_name)].benefit_base

    def rider_benefit(self, rider_name):
        return self._rider_benefits[self._acct.rider_name_index(rider_name)]

    def rider_benefits_dict(self):
        return {r.rider_name: self._rider_benefits[self._acct.rider_name_index(r.rider_name)]
                for r in self._acct.riders}

    @property
    def attained_age(self):
        return self._attained_age

    @property
    def acct_value(self):
        return self._acct_value

    @property
    def acct_value_pre_fee(self):
        return self._acct_value_pre_fee

    @acct_value.setter
    def acct_value(self, acct_value):
        self._acct_value = acct_value

    @property
    def riders(self):
        return self._riders

    @property
    def fees(self):
        return self._fees

    @property
    def date(self):
        return self._date

    @property
    def year_frac(self):
        return self._year_frac

    @property
    def attained_age(self):
        return self._attained_age

    @property
    def duration(self):
        return self._duration

    @property
    def total_rider_fees(self):
        return sum(self._rider_fees)

    @property
    def total_non_rider_fees(self):
        return sum(self._non_rider_fees)

    @property
    def date(self):
        return self._date

    @property
    def anniv_flag(self):
        return self._anniv_flag

    @property
    def rider_fee_vec(self):
        return self._rider_fees

    @property
    def non_rider_fee_vec(self):
        return self._non_rider_fees


# -- Example --
if __name__ == "__main__":
    import datetime as dt
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

    # This is a temporary code to check the results, the code plays the role of a
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

    df.to_csv('C:\\Temp\\rs.csv')
