"""
Objects for dealing with

This module provides a number of objects for bond instruments

Classes
-------
- `AssBondBase`: presenting a plain vanilla textbook bond
- `AssBondBaseIter`: iterator is used to roll bond forward from t_0 to t_1, to determine, what needs to be
                     paid out, mostly
"""

from abc import ABCMeta, abstractmethod
import datetime as dt

import pandas as pd

import Infra.IndexProvider as ip
import lib.constants as const
from lib.utils import dcf_act_act
from lib.constants import DAYS_PER_YEAR
from Managers.ScenarioManager import FixRateEngine, ScenarioGenerator
from Managers.MarketDataManager import MARKET_DATA_MANAGER


class AssAsset(object):
    pass


class AssBondBase(AssAsset):
    __metaclass__ = ABCMeta

    def __init__(self,
                 face=100,
                 frequency=4,
                 issue_date=dt.date(2014, 4, 15),
                 expiration_date=dt.date(2016, 4, 15),
                 dcf=dcf_act_act,
                 pricing_model=None,
                 name='',
                 ):

        if isinstance(face, int):
            face = float(face)

        assert isinstance(face, float), "Face value must be float"
        assert isinstance(issue_date, dt.date), "Issue Date should be of datetime.date"
        assert isinstance(expiration_date, dt.date), "Expiration Date should be of datetime.date"
        assert isinstance(frequency, int), "Frequency needs to be an int"
        assert frequency in const.BOND_COUPON_FREQUENCY, "Frequency not in %s" % const.BOND_COUPON_FREQUENCY
        assert isinstance(name, str), "Name of the bond should be in str"

        self._face = face
        self._frequency = frequency  # number of payment per year
        self._issue_date = issue_date
        self._expiration_date = expiration_date
        self._dcf = dcf
        self._pricing_model = pricing_model  # I don't really need this now since I only care about CF
        self._name = name

    def coupon_schedule(self):
        """ a list of coupon dates
        """

        to_add = self._issue_date + pd.DateOffset(months=int(12/self._frequency))
        to_add = to_add.date()
        coupon_dates = []
        while to_add <= self._expiration_date:
            coupon_dates.append(to_add)
            to_add += pd.DateOffset(months=int(12/self._frequency))
            to_add = to_add.date()

        coupon_dates = [x for x in coupon_dates if self._issue_date < x <= self._expiration_date]
        if coupon_dates[-1] != self._expiration_date:
            print('warning: the last coupon date calculated is not the same as expiration date')

        return coupon_dates

    @property
    def issue_date(self):
        return self._issue_date

    @property
    def expiration_date(self):
        return self._expiration_date

    @property
    def face(self):
        return self._face

    @property
    def name(self):
        return self._name

    @abstractmethod
    def coupon_rate(self, payment_date):
        """
        abstract method: giving a payment day, it will return a coupon rate,
        note that it can be implemented fixing on either the beginning of the period or end of period
        :param payment_date: coupon payment date
        :return: a float representing annualized coupon rate
        """
        return -1

    def bond_iterator(self):
        return AssBondBaseIter(self)

    def cash_flow_interest(self, payment_date):
        if payment_date not in self.coupon_schedule():
            raise NameError('%s is not a payment date' % payment_date)

        prev_coupon_index = ([self._issue_date] + self.coupon_schedule()).index(payment_date)
        prev_coupon_date = ([self._issue_date] + self.coupon_schedule())[prev_coupon_index - 1]

        dcf = abs(self._dcf(prev_coupon_date, payment_date))
        this_coupon = self.face * dcf * self.coupon_rate(payment_date)
        return this_coupon

    def cash_flow_principal(self, payment_date):
        if payment_date not in self.coupon_schedule():
            raise NameError('%s is not a payment date' % payment_date)

        if payment_date == self.expiration_date:
            return self.face
        else:
            return 0.

    def cash_flow_total(self, payment_date):
        return self.cash_flow_principal(payment_date) + self.cash_flow_interest(payment_date)

    def cash_flow_schedule(self):
        dates = self.coupon_schedule()
        results = [[self.cash_flow_interest(x), self.cash_flow_principal(x), self.cash_flow_total(x)] for x in dates]
        df = pd.DataFrame(results, index=dates, columns=['Interest', 'Principal', 'Total'])
        return df

    def __str__(self):
        return self.name


class AssBondFixRate(AssBondBase):

    def __init__(self,
                 face=100,
                 frequency=4,
                 coupon_rate=0.05,
                 issue_date=dt.date(2014, 4, 15),
                 expiration_date=dt.date(2016, 4, 15),
                 dcf=dcf_act_act,
                 pricing_model=None,
                 name=''):

        name_fix = 'Fix Coupon %s%%' % (coupon_rate * 100) if not name else name
        super(AssBondFixRate, self).__init__(face, frequency, issue_date, expiration_date, dcf, pricing_model, name_fix)

        self._fix_coupon_rate = coupon_rate

    def coupon_rate(self, payment_date):
        return self._fix_coupon_rate


class AssBondFloater(AssBondBase):

    def __init__(self,
                 face=100,
                 frequency=4,
                 rate_index_name='LIBOR_3M',
                 spread=0.0,
                 issue_date=dt.date(2014, 4, 15),
                 expiration_date=dt.date(2016, 4, 15),
                 dcf=dcf_act_act,
                 pricing_model=None,
                 name=''):

        name_floater = 'Floater %s + %sbps' % (rate_index_name, spread * 10000) if not name else name
        super(AssBondFloater, self).__init__(face, frequency, issue_date, expiration_date, dcf, pricing_model, name_floater)

        self._rate_index_name = rate_index_name
        self._spread = spread

    def coupon_rate(self, payment_date):
        # TODO: add inareas and the other mode: define two separate functions and add in a bond input, make a switch
        # so far this is fixing on payment date
        idx = MARKET_DATA_MANAGER.get(self._rate_index_name)
        return idx.index_value(payment_date) + self._spread


class AssBondBaseIter(object):
    def __init__(self,
                 bond,
                 ):
        self._bond = bond
        self._prev_date = None
        self._date = bond.issue_date
        self._live = 0.
        self._year_frac = None

        self._next_payment_date = self._bond.coupon_schedule()[0]
        self._cash_flow_schedule_to_come = self._bond.cash_flow_schedule()

        self._cash_flow_matured = self._bond.cash_flow_schedule().loc[[]]  # grab empty roles but retain the columns

    def next(self, _next_date=None, _year_frac=None):
        # 1) to capture age of bond
        # 2) to capture payments from to _next_date (what about accrual?)
        _prev_date = self._date
        self._prev_date = _prev_date
        self._date = _next_date
        self._year_frac = _year_frac or (self._date - _prev_date).days / DAYS_PER_YEAR
        self._live = (self._date - self._bond.issue_date).days / DAYS_PER_YEAR

        schedule = self._bond.coupon_schedule()
        to_come = [x for x in schedule if x > self._date]
        self._next_payment_date = to_come[0] if len(to_come) else None

        cf_schedule = self._bond.cash_flow_schedule()
        self._cash_flow_schedule_to_come = cf_schedule.loc[[d for d in cf_schedule.index if d > self._date]]
        self._cash_flow_matured = cf_schedule.loc[[d for d in cf_schedule.index if _prev_date < d <= self._date]]

    @property
    def cash_flow_schedule_to_come(self):
        return self._cash_flow_schedule_to_come

    @property
    def cash_flow_matured(self):
        return self._cash_flow_matured

    def __str__(self):
        msg = '%s: prev_date: %s --> date: %s\n' % (self._bond.name, self._prev_date, self._date)
        msg += 'CF Matured:\n'
        msg += str(self._cash_flow_matured)
        msg += '\nCF to Come:\n'
        msg += str(self._cash_flow_schedule_to_come)
        msg += '\n'
        return msg


# -- Example --
if __name__ == "__main__":
    MARKET_DATA_MANAGER.reset()
    MARKET_DATA_MANAGER.setup(dt.date(2014, 4, 15))

    # 1/ create a fixed rate bond
    b = AssBondFixRate(face=100,
                       coupon_rate=0.05,
                       frequency=4,
                       issue_date=dt.date(2014, 4, 15),
                       expiration_date=dt.date(2016, 4, 15),
                       dcf=dcf_act_act,
                       pricing_model=None,
                       name='Sample Bond 1',
                       )

    # or a "float rate bond" but provided with a scenario generator gives you flat rate
    init_df = pd.TimeSeries(data=[[0.0]], index=pd.date_range(start=dt.date(2014, 4, 15), periods=1, freq='D').date,
                            name=['LIBOR_3M'])
    ir_index = ip.IndexProvider(init_df, 'LIBOR_3M')
    sim_engine = FixRateEngine(0.05)
    scen_gen = ScenarioGenerator([ir_index], sim_engine, **{'max_time_step': 5. / 252})

    # a LIBOR 3M +50 bps floater
    b = AssBondFloater(rate_index_name='LIBOR_3M', spread=0.005)

    # 1.5/ print coupon rate at any fixing date
    print(b.coupon_rate(dt.date(2014, 6, 12)))

    # 2/ create bond iterator
    it = b.bond_iterator()

    # 3/ print essentials at initial step
    print(it)

    # 4/ evolve, and print
    it.next(_next_date=b.coupon_schedule()[1])
    print(it)



