"""This module provides classes for the credit rates"""
from abc import ABCMeta, abstractmethod
import datetime

import numpy as np
import pandas

from Infra.IndexProvider import IndexProvider
from lib.constants import EPSILON_WEIGHTS
from Managers.MarketDataManager import MARKET_DATA_MANAGER


class InsCreditRate(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def credit_rate(self, period=None, dt=None, prev_rate=None):
        pass

    def set_index_manager(self, model):
        pass


class InsCreditRateFixed(InsCreditRate):
    def __init__(self, credit_rate=None):
        self._m_credit_rate = credit_rate

    def credit_rate(self, period=None, dt=None, prev_rate=None):
        if self._m_credit_rate is None:
            raise Exception('The credit rate is not yet set up!')
        return self._m_credit_rate

    def set_credit_rate(self, credit_rate):
        assert credit_rate >= 0.0, "Credit rate must be above 0!"
        self._m_credit_rate = credit_rate

    def inv_index(self, start_date, periods, step_per_year):
        """
        params:
            start_date: starting date of the index, currently only support the beginning of the month
            periods: period of the index to represents, if
            step_per_year: number of frequency of calculation, the value in between is not calculated,
            and will be linearly interpolated if queried
            annual_rate: nominal rate of return per year
        return:
            An IndexProvider object representing the value of $1 invested on start_date at the beginning of each month
        """
        _supported_steps = (1, 2, 3, 4, 6, 12)
        if step_per_year in _supported_steps:
            period_rate = self._m_credit_rate / step_per_year
            freq = "%dMS" % (12 // step_per_year)
            index_dates = pandas.date_range(start_date, periods=periods, freq=freq).date
            index_value = np.array([1.0] + [1 + period_rate] * (periods - 1)).cumprod()
            return IndexProvider(pandas.TimeSeries(index_value, index_dates))
        else:
            raise NotImplementedError(
                "Currently only support %d, %d, %d, %d, %d, or %d step(s) per year" % _supported_steps)


class InsCreditRateFloating(InsCreditRate):
    def __init__(self, index_name, index_provider=None, look_back=None):
        look_back = look_back or datetime.timedelta(0)
        assert isinstance(look_back, datetime.timedelta), "The look_back needs to be a datetime.timedelta instance!"
        self._look_back = look_back
        self._index_name = index_name
        self._index_provider = index_provider

    def credit_rate(self, period=None, dt=None, prev_rate=None):
        index_dt = dt + self._look_back
        if not self._index_provider:
            self._index_provider = MARKET_DATA_MANAGER.get_index(self._index_name)
        return self._index_provider.index_value(index_dt)



class InsCreditRateSpread(InsCreditRate):
    def __init__(self, inner_credit_rate, spread):
        assert isinstance(inner_credit_rate, InsCreditRate), "inner_credit_rate must be a InsCreditRate instance"
        self._m_inner_credit_rate = inner_credit_rate
        self._m_spread = spread

    def credit_rate(self, period=None, dt=None, prev_rate=None):
        return self._m_inner_credit_rate.credit_rate(period, dt, prev_rate - self._m_spread) + self._m_spread


class InsCreditRateCapFloor(InsCreditRate):
    def __init__(self, inner_credit_rate, level, is_cap):
        assert isinstance(inner_credit_rate, InsCreditRate), "inner_credit_rate must be a InsCreditRate instance"
        self._m_inner_rate = inner_credit_rate
        self._m_level = level
        self._m_is_cap = is_cap

    def credit_rate(self, period=None, dt=None, prev_rate=None):
        inner_rate = self._m_inner_rate.credit_rate(period, dt, prev_rate)
        if self._m_is_cap:
            return min(self._m_level, inner_rate)
        else:
            return max(self._m_level, inner_rate)


class InsCreditRateComposite(InsCreditRate):
    def __init__(self, first, second, period):
        assert isinstance(first, InsCreditRate), "first credit rate must be a InsCreditRate instance"
        assert isinstance(second, InsCreditRate), "second credit rate must be a InsCreditRate instance"
        self._m_first = first
        self._m_second = second
        self._m_period = period

    def credit_rate(self, period=0.0, dt=0.0, prev_rate=0.0):
        if period <= self._m_period:
            return self._m_first.credit_rate(period, dt, prev_rate)
        else:
            return self._m_second.credit_rate(period, dt, prev_rate)


class InsCreditRateMutualFunds(InsCreditRate):
    def __init__(self, fund_info):
        """
        initialize a crediting object for investment accounts in mutual funds, modeled as a
        linear combination of invest-able indices to be provided by scen_gen
        :param fund_info: a structure with key = mutual fund name, values has index weights, and fund fee
        Example:
        fund_info = {'Fund A' :
                                {'Allocations': {
                                                    'S&P' : 0.5,
                                                    'RUS' : 0.3,
                                                    'NQX' : 0.1,
                                                    'EMG  : 0.1,
                                                },
                                 'Management Fee' : 0.01
                                 'Description' : 'blah blah'
                                }
                    }

        :return: an index provider (multiple assets), with asset names = keys of fund info
        :scen_gen: need a scenario generator to generate the underlying index values
        """
        assert fund_info
        self._fund_info = fund_info

        # perform some integrity check on fund info

        # collect all keys of tradable indices
        all_indices = []

        for fund_name, fund_i in fund_info.items():
            assert 'Allocations' in fund_i, "%s does not have an allocation" % fund_name
            total_weights = sum(fund_i['Allocations'].values())
            assert abs(total_weights - 1.0) <= EPSILON_WEIGHTS, "%s has a total weight of %s instead of 100%%" % \
                                                                (fund_name, total_weights)

            # check the index names are provided by the scenario gen
            #assert all(x in self._scenario_generator.index_names for x in fund_i['Allocations'].keys())

            all_indices += fund_i['Allocations'].keys()

        all_indices = list(set(all_indices))

        regr = dict([(x, [y['Allocations'].get(z, 0) for z in all_indices]) for x, y in fund_info.items()])

        self._regressors = pandas.DataFrame(data=regr, index=all_indices)
        self._assets = all_indices

    @property
    def regressors(self):
        return self._regressors

    @property
    def assets(self):
        return self._assets

    def credit_rate(self, period=None, dt=None, prev_rate=None):
        pass

    def inv_index(self, start_date, periods, step_per_year):
        """
        params:
            start_date: starting date of the index, currently only support the beginning of the month
            periods: period of the index to represents, if
            step_per_year: number of frequency of calculation, the value in between is not calculated,
            and will be linearly interpolated if queried
            annual_rate: nominal rate of return per year
        return:
            An IndexProvider object representing the value of $1 invested on start_date at the beginning of each month
        """

        # TODO: I dont think we need all these arguments, paths should be handled purely in scen gen
        _supported_steps = (1, 2, 3, 4, 6, 12)
        if step_per_year in _supported_steps:
            freq = "%dMS" % (12 // step_per_year)
            index_dates = pandas.date_range(start_date, periods=periods, freq=freq)

            ts = []
            for idx in self._assets:
                mdp = MARKET_DATA_MANAGER.get(idx)
                mdp.index_value(index_dates[-1].date())
                ts.append(MARKET_DATA_MANAGER.get_index(idx).data)

            # pandas.DataFrame({'fund A': [1, 0], 'fund B': [0, 1 ]}, index = ['stock A', 'stock B']) -> self.regressors
            # return a dot production which transform tradable index return dataframe to that of fund returns
            return IndexProvider(pandas.concat(ts, join='outer', axis = 1).dot(self.regressors))
        else:
            raise NotImplementedError(
                "Currently only support %d, %d, %d, %d, %d, or %d step(s) per year" % _supported_steps)


def main():
    start_date = datetime.date(2013, 2, 1)
    MARKET_DATA_MANAGER.reset()
    MARKET_DATA_MANAGER.setup(start_date)
    credit_rate = 0.03
    step_per_year = 12
    periods = 360
    credit_rider = InsCreditRateFixed(credit_rate)
    inv_index = credit_rider.inv_index(start_date, periods, step_per_year)
    one_date = datetime.date(2013, 12, 1)
    print(inv_index.data[one_date])
    print(inv_index.data)
    print(isinstance(inv_index.data, pandas.Series))

    # Test Multiple equity paths case
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
                             'stock A': 0.5,
                             'stock B': 0.5,
                         },
                         'Management Fee': 0.01,
                         'Description': 'blah blah',
                     },
                 }

    ic = InsCreditRateMutualFunds(fund_info=fund_info)
    index = ic.inv_index(datetime.date(2014, 1, 10), periods=360, step_per_year=12)
    # print([ 1,2,3] * index.index_value(datetime.date(2014,2,10))/index.index_value(datetime.date(2014,1,10)))
    print(index.data)
    print(type(index.data[['Fund B', 'Fund A']]))


# Example
if __name__ == '__main__':
    main()
    import pandas as pd
    import datetime as dt

    fc = InsCreditRateFloating(dt.timedelta(days=-30), 'LIBOR_3M')
    for td in pd.date_range(start=dt.date(1999, 12, 1), periods=200, freq='M'):
        print(fc.credit_rate(dt=td))

