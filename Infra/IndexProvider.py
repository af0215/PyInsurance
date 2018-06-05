"""
IndexProvider is used to handle the indices such as equity index, rate index, and etc.
Currently, we build it based on pandas TimeSeries and DataFrame
"""
import datetime as dt
from bisect import bisect_left

import pandas as pd
import numpy as np

from lib.constants import DAYS_PER_YEAR, BDAYS_PER_YEAR
from lib.calendarfns import day_count


class IndexProvider(object):
    def __init__(self, data, index_name=None):
        if isinstance(data, pd.TimeSeries):
            self._n_assets = 1
        elif isinstance(data, pd.DataFrame):
            self._n_assets = data.shape[1]
        else:
            raise Exception("Input data of {} is not supported".format(type(data)))
        self._data = data
        self._o_dates = [d.toordinal() for d in self._data.index]
        self._index_name = index_name

        if type(self._data.index[0]) is dt.date:
            self.index_type = 'date'
        else:
            raise Exception('The index of the data is not of type datetime.date')
        #TODO: Here I assume that the index is dt.date, need to check later

    def index_value(self, date):
        date_ord = date.toordinal()
        dates = self._o_dates

        n_dates = len(dates)
        i = bisect_left(dates, date_ord)
        if i >= n_dates or (i == 0 and date_ord <= dates[0]):
            if self._n_assets == 1:
                return self._data[0] if i == 0 else self._data[-1]
            else:
                return np.array(self._data.ix[0]) if i == 0 else np.array(self._data.ix[-1])
        temp = dates[i] - dates[i-1]
        if self._n_assets == 1:
            return (self._data[i]*(date_ord - dates[i-1]) + self._data[i-1]*(dates[i] - date_ord)) / temp
        else:
            return (np.array(self._data.ix[i])*(date_ord - dates[i-1])
                    + np.array(self._data.ix[i-1])*(dates[i] - date_ord)) / temp

    def append(self, new_data):
        self._data = self._data.append(new_data)

    def index_time_rel_from(self, from_date, freq='D', days_per_year=None):
        if freq == 'D':
            days_per_year = days_per_year or DAYS_PER_YEAR
            return np.array(day_count(from_date, end_date)/float(days_per_year) for end_date in self._data.index.date)
        if freq == 'B':
            days_per_year = days_per_year or BDAYS_PER_YEAR
            return np.array([day_count(from_date, end_date, freq='B')/float(days_per_year)
                            for end_date in self._data.index])

    def set_value_at(self, date, value):
        self._data.ix[date] = value
        self._data.sort_index()

    def truncate(self, end_date):
        """interpolate the value at the truncation date and drop the index data after this date"""
        self.set_value_at(end_date, self.index_value(end_date))
        self._data = self._data[self._data.index <= end_date]

    def __str__(self):
        return self.index_name or 'unnamed index'

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        """
        some check is needed here
        """
        self._data = new_data

    @property
    def n_assets(self):
        return self._n_assets

    @property
    def knots(self):
        return self.data.index.values

    @property
    def values(self):
        return self.data.values

    @property
    def index_name(self):
        return self._index_name

    @index_name.setter
    def index_name(self, name):
        self._index_name = name


class InsCurve(IndexProvider):
    """
    The purpose of this class is to facilitate the diddles.
    The behavior of the InsCurve should be equivalent to IndexProvider class to the diddle operations
    """
    #TODO: may need to combine it to the IndexProvider, but it might make sense to have a separate class
    def __inti__(self, knots, values):
        if not isinstance(knots, np.ndarray):
            knots = np.array(knots)
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if knots.size != values.size:
            raise ValueError('Length of knots should be same as the length of values!')
        if any(knots != np.sort(knots)):
            raise ValueError('x should be sorted ascending!')
        self._data = pd.TimeSeries(index=knots, data=values)
        self._n_assets = 1
        if isinstance(knots[0], dt.date):
            raise Exception('For index of type datetime.date, one should use IndexProvider')
        else:
            self.index_type = 'other'

    def index_value(self, value_at):
        knots = self.knots
        n_knots = len(knots)

        i = bisect_left(knots, value_at)
        if i >= n_knots or (i == 0 and value_at <= knots[0]):
            return self._data.ix[knots[0]] if i == 0 else self._data.ix[knots[-1]]
        temp = knots[i] - knots[i-1]
        return (self._data.ix[knots[i]]*(value_at - knots[i-1]) + self._data.ix[knots[i-1]]*(knots[i] - value_at))/temp


class FeeRateIndex(IndexProvider):
    def __init__(self, data):
        IndexProvider.__init__(self, data)
        if not isinstance(data, pd.TimeSeries):
            raise Exception("Expect a Time Series")
        self._data = data
        self._index_name = self._data.index.name
        if self._index_name == "Date":
            self._o_date = [d.toordinal() for d in self._data.index]

    def index_value(self, x):
        if self.index_name == "Date":
            xs = self._o_date
            x = x.toordinal()
        else:
            xs = list(self._data.index)
        n_xs = len(xs)
        i = bisect_left(xs, x)
        if i >= n_xs or (i == 0 and x <= xs[0]):
            return self._data[0] if i == 0 else self._data[-1]

        temp = xs[i] - xs[i-1]
        return (self._data[i]*(x - xs[i-1]) + self._data[i-1]*(xs[i] - x)) / temp

    @property
    def index_name(self):
        return self._index_name


class FixRateIndex(IndexProvider):
    """simple fix rate index"""
    def __init__(self, fixed_rate):
        data = pd.Series([fixed_rate], [dt.date.min])
        IndexProvider.__init__(self, data)
        self._data = data
        self._index_name = self._data.index.name
        if self._index_name == 'Date':
            self._o_dates = [d.toordinal() for d in self._data.index]

    def index_value(self, date):
        return self._data[0]

# Example
if __name__ == '__main__':
    # This is to test time series
    start_date = dt.date(2014, 11, 11)
    dates = pd.date_range(start_date, periods=120, freq="MS").date
    index_values = np.random.randn(120)
    index_data = pd.TimeSeries(index_values, index=dates)
    inv_index = IndexProvider(index_data)
    test_date = dt.date(2015, 1, 1)
    print(inv_index.index_value(test_date))

    # This is to test DataFrame
    start_date = dt.date(2014, 11, 11)
    dates = pd.date_range(start_date, periods=120, freq="MS").date
    index_values = np.random.randn(360).reshape(120, 3)
    index_data = pd.DataFrame(index_values, index=dates)
    inv_index = IndexProvider(index_data)
    test_date = dt.date(2015, 1, 1)
    print(inv_index.index_value(test_date))


