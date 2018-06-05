"""
This is the model to generate the asset paths
"""

from math import floor
import datetime as dt

import numpy as np
import pandas as pd

from lib.utils import is_corr_matrix
from lib.constants import BDAYS_PER_YEAR, DAYS_PER_YEAR
from lib.calendarfns import day_count
from lib.insurance import InsStepFunc
import Infra.IndexProvider as ip


"""
    alternatively, can explicitly use numpy.linalg.cholesky
    for A = LL', Cholesky(A) returns L, then one can generate the correlate normal distribution with cholesky
"""


# TODO: eventually, we should assign each index to an engine, so each engine and a list of index will be in a generator
class ScenarioGenerator(object):
    def __init__(self, indices, engine, **kwargs):
        """
            indices:  is used to provide the historical path if exists, and store the simulated path for each index
            engine:  is used to generate the simulated path, model related
                information (e.g., correlation) is stored in engine
            kwargs: store other simulation related information such as min/max time step of simulation and etc.

            The simulator can mix historical/deterministic with simulated path by only simulate for the dates after
            the last available date from the index. Therefore, the passed-by-reference index must have an initial info
        """
        self._max_dt = kwargs.get('max_time_step', 1./BDAYS_PER_YEAR)
        self._day_freq = kwargs.get('day_freq', 'B')

        self._days_per_year = kwargs.get('days_per_year') or \
            (DAYS_PER_YEAR if self._day_freq == 'D' else BDAYS_PER_YEAR)
        self._max_bday = max(floor(self._max_dt * BDAYS_PER_YEAR), 1)
        self._max_dt = self._max_bday/float(self._days_per_year)
        self._calc_t = False

        for idx in indices:
            if not isinstance(idx, ip.IndexProvider):
                raise Exception('Index of type {} is not supported'.format(type(idx)))
        self._indices = indices

        self._engine = engine
        if self._engine.is_time_dependent:
            self._init_date = kwargs['initial_date']
            self._engine.set_rel_time(self._init_date, self._day_freq, self._days_per_year)
            self._calc_t = True

        # TODO: we temporarily set up the cut off date to pricing date before we find a better solution
        self.align_index(kwargs.get('Pricing Date'))

    def align_index(self, cut_off_date=None):
        if cut_off_date is None:
            cut_off_date = min(ts.data.index[-1] for ts in self._indices)
        for ts in self._indices:
            ts.truncate(end_date=cut_off_date)

    def next(self, _next_date):
        """
            This function setup the simulation stepes and populate the index from the last available
        """
        # TODO: Be cautious with the fact that in liability, one may use Actual/360 and for simulation such as Stock
        # price, business-day/BDAY_PER_YEAR is more natural
        start_date = self._indices[0].data.index[-1]
        if self._calc_t:
            _t = day_count(self._init_date, start_date, freq=self._day_freq)/self._days_per_year
        else:
            _t = None

        if start_date >= _next_date:
            return
        else:
            sim_dates = pd.date_range(start=start_date, end=_next_date,
                                      freq='%dB' % self._max_bday, closed='right').date
            sim_steps = np.array([self._max_dt] * sim_dates.size)
        if sim_dates[-1] != _next_date:
            n_bday = pd.date_range(start=sim_dates[-1], end=_next_date, freq='B', closed='right').date.size
            if n_bday:
                sim_steps = np.append(sim_steps, n_bday/float(self._days_per_year))
                sim_dates = np.append(sim_dates, _next_date)
            else:
                sim_dates[-1] = _next_date

        init_values = np.array([ts.data.ix[-1] for ts in self._indices])
        sim_rs = self._engine.simulate(init_values, sim_steps, _t)

        if len(self._indices) == 1:
            self._indices[0].append(pd.TimeSeries(data=sim_rs,
                                    index=sim_dates, name=self._indices[0].data.name)
                      )
        else:
            for i, ts in enumerate(self._indices):
                ts.append(pd.TimeSeries(data=sim_rs[:, i],
                                        index=sim_dates, name=ts.data.name)
                )

    @property
    def indices(self):
        return self._indices

    @property
    def index_names(self):
        """return all the name of indices"""
        return [ x.index_name for x in self.indices]

    @property
    def index(self):
        """return a dataframe, keep API the same after introducing .indices, which is a list ot ts"""
        result = pd.concat([x.data for x in self.indices], join='outer', axis = 1)
        result.columns = self.index_names
        result = ip.IndexProvider(result)
        return result

class Engine(object):
    @property
    def is_time_dependent(self):
        return False


class EqBSEngine(Engine):
    def __init__(self, mu, vol, corr=None):
        self._n_asset = mu.size
        self._mu = mu
        self._vol = vol
        self._corr = np.identity(self._n_asset) if corr is None else corr
        assert self._n_asset == self._vol.size, "The size of the asset should match the size of volatilities!"
        assert self._n_asset == self._mu.size, "The size of the asset should match the size of drifts!"
        assert self._corr.shape == (self._n_asset, self._n_asset), "Incorrect dimension of the correlation matrix!"
        assert is_corr_matrix(self._corr), "The correlation matrix provided is not qualified!"
        self._U = np.linalg.cholesky(self._corr)

    def simulate(self, init_value, sim_steps, _t=None):
        w = np.random.randn(sim_steps.size, self._n_asset).dot(self._U.transpose())
        log_move = np.array([self._mu * d_t - 0.5 * self._cum_var(d_t, _t) + np.sqrt(self._cum_var(d_t, _t)) * dw
                             for d_t, dw in zip(sim_steps, w)])
        return np.array([init_value * np.exp(log_shift) for log_shift in log_move.cumsum(axis=0)])

    def _cum_var(self, d_t, t=None):
        """
            returns sqrt( S(t to t+delta) vol * vol * dt )
        """
        return d_t * self._vol * self._vol


class EqBSTermEngine(EqBSEngine):
    def __init__(self, mu, vol, corr=None):
        super(EqBSTermEngine, self).__init__(mu, vol, corr=corr)
        self._var_funcs = None
        self._t_0 = None

    def set_rel_time(self, init_date, freq, days_per_year):
        self._t_0 = init_date
        self._var_funcs = [InsStepFunc(
            vol_crv.index_time_rel_from(init_date, freq=freq, days_per_year=days_per_year),
            np.insert(np.power(vol_crv.data.values, 2), 0, 0.)) for vol_crv in self._vol]

    def _cum_var(self, d_t, t=None):
        return np.array([var_func.integral(t, d_t) for var_func in self._var_funcs])

    @property
    def is_time_dependent(self):
        return True


class IRHWEngine(Engine):
    """
        Implement a short rate engine with HW model (using naive discretization)
        currently it is actually a shifted Vasicek model with constant floor
        consider only single asset
        dr' = [level-alpha * r']dt + sigma * dW, r' = r - r_min
    """
    def __init__(self, level, alpha, sigma, r_min=0.):
        self._level = level
        self._alpha = alpha
        self._sigma = sigma
        self._r_min = r_min
        self._theta = self._level * self._alpha

    def simulate(self, init_value, sim_steps, _=None):
        dw = np.random.randn(sim_steps.size)
        rs = np.zeros(sim_steps.size)
        for i, d_t in enumerate(sim_steps):
            r = (rs[i-1] if i else init_value) - self._r_min
            rs[i] = self._r_min + max(0, r+(self._theta - self._alpha*r)*d_t+self._sigma*np.sqrt(d_t)*dw[i])
        return rs


class EqIREngine(Engine):
    """
        Correlated equity and interest rate simulation. Place holder for future work.
    """


class FixRateEngine(Engine):
    """
    dummy engine as a show of concept
    """
    def __init__(self, fixed_rate):
        self._fixed_rate = fixed_rate

    def simulate(self, _, sim_steps, _t=None):
        del _t
        return np.array(self._fixed_rate + sim_steps * 0.0)


class VolSurface(object):
    def __init__(self, _vol):
        if type(_vol) in (float, int, np.float32, np.float16, np.float64):
            self._vol_type = 'const'
        elif isinstance(_vol, pd.Series):
            self._vol_type = 'term'
        elif isinstance(_vol, dict):
            # Here, we expect to receive eithe
            # 1. regular grid:r
            #    _vol = {'coordinator': ('tenor', 'strike'), 'tenor': [0.1, 0.2, 0.3],
            #            'strike': [10, 11, 12], 'vol matrix': [[], [], []]} or
            #    _vol = {'coordinator': ('tenor', 'strike'), 'tenor': [0.1, 0.2, 0.3],
            #            'moneyness': [10, 11, 12], 'vol matrix': [[], [], []]} or
            # 2. irregular grid:
            #    _vol = {'coordinator': ('tenor', 'strike'), (0.2, 20): 0.2 ...}
            self._vol_type = 'surface'
            self._vol_coordinate = _vol['coordinate']
        else:
            raise Exception('The data is not in the required data type!')
        self._vol = _vol
        self._vol_coordinates = None or self._vol_coordinates
        if self._vol_coordinate[1] not in ('moneyness', 'strike'):
            raise Exception('The second coordinate of the vol surface is neither MONEYNESS or STRIKE!')
        if self._vol_coordinate[0] != 'tenor':
            raise Exception('The first coordinate of the vol surface is not TENOR!')

    def __call__(self, tenor=None, moneyness=None, strike=None):
        if self._vol_type == 'const':
            return self._vol
        if self._vol_type == 'term':
            if tenor:
                return self._value_at(tenor)
            else:
                raise Exception('Tenor is not specified for a vol term structure!')
        if self._vol_type == 'surface':
            point = tenor, moneyness if self._vol_coordinate[1] == 'moneyness' else tenor, strike
            if None in point:
                raise Exception('MONEYNESS or STRIKE or BOTH are None!')
            return self._value_at(point)

    @staticmethod
    def _value_at(point):
        if isinstance(point, tuple):
            pass
            # TODO: add surface interpolation method here, maybe it is better to cache the interpolated surface
        else:
            pass
            # TODO: add an interpolation scheme here for the term structure


def main():


    # # Test single asset
    # init_df = pd.DataFrame(data=np.exp(np.random.randn(10)),
    #                        index=pd.date_range(start=dt.date(2014, 1, 1), periods=10, freq='D').date,
    #                        columns=['stock A'])
    # eq_index = ip.IndexProvider(init_df)
    # print(eq_index.data)
    # sim_engine = EqBSEngine(np.array([0.02]),
    #                         np.array([0.2]))
    # simulator = ScenarioGenerator(eq_index, sim_engine, **{'max_time_step': 5./BDAYS_PER_YEAR})
    # simulator.next(dt.date(2014, 3, 1))
    # print(eq_index.data)

    # # Test Multiple asset case
    # init_df = pd.DataFrame(data=np.exp(np.random.randn(10, 2)),
    #                        index=pd.date_range(start=dt.date(2014, 1, 1), periods=10, freq='D').date,
    #                        columns=['stock A', 'stock B'])
    # eq_index = ip.IndexProvider(init_df)
    # print(eq_index.data)
    # sim_engine = EqBSEngine(np.array([0.02, 0.02]), np.array([0.2, 0.25]), corr=np.array([[1., 0.3], [0.3, 1.]]))
    # simulator = ScenarioGenerator(eq_index, sim_engine, **{'max_time_step': 5./BDAYS_PER_YEAR})
    # simulator.next(dt.date(2014, 3, 1))
    # print(eq_index.data)

    # Test Term Structure
    data = [pd.TimeSeries(data=np.exp(np.random.randn(10)),
                          index=pd.date_range(start=dt.date(2014, 1, 1), periods=10, freq='D').date),
            pd.TimeSeries(data=np.exp(np.random.randn(10)),
                          index=pd.date_range(start=dt.date(2014, 1, 1), periods=10, freq='D').date)]

    vol_term = pd.TimeSeries(data=np.array([0.2, 0.25, 0.3, 0.32, 0.35, 0.4]),
                             index=[dt.date(2014, 1, 1), dt.date(2014, 7, 1), dt.date(2015, 1, 1),
                                    dt.date(2016, 1, 1), dt.date(2019, 1, 1), dt.date(2024, 1, 1)])
    eq_indices = [ip.IndexProvider(data[0], 'index A'), ip.IndexProvider(data[1], 'index B')]

    print(eq_indices[0].index_name)
    print(eq_indices[0].data)
    print(eq_indices[1].index_name)
    print(eq_indices[1].data)

    sim_engine = EqBSTermEngine(np.array([0.02]), np.array([ip.IndexProvider(vol_term)]))
    simulator = ScenarioGenerator(eq_indices, sim_engine,
                                  **{'max_time_step': 5./BDAYS_PER_YEAR, 'initial_date': dt.date(2014, 1, 1)})
    simulator.next(dt.date(2020, 12, 31))

    print(eq_indices[0].index_name)
    print(eq_indices[0].data)
    print(eq_indices[1].index_name)
    print(eq_indices[1].data)


    # Test Rates

    # init_df = pd.DataFrame(data=[[0.0]], index=pd.date_range(start=dt.date(2014,1,1).date,
    #                        periods=1, freq='D').date, columns=['RateA'])
    # ir_index = ip.IndexProvider(init_df)
    # sim_engine = FixRateEngine(0.05)
    # simulator = ScenarioGenerator(ir_index, sim_engine, **{'max_time_step': 5./BDAY_PER_YEAR})
    # simulator.next(dt.date(2014, 3, 1))
    # print(ir_index.data)


if __name__ == '__main__':
    main()