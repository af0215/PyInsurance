import os
import datetime
import pandas
import numpy
from utils.database import pickle_load
from Infra.IndexProvider import IndexProvider

"""
We define a class to handle the market data
"""
PROJECT_ROOT = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
MARKET_DATA_DB = PROJECT_ROOT + '/pickle_db/market_data/'


def singleton(OrigCls):
    """
    Make a singleton class
    """
    class NewCls:
        _singleton_obj = None

        def __new__(cls):
            if not cls._singleton_obj:
                cls._singleton_obj = OrigCls()
            return cls._singleton_obj
    return NewCls


@singleton
class MarketDataManager(object):
    def __init__(self, market_data_date=datetime.date.today()):
        self._index_table = {}
        self._scen_gen_table = {}  # a table paring index with their corresponding scen generator, for projection
        self._mkt_data_provider_table = {}  # wrapper on index provider, backed up by scen gen
        self._market_data_date = market_data_date  # this is used as a cut of between historical and projection

    def get_index(self, idx):
        if idx not in self._index_table:
            self._load_index(idx)
            self._index_table[idx].truncate(self._market_data_date)
        return self._index_table.get(idx, None)

    def _load_index(self, index_name):
        """
        currently implement with pickle_db. In the future, will replace with more flexibility by using data adaptor
        """
        try:
            _index = pickle_load(index_name, db_path=MARKET_DATA_DB)
            assert isinstance(_index, IndexProvider), 'Yo, the loaded %s is not an index provider' % index_name
            _index.index_name = index_name  # making name in this Mgr and name on index provider consistent
            self._index_table[index_name] = _index
        except FileNotFoundError:
            print('{} is not found in the database'.format(index_name))

    @property
    def market_data_date(self):
        return self._market_data_date

    @property
    def index_table(self):
        return self._index_table

    @market_data_date.setter
    def market_data_date(self, date):
        assert type(date) is datetime.date, "%s is not a date" % date
        self._market_data_date = date
        print("NEED TO RELOAD ALL INDICES, YOU WILL LOSE PROJECTED VALUES")
        for idx in self._index_table.keys():
            self._load_index(idx)
            self._index_table[idx].truncate(self._market_data_date)
            print('Index: %s is reloaded and set to Market Data Date %s' % (idx, self._market_data_date))

    @property
    def scen_gen_table(self):
        return self._scen_gen_table

    def get_mkt_data_provider(self, idx):
        if idx not in self._mkt_data_provider_table:
            if idx in self._index_table and idx in self._scen_gen_table:
                self._mkt_data_provider_table[idx] = MarketDataProvider(self._index_table[idx],
                                                                        scen_gen=self._scen_gen_table[idx])
            else:
                raise ValueError('%s is %s Index Table, %s Scenario Generator Table' %
                                 (idx,
                                  'in' if idx in self._index_table else 'not in',
                                  'in' if idx in self._scen_gen_table else 'not in'))
        return self._mkt_data_provider_table.get(idx, None)

    def get(self, idx):
        return self.get_mkt_data_provider(idx)

    def reset(self):
        self._index_table = {}
        self._scen_gen_table = {}  # a table paring index with their corresponding scen generator, for projection
        self._mkt_data_provider_table = {}  # wrapper on index provider, backed up by scen gen

    def setup(self, date):
        from Managers.ScenarioManager import EqBSEngine, ScenarioGenerator, FixRateEngine
        from lib.constants import BDAYS_PER_YEAR
        self.reset()
        self.market_data_date = date

        libor_3m = IndexProvider(pandas.TimeSeries(index=[date], data=[0.05]), index_name='LIBOR_3M')
        ir_eng = FixRateEngine(0.05)
        scen_gen_libor = ScenarioGenerator([libor_3m], ir_eng, **{'max_time_step': 5. / BDAYS_PER_YEAR})
        self._index_table['LIBOR_3M'] = libor_3m
        self._scen_gen_table['LIBOR_3M'] = scen_gen_libor

        init_df = [pandas.TimeSeries(data=[100], index=[date], name='stock A'),
                   pandas.TimeSeries(data=[100], index=[date], name='stock B')]
        eq_index = [IndexProvider(init_df[0], 'stock A'), IndexProvider(init_df[1], 'stock B')]
        sim_engine = EqBSEngine(numpy.array([0.02, 0.02]), numpy.array([0.0, 0.0]), corr=numpy.array([[1., 0.3], [0.3, 1.]]))
        scen_gen = ScenarioGenerator(eq_index, sim_engine, **{'max_time_step': 5. / BDAYS_PER_YEAR})

        self._index_table['stock A'] = eq_index[0]
        self._index_table['stock B'] = eq_index[1]
        self._scen_gen_table['stock A'] = scen_gen
        self._scen_gen_table['stock B'] = scen_gen


@singleton
class SimulationManager(object):
    def __init__(self):
        self._sim_engine_table = {}

    def get_sim_engine(self, index_name):
        return self._sim_engine_table.get(index_name, None)

    def set_sim_engine(self, index_name, engine):
        self._sim_engine_table[index_name] = engine


class MarketDataProvider(object):
    """a wrapper around index provider, which a backup of Scenario Gen, such that projected values can be populated """

    def __init__(self, index, scen_gen):
        self._index = index
        self._scen_gen = scen_gen

    def __str__(self):
        return 'MarketDataProvider -> %s : %s' % (self._index, self._scen_gen)

    def index_value(self, date):
        if date not in self._index.data.index:
            self._scen_gen.next(date)
            print('wow, %s not in %s, I generated: %s' % (date, self._index.index_name, self._index.data[date]))

        # TODO: think through: cases of date vs [ begin of historical data, end of historical data]
        return self._index.index_value(date)

    @property
    def index(self):
        return self._index

MARKET_DATA_MANAGER = MarketDataManager()

# ---------- See the example on how all these work together ----------------
if __name__ == '__main__':
    from utils.database import pickle_save
    import pandas as pd
    import datetime as dt
    import numpy as np
    from Managers.ScenarioManager import EqBSEngine, ScenarioGenerator, FixRateEngine
    from lib.constants import BDAYS_PER_YEAR

    sample_credit_curve = IndexProvider(
        pd.TimeSeries(index=pd.date_range(start=dt.date(2000, 1, 1), periods=600, freq='MS').date,
                      data=[0.03]*600)
    )
    pickle_save(sample_credit_curve, 'sample_credit_curve', db_path=MARKET_DATA_DB)


    MARKET_DATA_MANAGER.reset()

    # =========== test re-set market data date ================
    print(MARKET_DATA_MANAGER.get_index('fake_libor_3m').data)
    MARKET_DATA_MANAGER.market_data_date = dt.date(2008, 1, 1)
    print(MARKET_DATA_MANAGER.get_index('fake_libor_3m').data)

    # ========== test scen gen table =============
    print(MARKET_DATA_MANAGER.scen_gen_table)
    eng = FixRateEngine(0.05)
    scen_gen_libor = ScenarioGenerator([MARKET_DATA_MANAGER.get_index('fake_libor_3m')], eng, **{'max_time_step': 5. / BDAYS_PER_YEAR})
    MARKET_DATA_MANAGER.scen_gen_table['fake_libor_3m'] = scen_gen_libor
    print(MARKET_DATA_MANAGER.scen_gen_table)

    # ========== test mkt data provider interface =============
    print(MARKET_DATA_MANAGER.get_mkt_data_provider('fake_libor_3m'))
    mgrlibor = MARKET_DATA_MANAGER.get('fake_libor_3m')
    print('fake libor 3m: initial : %s' % mgrlibor.index_value(dt.date(2008, 1, 1)))
    print('fake libor 3m: 2009/1/1: %s' % mgrlibor.index_value(dt.date(2009, 1, 1)))

    try:
        MARKET_DATA_MANAGER.index_table['fake_libor_6m'] = 'dummy'
        MARKET_DATA_MANAGER.get_mkt_data_provider('fake_libor_6m')
    except ValueError as err:
        print(err)

    try:
        MARKET_DATA_MANAGER.scen_gen_table['fake_libor_1y'] = 'dummy'
        MARKET_DATA_MANAGER.get_mkt_data_provider('fake_libor_1y')
    except ValueError as err:
        print(err)

    # ========== test mkt data provider =============
    # For now, we assume the init_date is month begin
    step_per_year = 12
    periods = 360
    init_date = dt.date(2013, 2, 1)
    pricing_date = init_date
    # Set up the investment index
    # credit_rider = isr.InsCreditRateFixed(credit_rate)

    # set up the mutual fund return index
    init_df = [pd.TimeSeries(data=[100], index=[init_date], name='stock A'),
               pd.TimeSeries(data=[100], index=[init_date], name='stock B')]
    eq_index = [IndexProvider(init_df[0], 'stock A'), IndexProvider(init_df[1], 'stock B')]
    sim_engine = EqBSEngine(np.array([0.02, 0.02]), np.array([0.2, 0.25]), corr=np.array([[1., 0.3], [0.3, 1.]]))
    scen_gen = ScenarioGenerator(eq_index, sim_engine, **{'max_time_step': 5. / BDAYS_PER_YEAR})

    pickle_save(eq_index[0], 'stock A', db_path=MARKET_DATA_DB)
    pickle_save(eq_index[1], 'stock B', db_path=MARKET_DATA_DB)

    MARKET_DATA_MANAGER.index_table['stock A'] = eq_index[0]
    MARKET_DATA_MANAGER.index_table['stock B'] = eq_index[1]
    MARKET_DATA_MANAGER.scen_gen_table['stock A'] = scen_gen
    MARKET_DATA_MANAGER.scen_gen_table['stock B'] = scen_gen

    mdpa = MARKET_DATA_MANAGER.get('stock A')
    print(mdpa.index_value(init_date))
    print(mdpa.index_value(dt.date(2014, 2, 1)))

    mdpb = MARKET_DATA_MANAGER.get('stock B')
    print(mdpb.index_value(init_date))
    print(mdpb.index_value(dt.date(2014, 2, 1)))
    print(mdpb.index_value(init_date))


