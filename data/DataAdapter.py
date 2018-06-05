"""
Adapter to collect data, from public sources such as Quandl
"""

import Quandl
import os
import pandas

DEFAULT_TOKEN = "CBXF8eUWzYrcCH9zrvx6"
PROJECT_ROOT = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
MARKET_DATA_DB = PROJECT_ROOT + '/pickle_db/market_data/'


class DataAdapter(object):
    def get_level_timeseries(self, ticker):
        pass


class QuandlAdapter(DataAdapter):
    def __init__(self, **kwargs):

        if 'authtoken' in kwargs:
            self._authtoken = kwargs.get('authtoken')
        else:
            self._authtoken = DEFAULT_TOKEN

    def get_level_timeseries(self, ticker, field=None):
        lvl = Quandl.get(ticker, authtoken=self._authtoken)
        if isinstance(lvl, pandas.DataFrame):
            cols = lvl.columns
            if len(cols) == 1:
                lvl = lvl[cols[0]]
            elif field in cols:
                lvl = lvl[field]
            else:
                raise Exception('{} is not in the dataframe'.format(field))
        if hasattr(lvl.index, 'date'):
            lvl.index = lvl.index.date
        return lvl

# ------ example --------------
if __name__ == '__main__':
    from utils.database import pickle_save
    from Infra.IndexProvider import IndexProvider
    sample_adapter = QuandlAdapter()
    libor_6m = IndexProvider(sample_adapter.get_level_timeseries('ODA/USA_FLIBOR6'))
    pickle_save(libor_6m, 'libor_6m', db_path=MARKET_DATA_DB)
