'''
    Place to hold all manager classes.
'''
from lib.constants import DAYS_PER_YEAR
import pandas as pd


class ProjectionManager(object):
    """
        class to setup the projection, such as timeline, scenarios, and etc
    """
    def __init__(self, aggregator, model_iter, **kwargs):
        self._aggregator = aggregator
        self._model_iter = model_iter
        self._params = kwargs
        self._timeline = None
        self._setup()
        self._day_per_year = self._params.get('days per year') or DAYS_PER_YEAR

    def _setup(self):
        if 'periods' not in self._params:
            raise Exception('Periods of projection is not specified!')
        self._periods = self._params.get('periods')
        if 'frequency' not in self._params:
            raise Exception('Frequency of projection is not specified! (Using strings allowed by pandas.date_range() )')
        self._frequency = self._params.get('frequency')
        if 'pricing date' not in self._params:
            raise Exception('Pricing date is not specified!')
        self._pricing_date = self._params.get('pricing date')
        self._timeline = pd.date_range(self._pricing_date, periods=self._periods, freq=self._frequency).date

    def run(self):
        prev_date = self._pricing_date
        for d in self._timeline:
            if d != self._pricing_date:
                year_frac = (d - prev_date).days / DAYS_PER_YEAR
                self._model_iter.next(d, year_frac)
            self._aggregator.collect_element(self._model_iter)
            prev_date = d






