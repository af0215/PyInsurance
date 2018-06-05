import datetime as dt

import numpy as np

from Securities.AssBond import AssBondBase, AssBondFixRate, AssBondFloater
from lib.constants import BOND_MODEL_STATES
from Infra.CurveAggregator import create_curve_aggregator
from Managers.ProjectionManager import ProjectionManager


class AssModelBase(object):
    def __init__(self,
                 asset,
                 credit_model,
                 state_names=BOND_MODEL_STATES,
                 accrual=lambda df,cfname: df.sum(0)[cfname],
                 init_state_weights=None,
                 state_history=None,  # index recording the known states, either historical or future projection
                 use_discrete=False):  # indTingicate if stochastic model is used instead of probabilistic model
        """
        start a model projection model, these information are states that will be evolved by iterator.
        I plan to implement this as a simple asset evolution model with an external credit model
        (similar to that of a liability model)
        """
        # so far bond base is the only support asset in this simple model
        assert isinstance(asset, AssBondBase)
        self._asset = asset
        self._state_names = state_names
        self._num_states = len(self._state_names)
        self._state_history = state_history
        self._use_discrete = use_discrete
        self._credit_model = credit_model
        self.accrual = accrual
        self._init_state_weights = init_state_weights

    def trans_matrix(self, model_iter):
        # Key function here to generate the transition matrix for the model iterator
        trans_matrix = np.zeros((self._num_states, self._num_states))

        p_1 = self._credit_model.prob(model_iter)  # probability of mortality given the time period
        # Assuming all the transitions are continuous and independent
        #lambda_1 = np.log(1-p_1)  # default intensity

        trans_matrix[self._state_names.index('Current'), self._state_names.index('Default')] = p_1

        trans_matrix[self._state_names.index('Current'), self._state_names.index('Current')] = 1-p_1

        trans_matrix[self._state_names.index('Default'), self._state_names.index('Default')] = 1.0

        return trans_matrix

    def create_iterator(self, pricing_date):
        return AssModelIter(self, self._asset, pricing_date, self._init_state_weights)

    @property
    def num_states(self):
        return self._num_states

    @property
    def state_names(self):
        return self._state_names

    def __str__(self):
        return 'Base AssModel with Credit Model %s' % self._credit_model



class AssModelIter(object):
    def __init__(self,
                 model,
                 asset,
                 pricing_date,
                 state_weights=None):
        """
        The model iterator is used to evolve the model
        """
        assert isinstance(asset, AssBondBase)
        self._model = model
        self._asset = asset
        self._pricing_date = pricing_date
        self._trans_matrix = np.zeros((self._model.num_states, self._model.num_states))
        if state_weights is None:
            self._state_weights = np.zeros(len(self._model.state_names))
            self._state_weights[model.state_names.index("Current")] = 1.0
        else:
            self._state_weights = state_weights
        self._assetIter = asset.bond_iterator()
        self._next_date = None
        self._year_frac = None
        self._new_default = 0.0
        self._date = pricing_date

    def advance(self):
        # this is the key function to forward the states transition
        self._trans_matrix = self._model.trans_matrix(self)
        state_weights = np.dot(self._trans_matrix, self._state_weights)
        self._state_weights = state_weights

    def next(self, _next_date, _year_frac):
        """
            this is the key function to calculate things like mgmt/rider/etc fees collected
            surrender charge collected and etc

            The implementation will follow the steps below
            1. Advance the state distribution using end of previous period account info and etc
            2. since we assume that all lapse/mortality happens on the end of the period, we evolve the account
            3. calculation the cash-flows (aggregated by aggregator) on the account
        """
        self._next_date = _next_date
        self._year_frac = _year_frac


        # Step 1: Advance state distribution
        _prev_state_weights = self._state_weights.copy()
        self.advance()
        _delta_weights = self._state_weights - _prev_state_weights

        # Step 2: Evolve the 'active' account
        self._assetIter.next(_next_date, _year_frac)

        # Step 3: Calculate the state change
        self._new_default = _delta_weights[self._model.state_names.index('Default')]


        # Step 4: calculate behavior related transactions (some calc should happen in aggregator)

        # wrap-up
        self._date = _next_date

    def m_trans_matrix(self):
        return self._trans_matrix

    def m_weight_active(self):
        return self._state_weights[self._model.state_names.index('Current')]

    def m_new_default(self):
        return self._new_default

    def _cf_accrual(self, cfname):
        return self._model.accrual(self._assetIter.cash_flow_matured, cfname)

    #TODO: well, the cf should really scaled by survival probility here, its should be self sustained in model
    def m_cf_total(self):
        return self._cf_accrual('Total')

    def m_cf_interest(self):
        return self._cf_accrual('Interest')

    def m_cf_principal(self):
        return self._cf_accrual('Principal')

    @property
    def date(self):
        return self._date

    # @property
    # def model(self):
    #     return self._model


class NoDefaultCreditModel(object):
    """ modeled after InsLapseModel
    """
    def prob(self, _model_iter):
        return 0

    def __str__(self):
        return "No Default"

if __name__ == '__main__':

    from lib.utils import dcf_act_act
    from Managers.MarketDataManager import MARKET_DATA_MANAGER
    MARKET_DATA_MANAGER.reset()
    MARKET_DATA_MANAGER.setup(dt.date(2014,4,15))
    # Setup a bond
    asset = AssBondFixRate(face=100,
                           coupon_rate=0.05,
                           frequency=4,
                           issue_date=dt.date(2014, 4, 15),
                           expiration_date=dt.date(2016, 4, 15),
                           dcf=dcf_act_act,
                           pricing_model=None,
                           name='Sample Bond 1',
                           )

    # or a "float rate bond" but provided with a scenario generator gives you flat rate

    # a LIBOR 3M +50 bps floater
    asset = AssBondFloater(rate_index_name='LIBOR_3M', spread=0.005)



    asset_iter = asset.bond_iterator()

    # Setup a credit model using lapse static
    credit_model = NoDefaultCreditModel()

    # Setup Asset Model
    pricing_date = dt.date(2014, 4, 15)
    model = AssModelBase(asset, credit_model)
    model_iter = model.create_iterator(pricing_date)

    metrics = ['Date', 'CF_In:Interest', 'CF_In:Principal']

    crv_aggregator = create_curve_aggregator(metrics)

    params = {'pricing date': pricing_date, 'periods': 60, 'frequency': 'MS'}
    proj_mgr = ProjectionManager(crv_aggregator, model_iter, **params)
    proj_mgr.run()

    df = crv_aggregator.to_dataframe()
    print(df)