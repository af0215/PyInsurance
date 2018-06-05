"""
Test: Insurance.test.TestInsModelFA
"""

import datetime as dt
import numpy as np
from Account.InsAcct import InsAcct
from lib.constants import FA_MODEL_STATES, FA_MODEL_STATE_INDICES
import Products.InsCreditRate as isr
import Products.InsFee as mif
import Products.InsRider as mir
from lib.utils import extract_strict
from Models.InsMortModel import InsMortModel
from Models.InsLapseModel import SurrenderCharge, LapseDynamic
from lib.insurance import InsStepFunc, linear_comp_bounded
from Infra.CurveAggregator import create_curve_aggregator
from Managers.ProjectionManager import ProjectionManager
from Products.InsProduct import InsProduct

OUTPUT_PATH = 'Output/FA'


class InsModelFA(object):
    def __init__(self,
                 acct,
                 lapse_model,
                 mortality_model,
                 surrender_charge,
                 state_names=FA_MODEL_STATES,
                 state_indices=FA_MODEL_STATE_INDICES,
                 init_state_weights=None,
                 state_history=None,  # index recording the known states, either historical or future projection
                 use_discrete=False):  # indicate if stochastic model is used instead of probabilistic model
        """
        Initiate the Fixed Annuity model. Model contains the updated model assumptions such as (dynamic) lapse rate,
        (dynamic) utilization rate, and etc. These information are updated in each projection period by model iterator
        using information from both model and account(InsAcct).
        """
        assert isinstance(acct, InsAcct)
        self._acct = acct
        self._state_names = state_names
        self._state_name_indices = state_indices
        self._num_states = len(self._state_names)
        self._state_history = state_history
        self._use_discrete = use_discrete
        self._lapse_model = lapse_model
        self._mortality_model = mortality_model
        self._surrender_charge = surrender_charge
        self._init_state_weights = init_state_weights

    def trans_matrix(self, model_iter):
        # Key function here to generate the transition matrix for the model iterator
        trans_matrix = np.zeros((self._num_states, self._num_states))

        p_1 = self._mortality_model.prob(model_iter)  # probability of mortality given the time period
        p_2 = self._lapse_model.prob(model_iter)
        # Assuming all the transitions are continuous and independent
        lambda_1 = -np.log(1-p_1)
        lambda_2 = -np.log(1-p_2)
        trans_matrix[self._state_name_indices['Dead'], self._state_name_indices['Active']] \
            = (p_1+p_2-p_1*p_2)*lambda_1/(lambda_1+lambda_2)
        trans_matrix[self._state_name_indices['Surrendered'], self._state_name_indices['Active']] \
            = (p_1+p_2-p_1*p_2)*lambda_2/(lambda_1+lambda_2)
        trans_matrix[self._state_name_indices['Active'], self._state_name_indices['Active']] \
            = 1-p_1-p_2+p_1*p_2

        trans_matrix[self._state_name_indices['Surrendered'], self._state_name_indices['Surrendered']] = 1.0
        trans_matrix[self._state_name_indices['Dead'], self._state_name_indices['Dead']] = 1.0

        return trans_matrix

    def surrender_charge(self, model_iter):
        return self._surrender_charge(model_iter)

    def create_iterator(self, pricing_date):
        return InsModelFAIter(self, self._acct, pricing_date, self._init_state_weights)

    @property
    def num_states(self):
        return self._num_states

    @property
    def state_name_indices(self):
        return self._state_name_indices

    @property
    def state_names(self):
        return self._state_names


class InsModelFAIter(object):
    def __init__(self,
                 model,
                 acct,
                 pricing_date,
                 state_weights=None):
        """
        The model iterator is used to evolve the model (InsModelFA)
        """
        assert isinstance(acct, InsAcct)
        self._model = model
        self._acct = acct
        self._pricing_date = pricing_date
        self._trans_matrix = np.zeros((self._model.num_states, self._model.num_states))
        if state_weights is None:
            self._state_weights = np.zeros(len(self._model.state_names))
            self._state_weights[model.state_name_indices["Active"]] = 1.0
        else:
            self._state_weights = state_weights
        self._schedule = acct.acct_iterator()
        self._next_date = None
        self._year_frac = None
        self._date = pricing_date
        self._new_death = 0.0
        self._new_lapse = 0.0
        self._prev_benefits_dict = {}

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

        # record some information for aggregator
        self._prev_benefits_dict = self._schedule.rider_benefits_dict()
        # Step 1: Advance state distribution
        _prev_state_weights = self._state_weights.copy()
        self.advance()
        _delta_weights = self._state_weights - _prev_state_weights

        # Step 2: Evolve the 'active' account
        self._schedule.next(_next_date, _year_frac)

        # Step 3: Calculate the state change
        self._new_death = _delta_weights[self._model.state_name_indices['Dead']]
        self._new_lapse = _delta_weights[self._model.state_name_indices['Surrendered']]

        # Step 4: calculate behavior related transactions (some calc should happen in aggregator)

        # wrap-up
        self._date = _next_date

    def m_trans_matrix(self):
        return self._trans_matrix

    def m_weight_active(self):
        return self._state_weights[self._model.state_name_indices['Active']]

    def m_new_death(self):
        return self._new_death

    def m_new_lapse(self):
        return self._new_lapse

    def m_collected_fees(self):
        pass

    def m_paid_benefits(self):
        pass

    def m_acct_value(self):
        return self._schedule.acct_value

    def m_non_rider_fee(self, fee_name):
        return self._schedule.non_rider_fee(fee_name)

    def m_rider_fee(self, fee_name):
        return self._schedule.rider_fee(fee_name)

    def m_prev_rider_benefits(self, rider_name):
        return self._prev_benefits_dict.get(rider_name, 0.0)

    def m_rider_benefit(self, rider_name):
        return self._schedule.rider_benefit(rider_name)

    def m_rider_benefit_base(self, rider_name):
        return self._schedule.rider_benefit_base(rider_name)

    def surrender_charge(self):
        return self._model.surrender_charge(self)

    @property
    def date(self):
        return self._date

    @property
    def attained_age(self):
        return self._schedule.attained_age

    @property
    def duration(self):
        return self._schedule.duration

    @property
    def year_frac(self):
        return self._year_frac

    @property
    def m_state_weights(self):
        return self._state_weights

    @property
    def anniv_flag(self):
        return self._schedule.anniv_flag

    # @property
    # def model(self):
    #     return self._model


if __name__ == '__main__':
    raw_input = {"Acct Value": 1344581.6,
                 "Attained Age": 52.8,
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
                 "Asset Values": [1344581.6]}

    # For now, we assume the init_date is month begin
    step_per_year = 12
    credit_rate = 0.03
    periods = 360
    init_date = dt.date(2013, 2, 1)
    pricing_date = init_date
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

    # Setup lapse function and lapse model
    xs = [0]
    ys = [0.0, 0.1]
    shock_func = linear_comp_bounded(1, 0, floor=0.5, cap=1.5)
    # lapse_model = LapseStatic(InsStepFunc(xs, ys))
    lapse_model = LapseDynamic(InsStepFunc(xs, ys), shock_func, rider_name='UWL')

    # Setup surrender charge
    xs = [0]
    ys = [100, 100]
    fixed_charge_func = InsStepFunc(xs, ys)
    xs = [0, 1, 2]
    ys = [0.0, 0.3, 0.2, 0.0]
    pct_charge_func = InsStepFunc(xs, ys)
    surrender_charge = SurrenderCharge(fixed_charge_func, pct_charge_func)

    # Setup mortality function and mortality model
    xs = [x for x in range(0, 100)]
    ys = [0.01] * 100
    ys.append(float('inf'))
    mort_model = InsMortModel(InsStepFunc(xs, ys))

    # Setup FA Model
    model = InsModelFA(acct, lapse_model, mort_model, surrender_charge)
    model_iter = model.create_iterator(pricing_date)

    # model iterator to evolve the model_iter to move forward
    metrics = ['Account Value',
               'Active Population',
               'Benefit Base.UWL',
               'Rider Fee.UWL',
               'Benefit.UWL',
               'Fee.Mgmt Fee',
               'Fee.Booking Fee',
               'Date',
               'Attained Age',
               'Anniv Flag',
               'Death',
               'Lapse',
               'Paid Benefit.UWL',
               'Surrender Charge',
               ]
    crv_aggregator = create_curve_aggregator(metrics)

    params = {'pricing date': init_date, 'periods': 60, 'frequency': 'MS'}
    proj_mgr = ProjectionManager(crv_aggregator, model_iter, **params)
    proj_mgr.run()

    df = crv_aggregator.to_dataframe()
    df[['Rider Fee.UWL', 'Fee.Mgmt Fee', 'Fee.Booking Fee', 'Surrender Charge']].plot(kind='bar', stacked=True)

    import os
    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    df.to_csv(OUTPUT_PATH + '/rider_rs.csv')
    print('output saved to %s' % OUTPUT_PATH + '/rider_rs.csv')








