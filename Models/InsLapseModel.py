"""Models for calculation of lapse rate"""

from abc import ABCMeta, abstractmethod
import numpy as np


class LapseBasic(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def prob(self, model_iter):
        pass


class LapseStatic(LapseBasic):
    def __init__(self, lapse_func):
        """
            lapse_func:
        """
        self._lapse_func = lapse_func

    def prob(self, model_iter):
        return 1.0 - np.exp(-1 * self._lapse_func.integral(model_iter.duration, model_iter.year_frac))


class LapseDynamic(LapseBasic):

    def __init__(self, base_lapse_func, dynamic_shock, rider_name):
        """
        lapse_func:     base lapse function, usually a InsStepFunc object
        dynamic_shock:  a functional, can use lib.insurance.linear_comp_bounded(scalar, shift, floor, cap)
        rider_name:     rider name of the contract whose death benefit will be used to calculate ratio,
                        a function of moneyness = AV/DB
        """
        super(LapseDynamic, self).__init__()
        self._base_lapse_func = base_lapse_func
        self._dynamic_shock = dynamic_shock(lambda x: x)
        self._rider_name = rider_name

    def prob(self, model_iter):
        moneyness = model_iter.m_acct_value()/model_iter.m_rider_benefit(self._rider_name)
        ratio = self._dynamic_shock(moneyness)
        return 1.0 - np.exp(-1.0 * self._base_lapse_func.integral(model_iter.duration, model_iter.year_frac) * ratio)


class SurrenderCharge(object):
    def __init__(self, fixed_charge_func=None, pct_charge_func=None):
        self._fixed_charge = fixed_charge_func
        self._pct_charge = pct_charge_func

    def __call__(self, model_iter):
        return self._fixed_charge(model_iter.duration) \
            + model_iter.m_acct_value() * self._pct_charge(model_iter.duration)



