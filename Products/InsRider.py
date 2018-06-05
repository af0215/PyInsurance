"""Class of insurance riders."""

from abc import ABCMeta, abstractmethod

import Infra.IndexProvider as mip

# TODO: may use the digits to represents rider type
# TODO: 1. need to consider two separate functions to reflect 1) change of benefit base and 2) benefit payout!


class InsRiderBase(object):
    __metaclass__ = ABCMeta

    """
    Base API for insurance riders
    """
    @abstractmethod
    def __init__(self):
        self._rider_type = None
        self._benefit_base = None
        self._rider_name = None
        self._fee_rate = None

    @abstractmethod
    def benefit(self, acct_iter):
        pass

    @abstractmethod
    def set_benefit_base(self, acct_iter):
        pass

    @abstractmethod
    def rider_fee(self, acct_iter):
        pass

    @abstractmethod
    def bind_to_account(self, cell):
        """
        update the account specific information, such as ROP and etc.
        """
        pass

    def set_fee_rate(self, fee_rate):
        if fee_rate < 0.0:
            raise Exception('Fee Rate must be non-negative!')
        self._fee_rate = fee_rate

    @property
    def rider_type(self):
        return self._rider_type

    @property
    def benefit_base(self):
        return self._benefit_base

    @property
    def fee_rate(self):
        return self._fee_rate

    @property
    def rider_name(self):
        return self._rider_name


class InsRiderDB(InsRiderBase):
    def __init__(self, benefit_base=None, fee_rate=None, rider_name=None):
        super(InsRiderDB, self).__init__()
        # fee_rate is annualized
        self._benefit_base = benefit_base
        self._rider_type = "Death Benefit"
        self._fee_rate = fee_rate
        self._rider_name = rider_name

    def benefit(self, acct_iter):
        if self._benefit_base is None:
            raise Exception('Benefit base is not yet set up!')
        return self._benefit_base

    def rider_fee(self, acct_iter):
        if self._fee_rate is None:
            raise Exception('Rider fee is not yet set up!')
        if isinstance(self._fee_rate, float):
            return self._benefit_base * self._fee_rate * acct_iter.year_frac
        if isinstance(self._fee_rate, mip.FeeRateIndex):
            if self._fee_rate.index_name == "Date":
                return self._benefit_base * self._fee_rate.index_vlaue(acct_iter.date) * acct_iter.year_frac
            if self._fee_rate.index_name == "Age":
                return self._benefit_base * self._fee_rate.index_vlaue(acct_iter.attained_age) * acct_iter.year_frac
            if self._fee_rate.index_name == "Duration":
                return self._benefit_base * self._fee_rate.index_vlaue(acct_iter.duration) * acct_iter.year_frac

    def set_benefit_base(self, acct_iter):
        self._benefit_base = acct_iter.acct_value

    def update_benefit_base(self, updated_benefit_base):
        self._benefit_base = updated_benefit_base

