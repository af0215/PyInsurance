"""This module provides classes for the non-rider based fees, e.g., management fees"""
# For the fee type class, we will assume the fee charge information required to calculate the fees are
# available from account iterator, therefore are not necessarily recorded in the fee object. This is different
# in the rider object, which needs to record the benefit/fee base on its own
from abc import ABCMeta, abstractmethod
from enum import Enum


class FeeType(Enum):
    constant = 1
    proportional = 2
    composite = 3


class InsFee(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fee(self, acct_iter):
        pass

    @property
    def fee_name(self):
        return self._fee_name


class InsFeeConst(InsFee):
    def __init__(self, fee_amt=None, fee_name=None):
        self._fee_amt = fee_amt
        self._fee_name = fee_name

    def fee(self, acct_iter):
        if acct_iter is None:
            raise ValueError("acct_iter cannot be None")
        if self._fee_amt is None:
            raise Exception('Fee amount is not yet set up!')
        return self._fee_amt * acct_iter.year_frac

    def set_fee(self, fee_amt):
        self._fee_amt = fee_amt

    @staticmethod
    def fee_type():
        return FeeType.constant.value


class InsFeeProp(InsFee):
    def __init__(self, fee_rate=None, fee_name=None):
        self._fee_rate = fee_rate
        self._fee_name = fee_name

    @property
    def fee_rate(self):
        if self._fee_rate is None:
            raise Exception('Fee rate is not yet set up!')
        return self._fee_rate

    @fee_rate.setter
    def fee_rate(self, fee_rate):
        self._fee_rate = fee_rate

    def set_fee_rate(self, fee_rate):
        if fee_rate < 0.0:
            raise Exception('Fee Rate must be non-negative!')
        self._fee_rate = fee_rate

    def fee(self, acct_iter):
        if acct_iter is None:
            raise ValueError("acct_iter cannot be None")
        return acct_iter.acct_value * self._fee_rate * acct_iter.year_frac

    @staticmethod
    def fee_type():
        return FeeType.proportional.value


class InsFeeComposite(InsFee):
    def __init__(self, fee_const, fee_prop, fee_name=None):
        self._fee_const = fee_const
        self._fee_prop = fee_prop
        self._fee_name = fee_name

    def fee(self, acct_iter=None):
        return self._fee_const.fee() + self._fee_prop.fee(acct_iter)

    @property
    def fee_rate(self):
        return self._fee_prop.fee_rate()

    @fee_rate.setter
    def fee_rate(self, fee_rate):
        self._fee_prop.fee_rate = fee_rate

    @staticmethod
    def fee_type():
        return FeeType.composite.value

# Examples:
if __name__ == "__main__":
    mgmt_fee = InsFeeProp(0.03)

