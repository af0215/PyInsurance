"""
Aggregator is used to collect the projected cash-flows, populations, ane etc,
from model iterator (e.g., InsModelFA) after each step of projection.
"""

from abc import ABCMeta, abstractmethod
import pandas as pd
from lib.decorators import scale_by_active_from_arg, scale_by_death_from_arg, scale_by_lapse_from_arg
from lib.decorators import identity_decorator, negative_decorator
import types


class InsCurveAggregator:
    def __init__(self, acct_metrics):
        self.model_elements = dict()
        for elem in acct_metrics:
            self.model_elements[elem] = create_element_aggregator(elem)

    def collect_element(self, model_iter):
        for _k, agt in self.model_elements.items():
            agt.collect_element(model_iter)

    def clear(self):
        del self.model_elements

    def to_dataframe(self):
        uniform_length = len(set(len(v.m_values) for k, v in self.model_elements.items())) == 1
        length_by_name = dict(dict((k, len(v.m_values)) for k, v in self.model_elements.items()))
        assert uniform_length, "collected are of different length %s" % length_by_name
        df = pd.DataFrame.from_dict(dict((k, v.m_values) for k, v in self.model_elements.items()))
        if 'Date' in self.model_elements:
            df.set_index(['Date'], inplace=True)
        if 'Anniv Flag' in self.model_elements:
            df['Anniv Flag'] = df['Anniv Flag'].astype(int)
        return df


class InsElementAggregator(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.m_pool_dates = []

    @abstractmethod
    def collect_element(self, model_iter):
        pass

    def clear(self):
        pass


class InsElementAggregatorBasic(InsElementAggregator):
    def __init__(self):
        InsElementAggregator.__init__(self)
        self.m_date_index = 0
        self.m_values = []

    def element_value(self, model_iter):
        pass

    def collect_element(self, model_iter):
        value = self.element_value(model_iter)
        if value is not None:
            self.m_values.append(value)

    def add_element(self, model_iter):
        pass

    def clear(self):
        pass

    def get_element(self):
        pass


class InsElementAggregatorProp(InsElementAggregatorBasic):
    def __init__(self, prop_name):
        super(InsElementAggregatorProp, self).__init__()
        self.prop_name = prop_name

    def element_value(self, model_iter):
        if hasattr(model_iter, '_'.join(self.prop_name.split())):
            return getattr(model_iter, '_'.join(self.prop_name.split()))
        else:
            raise Exception("model_iter doesnt have %s" % '_'.join(self.prop_name.split()))


def create_curve_aggregator(acct_metrics):
    return InsCurveAggregator(acct_metrics)


def element_aggregator_generator(attr_name, *args, scalar=identity_decorator, **kwargs):

    aggr = InsElementAggregatorBasic()

    def element_value(self, model_iter):
        from Models.AssPortfolioModel import AssPortfolioModelBase

        if isinstance(model_iter, AssPortfolioModelBase):
            result = [] #return a list here
            for itr in model_iter.model_iters:
                """ loop thru all the available model_iters and return a list """
                if hasattr(itr, attr_name):
                    result.append(getattr(itr, attr_name)(*args, **kwargs))
                else:
                    raise Exception("model_iter doesnt have %s" % attr_name)
            return result
        else:
            if hasattr(model_iter, attr_name):
                return getattr(model_iter, attr_name)(*args, **kwargs)
            else:
                raise Exception("model_iter doesnt have %s" % attr_name)

    func = element_value
    if isinstance(scalar, list):
        for deco in scalar:
            func = deco(func)
    else:
        func = scalar(func)
    # here dynamically re-bind the object (instance) method
    aggr.element_value = types.MethodType(func, aggr)
    return aggr


def create_element_aggregator(elem):
    if elem == "Account Value":
        return element_aggregator_generator('m_acct_value')
    elif elem == "Active Population":
        return element_aggregator_generator('m_weight_active')
    elif elem == "Death":
        return element_aggregator_generator('m_new_death')
    elif elem == "Lapse":
        return element_aggregator_generator('m_new_lapse')
    elif elem.startswith('Rider Fee.'):
        return element_aggregator_generator('m_rider_fee', elem.split('.')[1])
    elif elem.startswith('Fee.'):
        return element_aggregator_generator('m_non_rider_fee', elem.split('.')[1])
    elif elem.lower() in ("date", "anniv flag", "attained age"):
        return InsElementAggregatorProp(elem.lower())
    elif elem.startswith("Benefit Base."):
        return element_aggregator_generator('m_rider_benefit_base', elem.split('.')[1])
    elif elem.startswith("Benefit."):
        return element_aggregator_generator('m_rider_benefit', elem.split('.')[1])
    elif elem.startswith("Collected Fee."):
        return element_aggregator_generator('m_non_rider_fee', elem.split('.')[1], scalar=scale_by_active_from_arg)
    elif elem.startswith("Collected Rider Fee."):
        return element_aggregator_generator('m_rider_fee', elem.split('.')[1], scalar=scale_by_active_from_arg)
    elif elem.startswith("Paid Benefit."):
        return element_aggregator_generator('m_prev_rider_benefits', elem.split('.')[1], scalar=scale_by_death_from_arg)
    elif elem == 'Surrender Charge':
        return element_aggregator_generator('surrender_charge', scalar=scale_by_lapse_from_arg)
    elif elem == 'CF_In:Interest':
        return element_aggregator_generator('m_cf_interest', scalar=scale_by_active_from_arg)
    elif elem == 'CF_In:Principal':
        return element_aggregator_generator('m_cf_principal', scalar=scale_by_active_from_arg)
    elif elem.startswith("CF_In:Fee."):
        return element_aggregator_generator('m_non_rider_fee', elem.split('.')[1], scalar=scale_by_active_from_arg)
    elif elem.startswith("CF_In:Rider Fee."):
        return element_aggregator_generator('m_rider_fee', elem.split('.')[1], scalar=scale_by_active_from_arg)
    elif elem.startswith("CF_Out:Benefit."):
        return element_aggregator_generator('m_prev_rider_benefits', elem.split('.')[1], scalar=[scale_by_death_from_arg, negative_decorator])
    elif elem == 'CF_In:Surrender Charge':
        return element_aggregator_generator('surrender_charge', scalar=scale_by_lapse_from_arg)
    else:
        raise NotImplementedError("aggregator for {} is NOT implemented yet!".format(elem))

