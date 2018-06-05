__author__ = 'Ting'

"""
Test:         Insurance.test.TestInsUtils.py
Description:
"""

import pandas as pd
import numpy as np
import Infra.IndexProvider as ip
import lib.constants as const


def extract_or_default(container, tag, default_value=None):
    """
    if tag exists in container, return the associated value,
    return default_value otherwise.
    """

    if tag in container:
        return container.get(tag)
    else:
        return default_value


def extract_strict(container, tag):
    """
    if tag exists in container, return the associated value,
    else raise exception
    """
    if tag in container:
        return container.get(tag)
    else:
        raise Exception("{} is not in the container".format(tag))


def next_date(curr_date, step_per_year):
    if step_per_year == 12:
        return pd.date_range(curr_date, periods=2, freq='MS')[1].date()
    if step_per_year == 4:
        return pd.date_range(curr_date, periods=2, freq='3MS')[1].date()
    if step_per_year == 1:
        return pd.date_range(curr_date, periods=2, freq='12MS')[1].date()


def inv_return(_prev_date, _curr_date, asset_values, inv_index):
    assert inv_index, ip.IndexProvider
    if inv_index.n_assets == len(asset_values):
        return asset_values * (inv_index.index_value(_curr_date)/inv_index.index_value(_prev_date) - 1.0)
    else:
        raise ValueError("# of assets indexes is different from # of assets!")


def check_equal(a, b, tol=1e-10):
    return abs(a-b) < tol


def dcf_act_act(start_date, end_date):
    return float((end_date - start_date).days) / const.DAYS_PER_YEAR


def is_corr_matrix(mtr):
    return np.all(mtr == mtr.transpose()) and np.all(np.linalg.eigvals(mtr) >= 0) and np.all(np.diagonal(mtr) == 1.0)