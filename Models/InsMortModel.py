"""
    Test: Insurance.test.TestInsMortModel.py
"""

import numpy as np


class InsMortModel(object):
    """
        This is the mortality model.
    """
    def __init__(self, mort_func):
        """
            The mortality model is initialized by the mort_func, which provides the
             aggregate mortality intensity
        """
        self._mort_func = mort_func

    def prob(self, model_iter):
        return 1.0 - np.exp(-1 * self._mort_func.integral(model_iter.attained_age, model_iter.year_frac))


if __name__ == '__main__':
    from lib.insurance import InsStepFunc
    from collections import namedtuple
    x = np.array(range(0, 90))
    y = np.array(range(0, 90)) * 0.01
    y = np.insert(y, y.size, float('inf'))
    mort_func = InsStepFunc(x, y)
    mort_model = InsMortModel(mort_func)
    for age in range(0, 90):
        _prob = mort_model.prob(namedtuple('mock_iter', ['attained_age', 'year_frac'])(age, 100-age))
        print(_prob)


