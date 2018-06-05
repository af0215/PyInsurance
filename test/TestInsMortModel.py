__author__ = 'Ting'

import unittest
import numpy as np
from Models.InsMortModel import InsMortModel
from lib.insurance import InsStepFunc
from collections import namedtuple


class TestInsMortModel(unittest.TestCase):
    def setUp(self):
        pass

    def test_prob_range(self):
        """
            Test the probability is between 0 and 1.
        """
        x = np.random.uniform(0, 90, 10)
        x.sort()
        y = np.random.uniform(0, 4, 10)
        y = np.insert(y, y.size, float('inf'))
        mort_func = InsStepFunc(x, y)
        mort_model = InsMortModel(mort_func)
        for age, year_frac in zip(np.random.uniform(0, 90, 100), np.random.uniform(0, 1, 100)):
            _prob = mort_model.prob(namedtuple('mock_iter', ['attained_age', 'year_frac'])(age, year_frac))
            self.assertTrue(0.0 <= _prob <= 1.0)

    def test_prob_mono(self):
        """
            Test the probability is monotonic w.r.t year_frac.
        """
        x = np.array(range(0, 90))
        y = np.array(range(0, 90)) * 0.01
        y = np.insert(y, y.size, float('inf'))
        mort_func = InsStepFunc(x, y)
        mort_model = InsMortModel(mort_func)
        _probs = np.array([mort_model.prob(namedtuple('mock_iter', ['attained_age', 'year_frac'])(0, year_frac))
                           for year_frac in range(0, 90)])
        self.assertTrue(all((_probs[1:] - _probs[:-1]) >= 0.0))

    def test_prob_limit(self):
        """
            Test the limit case:
            1. When duration is zero, the prob is zero
            2. When duration is over the upper limit of age, prob is 1
        """
        x = np.array(range(0, 90))
        y = np.array(range(0, 90)) * 0.01
        y = np.insert(y, y.size, float('inf'))
        mort_func = InsStepFunc(x, y)
        mort_model = InsMortModel(mort_func)
        for age in range(0, 100):
            _prob = mort_model.prob(namedtuple('mock_iter', ['attained_age', 'year_frac'])(age, 0.0))
            self.assertAlmostEqual(_prob, 0.0, 10)
            _prob = mort_model.prob(namedtuple('mock_iter', ['attained_age', 'year_frac'])(age, 100-age))
            self.assertAlmostEqual(_prob, 1.0, 10)

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()