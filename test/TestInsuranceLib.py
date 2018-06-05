__author__ = 'Ting'

import unittest
from lib.insurance import create_step_func, create_step_func_cum
import numpy as np


class TestInsuranceLib(unittest.TestCase):
    def setUp(self):
        xs = np.array([0, 1, 2])
        ys = np.array([-1, 1, 5, 10])
        self.f1 = create_step_func(xs, ys)
        self.f2 = create_step_func_cum(xs, ys)

    def test_step_func(self):
        rs1 = np.array([self.f1(x) for x in [0, 1, 2, -10, 1.5, 10]])
        rs2 = np.array([self.f2(*x) for x in [(0, 2), (-1, 3), (-1, 0.5), (3, 5)]])
        self.assertTrue(all(rs1 == np.array([1, 5, 10, -1, 5, 10])))
        self.assertTrue(all(rs2 == np.array([6, 5, -0.5, 50])))

    def test_step_func_cum(self):
        #Todo: add some test here
        pass

    def tearDown(self):
        pass


