__author__ = 'Ting'

import unittest

import numpy as np

import Managers.ScenarioManager as sm


class TestScenarioManger(unittest.TestCase):
    def setUp(self):
        pass

    # EqBSEngine Tests
    def test_eq_bs_zero_vol(self):
        """
            If vol is zero, give the expected deterministic return; The two paths with same drifts have the same return
        """
        engine = sm.EqBSEngine(np.array([0., 0.]),
                               np.array([0., 0.]),
                               corr=np.array([[1., 0.3], [0.3, 1.]]))
        steps = np.array([5./252]*50)
        path = engine.simulate(np.array([1.0, 1.0]), steps)
        self.assertTrue(np.all(v[0] == v[1] for v in path))

        engine = sm.EqBSEngine(np.array([0.]),
                               np.array([0.]))
        steps = np.array([5./252]*50)
        path = engine.simulate(1.0, steps)
        self.assertTrue(np.all(path == 1.))

    def test_eq_bs_cor(self):
        """
            When the correlation is very close to 1, the return is almost the same
        """
        engine = sm.EqBSEngine(np.array([0.02, 0.02]),
                               np.array([0.2, 0.2]),
                               corr=np.array([[1., 0.99999999],
                                              [0.99999999, 1.]]))
        steps = np.array([5./252]*50)
        path = engine.simulate(np.array([1.0, 1.0]), steps)
        rt = [path[i]/path[i-1] for i in range(1, len(path))]
        rel_diff = [ abs(v[1]-v[0])/(abs(v[1])+abs(v[0])) for v in rt]
        self.assertAlmostEqual(max(rel_diff), 0.0, places=3)

    # EqBSTermEngine Tests

    # IRHWEngine Tests
    def test_ir_hw_zero_vol(self):
        """
            When the vol is zero and the current level is at the mean return level, it stays at this level.
        """
        engine = sm.IRHWEngine(0.1, 0.2, 0.)
        steps = np.array([5./252]*50)
        r_0 = 0.1
        path = engine.simulate(r_0, steps)
        self.assertTrue(np.all(path == r_0))

    # FixRateEngine Tests
    def test_fix_rate(self):
        """
        create a fix rate engine and make sure it stays fixed at given rate
        """

        engine = sm.FixRateEngine(0.05)
        steps = np.array([5./252]*50)
        path = engine.simulate(None, steps)
        self.assertTrue(np.all(path==0.05))

    def tearDown(self):
        pass

