"""
Test:           Insurance.test.TestInsAcct.py
Description:    Utility related functions, e.g., fractional mortality interpolation, dynamic lapse calculation, and etc.
"""

import bisect
import numpy as np
from abc import ABCMeta, abstractmethod


class InsFunc(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def integral(self, x, dx):
        pass


class InsStepFunc(InsFunc):
    def __init__(self, xs, ys):
        self._xs = xs.copy()
        self._ys = ys.copy()
        self._f = create_step_func(self._xs, self._ys)
        self._sf = create_step_func_cum(self._xs, self._ys)

    def __call__(self, x):
        return self._f(x)

    def integral(self, x, dx):
        return self._sf(x, dx)


def create_step_func(xs, ys):
    if not isinstance(xs, np.ndarray):
        xs = np.array(xs)
    if not isinstance(ys, np.ndarray):
        ys = np.array(ys)
    if xs.size != (ys.size-1):
        raise ValueError('Length of x should be 1 less than length of y!')
    if any(xs != np.sort(xs)):
        raise ValueError('x should be sorted ascending!')

    def f(x):
        i = bisect.bisect(xs, x)
        return ys[i]

    return f


def create_step_func_cum(xs, ys):
    if len(xs) != len(ys)-1:
        raise ValueError('Length of x should be 1 less than length of y!')
    if not isinstance(xs, np.ndarray):
        xs = np.array(xs)
    if not isinstance(ys, np.ndarray):
        ys = np.array(ys)
    if any(xs != np.sort(xs)):
        raise ValueError('x should be sorted ascending!')

    area = (xs[1:]-xs[:-1]) * ys[1:-1]

    def sf(x, dx):
        if dx < 0:
            raise ValueError('dx cannot be negative')
        elif dx == 0.0:
            return 0.0
        i = bisect.bisect(xs, x)
        j = bisect.bisect(xs, x+dx)
        if i == j:
            return dx * ys[i]
        else:
            s = sum(area[i:j-1])
            dx_i = xs[i] - x
            dx_j = x + dx - xs[j-1]
            ds_i = 0 if dx_i == 0 else dx_i * ys[i]
            ds_j = 0 if dx_j == 0 else dx_j * ys[j]
            s += ds_i + ds_j
            return s
    return sf


def linear_transfer(slope, shift):
    """
        This function generates a decorator to generate g(x)=f(slope*x + shift)
    """
    # TODO: not tested
    def transfer_f(original_f):
        def transferred_f(x):
            return original_f(slope*x+shift)
        return transferred_f
    return transfer_f


def linear_transfer_integral(slope=1., shift=0.):
    """
        This function generates a decorator to get the integral function sg(x, dx) from sf(x, dx)
        for g(x)=f(slope*x + shift).
    """
    # TODO: not tested
    def transfer_sf(original_sf):
        def transferred_sf(x, dx):
            return original_sf(slope*x+shift, slope*dx)/slope
        return transferred_sf
    return transfer_sf


def linear_comp_bounded(scalar, shift, cap=np.PINF, floor=np.NINF):
    """
        This function generates a decorator to generate g(x)= scalar * f(x) + shift
    """
    # TODO: not tested
    def compose_f(original_f):
        def composed_f(x):
            return min(max(floor, scalar * original_f(x) + shift), cap)
        return composed_f
    return compose_f


if __name__ == '__main__':
    @linear_comp_bounded(2, 3, 10, 0)
    def f(x):
        return x
    print(f(0), f(3), f(10))
