# -*- coding: utf-8 -*-

import numpy as np
from .core import make_monadic_dyadic_scalar_f

add = make_monadic_dyadic_scalar_f(lambda x: x, np.add)


sub = make_monadic_dyadic_scalar_f(np.negative, np.subtract)


def _direction(right):
    with np.errstate(divide='ignore', invalid='ignore'):
        return right/abs(right)
mul = make_monadic_dyadic_scalar_f(_direction, np.multiply)


def _reciprocal(right):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1./right
div = make_monadic_dyadic_scalar_f(_reciprocal, np.divide)


def _residue(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.mod(right, left) + (left == 0)*right
residue = make_monadic_dyadic_scalar_f(np.abs, _residue)

min = make_monadic_dyadic_scalar_f(np.floor, np.min)


max = make_monadic_dyadic_scalar_f(np.ceil, np.max)


power = make_monadic_dyadic_scalar_f(np.exp, np.power)


log = make_monadic_dyadic_scalar_f(np.log,
        lambda left, right: np.log(right)/np.log(left))
