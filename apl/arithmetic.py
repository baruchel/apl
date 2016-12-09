# -*- coding: utf-8 -*-

import numpy as np
from .core import make_monadic_dyadic_scalar_f

add = make_monadic_dyadic_scalar_f(lambda x: x, np.add)


sub = make_monadic_dyadic_scalar_f(np.negative, np.subtract)



def _direction(_right):
    with np.errstate(divide='ignore', invalid='ignore'):
        return _right/abs(_right)
mul = make_monadic_dyadic_scalar_f(_direction, np.multiply)


def _reciprocal(_right):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1./_right
div = make_monadic_dyadic_scalar_f(_reciprocal, np.divide)
