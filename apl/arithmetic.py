# -*- coding: utf-8 -*-

import numpy as np
from .internal import make_monadic_dyadic_scalar_f

add = make_monadic_dyadic_scalar_f(lambda x: x, np.add)


sub = make_monadic_dyadic_scalar_f(np.negative, np.subtract)
