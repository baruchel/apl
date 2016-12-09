"""
An implementation of the APL programming language.
"""

__version__ = '0.1'

import numpy as np

from .internal import (
          _apl, AplArray,
        )

from .core import (
          index, rho
        )

from .arithmetic import (
          add, sub, mul, div
        )

from .parse import parse_line

def APL(x):
    """
    Return an array to be used with the APL module.
    This type is basically a Numpy array with some internal
    new features.
    """
    if isinstance(x, AplArray):
        return _apl(np.array(x), stops = x.__apl_stops__)
    if isinstance(x, (np.integer, int,
                                 np.floating, float,
                                 np.complexfloating, complex)):
        return _apl(np.array([x])) # scalar
    return _apl(np.array(x))

__all__ = ['APL', 'index', 'rho',
           'add', 'sub', 'mul', 'div',
           # to be removed later
           'parse_line'
          ]
