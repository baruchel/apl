"""
An implementation of the APL programming language.
"""

__version__ = '0.1'

import numpy as np

from .core import (
          # internal methods
          _apl,
          # public methods
          index, rho
        )

from .parse import parse_line

def APL(x):
    """
    Return an array to be used with the APL module.
    This type is basically a Numpy array with some internal
    new features.
    """
    return _apl(np.array(x))

__all__ = ['APL', 'index', 'rho',
           # to be removed later
           'parse_line'
          ]