"""
An implementation of the APL programming language.
"""

__version__ = '0.1'

import numpy as np

import core

def APL(x):
    """
    Return an array to be used with the APL module.
    This type is basically a Numpy array with some internal
    new features.
    """
    return core._apl(np.array(x))

__all__ = ['APL']
