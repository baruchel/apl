# -*- coding: utf-8 -*-

import numpy as np
from .internal import (
        DomainError, RankError,
        _apl, AplArray,
        _apl_ensure, _apl_vector_ensure, _apl_raw_vector_ensure,
        _apl_disclose_ensure # TODO: remove
        )


apl_offset = 0


def rho(_right, _left=None):
    if _left == None: # monadic
        return _right.apl_rho()
    else: # dyadic
        # Right argument
        _right, _, stops, tailshape = _apl_ensure(_right)
        # Left argument
        _left, _, stops2, _ = _apl_raw_vector_ensure(_left)
        _left = tuple(_left)
        n = np.prod(_left) * np.prod(tailshape)
        _right = np.tile(_right.flatten(),
                      np.ceil(float(n) / np.prod(_right.shape)))
        _right = _apl(_right[:n].reshape(_left + tailshape))
        _right.__apl_stops__ = [ x - stops[0]+len(_left) for x in stops ]
        return _right
        

def index(_right, _left=None):
    if _left == None: # monadic
        if isinstance(_right, (np.integer, int)):
            return _apl(np.arange(apl_offset, _right+apl_offset))
        elif isinstance(_right, (np.floating, float)):
            if int(_right)==_right:
                return _apl(np.arange(apl_offset, int(_right)+apl_offset))
            else:
                raise DomainError(_right)
        elif isinstance(_right, (np.complexfloating, complex)):
            if _right.imag==0:
                return index(_right.real)
            else:
                raise DomainError(_right)
        elif isinstance(_right, (AplArray, np.ndarray)):
            s = _right.shape
            n = len(s)
            if n > 1:
                raise DomainError(_right)
            elif s[0] == 1: # identify as a scalar
                return index(_right[0])
            else:
                if not np.issubdtype(_right.dtype, np.integer):
                    # following line will raise an error for complex
                    #if np.any(np.mod(_right, 1) != 0):
                    tmp = _right.real.astype(np.int)
                    if np.all(_right == tmp):
                        _right = tmp
                    else:
                        raise DomainError(_right)
                return _apl(np.rollaxis(
                              np.indices(_right)+apl_offset, 0, s[0]+1),
                        stops = [s[0]])
        elif isinstance(_right, (tuple, list)):
            return index(np.array(_right))
        else:
            raise DomainError(_right)
    else: # dyadic
        _right, rho, stops, tailshape = _apl_ensure(_right)
        _left, rho2, stops2, ts2 = _apl_vector_ensure(_left)
        if tailshape != ts2:
            return _apl(np.array([rho2[0]] * int(np.prod(rho))).reshape(rho)
                         + apl_offset)
        _right = np.atleast_3d( _left.reshape( (1, rho2[0]) + ts2 )
                == _right.reshape( (np.prod(rho), 1) + ts2 ) ).all(2)
        tmp = np.argmax(_right, axis=1)
        tmp = (_right[:,0] == tmp)*rho2[0] + tmp + apl_offset
        return _apl(tmp.reshape(rho))


def make_monadic_dyadic_scalar_f(m, d):
    """
    Return a monadic/dyadic scalar function.
    """
    def f(_right, _left=None, _axis=[]):
        if _left == None: # monadic
            _right, _, stops, _ = _apl_ensure(_right)
            return _apl(m(_right), stops = stops)
        else: # dyadic
            if _axis:
                _axis, _, _, _ = _apl_raw_vector_ensure(_axis)
                if len(_axis) != len(set(_axis)):
                    raise RankError(_axis)
                _axis = [ x - apl_offset for x in _axis ]
            lstruct, rstruct = _left.apl_struct(), _right.apl_struct()

            # TODO: by default, add 1 to shape:
            # axis : conserver les dimensions nommées et mettre les autres à 1
            #    cf. manuel, p. 72
            # either one is scalar or rho=rho2
            # >>> a=np.array([[1,2],[3,4]])
            # >>> b=np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
            # >>> a.reshape((2,2,1))+b
            # a ← 2 2 ⍴ 1 2 3 4
            # b ← 2 2 ⍴ (1 2) (3 4) (5 6) (7 8)
            # a + b
            print("AX", _axis)
            
            pass # TODO
    return f
