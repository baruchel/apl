# -*- coding: utf-8 -*-

import numpy as np
from .internal import (
        DomainError, RankError, InvalidAxisError,
        _apl, AplArray,
        _apl_ensure, _apl_vector_ensure, _apl_raw_vector_ensure,
        )


apl_offset = 0


def rho(right, left=None):
    if left == None: # monadic
        return _apl_ensure(right).apl_rho()
    else: # dyadic
        # Right argument
        right = _apl_ensure(right)
        stops = right.__apl_stops__
        tailshape = right.shape[stops[0]:] if stops else tuple([])
        # Left argument
        left = _apl_raw_vector_ensure(left)
        left = tuple(left)
        n = np.prod(left) * np.prod(tailshape)
        right = np.tile(right.flatten(),
                      np.ceil(float(n) / np.prod(right.shape)))
        right = _apl(right[:n].reshape(left + tailshape))
        right.__apl_stops__ = [ x - stops[0]+len(left) for x in stops ]
        return right
        

def index(right, left=None):
    if left == None: # monadic
        if isinstance(right, (np.integer, int)):
            return _apl(np.arange(apl_offset, right+apl_offset))
        elif isinstance(right, (np.floating, float)):
            if int(right)==right:
                return _apl(np.arange(apl_offset, int(right)+apl_offset))
            else:
                raise DomainError(right)
        elif isinstance(right, (np.complexfloating, complex)):
            if right.imag==0:
                return index(right.real)
            else:
                raise DomainError(right)
        elif isinstance(right, (AplArray, np.ndarray)):
            s = right.shape
            n = len(s)
            if n > 1:
                raise DomainError(right)
            elif s[0] == 1: # identify as a scalar
                return index(right[0])
            else:
                if not np.issubdtype(right.dtype, np.integer):
                    # following line will raise an error for complex
                    #if np.any(np.mod(right, 1) != 0):
                    tmp = right.real.astype(np.int)
                    if np.all(right == tmp):
                        right = tmp
                    else:
                        raise DomainError(right)
                return _apl(np.rollaxis(
                              np.indices(right)+apl_offset, 0, s[0]+1),
                        stops = [s[0]])
        elif isinstance(right, (tuple, list)):
            return index(np.array(right))
        else:
            raise DomainError(right)
    else: # dyadic
        right = _apl_ensure(right)
        stops = right.__apl_stops__
        if stops:
            rho = right.shape[:stops[0]]
            tailshape = right.shape[stops[0]:]
        else:
            rho = right.shape
            tailshape = tuple([])
        left = _apl_vector_ensure(left)
        stops2 = left.__apl_stops__
        if stops2:
            rho2 = left.shape[:stops2[0]]
            ts2 = left.shape[stops2[0]:]
        else:
            rho2 = left.shape
            ts2 = tuple([])
        if tailshape != ts2:
            return _apl(np.array([rho2[0]] * int(np.prod(rho))).reshape(rho)
                         + apl_offset)
        right = np.atleast_3d( left.reshape( (1, rho2[0]) + ts2 )
                == right.reshape( (np.prod(rho), 1) + ts2 ) ).all(2)
        tmp = np.argmax(right, axis=1)
        tmp = (right[:,0] == tmp)*rho2[0] + tmp + apl_offset
        return _apl(tmp.reshape(rho))


def make_monadic_dyadic_scalar_f(m, d):
    """
    Return a monadic/dyadic scalar function.
    For dyadic functions, the order of the arguments should be (left, right).
    """
    def f(right, left=None, _axis=[]):
        if left == None: # monadic
            right, _, stops, _ = _apl_ensure(right)
            return _apl(m(right), stops = stops)
        else: # dyadic
            if _axis:
                _axis = _apl_raw_vector_ensure(_axis)
                if len(_axis) != len(set(_axis)):
                    raise RankError(_axis.apl_struct())
                _axis = [ x - apl_offset for x in _axis ]
            right = _apl_ensure(right)
            left = _apl_ensure(left)
            lstruct, rstruct = left.apl_struct(), right.apl_struct()
            ln, rn = len(lstruct), len(rstruct)
            ls, rs = tuple([]), tuple([])
            stops = []
            for i in range(max(ln, rn)):
                if i < ln and lstruct[i]:
                    if i < rn and rstruct[i]:
                        if _axis:
                            if _axis[-1] < len(lstruct[i]):
                                A = tuple(lstruct[i][j] for j in _axis)
                                if A == rstruct[i]:
                                    S = [1]*len(lstruct[i])
                                    for j in _axis: S[j] = lstruct[i][j]
                                    ls += lstruct[i]
                                    rs += tuple(S)
                                    _axis = None
                                    stops.append(len(ls))
                                    continue
                            if _axis[-1] < len(rstruct[i]):
                                A = tuple(rstruct[i][j] for j in _axis)
                                if A == lstruct[i]:
                                    S = [1]*len(rstruct[i])
                                    for j in _axis: S[j] = rstruct[i][j]
                                    ls += tuple(S)
                                    rs += rstruct[i]
                                    _axis = None
                                    stops.append(len(ls))
                                    continue
                            raise InvalidAxisError(_axis)
                        if lstruct[i] == rstruct[i]:
                            ls += lstruct[i]
                            rs += rstruct[i]
                        elif all(x==1 for x in lstruct[i]):
                            ls += (1,) * len(rstruct[i])
                            rs += rstruct[i]
                        elif all(x==1 for x in rstruct[i]):
                            ls += lstruct[i]
                            rs += (1,) * len(lstruct[i])
                        else: raise RankError(left.apl_struct(),
                                              right.apl_struct())
                    else:
                        ls += lstruct[i]
                        rs += (1,) * len(lstruct[i])
                    _axis = None
                else:
                    if i < rn and rstruct[i]:
                        rs += rstruct[i]
                        ls += (1,) * len(rstruct[i])
                        _axis = None
                stops.append(len(ls))
            stops.pop()
            return _apl(
                d(left.reshape(ls), right.reshape(rs)),
                stops = stops)
    return f
