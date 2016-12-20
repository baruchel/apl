# -*- coding: utf-8 -*-

import numpy as np

class AplError(Exception):
    pass
class DomainError(AplError):
    pass
class RankError(AplError):
    pass
class InvalidAxisError(AplError):
    pass



class AplArray(np.ndarray):
    def apl_rho(self):
        if len(self.__apl_stops__) == 0:
            return _apl(np.array(self.shape))
        return _apl(np.array(self.shape[:self.__apl_stops__[0]]))
    def apl_struct(self):
        s, struct = 0, []
        for i in (self.__apl_stops__ + [len(self.shape)]):
            struct.append((self.shape)[s:i])
            s = i
        return struct
    def apl_pretty_struct(self, sep="x"):
        s = ""
        a = self.apl_struct()
        for k in a:
            s += "(" + ("".join([str(x) + sep for x in k]))
        return s[1:-len(sep)] + (")" * (len(a)-1))
    def __repr__(self):
        N = np.ndarray.__repr__(self)
        s = self.apl_pretty_struct()
        return N + "\n    with APL structure: " + s


def _apl(a, stops=[]):
    a = a.view(AplArray)
    a.__apl_stops__ = stops
    return a


def _apl_ensure(right):
    if isinstance(right, AplArray): return right
    elif isinstance(right, np.ndarray): return _apl(right)
    elif isinstance(right, (np.integer, int,
                             np.floating, float,
                             np.complexfloating, complex)):
        return _apl(np.array([right]))
    else: # list, tuple, range, etc.
        try:
            return _apl(np.array(right))
        except:
            raise TypeError(_right)


def _apl_vector_ensure(right):
    """
    Return an array after having checked it is a vector.
    """
    right = _apl_ensure(right)
    rho = right.apl_rho()
    if len(rho) > 1:
        raise RankError(right.apl_struct())
    return right

def _apl_raw_vector_ensure(right):
    """
    Return an array after having checked it is a vector of scalars.
    """
    right = _apl_ensure(right)
    rho = right.apl_rho()
    if len(rho) > 1:
        raise RankError(_right.apl_struct()) # TODO?
    if len(right.__apl_stops__):
        raise RankError(_right.apl_struct()) # TODO?
    return right
