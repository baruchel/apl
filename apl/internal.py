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


def _apl_ensure(_right):
    if isinstance(_right, AplArray):
        if len(_right.__apl_stops__):
            stops = _right.__apl_stops__
            rho = _right.shape[:stops[0]]
            tailshape = _right.shape[stops[0]:]
        else:
            stops = []
            rho, tailshape = _right.shape, tuple([])
    elif isinstance(_right, np.ndarray):
        stops = []
        _right = _apl(np.array([_right]))
        rho, tailshape = _right.shape, tuple([])
    elif isinstance(_right, (np.integer, int,
                             np.floating, float,
                             np.complexfloating, complex)):
        stops = []
        _right = _apl(np.array([_right]))
        rho, tailshape = (1,), tuple([])
    else: # list, tuple, range, etc.
        try:
            stops = []
            _right = _apl(np.array(_right))
            rho, tailshape = _right.shape, tuple([])
        except:
            raise TypeError(_right)
    return _right, rho, stops, tailshape


def _apl_vector_ensure(_right):
    """
    Return common parameters and checks that array is a vector.
    """
    _right, rho, stops, tailshape = _apl_ensure(_right)
    if len(rho) > 1:
        raise RankError(_right.apl_struct())
    return _right, rho, stops, tailshape

def _apl_raw_vector_ensure(_right):
    """
    Return common parameters and checks that array is a vector of scalars.
    """
    _right, rho, stops, tailshape = _apl_ensure(_right)
    if len(rho) > 1:
        raise RankError(_right.apl_struct()) # TODO?
    if len(stops):
        raise RankError(_right.apl_struct()) # TODO?
    return _right, rho, stops, tailshape
