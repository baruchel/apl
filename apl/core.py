# -*- coding: utf-8 -*-

import numpy as np

apl_offset = 0

class DomainError(Exception):
    pass
class RankError(Exception):
    pass

class AplArray(np.ndarray):
    def apl_rho(self):
        if len(self.__apl_stops__) == 0:
            return _apl(np.array(self.shape))
        return _apl(np.array(self.shape[:self.__apl_stops__[0]]))

def _apl(a):
    a = a.view(AplArray)
    a.__apl_stops__ = []
    return a

def rho(_right, _left=None):
    if _left == None: # monadic
        return _right.apl_rho()
    else: # dyadic
        # Right argument
        if isinstance(_right, AplArray):
            if len(_right.__apl_stops__):
                stops = _right.__apl_stops__
                r = _right.shape[stops[0]:]
            else:
                stops = []
                r = tuple([])
        elif isinstance(_right, np.ndarray):
            stops = []
            r = tuple([])
        elif isinstance(_right, (tuple, list)):
            _right = np.array(_right)
            stops = []
            r = tuple([])
        elif isinstance(_right, (np.integer, int,
                                 np.floating, float,
                                 np.complexfloating, complex)):
            _right = np.array([_right])
            stops = []
            r = tuple([])
        else: raise DomainError(_right)
        # Left argument
        if isinstance(_left, AplArray):
            if len(_left.__apl_stops__) or len(_left.shape) > 1:
                raise ValueError(_left)
            tmp = _left.real.astype(np.int)
            if np.all(_left == tmp): _left = tmp
            else: raise DomainError(_left)
            _left = tuple(_left)
        elif isinstance(_left, np.ndarray):
            if len(_left.shape) > 1: raise ValueError(_left)
            tmp = _left.real.astype(np.int)
            if np.all(_left == tmp): _left = tmp
            else: raise DomainError(_left)
            _left = tuple(_left)
        elif isinstance(_left, (tuple, list)):
            return rho(_right, np.array(_left))
        elif isinstance(_left, (np.integer, int,
                                 np.floating, float,
                                 np.complexfloating, complex)):
            if _left == int(_left): _left = (_left,)
            else: raise DomainError(_left)
        else: raise DomainError(_left)
        n = np.prod(_left) * np.prod(r)
        _right = np.tile(_right.flatten(),
                      np.ceil(float(n) / np.prod(_right.shape)))
        _right = _apl(_right[:n].reshape(_left + r))
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
                a = np.rollaxis(np.indices(_right)+apl_offset, 0, s[0]+1)
                a = _apl(a)
                a.__apl_stops__ = [s[0]]
                return a
        elif isinstance(_right, (tuple, list)):
            return index(np.array(_right))
        else:
            raise DomainError(_right)
    else: # dyadic
        pass # TODO
