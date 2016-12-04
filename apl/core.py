# -*- coding: utf-8 -*-

import numpy as np

apl_offset = 0

class DomainError(Exception):
    pass

class APLArray(np.ndarray):
    def apl_rho(self):
        if len(self.__apl_stops__) == 0:
            return _apl(np.array(self.shape))
        return _apl(np.array(self.shape[:self.__apl_stops__[0]]))

def _apl(a):
    a = a.view(APLArray)
    a.__apl_stops__ = []
    return a

def rho(_right, _left=None):
    if _left == None: # monadic
        return _right.apl_rho()

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
        elif isinstance(_right, (APLArray, np.ndarray)):
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

        
