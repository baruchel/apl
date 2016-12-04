# -*- coding: utf-8 -*-

import numpy as np
from apl import *

import apl.core

print(index(5))
print(index(5.0))
print(index(5.+0j))

apl.core.apl_offset = 1
print(index(5))
print(index(5.0))
print(index(5.+0j))

apl.core.apl_offset = 0
print(index((2,3)))
print(index(np.array([2,3])))
print(index(APL([2,3])))

a = index(APL((2,3,4)))
print(rho(a))

print(rho([1,0,0,0], [3,3]))
print(rho(rho(rho(APL([[1,2],[3,4]])))))


print(parse_line(u"⍳(3J3+.5j¯3)-.5"))
