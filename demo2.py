from apl import *
import apl.core
apl.core.apl_offset = 1

a = rho(index(4), [2,2])
a.__apl_stops__.append(0)
a.__apl_stops__.append(0)
b = rho(index(4), [2,2])
c = add(a,b)
print(c.__repr__())

a = rho(index(6), [2,3])
a.__apl_stops__.append(0)
a = rho(a, [5])
b = rho(index(10), [2, 5])
c = add(a,b, _axis=[2])
print(c.__repr__())
