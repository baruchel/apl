from apl import *


# Make apl_offset=1
import apl.core
apl.core.apl_offset = 1

a = rho(index(4), [2,2])
a.__apl_stops__.append(0) # Enclose (not yet implemented)
a.__apl_stops__.append(0) # Enclose (not yet implemented)
b = rho(index(4), [2,2])
c = add(a,b)
print(c.__repr__())

a = rho(index(6), [2,3])
a.__apl_stops__.append(0) # Enclose (not yet implemented)
a = rho(a, [5])
b = rho(index(10), [2, 5])
c = add(a,b, axis=[2])
print(c.__repr__())
