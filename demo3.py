# -*- coding: utf-8 -*-

import numpy as np
from apl import *

import apl.core

a = index(APL((2,3,4)))
print(a.apl_pretty_struct(sep=u"×"))
print(a.apl_pretty_struct(sep=u" × "))
print(a.apl_pretty_struct(sep=u", "))
a.__apl_stops__.insert(0, 0) # enclose (not implemented yet)
print(a.apl_pretty_struct(sep=u"×"))
print(a.apl_pretty_struct(sep=u" × "))
print(a.apl_pretty_struct(sep=u", "))
