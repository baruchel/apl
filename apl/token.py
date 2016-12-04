# -*- coding: utf-8 -*-

import re

__symbols__  = u"()[]{}"
__symbols__ += u"+-×÷"
__symbols__ += u"⍺⍵"
__symbols__ += u"⍳⍴"
__symbols__ += u"←"

__name__  = u"([A-Za-z∆⍙][A-Za-z0-9∆⍙¯_]*)"  # Constructed Names
# TODO: negative numbers
__name__ += u"|(¯?[0-9]+\.[0-9]*)|(¯?\.?[0-9]+)" # Floats or Integers
__name__ = re.compile(__name__)

class __APL_type__:
    def __init__(self, t):
        self.name = "APL." + t
    def __repr__(self):
        return self.name
    def __str__(self):
        return self.name
SymbolType   = __APL_type__("symbol")
IntegerType  = __APL_type__("int")
FloatType    = __APL_type__("float")
ComplexType  = __APL_type__("complex")
StringType   = __APL_type__("str")
VariableType = __APL_type__("variable")

def tokenize(s):
    """
    Return the list of tokens from a line of APL code.
    The string must be a single line of code.

    NB. A complex number is returned with a uppercase J in all cases.
    """
    i = 0
    t = []
    while i < len(s):
        # comment
        if s[i] == u"⍝":
            return t
        # space
        elif s[i] == u" ":
            i += 1
        # symbol
        elif s[i] in __symbols__:
            t.append((SymbolType, s[i]))
            i += 1
        # quote
        # TODO: escape (not in Dyalog APL or NGN APL)?
        elif s[i] in u"'\"":
            try:
                j = s.index(s[i], i+1)
                t.append((StringType, s[i:j+1]))
                i = j + 1
            except ValueError:
                raise SyntaxError(s[i:] + " (col. " + str(i) + ")")
        # misc
        else:
            m = __name__.match(s[i:])
            if m == None:
                raise SyntaxError(s[i:] + " (col. " + str(i) + ")")
            else:
                m = m.group()
                i += len(m)
                # Complex number
                if i+1 <len(s) and s[i] in u"jJ":
                    i += 1
                    n = __name__.match(s[i:])
                    if n == None:
                        raise SyntaxError(s[i:] + " (col. " + str(i) + ")")
                    else:
                        n = n.group()
                        t.append((ComplexType, m + u"J" + n))
                        i += len(n)
                # Float, Integer or Variable
                elif m[0] in u"¯.0123456789":
                    if u"." in m:
                        t.append((FloatType, m))
                    else:
                        t.append((IntegerType, m))
                else:
                    t.append((VariableType, m))
                # No constructed name following a number:
                if i < len(s) and None != __name__.match(s[i:]):
                    raise SyntaxError(s[i:] + " (col. " + str(i) + ")")
    return t
