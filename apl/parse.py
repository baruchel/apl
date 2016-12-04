# -*- coding: utf-8 -*-

import apl.token as token

def check_delimiters(tokens):
    """
    Check if corresponding {([])} symbols are correct.
    """
    stack = []
    for ty, to in tokens:
        if ty == token.SymbolType and to in u"{([":
            stack.append(to)
        if ty == token.SymbolType and to in u"])}":
            if len(stack) == 0:
                raise SyntaxError("No corresponding delimiter for " + to)
            elif (    (to == u"]" and stack[-1] != u"[")
                   or (to == u")" and stack[-1] != u"(")
                   or (to == u"}" and stack[-1] != u"{")):
                raise SyntaxError(stack[-1] + " delimiter"
                                  + " can not be closed with " + to)
            stack.pop()
    if len(stack):
        raise SyntaxError(stack[-1] + " delimiter is not closed")


def clean_numbers(tokens):
    """
    Map APL numbers to Python numbers.
    """
    t = []
    for ty, to in tokens:
        if ty in (token.IntegerType, token.FloatType, token.ComplexType):
            to = to.replace(u"¯", u"-")
            if ty == token.ComplexType:
                i = to.index(u"J")
                if to[i+1] == u"-":
                    to = "("+to[:i]+to[i+1:]+"j)"
                else:
                    to = "("+to[:i]+"+"+to[i+1:]+"j)"
        t.append((ty, to))
    return t

def parse_line(l):
    tokens = token.tokenize(l)
    check_delimiters(tokens)
    tokens = clean_numbers(tokens)
    return tokens
