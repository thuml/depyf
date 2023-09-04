from depyf import decompile, Decompiler
import unittest
import dis
import sys
from dataclasses import dataclass
from copy import deepcopy

def test_deadcode_removal():
    def f():
        a = {'what': 'ever'}
        del a['what']
        a[1] = 2
        return len(a)
        blabla = 5
        return blabla
    assert 'blabla' not in decompile(f)

def test_while2():
    def f(a):
        while a < 5:
            a += 1
            if a > 3:
                continue
            else:
                break
        return 2 * a
    scope = {}
    exec(decompile(f), scope)
    for a in range(10):
        assert f(a) == scope['f'](a)

def test_while_else():
    def f(a, b):
        while a < 5 and b < 0:
            a += 1
        else:
            a = 1
            b += 1
        return a * b
    scope = {}
    exec(decompile(f), scope)
    for a in [4, 6]:
        for b in [-3, 4]:
            assert f(a, b) == scope['f'](a, b)

def test_rename():
    def f(a, b):
        return a * b
    scope = {}
    exec(Decompiler(f).decompile(overwite_fn_name="g"), scope)
    lambda_f = lambda a, b: a * b
    exec(Decompiler(lambda_f).decompile(overwite_fn_name="lam"), scope)
    a = 1
    b = 2
    assert f(a, b) == scope['g'](a, b)
    assert f(a, b) == scope['lam'](a, b)


def test_shortcircuit():
    def f(a, b):
        if a > 0 and b > 0:
            return a + b
        elif a > 1 or b > 2:
            return a - b
        else:
            return 2
    scope = {}
    exec(decompile(f), scope)
    for a in [-1, 0, 1, 2]:
        for b in [-1, 0, 1, 2]:
            assert f(a, b) == scope['f'](a, b)
