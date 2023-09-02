from depyf import decompile
import unittest
import dis
import sys
from dataclasses import dataclass
from copy import deepcopy

from depyf import decompile

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

def test_shortcircuit():
    def f(a, b):
        if a > 0 and b > 0:
            return a + b
        elif a > 0 or b > 0:
            return a - b
        else:
            return 2
    scope = {}
    exec(decompile(f), scope)
    for a in [-1, 1]:
        for b in [-2, 2]:
            assert f(a, b) == scope['f'](a, b)
