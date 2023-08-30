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
