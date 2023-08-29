from depyf import decompile
import unittest
import dis
import sys
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class Point:
    x: int
    y: int
    def __matmul__(self, other):
        return self.x * other.x + self.y * other.y

    def __imatmul__(self, other):
        self.x = self.x * other.x
        self.y = self.y * other.y
        return self

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        else:
            raise IndexError("Point only has two dimensions")


point = Point(1, 2)
data_map = {1: 2}


def test_UNARY_POSITIVE():
    def f():
        x = 1
        return +x
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_UNARY_NEGATIVE():
    def f():
        x = 1
        return -x
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_UNARY_NOT():
    def f():
        x = 1
        return not x
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_UNARY_INVERT():
    def f():
        x = 1
        return ~x
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BINARY_POWER():
    def f():
        a = 2
        b = 3
        return (a ** b) ** a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BINARY_MULTIPLY():
    def f():
        a = 2
        b = 3
        return (a ** b) ** a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BINARY_MATRIX_MULTIPLY():
    def f():
        return point @ point
    ans = f()
    scope = {'point': point}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BINARY_FLOOR_DIVIDE():
    def f():
        a = 2
        b = 3
        return (a // b) // a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BINARY_TRUE_DIVIDE():
    def f():
        a = 2
        b = 3
        return (a / b) / a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BINARY_MODULO():
    def f():
        a = 2
        b = 3
        return (a % b) % a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BINARY_ADD():
    def f():
        a = 2
        b = 3
        return (a + b) + a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BINARY_SUBTRACT():
    def f():
        a = 2
        b = 3
        return (a - b) - a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BINARY_SUBSCR():
    def f():
        a = (1, 2, 3)
        return a[0]
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BINARY_LSHIFT():
    def f():
        a = 2
        b = 3
        return (a << b) << a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BINARY_RSHIFT():
    def f():
        a = 2
        b = 3
        return (a >> b) >> a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BINARY_AND():
    def f():
        a = 2
        b = 3
        return (a & b) & a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BINARY_XOR():
    def f():
        a = 2
        b = 3
        return (a ^ b) ^ a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BINARY_OR():
    def f():
        a = 2
        b = 3
        return (a | b) | a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_INPLACE_POWER():
    def f():
        a = 2
        a **= 3
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_INPLACE_MULTIPLY():
    def f():
        a = 2
        a *= 3
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_INPLACE_MATRIX_MULTIPLY():
    def f():
        point = Point(1, 2)
        point @= point
        return point
    ans = f()
    scope = {'Point': Point}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_INPLACE_FLOOR_DIVIDE():
    def f():
        a = 2
        a //= 3
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_INPLACE_TRUE_DIVIDE():
    def f():
        a = 2
        a /= 3
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_INPLACE_MODULO():
    def f():
        a = 2
        a %= 3
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_INPLACE_ADD():
    def f():
        a = 2
        a += 3
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_INPLACE_SUBTRACT():
    def f():
        a = 2
        a -= 3
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_INPLACE_LSHIFT():
    def f():
        a = 2
        a <<= 3
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_INPLACE_RSHIFT():
    def f():
        a = 2
        a >>= 3
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_INPLACE_AND():
    def f():
        a = 2
        a &= 3
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_INPLACE_XOR():
    def f():
        a = 2
        a ^= 3
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_INPLACE_OR():
    def f():
        a = 2
        a |= 3
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_STORE_SUBSCR():
    def f():
        point = Point(1, 2)
        point[0] = 3
        return point
    ans = f()
    scope = {'Point': Point}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_DELETE_SUBSCR():
    def f():
        a = deepcopy(data_map)
        del a[1]
        return a
    ans = f()
    scope = {'data_map': data_map, 'deepcopy': deepcopy}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_YIELD_VALUE():
    def f():
        yield 1
        yield 2
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert list(scope['f']()) == list(ans)


def test_GET_LEN():
    def f():
        return len((1, 2, 3))
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_STORE_GLOBAL():
    def f():
        global len
        len = 1
        return len
    ans = f()
    scope = {'len': len}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_DELETE_NAME():
    def f():
        a = 1
        del a
        a = 2
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_UNPACK_SEQUENCE():
    def f():
        a, b = (1, 2)
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_UNPACK_EX():
    def f():
        a, *b = (1, 2, 3)
        return b
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_STORE_ATTR():
    def f():
        point = Point(1, 2)
        point.x = 5
        return point
    ans = f()
    scope = {'Point': Point}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_DELETE_ATTR():
    def f():
        point = Point(1, 2)
        del point.x
        point.x = 9
        return point
    ans = f()
    scope = {'Point': Point}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BUILD_TUPLE():
    def f():
        a = 1
        b = 2
        return (a, b), (a,)
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BUILD_TUPLE_UNPACK():
    def f():
        a = [1, 2, 3]
        b = {4, 5, 6}
        return (*a, *b), (*a,)
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BUILD_LIST_UNPACK():
    def f():
        a = [1, 2, 3]
        b = {4, 5, 6}
        return [*a, *b], [*a,]
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BUILD_SET_UNPACK():
    def f():
        a = [1, 2, 3]
        b = {4, 5, 6}
        return {*a, *b}, {*a,}
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BUILD_MAP_UNPACK():
    def f():
        a = {1: 2}
        b = {3: 4}
        return {**a, **b}, {**a,}
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans



def test_BUILD_LIST():
    def f():
        a = 1
        b = 2
        return [a, b], [a]
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BUILD_SET():
    def f():
        a = 1
        b = 2
        return {a, b}, {a}
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BUILD_MAP():
    def f():
        a = 1
        b = 2
        return {a: 1, 2: 3}, {b: a}
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BUILD_CONST_KEY_MAP():
    def f():
        return {5: 1, 2: 3}
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_LIST_TO_TUPLE():
    # not clear how to test this
    def f():
        return
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_LIST_EXTEND():
    def f():
        return [1, 2, 3]
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_SET_UPDATE():
    def f():
        return {1, 2, 3}
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_DICT_UPDATE():
    def f():
        a = {1: 2}
        b = {'a': 4}
        return {**a, **b}
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_DICT_MERGE():
    def f():
        a = {1: 2}
        b = {'a': 4}
        a.update(**b)
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_CALL_FUNCTION_EX():
    def func(*args, **kwargs):
        return (args, kwargs)
    def f():
        a = [1, 2, 3]
        b = {'a': 4}
        ans1 = func(*a)
        ans2 = func(**b)
        ans3 = func(*a, **b)
        ans4 = func()
        ans5 = func(1, 2, 3, *a)
        ans6 = func(1, 2, 3, **b)
        ans7 = func(b=2, **b)
        return ans1, ans2, ans3, ans4, ans5, ans6, ans7
    ans = f()
    scope = {'func': func}
    exec(decompile(f), scope)
    assert scope['f']() == ans



def test_LOAD_ATTR():
    def f():
        point = Point(1, 2)
        return point.x
    ans = f()
    scope = {'Point': Point}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_COMPARE_OP():
    def f():
        return (3 == 3) + (1 < 2) + (2 > 1) + (2 >= 2) + (1 <= 2) + (1 != 2)
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_IS_OP():
    def f():
        return (int is int), (int is not float)
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_CONTAINS_OP():
    def f():
        return (1 in [1, 2, 3]), (5 not in (6, 7, 4))
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_IMPORT_NAME():
    def f():
        from math import sqrt
        import functools
        return functools.partial(sqrt, 0.3)()
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_BUILD_SLICE():
    def f():
        a = [1, 2, 3]
        return a[:] + a[1:] + a[:2] + a[1:2] + a[::-1]
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_FORMAT_VALUE():
    def f():
        a = 1
        b = 2
        c = 3
        return f"{a} {b!r} {b!s} {b!a} {c:.2f}"
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_ROT_TWO():
    def f():
        a = 1
        b = 2
        c = 3
        d = 4
        e = 5
        a, b = b, a
        a, b, c = c, b, a
        a, b, c, d = d, c, b, a
        a, b, c, d, e = e, d, c, b, a
        return a
    ans = f()
    scope = {}
    exec(decompile(f), scope)
    assert scope['f']() == ans


def test_IF():
    def f(a):
        if a == 0:
            return 0
        elif a == 1:
            return 1
        else:
            return 2
    scope = {}
    exec(decompile(f), scope)
    for a in range(10):
        assert scope['f'](a) == f(a)


def test_IF_NONE():
    def f(a):
        if a is None:
            return 0
        elif a is not None:
            return 1
    scope = {}
    exec(decompile(f), scope)
    for a in range(10):
        assert scope['f'](a) == f(a)


def test_WHILE():
    def f(a):
        while a < 5:
            a += 1
            break
        return a
    scope = {}
    exec(decompile(f), scope)
    for a in range(10):
        assert scope['f'](a) == f(a)
