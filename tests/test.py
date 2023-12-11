from depyf import decompile
from depyf.utils import collect_all_code_objects
from depyf.code_transform import prepare_freevars_for_compile
import unittest
import dis
import sys
import inspect
import py_compile
import subprocess
from dataclasses import dataclass
from copy import deepcopy

from contextlib import contextmanager


def decompile_by_depyf(func):
    old_code = func.__code__
    decompile_path = "./decompiled_code.py"
    # first step, decompile the code
    src = decompile(old_code)
    new_src = prepare_freevars_for_compile(old_code, src)
    with open(decompile_path, "w") as f:
        f.write(new_src)

    # second step, compile the code
    tmp_code = compile(new_src, filename=old_code.co_filename, mode="exec")
    new_code = [x for x in collect_all_code_objects(tmp_code) if x.co_name == old_code.co_name][0]
    return new_code

def generate_pyc_and_get_decompiled_code(func, decompiler):
    old_code = func.__code__
    py_path = "./generated_code.py"
    decompile_path = "./decompiled_code.py"

    # first step, get the source code and remove indentation 
    src = inspect.getsource(func)
    src = src.split("\n")
    strip = 0
    while src[0][strip] == ' ' and strip < src[0].__len__():
        strip += 1
    src = "\n".join([line[strip:] for line in src])

    # second step, generate pyc and get decompiled code
    pyc_path = py_compile.compile(py_path)
    if pyc_path == None:
        raise Exception("Fail to compile")
    output = subprocess.check_output(decompiler + " " + pyc_path, shell=True)
    new_src = output.decode()

    # second step, compile the code
    tmp_code = compile(new_src, filename=old_code.co_filename, mode="exec")
    new_code = [x for x in collect_all_code_objects(tmp_code) if x.co_name == old_code.co_name][0]
    return new_code

def decompile_by_uncompyle6(func):
    return generate_pyc_and_get_decompiled_code(func, decompiler="uncompyle6")

def decompile_by_decompyle3(func):
    return generate_pyc_and_get_decompiled_code(func, decompiler="decompyle3")

def decompile_by_pycdc(func):
    return generate_pyc_and_get_decompiled_code(func, decompiler="./pycdc")

decompile_fn = decompile_by_depyf

@contextmanager
def replace_code_by_decompile_and_compile(func):
    old_code = func.__code__

    new_code = decompile_fn(func)

    # third step, replace the code
    func.__code__ = new_code
    try:
        yield
    finally:
        # restore the code
        func.__code__ = old_code

@contextmanager
def compile_extract_code_and_decompile(func):
    filename = "tmp_" + func.__code__.co_name + ".py"

    # first step, store the code of func .pys
    code = inspect.getsource(func)
    with open(filename, "w") as f:
        f.write(code)

    # second step, generate .pyc
    pyc_path = py_compile.compile(filename)

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
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_UNARY_NEGATIVE():
    def f():
        x = 1
        return -x

    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_UNARY_NOT():
    def f():
        x = 1
        return not x

    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_UNARY_INVERT():
    def f():
        x = 1
        return ~x

    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BINARY_POWER():
    def f():
        a = 2
        b = 3
        return (a ** b) ** a

    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BINARY_MULTIPLY():
    def f():
        a = 2
        b = 3
        return (a ** b) ** a

    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BINARY_MATRIX_MULTIPLY():
    def f():
        return point @ point

    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BINARY_FLOOR_DIVIDE():
    def f():
        a = 2
        b = 3
        return (a // b) // a

    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BINARY_TRUE_DIVIDE():
    def f():
        a = 2
        b = 3
        return (a / b) / a

    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BINARY_MODULO():
    def f():
        a = 2
        b = 3
        return (a % b) % a

    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BINARY_ADD():
    def f():
        a = 2
        b = 3
        return (a + b) + a

    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BINARY_SUBTRACT():
    def f():
        a = 2
        b = 3
        return (a - b) - a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BINARY_SUBSCR():
    def f():
        a = (1, 2, 3)
        return a[0]
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BINARY_LSHIFT():
    def f():
        a = 2
        b = 3
        return (a << b) << a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BINARY_RSHIFT():
    def f():
        a = 2
        b = 3
        return (a >> b) >> a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BINARY_AND():
    def f():
        a = 2
        b = 3
        return (a & b) & a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BINARY_XOR():
    def f():
        a = 2
        b = 3
        return (a ^ b) ^ a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BINARY_OR():
    def f():
        a = 2
        b = 3
        return (a | b) | a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_INPLACE_POWER():
    def f():
        a = 2
        a **= 3
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_INPLACE_MULTIPLY():
    def f():
        a = 2
        a *= 3
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_INPLACE_MATRIX_MULTIPLY():
    def f():
        point = Point(1, 2)
        point @= point
        return point
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_INPLACE_FLOOR_DIVIDE():
    def f():
        a = 2
        a //= 3
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_INPLACE_TRUE_DIVIDE():
    def f():
        a = 2
        a /= 3
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_INPLACE_MODULO():
    def f():
        a = 2
        a %= 3
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_INPLACE_ADD():
    def f():
        a = 2
        a += 3
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_INPLACE_SUBTRACT():
    def f():
        a = 2
        a -= 3
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_INPLACE_LSHIFT():
    def f():
        a = 2
        a <<= 3
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_INPLACE_RSHIFT():
    def f():
        a = 2
        a >>= 3
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_INPLACE_AND():
    def f():
        a = 2
        a &= 3
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_INPLACE_XOR():
    def f():
        a = 2
        a ^= 3
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_INPLACE_OR():
    def f():
        a = 2
        a |= 3
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_STORE_SUBSCR():
    def f():
        point = Point(1, 2)
        point[0] = 3
        return point
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_DELETE_SUBSCR():
    def f():
        a = deepcopy(data_map)
        del a[1]
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_GET_LEN():
    def f():
        return len((1, 2, 3))
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_STORE_GLOBAL():
    def f():
        global len
        len = 1
        return len
    with replace_code_by_decompile_and_compile(f):
        # test side effects
        global len
        original_len = len
        f()
        assert len == 1
        len = original_len


def test_DELETE_NAME():
    def f():
        a = 1
        del a
        a = 2
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_UNPACK_SEQUENCE():
    def f():
        a, b = (1, 2)
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_UNPACK_EX():
    def f():
        a, *b = (1, 2, 3)
        return b
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_STORE_ATTR():
    def f():
        point = Point(1, 2)
        point.x = 5
        return point
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_DELETE_ATTR():
    def f():
        point = Point(1, 2)
        del point.x
        point.x = 9
        return point
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BUILD_TUPLE():
    def f():
        a = 1
        b = 2
        return (a, b), (a,)
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BUILD_TUPLE_UNPACK():
    def f():
        a = [1, 2, 3]
        b = {4, 5, 6}
        return (*a, *b), (*a,)
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BUILD_LIST_UNPACK():
    def f():
        a = [1, 2, 3]
        b = {4, 5, 6}
        return [*a, *b], [*a,]
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BUILD_SET_UNPACK():
    def f():
        a = [1, 2, 3]
        b = {4, 5, 6}
        return {*a, *b}, {*a, }
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BUILD_MAP_UNPACK():
    def f():
        a = {1: 2}
        b = {3: 4}
        return {**a, **b}, {**a, }
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BUILD_LIST():
    def f():
        a = 1
        b = 2
        return [a, b], [a]
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BUILD_SET():
    def f():
        a = 1
        b = 2
        return {a, b}, {a}
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BUILD_MAP():
    def f():
        a = 1
        b = 2
        return {a: 1, 2: 3}, {b: a}
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BUILD_CONST_KEY_MAP():
    def f():
        return {5: 1, 2: 3}
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_LIST_TO_TUPLE():
    # not clear how to test this
    def f():
        return
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_LIST_EXTEND():
    def f():
        return [1, 2, 3]
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_SET_UPDATE():
    def f():
        return {1, 2, 3}
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_DICT_UPDATE():
    def f():
        a = {1: 2}
        b = {'a': 4}
        return {**a, **b}
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_DICT_MERGE():
    def f():
        a = {1: 2}
        b = {'a': 4}
        a.update(**b)
        return a
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_CALL_FUNCTION_NORMAL():
    def func(a, b, c=1):
        return (a, b, c)
    def f():
        a = [1, 2, 3]
        b = {'a': 4}
        ans1 = func(1, a, b)
        ans2 = func(b, 1, a)
        ans3 = func(a=a, b=b)
        ans4 = func(a=b, b=a)
        return ans1, ans2, ans3, ans4
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_function_signature():
    def func(a, b, c=1, *, d=2):
        return (a, b, c, d)

    a = [1, 2, 3]
    b = {'a': 4}
    ans1 = func(1, a, b, d=5)
    ans2 = func(b, 1, a, d=5)
    with replace_code_by_decompile_and_compile(func):
        assert func(1, a, b, d=5) == ans1
        assert func(b, 1, a, d=5) == ans2


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
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_var_args():
    def func(*args, **kwargs):
        return (args, kwargs)

    a = [1, 2, 3]
    b = {'a': 4}
    ans = func(1, a, b, d=5)
    with replace_code_by_decompile_and_compile(func):
        assert func(1, a, b, d=5) == ans


def test_complex_signature():
    def func(a, b, *args, **kwargs):
        return (a, b, args, kwargs)

    a = [1, 2, 3]
    b = {'a': 4}
    ans = func(1, a, b, d=5)
    with replace_code_by_decompile_and_compile(func):
        assert func(1, a, b, d=5) == ans


def test_LOAD_ATTR():
    def f():
        point = Point(1, 2)
        return point.x
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_COMPARE_OP():
    def f():
        return (3 == 3) + (1 < 2) + (2 > 1) + (2 >= 2) + (1 <= 2) + (1 != 2)
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_IS_OP():
    def f():
        return (int is int), (int is not float)
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_CONTAINS_OP():
    def f():
        return (1 in [1, 2, 3]), (5 not in (6, 7, 4))
    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_IMPORT_NAME():
    def f():
        from math import sqrt
        import functools
        return functools.partial(sqrt, 0.3)()

    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_BUILD_SLICE():
    def f():
        a = [1, 2, 3]
        return a[:] + a[1:] + a[:2] + a[1:2] + a[::-1]

    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_LIST_COMP():
    def f(a):
        return [i ** 2 for i in range(a)]

    ans = [f(i) for i in range(10)]
    with replace_code_by_decompile_and_compile(f):
        assert [f(i) for i in range(10)] == ans


def test_SET_COMP():
    def f(a):
        return {i ** 2 for i in range(a)}

    ans = [f(i) for i in range(10)]
    with replace_code_by_decompile_and_compile(f):
        assert [f(i) for i in range(10)] == ans


def test_MAP_COMP():
    def f(a):
        return {i: i ** 2 for i in range(a)}

    ans = [f(i) for i in range(10)]
    with replace_code_by_decompile_and_compile(f):
        assert [f(i) for i in range(10)] == ans


def test_NESTED_COMP():
    def f(a):
        return [{x: {_ for _ in range(x)} for x in range(i)} for i in range(a)]

    ans = [f(i) for i in range(10)]
    with replace_code_by_decompile_and_compile(f):
        assert [f(i) for i in range(10)] == ans


def test_FORMAT_VALUE():
    def f():
        a = 1
        b = 2
        c = 3
        return f"{a} {b!r} {b!s} {b!a} {c:.2f}"

    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


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
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans


def test_IF():
    def f(a):
        if a == 0:
            return 0
        elif a == 1:
            return 1
        else:
            return 2

    ans = [f(i) for i in range(10)]
    with replace_code_by_decompile_and_compile(f):
        assert [f(i) for i in range(10)] == ans


def test_compound_IF_and():
    def f(a, b):
        c = 1
        if a > 0 and b > 1:
            c += 1
        else:
            c += 2
        c += 3
        return c

    ans = [f(a, b) for a in range(-3, 3) for b in range(-3, 3)]
    with replace_code_by_decompile_and_compile(f):
        assert [f(a, b) for a in range(-3, 3) for b in range(-3, 3)] == ans


def test_compound_IF_or():
    def f(a, b):
        c = 1
        if a > 0 or b > 1:
            c += 1
        else:
            c += 2
        c += 3
        return c

    ans = [f(a, b) for a in range(-3, 3) for b in range(-3, 3)]
    with replace_code_by_decompile_and_compile(f):
        assert [f(a, b) for a in range(-3, 3) for b in range(-3, 3)] == ans


def test_IF_NONE():
    def f(a):
        if a is None:
            return 0
        elif a is not None:
            return 1

    ans = [f(i) for i in range(10)]
    with replace_code_by_decompile_and_compile(f):
        assert [f(i) for i in range(10)] == ans


def test_MAKE_FUNCTION():
    def f(a):
        def g(b=3):
            return a + b
        return g(2)

    ans = [f(i) for i in range(10)]
    with replace_code_by_decompile_and_compile(f):
        assert [f(i) for i in range(10)] == ans


def test_simple_try():
    def f(a):
        try:
            a += 1
        finally:
            a += 2
        a += 3
        return a

    ans = [f(i) for i in range(10)]
    with replace_code_by_decompile_and_compile(f):
        assert [f(i) for i in range(10)] == ans


def test_simple_for():
    def f(a):
        for i in range(5):
            a += i
        return a

    ans = [f(i) for i in range(10)]
    with replace_code_by_decompile_and_compile(f):
        assert [f(i) for i in range(10)] == ans


class A:
    def f(self):
        return __class__

def test_class_method():
    a = A()
    ans = a.f()
    with replace_code_by_decompile_and_compile(A.f):
        assert a.f() == ans


def test_EXTENDED_ARG():
    def f():
        a_658 = 658
        a_358 = 358
        a_308 = 308
        a_344 = 344
        a_640 = 640
        a_256 = 256
        a_347 = 347
        a_424 = 424
        a_336 = 336
        a_539 = 539
        a_56 = 56
        a_680 = 680
        a_860 = 860
        a_580 = 580
        a_230 = 230
        a_992 = 992
        a_433 = 433
        a_892 = 892
        a_888 = 888
        a_841 = 841
        a_748 = 748
        a_690 = 690
        a_166 = 166
        a_118 = 118
        a_165 = 165
        a_842 = 842
        a_575 = 575
        a_556 = 556
        a_402 = 402
        a_640 = 640
        a_154 = 154
        a_979 = 979
        a_350 = 350
        a_378 = 378
        a_742 = 742
        a_579 = 579
        a_830 = 830
        a_748 = 748
        a_688 = 688
        a_884 = 884
        a_118 = 118
        a_815 = 815
        a_413 = 413
        a_963 = 963
        a_864 = 864
        a_296 = 296
        a_552 = 552
        a_245 = 245
        a_969 = 969
        a_999 = 999
        a_71 = 71
        a_83 = 83
        a_964 = 964
        a_169 = 169
        a_48 = 48
        a_151 = 151
        a_100 = 100
        a_30 = 30
        a_478 = 478
        a_458 = 458
        a_233 = 233
        a_800 = 800
        a_191 = 191
        a_286 = 286
        a_488 = 488
        a_101 = 101
        a_456 = 456
        a_478 = 478
        a_347 = 347
        a_853 = 853
        a_961 = 961
        a_572 = 572
        a_407 = 407
        a_353 = 353
        a_79 = 79
        a_514 = 514
        a_801 = 801
        a_382 = 382
        a_898 = 898
        a_979 = 979
        a_696 = 696
        a_751 = 751
        a_367 = 367
        a_380 = 380
        a_52 = 52
        a_445 = 445
        a_321 = 321
        a_728 = 728
        a_19 = 19
        a_64 = 64
        a_679 = 679
        a_46 = 46
        a_402 = 402
        a_199 = 199
        a_479 = 479
        a_370 = 370
        a_768 = 768
        a_988 = 988
        a_205 = 205
        a_19 = 19
        a_125 = 125
        a_821 = 821
        a_335 = 335
        a_816 = 816
        a_135 = 135
        a_210 = 210
        a_212 = 212
        a_926 = 926
        a_726 = 726
        a_384 = 384
        a_279 = 279
        a_157 = 157
        a_457 = 457
        a_595 = 595
        a_184 = 184
        a_410 = 410
        a_375 = 375
        a_981 = 981
        a_154 = 154
        a_489 = 489
        a_209 = 209
        a_186 = 186
        a_784 = 784
        a_317 = 317
        a_110 = 110
        a_251 = 251
        a_752 = 752
        a_138 = 138
        a_119 = 119
        a_100 = 100
        a_74 = 74
        a_951 = 951
        a_3 = 3
        a_987 = 987
        a_655 = 655
        a_422 = 422
        a_89 = 89
        a_930 = 930
        a_235 = 235
        a_973 = 973
        a_391 = 391
        a_956 = 956
        a_641 = 641
        a_658 = 658
        a_458 = 458
        a_883 = 883
        a_440 = 440
        a_369 = 369
        a_99 = 99
        a_503 = 503
        a_94 = 94
        a_195 = 195
        a_403 = 403
        a_182 = 182
        a_192 = 192
        a_196 = 196
        a_903 = 903
        a_108 = 108
        a_224 = 224
        a_891 = 891
        a_922 = 922
        a_278 = 278
        a_201 = 201
        a_485 = 485
        a_200 = 200
        a_28 = 28
        a_249 = 249
        a_578 = 578
        a_857 = 857
        a_411 = 411
        a_807 = 807
        a_267 = 267
        a_667 = 667
        a_53 = 53
        a_907 = 907
        a_26 = 26
        a_273 = 273
        a_672 = 672
        a_945 = 945
        a_931 = 931
        a_318 = 318
        a_795 = 795
        a_439 = 439
        a_965 = 965
        a_939 = 939
        a_891 = 891
        a_572 = 572
        a_283 = 283
        a_706 = 706
        a_124 = 124
        a_383 = 383
        a_615 = 615
        a_883 = 883
        a_348 = 348
        a_805 = 805
        a_9 = 9
        a_970 = 970
        a_698 = 698
        a_526 = 526
        a_125 = 125
        a_621 = 621
        a_826 = 826
        a_273 = 273
        a_340 = 340
        a_781 = 781
        a_581 = 581
        a_112 = 112
        a_417 = 417
        a_632 = 632
        a_277 = 277
        a_432 = 432
        a_984 = 984
        a_437 = 437
        a_795 = 795
        a_437 = 437
        a_651 = 651
        a_288 = 288
        a_369 = 369
        a_722 = 722
        a_630 = 630
        a_65 = 65
        a_470 = 470
        a_307 = 307
        a_913 = 913
        a_666 = 666
        a_88 = 88
        a_326 = 326
        a_342 = 342
        a_912 = 912
        a_756 = 756
        a_540 = 540
        a_84 = 84
        a_595 = 595
        a_769 = 769
        a_607 = 607
        a_130 = 130
        a_556 = 556
        a_359 = 359
        a_237 = 237
        a_982 = 982
        a_274 = 274
        a_796 = 796
        a_604 = 604
        a_232 = 232
        a_559 = 559
        a_629 = 629
        a_329 = 329
        a_919 = 919
        a_61 = 61
        a_642 = 642
        a_127 = 127
        a_355 = 355
        a_694 = 694
        a_199 = 199
        a_261 = 261
        a_348 = 348
        a_319 = 319
        a_753 = 753
        a_621 = 621
        a_916 = 916
        a_601 = 601
        a_125 = 125
        a_854 = 854
        a_539 = 539
        a_212 = 212
        a_297 = 297
        a_736 = 736
        a_365 = 365
        a_965 = 965
        a_534 = 534
        a_472 = 472
        a_302 = 302
        a_513 = 513
        a_450 = 450
        a_370 = 370
        a_346 = 346
        a_492 = 492
        a_112 = 112
        a_457 = 457
        a_155 = 155
        a_423 = 423
        a_782 = 782
        a_941 = 941
        a_860 = 860
        a_932 = 932
        a_359 = 359
        a_329 = 329
        a_554 = 554
        a_299 = 299
        a_456 = 456
        a_663 = 663
        a_933 = 933
        a_826 = 826
        a_715 = 715
        a_319 = 319
        a_178 = 178
        a_551 = 551
        a_671 = 671
        a_41 = 41
        a_702 = 702
        a = 10
        return a

    ans = f()
    with replace_code_by_decompile_and_compile(f):
        assert f() == ans
