from depyf import decompile
import dis
import sys

"""We don't use test_* for this, because `pytest` changes the logic of `assert` statements
"""
if "RAISE_VARARGS" not in dis.opname or "LOAD_ASSERTION_ERROR" not in dis.opname:
    print("RAISE_VARARGS or LOAD_ASSERTION_ERROR not supported in this version of Python: {}".format(sys.version))
else:
    def f():
        assert 1 == 1, "1 is not equal to 1?"
        return
    ans = f()
    scope = {}
    exec(decompile(f.__code__), scope)
    assert scope['f']() == ans
