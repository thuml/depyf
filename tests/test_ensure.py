from depyf.utils import decompile_ensure

import asyncio

def f(a, b):
    try:
        return a + b
    finally:
        return a - b

print(decompile_ensure(f.__code__))
