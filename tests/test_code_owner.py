from functools import partial, lru_cache

def f(a, b):
    return a + b

class A:
    def __call__(self, a, b):
        return a + b

import depyf

print(depyf.decompile(partial(f, 1)))

print(depyf.decompile(lru_cache(None)(f)))

print(depyf.decompile(A()))
