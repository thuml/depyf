import torch
from depyf.explain.backend import eager, aot_eager

@torch.compile(backend=eager)
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

toy_example(torch.randn(10), torch.randn(10))

from depyf.explain import dump_src
src = dump_src(toy_example)
with open("./dump_src_dump_function.py", "w") as f:
    f.write(src)
