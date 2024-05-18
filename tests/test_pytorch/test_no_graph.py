import torch

@torch.compile
def f(x):
    s = "Hello" + x
    print(s)

@torch.compile
def g(a, b):
    assert a
    print(b)
    [a for _ in [None]]

import depyf
with depyf.prepare_debug("./temp_output"):
    f(", world!")
    g(1, 2)
