import torch

@torch.compile
def f(x):
    s = "hello " + x
    return s

import depyf
with depyf.prepare_debug("./temp_output"):
    f(", world!")
