import torch

@torch.compile
def f(x):
    s = "Hello" + x
    print(s)

import depyf
with depyf.prepare_debug("./temp_output"):
    f(", world!")
