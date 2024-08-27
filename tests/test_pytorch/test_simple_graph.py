import torch

@torch.compile
def fn():
    x = torch.randn(1, 10, requires_grad=True)
    y = torch.randn(10, 1)
    loss = torch.mm(x, y).sum()
    loss.backward()
    return x.grad

import depyf
with depyf.prepare_debug("./simple_output", log_bytecode=True, clean_wild_fx_code=False):
    fn()
