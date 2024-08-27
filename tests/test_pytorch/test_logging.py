import torch

@torch.compile
def fn():
    x = torch.randn(1, 10, requires_grad=True)
    y = torch.randn(10, 1)
    loss = torch.mm(x, y).sum()
    loss.backward()
    return x.grad

import depyf
depyf.install()
fn()
depyf.uninstall()
