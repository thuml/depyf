import torch
from depyf.explain.backend import eager, aot_eager

class ToyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        x = a / (torch.abs(a) + 1)
        if b.sum() < 0:
            b = b * -1
        return x * b

toy_module = ToyModule()

toy_module = torch.compile(toy_module)

import depyf
# uncomment the following line to allow you to set breakpoints in the dumped source code
# with depyf.prepare_debug(toy_module, "./dump_src_debug_module"):
with depyf.prepare_debug(toy_module, "./dump_src_debug_module", pause=False):
    for _ in range(100):
        toy_module(torch.randn(10), torch.randn(10))

with depyf.debug():
    toy_module(torch.randn(10), torch.randn(10))
