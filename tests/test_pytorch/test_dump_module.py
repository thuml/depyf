import torch

class ToyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        x = a / (torch.abs(a) + 1)
        if b.sum() < 0:
            b = b * -1
        return x * b

toy_module = ToyModule()

toy_module = torch.compile(toy_module, backend="eager")

toy_module(torch.randn(10), torch.randn(10))

from depyf.explain import dump_src
src = dump_src(toy_module)
with open("./dump_src_dump_module.py", "w") as f:
    f.write(src)
