import torch

@torch.compile(backend="aot_eager")
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

import depyf
# uncomment the following line to allow you to set breakpoints in the dumped source code
# with depyf.prepare_debug(toy_example, "./dump_src_debug_function"):
with depyf.prepare_debug(toy_example, "./dump_src_debug_function_aot"):
    for _ in range(100):
        toy_example(torch.randn(10, requires_grad=True), torch.randn(10, requires_grad=True))

with depyf.debug():
    toy_example(torch.randn(10, requires_grad=True), torch.randn(10, requires_grad=True))
