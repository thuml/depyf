import torch

def toy_function(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

class ToyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        x = a / (torch.abs(a) + 1)
        if b.sum() < 0:
            b = b * -1
        return x * b


import os
backend = os.environ.get("TORCH_COMPILE_BACKEND", "eager")
requires_grad = os.environ.get("REQUIRES_GRAD", "0") != "0"
dynamic_shape = os.environ.get("DYNAMIC_SHAPE", "0") != "0"
compile_type = os.environ.get("COMPILE_TYPE", "function")
usage_type = os.environ.get("USAGE_TYPE", "debug")

assert backend in ["eager", "aot_eager", "inductor"]

input1 = torch.ones(10, requires_grad=requires_grad), torch.ones(10, requires_grad=requires_grad)

input2 = torch.ones(8, requires_grad=requires_grad), - torch.ones(8, requires_grad=requires_grad)

if compile_type == "function":
    target = toy_function
elif compile_type == "module":
    target = ToyModule()
else:
    raise ValueError("Unknown compile_type: {}".format(compile_type))

target = torch.compile(target, backend=backend)

def warmup():
    for _ in range(100):
        target(*input1)
        if dynamic_shape:
            target(*input2)

def call():
    target(*input1)

description = f"{usage_type}_{compile_type}_{backend}"
description += "_with_dynamic_shape" if dynamic_shape else "_without_dynamic_shape"
description += "_with_grad" if requires_grad else "_without_grad"

from depyf.utils import safe_create_directory

import os
if not os.path.exists("./depyf_output/"):
    safe_create_directory("./depyf_output/")

import depyf
with depyf.prepare_debug(f"./depyf_output/{description}"):
    warmup()
