from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s52)", arg1_1: "f32[s52]", arg2_1: "f32[s52]"):
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:6 in torch_dynamo_resume_in_toy_function_at_5, code: b = b * -1
        mul: "f32[s52]" = torch.ops.aten.mul.Tensor(arg1_1, -1);  arg1_1 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:7 in torch_dynamo_resume_in_toy_function_at_5, code: return x * b
        mul_2: "f32[s52]" = torch.ops.aten.mul.Tensor(arg2_1, mul);  arg2_1 = mul = None
        return (mul_2,)
        