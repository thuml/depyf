from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s52)", primals_2: "f32[s52]", primals_3: "f32[s52]"):
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:6 in torch_dynamo_resume_in_toy_function_at_5, code: b = b * -1
        mul: "f32[s52]" = torch.ops.aten.mul.Tensor(primals_2, -1)
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:7 in torch_dynamo_resume_in_toy_function_at_5, code: return x * b
        mul_2: "f32[s52]" = torch.ops.aten.mul.Tensor(primals_3, mul);  mul = None
        return (mul_2, primals_2, primals_3, primals_1)
        