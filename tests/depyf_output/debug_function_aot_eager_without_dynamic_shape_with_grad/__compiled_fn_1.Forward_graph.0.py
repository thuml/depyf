from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[10]", primals_2: "f32[10]"):
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function, code: x = a / (torch.abs(a) + 1)
        abs_1: "f32[10]" = torch.ops.aten.abs.default(primals_1)
        add: "f32[10]" = torch.ops.aten.add.Tensor(abs_1, 1);  abs_1 = None
        div: "f32[10]" = torch.ops.aten.div.Tensor(primals_1, add);  add = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:5 in toy_function, code: b = b * -1 + b.sum()
        mul: "f32[10]" = torch.ops.aten.mul.Tensor(primals_2, -1)
        sum_1: "f32[]" = torch.ops.aten.sum.default(primals_2)
        add_1: "f32[10]" = torch.ops.aten.add.Tensor(mul, sum_1);  mul = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:6 in toy_function, code: return x * b
        mul_1: "f32[10]" = torch.ops.aten.mul.Tensor(div, add_1);  div = add_1 = None
        return (mul_1, primals_1, primals_2, sum_1)
        