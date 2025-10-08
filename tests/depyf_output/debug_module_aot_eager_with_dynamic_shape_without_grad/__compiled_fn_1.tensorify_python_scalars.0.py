from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[10]", arg1_1: "f32[10]"):
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward, code: x = a / (torch.abs(a) + 1)
        abs_1: "f32[10]" = torch.ops.aten.abs.default(arg0_1)
        add: "f32[10]" = torch.ops.aten.add.Tensor(abs_1, 1);  abs_1 = None
        div: "f32[10]" = torch.ops.aten.div.Tensor(arg0_1, add);  arg0_1 = add = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward, code: b = b * -1 + b.sum()
        mul: "f32[10]" = torch.ops.aten.mul.Tensor(arg1_1, -1)
        sum_1: "f32[]" = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
        add_1: "f32[10]" = torch.ops.aten.add.Tensor(mul, sum_1);  mul = sum_1 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:15 in forward, code: return x * b
        mul_1: "f32[10]" = torch.ops.aten.mul.Tensor(div, add_1);  div = add_1 = None
        return (mul_1,)
        