from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s52)", arg1_1: "f32[s52]", arg2_1: "f32[s52]"):
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function, code: x = a / (torch.abs(a) + 1)
        abs_1: "f32[s52]" = torch.ops.aten.abs.default(arg1_1)
        add_2: "f32[s52]" = torch.ops.aten.add.Tensor(abs_1, 1);  abs_1 = None
        div: "f32[s52]" = torch.ops.aten.div.Tensor(arg1_1, add_2);  arg1_1 = add_2 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:5 in toy_function, code: b = b * -1 + b.sum()
        mul_3: "f32[s52]" = torch.ops.aten.mul.Tensor(arg2_1, -1)
        sum_1: "f32[]" = torch.ops.aten.sum.default(arg2_1);  arg2_1 = None
        add_9: "f32[s52]" = torch.ops.aten.add.Tensor(mul_3, sum_1);  mul_3 = sum_1 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:6 in toy_function, code: return x * b
        mul_6: "f32[s52]" = torch.ops.aten.mul.Tensor(div, add_9);  div = add_9 = None
        return (mul_6,)
        