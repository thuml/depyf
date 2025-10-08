from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, L_a_: "f32[10]", L_b_: "f32[10]"):
        l_a_ = L_a_
        l_b_ = L_b_
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function, code: x = a / (torch.abs(a) + 1)
        abs_1: "f32[10]" = torch.abs(l_a_)
        add: "f32[10]" = abs_1 + 1;  abs_1 = None
        x: "f32[10]" = l_a_ / add;  l_a_ = add = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:5 in toy_function, code: b = b * -1 + b.sum()
        mul: "f32[10]" = l_b_ * -1
        sum_1: "f32[]" = l_b_.sum();  l_b_ = None
        b: "f32[10]" = mul + sum_1;  mul = sum_1 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:6 in toy_function, code: return x * b
        mul_1: "f32[10]" = x * b;  x = b = None
        return (mul_1,)
        