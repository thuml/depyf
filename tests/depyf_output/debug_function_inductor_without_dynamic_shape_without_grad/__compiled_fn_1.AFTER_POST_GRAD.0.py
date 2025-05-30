from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[10]", arg1_1: "f32[10]"):
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:5 in toy_function, code: if b.sum() < 0:
        sum_1: "f32[]" = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
        lt: "b8[]" = torch.ops.aten.lt.Scalar(sum_1, 0);  sum_1 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function, code: x = a / (torch.abs(a) + 1)
        abs_1: "f32[10]" = torch.ops.aten.abs.default(arg0_1)
        add: "f32[10]" = torch.ops.aten.add.Tensor(abs_1, 1);  abs_1 = None
        div: "f32[10]" = torch.ops.aten.div.Tensor(arg0_1, add);  arg0_1 = add = None
        return (lt, div)
        