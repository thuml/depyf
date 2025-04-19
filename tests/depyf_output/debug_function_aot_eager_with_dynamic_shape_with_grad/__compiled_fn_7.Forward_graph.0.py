from __future__ import annotations
import torch
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s97)", primals_2: "f32[s97]", primals_3: "Sym(s52)", primals_4: "f32[s52]"):
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function, code: x = a / (torch.abs(a) + 1)
        abs_1: "f32[s97]" = torch.ops.aten.abs.default(primals_2)
        add_2: "f32[s97]" = torch.ops.aten.add.Tensor(abs_1, 1);  abs_1 = None
        div: "f32[s97]" = torch.ops.aten.div.Tensor(primals_2, add_2);  add_2 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:5 in toy_function, code: if b.sum() < 0:
        sum_1: "f32[]" = torch.ops.aten.sum.default(primals_4);  primals_4 = None
        lt: "b8[]" = torch.ops.aten.lt.Scalar(sum_1, 0);  sum_1 = None
        return (lt, div, primals_2, primals_1)
        