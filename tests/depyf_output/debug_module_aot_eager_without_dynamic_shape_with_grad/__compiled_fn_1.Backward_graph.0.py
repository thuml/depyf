from __future__ import annotations
import torch
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[10]", tangents_1: "f32[10]"):
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward, code: x = a / (torch.abs(a) + 1)
        neg: "f32[10]" = torch.ops.aten.neg.default(tangents_1)
        abs_1: "f32[10]" = torch.ops.aten.abs.default(primals_1)
        add: "f32[10]" = torch.ops.aten.add.Tensor(abs_1, 1);  abs_1 = None
        div_1: "f32[10]" = torch.ops.aten.div.Tensor(primals_1, add)
        div_2: "f32[10]" = torch.ops.aten.div.Tensor(div_1, add);  div_1 = None
        mul: "f32[10]" = torch.ops.aten.mul.Tensor(neg, div_2);  neg = div_2 = None
        div_3: "f32[10]" = torch.ops.aten.div.Tensor(tangents_1, add);  tangents_1 = add = None
        sgn: "f32[10]" = torch.ops.aten.sgn.default(primals_1);  primals_1 = None
        mul_1: "f32[10]" = torch.ops.aten.mul.Tensor(mul, sgn);  mul = sgn = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward, code: x = a / (torch.abs(a) + 1)
        add_1: "f32[10]" = torch.ops.aten.add.Tensor(div_3, mul_1);  div_3 = mul_1 = None
        return (add_1, None)
        