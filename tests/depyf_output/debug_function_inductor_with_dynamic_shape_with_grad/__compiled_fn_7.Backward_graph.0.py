from __future__ import annotations
import torch
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s97)", primals_2: "f32[s97]", tangents_1: "f32[s97]"):
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function, code: x = a / (torch.abs(a) + 1)
        neg: "f32[s97]" = torch.ops.aten.neg.default(tangents_1)
        abs_1: "f32[s97]" = torch.ops.aten.abs.default(primals_2)
        add_2: "f32[s97]" = torch.ops.aten.add.Tensor(abs_1, 1);  abs_1 = None
        div_1: "f32[s97]" = torch.ops.aten.div.Tensor(primals_2, add_2)
        div_2: "f32[s97]" = torch.ops.aten.div.Tensor(div_1, add_2);  div_1 = None
        mul_3: "f32[s97]" = torch.ops.aten.mul.Tensor(neg, div_2);  neg = div_2 = None
        div_3: "f32[s97]" = torch.ops.aten.div.Tensor(tangents_1, add_2);  tangents_1 = add_2 = None
        sign: "f32[s97]" = torch.ops.aten.sign.default(primals_2);  primals_2 = None
        mul_4: "f32[s97]" = torch.ops.aten.mul.Tensor(mul_3, sign);  mul_3 = sign = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function, code: x = a / (torch.abs(a) + 1)
        add_7: "f32[s97]" = torch.ops.aten.add.Tensor(div_3, mul_4);  div_3 = mul_4 = None
        return (None, add_7, None, None)
        