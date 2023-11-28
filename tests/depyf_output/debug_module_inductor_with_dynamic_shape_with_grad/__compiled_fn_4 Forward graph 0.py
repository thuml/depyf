from __future__ import annotations



def forward(self, primals_1: "Sym(s0)", primals_2: "f32[s0]", primals_3: "Sym(s1)", primals_4: "f32[s1]"):
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14, code: x = a / (torch.abs(a) + 1)
    abs_1: "f32[s0]" = torch.ops.aten.abs.default(primals_2)
    add: "f32[s0]" = torch.ops.aten.add.Tensor(abs_1, 1);  abs_1 = None
    div: "f32[s0]" = torch.ops.aten.div.Tensor(primals_2, add);  add = None
    
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:15, code: if b.sum() < 0:
    sum_1: "f32[]" = torch.ops.aten.sum.default(primals_4);  primals_4 = None
    lt: "b8[]" = torch.ops.aten.lt.Scalar(sum_1, 0);  sum_1 = None
    return [div, lt, primals_2, div, primals_1]
    