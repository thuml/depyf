from __future__ import annotations



def forward(self, primals_1: "f32[10]", primals_2: "f32[10]", tangents_1: "f32[10]"):
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:17, code: return x * b
    mul_1: "f32[10]" = torch.ops.aten.mul.Tensor(tangents_1, primals_1);  primals_1 = None
    mul_2: "f32[10]" = torch.ops.aten.mul.Tensor(tangents_1, primals_2);  tangents_1 = primals_2 = None
    return [mul_2, mul_1]
    