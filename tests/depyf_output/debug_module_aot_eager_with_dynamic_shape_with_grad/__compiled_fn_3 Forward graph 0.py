from __future__ import annotations



def forward(self, primals_1: "f32[10]", primals_2: "f32[10]"):
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:17, code: return x * b
    mul: "f32[10]" = torch.ops.aten.mul.Tensor(primals_1, primals_2)
    return [mul, primals_1, primals_2]
    