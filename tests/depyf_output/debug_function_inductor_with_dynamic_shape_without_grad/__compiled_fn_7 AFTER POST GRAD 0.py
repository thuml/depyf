from __future__ import annotations



def forward(self, arg0_1: "f32[8]", arg1_1: "f32[8]"):
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:6, code: b = b * -1
    mul: "f32[8]" = torch.ops.aten.mul.Tensor(arg0_1, -1);  arg0_1 = None
    
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:7, code: return x * b
    mul_1: "f32[8]" = torch.ops.aten.mul.Tensor(arg1_1, mul);  arg1_1 = mul = None
    return (mul_1,)
    