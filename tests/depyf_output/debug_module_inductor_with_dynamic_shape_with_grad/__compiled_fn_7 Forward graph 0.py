from __future__ import annotations



def forward(self, primals_1: "f32[8]", primals_2: "f32[8]"):
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:16 in torch_dynamo_resume_in_forward_at_15, code: b = b * -1
    mul: "f32[8]" = torch.ops.aten.mul.Tensor(primals_1, -1)
    
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:17 in torch_dynamo_resume_in_forward_at_15, code: return x * b
    mul_1: "f32[8]" = torch.ops.aten.mul.Tensor(primals_2, mul);  mul = None
    return [mul_1, primals_1, primals_2]
    