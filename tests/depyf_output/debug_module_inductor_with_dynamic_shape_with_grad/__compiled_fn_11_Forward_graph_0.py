from __future__ import annotations



def forward(self, primals_1: "Sym(s0)", primals_2: "f32[s0]", primals_3: "f32[s0]"):
     # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:16 in torch_dynamo_resume_in_forward_at_15, code: b = b * -1
    mul: "f32[s0]" = torch.ops.aten.mul.Tensor(primals_2, -1)
    
     # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:17 in torch_dynamo_resume_in_forward_at_15, code: return x * b
    mul_2: "f32[s0]" = torch.ops.aten.mul.Tensor(primals_3, mul);  mul = None
    return (mul_2, primals_2, primals_3, primals_1)
    