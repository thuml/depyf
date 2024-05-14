from __future__ import annotations



def forward(self, primals_1: "f32[8]", primals_2: "f32[8]", tangents_1: "f32[8]"):
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:7 in torch_dynamo_resume_in_toy_function_at_5, code: return x * b
    mul_2: "f32[8]" = torch.ops.aten.mul.Tensor(tangents_1, primals_2);  primals_2 = None
    
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:6 in torch_dynamo_resume_in_toy_function_at_5, code: b = b * -1
    mul: "f32[8]" = torch.ops.aten.mul.Tensor(primals_1, -1);  primals_1 = None
    
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:7 in torch_dynamo_resume_in_toy_function_at_5, code: return x * b
    mul_3: "f32[8]" = torch.ops.aten.mul.Tensor(tangents_1, mul);  tangents_1 = mul = None
    
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:6 in torch_dynamo_resume_in_toy_function_at_5, code: b = b * -1
    mul_4: "f32[8]" = torch.ops.aten.mul.Tensor(mul_2, -1);  mul_2 = None
    return [mul_4, mul_3]
    