from __future__ import annotations
import torch
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s52)", primals_2: "f32[s52]", primals_3: "f32[s52]", tangents_1: "f32[s52]"):
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:7 in torch_dynamo_resume_in_toy_function_at_5, code: return x * b
        mul_4: "f32[s52]" = torch.ops.aten.mul.Tensor(tangents_1, primals_3);  primals_3 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:6 in torch_dynamo_resume_in_toy_function_at_5, code: b = b * -1
        mul: "f32[s52]" = torch.ops.aten.mul.Tensor(primals_2, -1);  primals_2 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:7 in torch_dynamo_resume_in_toy_function_at_5, code: return x * b
        mul_5: "f32[s52]" = torch.ops.aten.mul.Tensor(tangents_1, mul);  tangents_1 = mul = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:6 in torch_dynamo_resume_in_toy_function_at_5, code: b = b * -1
        mul_6: "f32[s52]" = torch.ops.aten.mul.Tensor(mul_4, -1);  mul_4 = None
        return (None, mul_6, mul_5)
        