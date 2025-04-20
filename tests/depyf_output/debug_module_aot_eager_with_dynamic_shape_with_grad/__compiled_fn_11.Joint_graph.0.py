from __future__ import annotations
import torch
from torch import device
class joint_helper(torch.nn.Module):
    def forward(self, primals, tangents):
        primals_1: "Sym(s52)"; primals_2: "f32[s52]"; primals_3: "f32[s52]"; tangents_1: "f32[s52]"; 
    
        primals_1, primals_2, primals_3, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:16 in torch_dynamo_resume_in_forward_at_15, code: b = b * -1
        mul: "f32[s52]" = torch.ops.aten.mul.Tensor(primals_2, -1);  primals_2 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:17 in torch_dynamo_resume_in_forward_at_15, code: return x * b
        mul_2: "f32[s52]" = torch.ops.aten.mul.Tensor(primals_3, mul)
        mul_4: "f32[s52]" = torch.ops.aten.mul.Tensor(tangents_1, primals_3);  primals_3 = None
        mul_5: "f32[s52]" = torch.ops.aten.mul.Tensor(tangents_1, mul);  tangents_1 = mul = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:16 in torch_dynamo_resume_in_forward_at_15, code: b = b * -1
        mul_6: "f32[s52]" = torch.ops.aten.mul.Tensor(mul_4, -1);  mul_4 = None
        return pytree.tree_unflatten([mul_2, None, mul_6, mul_5], self._out_spec)
        