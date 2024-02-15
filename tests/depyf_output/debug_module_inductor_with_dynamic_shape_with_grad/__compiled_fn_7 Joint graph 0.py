from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[8]"; primals_2: "f32[8]"; tangents_1: "f32[8]"; 

    primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:16 in torch_dynamo_resume_in_forward_at_15, code: b = b * -1
    mul: "f32[8]" = torch.ops.aten.mul.Tensor(primals_1, -1);  primals_1 = None
    
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:17 in torch_dynamo_resume_in_forward_at_15, code: return x * b
    mul_1: "f32[8]" = torch.ops.aten.mul.Tensor(primals_2, mul)
    mul_2: "f32[8]" = torch.ops.aten.mul.Tensor(tangents_1, primals_2);  primals_2 = None
    mul_3: "f32[8]" = torch.ops.aten.mul.Tensor(tangents_1, mul);  tangents_1 = mul = None
    
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:16 in torch_dynamo_resume_in_forward_at_15, code: b = b * -1
    mul_4: "f32[8]" = torch.ops.aten.mul.Tensor(mul_2, -1);  mul_2 = None
    return pytree.tree_unflatten([mul_1, mul_4, mul_3], self._out_spec)
    