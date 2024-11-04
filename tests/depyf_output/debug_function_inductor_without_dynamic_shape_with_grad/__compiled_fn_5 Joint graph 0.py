from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[10]"; primals_2: "f32[10]"; tangents_1: "f32[10]"; 

    primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
     # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:7 in torch_dynamo_resume_in_toy_function_at_5, code: return x * b
    mul: "f32[10]" = torch.ops.aten.mul.Tensor(primals_1, primals_2)
    mul_1: "f32[10]" = torch.ops.aten.mul.Tensor(tangents_1, primals_1);  primals_1 = None
    mul_2: "f32[10]" = torch.ops.aten.mul.Tensor(tangents_1, primals_2);  tangents_1 = primals_2 = None
    return pytree.tree_unflatten([mul, mul_2, mul_1], self._out_spec)
    