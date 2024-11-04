from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[10]"; primals_2: "f32[10]"; tangents_1: "f32[10]"; 

    primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
     # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function, code: x = a / (torch.abs(a) + 1)
    abs_1: "f32[10]" = torch.ops.aten.abs.default(primals_1)
    add: "f32[10]" = torch.ops.aten.add.Tensor(abs_1, 1);  abs_1 = None
    div: "f32[10]" = torch.ops.aten.div.Tensor(primals_1, add)
    
     # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:5 in toy_function, code: if b.sum() < 0:
    sum_1: "f32[]" = torch.ops.aten.sum.default(primals_2);  primals_2 = None
    lt: "b8[]" = torch.ops.aten.lt.Scalar(sum_1, 0);  sum_1 = None
    
     # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function, code: x = a / (torch.abs(a) + 1)
    neg: "f32[10]" = torch.ops.aten.neg.default(tangents_1)
    div_1: "f32[10]" = torch.ops.aten.div.Tensor(primals_1, add)
    div_2: "f32[10]" = torch.ops.aten.div.Tensor(div_1, add);  div_1 = None
    mul: "f32[10]" = torch.ops.aten.mul.Tensor(neg, div_2);  neg = div_2 = None
    div_3: "f32[10]" = torch.ops.aten.div.Tensor(tangents_1, add);  tangents_1 = add = None
    sign: "f32[10]" = torch.ops.aten.sign.default(primals_1);  primals_1 = None
    mul_1: "f32[10]" = torch.ops.aten.mul.Tensor(mul, sign);  mul = sign = None
    
     # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function, code: x = a / (torch.abs(a) + 1)
    add_1: "f32[10]" = torch.ops.aten.add.Tensor(div_3, mul_1);  div_3 = mul_1 = None
    return pytree.tree_unflatten([div, lt, add_1, None], self._out_spec)
    