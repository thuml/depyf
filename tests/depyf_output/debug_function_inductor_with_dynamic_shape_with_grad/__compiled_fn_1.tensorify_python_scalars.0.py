from __future__ import annotations
import torch
from torch import device
class inner_f(torch.nn.Module):
    def forward(self, primals, tangents):
        primals_1: "f32[10]"; primals_2: "f32[10]"; tangents_1: "f32[10]"; 
    
        primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function, code: x = a / (torch.abs(a) + 1)
        abs_1: "f32[10]" = torch.ops.aten.abs.default(primals_1)
        add: "f32[10]" = torch.ops.aten.add.Tensor(abs_1, 1);  abs_1 = None
        div: "f32[10]" = torch.ops.aten.div.Tensor(primals_1, add)
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:5 in toy_function, code: b = b * -1 + b.sum()
        mul: "f32[10]" = torch.ops.aten.mul.Tensor(primals_2, -1)
        sum_1: "f32[]" = torch.ops.aten.sum.default(primals_2);  primals_2 = None
        add_1: "f32[10]" = torch.ops.aten.add.Tensor(mul, sum_1);  mul = sum_1 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:6 in toy_function, code: return x * b
        mul_1: "f32[10]" = torch.ops.aten.mul.Tensor(div, add_1)
        mul_2: "f32[10]" = torch.ops.aten.mul.Tensor(tangents_1, div);  div = None
        mul_3: "f32[10]" = torch.ops.aten.mul.Tensor(tangents_1, add_1);  tangents_1 = add_1 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:5 in toy_function, code: b = b * -1 + b.sum()
        sum_2: "f32[]" = torch.ops.aten.sum.default(mul_2)
        expand: "f32[10]" = torch.ops.aten.expand.default(sum_2, [10]);  sum_2 = None
        mul_4: "f32[10]" = torch.ops.aten.mul.Tensor(mul_2, -1);  mul_2 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:5 in toy_function, code: b = b * -1 + b.sum()
        add_2: "f32[10]" = torch.ops.aten.add.Tensor(expand, mul_4);  expand = mul_4 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function, code: x = a / (torch.abs(a) + 1)
        neg: "f32[10]" = torch.ops.aten.neg.default(mul_3)
        div_1: "f32[10]" = torch.ops.aten.div.Tensor(primals_1, add)
        div_2: "f32[10]" = torch.ops.aten.div.Tensor(div_1, add);  div_1 = None
        mul_5: "f32[10]" = torch.ops.aten.mul.Tensor(neg, div_2);  neg = div_2 = None
        div_3: "f32[10]" = torch.ops.aten.div.Tensor(mul_3, add);  mul_3 = add = None
        sign: "f32[10]" = torch.ops.aten.sign.default(primals_1);  primals_1 = None
        mul_6: "f32[10]" = torch.ops.aten.mul.Tensor(mul_5, sign);  mul_5 = sign = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function, code: x = a / (torch.abs(a) + 1)
        add_3: "f32[10]" = torch.ops.aten.add.Tensor(div_3, mul_6);  div_3 = mul_6 = None
        return pytree.tree_unflatten([mul_1, add_3, add_2], self._out_spec)
        