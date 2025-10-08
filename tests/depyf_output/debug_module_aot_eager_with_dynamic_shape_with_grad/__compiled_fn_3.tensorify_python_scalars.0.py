from __future__ import annotations
import torch
from torch import device
class inner_f(torch.nn.Module):
    def forward(self, primals, tangents):
        primals_1: "Sym(s52)"; primals_2: "f32[s52]"; primals_3: "f32[s52]"; tangents_1: "f32[s52]"; 
    
        primals_1, primals_2, primals_3, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward, code: x = a / (torch.abs(a) + 1)
        abs_1: "f32[s52]" = torch.ops.aten.abs.default(primals_2)
        add_2: "f32[s52]" = torch.ops.aten.add.Tensor(abs_1, 1);  abs_1 = None
        div: "f32[s52]" = torch.ops.aten.div.Tensor(primals_2, add_2)
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward, code: b = b * -1 + b.sum()
        mul_3: "f32[s52]" = torch.ops.aten.mul.Tensor(primals_3, -1)
        sum_1: "f32[]" = torch.ops.aten.sum.default(primals_3);  primals_3 = None
        add_9: "f32[s52]" = torch.ops.aten.add.Tensor(mul_3, sum_1);  mul_3 = sum_1 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:15 in forward, code: return x * b
        mul_6: "f32[s52]" = torch.ops.aten.mul.Tensor(div, add_9)
        mul_8: "f32[s52]" = torch.ops.aten.mul.Tensor(tangents_1, div);  div = None
        mul_9: "f32[s52]" = torch.ops.aten.mul.Tensor(tangents_1, add_9);  tangents_1 = add_9 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward, code: b = b * -1 + b.sum()
        sum_2: "f32[]" = torch.ops.aten.sum.default(mul_8)
        expand: "f32[s52]" = torch.ops.aten.expand.default(sum_2, [primals_1]);  sum_2 = primals_1 = None
        mul_10: "f32[s52]" = torch.ops.aten.mul.Tensor(mul_8, -1);  mul_8 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward, code: b = b * -1 + b.sum()
        add_14: "f32[s52]" = torch.ops.aten.add.Tensor(expand, mul_10);  expand = mul_10 = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward, code: x = a / (torch.abs(a) + 1)
        neg: "f32[s52]" = torch.ops.aten.neg.default(mul_9)
        div_1: "f32[s52]" = torch.ops.aten.div.Tensor(primals_2, add_2)
        div_2: "f32[s52]" = torch.ops.aten.div.Tensor(div_1, add_2);  div_1 = None
        mul_11: "f32[s52]" = torch.ops.aten.mul.Tensor(neg, div_2);  neg = div_2 = None
        div_3: "f32[s52]" = torch.ops.aten.div.Tensor(mul_9, add_2);  mul_9 = add_2 = None
        sgn: "f32[s52]" = torch.ops.aten.sgn.default(primals_2);  primals_2 = None
        mul_12: "f32[s52]" = torch.ops.aten.mul.Tensor(mul_11, sgn);  mul_11 = sgn = None
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward, code: x = a / (torch.abs(a) + 1)
        add_15: "f32[s52]" = torch.ops.aten.add.Tensor(div_3, mul_12);  div_3 = mul_12 = None
        return pytree.tree_unflatten([mul_6, None, add_15, add_14], self._out_spec)
        