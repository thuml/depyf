from __future__ import annotations



def forward(self, s0 : torch.SymInt, L_a_ : torch.Tensor, s1 : torch.SymInt, L_b_ : torch.Tensor):
    l_a_ = L_a_
    l_b_ = L_b_
    
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function, code: x = a / (torch.abs(a) + 1)
    abs_1 = torch.abs(l_a_)
    add = abs_1 + 1;  abs_1 = None
    x = l_a_ / add;  l_a_ = add = None
    
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:5 in toy_function, code: if b.sum() < 0:
    sum_1 = l_b_.sum();  l_b_ = None
    lt = sum_1 < 0;  sum_1 = None
    return (x, lt)
    