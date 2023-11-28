from __future__ import annotations



def forward(self, L_b_ : torch.Tensor, L_x_ : torch.Tensor):
    l_b_ = L_b_
    l_x_ = L_x_
    
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:6, code: b = b * -1
    b = l_b_ * -1;  l_b_ = None
    
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:7, code: return x * b
    mul_1 = l_x_ * b;  l_x_ = b = None
    return (mul_1,)
    