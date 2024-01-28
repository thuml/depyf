from __future__ import annotations



def forward(self, L_x_ : torch.Tensor, L_b_ : torch.Tensor):
    l_x_ = L_x_
    l_b_ = L_b_
    
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:17 in torch_dynamo_resume_in_forward_at_15, code: return x * b
    mul = l_x_ * l_b_;  l_x_ = l_b_ = None
    return (mul,)
    