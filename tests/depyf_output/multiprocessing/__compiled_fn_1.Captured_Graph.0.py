from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]"):
        l_x_ = L_x_
        
         # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_mp.py:5 in f, code: return x + 1
        add: "f32[5]" = l_x_ + 1;  l_x_ = None
        return (add,)
        