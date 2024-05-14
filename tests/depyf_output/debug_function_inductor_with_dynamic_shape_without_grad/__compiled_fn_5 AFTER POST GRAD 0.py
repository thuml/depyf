from __future__ import annotations



def forward(self, arg0_1: "f32[10]", arg1_1: "f32[10]"):
    # File: /Users/youkaichao/data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:7 in torch_dynamo_resume_in_toy_function_at_5, code: return x * b
    mul: "f32[10]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    return (mul,)
    