import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        def forward_function(x):
            return self.linear(x)
        return forward_function(x)

model = LinearModel()
model = torch.compile(model, backend='eager')

x = torch.tensor([[1.0]])

import depyf
with depyf.prepare_debug("./cellvars"):
    model(x)
