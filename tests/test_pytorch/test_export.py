import torch
import sys
if sys.version_info.minor == 11:
    # it seems python 3.11 with pytorch + export has some problems,
    # skipping the test since this is not the major use case.
    exit(0)
import depyf

# make sure a very long variable name will not cause any problem
very_long_variable = "a" * 1000
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=8, nhead=2, batch_first=True),
            num_layers=6,
        )
        setattr(self, very_long_variable, encoder)

    def forward(self, x):
        encoder = getattr(self, very_long_variable)
        return encoder(x)

model = MyModel()
x = torch.randn(1, 10, 8)
with depyf.prepare_debug('export_output'):
    model_opt = torch.compile(model,fullgraph=True)
    model_opt(x)
    exported = torch.export.export(model,(x,))
    exported_model=exported.module()
