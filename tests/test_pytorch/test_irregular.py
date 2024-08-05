def f():
    print(len("Hello, world!"))
import dis
code = f.__code__

from types import CodeType

class Tmp:
    def __init__(self):
        self.called_count = 0
    
    def __len__(self):
        self.called_count += 1
        return 1

tmp = Tmp()

co_consts = code.co_consts
# replace the const string with a new object, to emulate the irregular behavior
# of torch.compile
new_consts = tuple(x if x is None else tmp for x in co_consts)

new_code = code.replace(co_consts=new_consts)

import torch

def convert_hook(old_code, compiled_code):
    if old_code is code:
        return new_code

handle = torch._dynamo.convert_frame.register_bytecode_hook(convert_hook)
f = torch.compile(f)

import depyf
with depyf.prepare_debug("test_irregular"):
    f()
    f()
    assert tmp.called_count == 2
