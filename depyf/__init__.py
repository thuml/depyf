from typing import List, Tuple, Dict, Union, Callable, Optional
from types import CodeType

from .decompiler import Decompiler

from .code_transform import structure_hash

def decompile(code: Union[CodeType, Callable]):
    """Decompile a code object or a function."""
    return Decompiler(code).decompile()


import types


def pytorch_bytecode_src_hook(code: types.CodeType, new_code: types.CodeType):
    import torch
    bytecode_log = torch._logging.getArtifactLogger(
        "torch._dynamo.convert_frame", "bytecode"
    )
    import logging

    if bytecode_log.isEnabledFor(logging.DEBUG):
        try:
            decompiled_src = decompile(new_code)
            bytecode_log.debug("possible source code:")
            bytecode_log.debug(decompiled_src)
        except Exception as e:
            bytecode_log.debug("Decompilation fails due to: %s", str(e))
        finally:
            bytecode_log.debug(
                "If you find the decompiled code is wrong,"
                "please submit an issue at "
                "https://github.com/thuml/depyf/issues."
            )

_handle = None

def install():
    import torch
    global _handle
    if _handle is not None:
        return
    _handle = torch._dynamo.convert_frame.register_bytecode_hook(pytorch_bytecode_src_hook)


def uninstall():
    global _handle
    if _handle is None:
        return
    _handle.remove()
    _handle = None

import os

__version__ = open(f"{os.path.dirname(__file__)}/VERSION.txt").read().strip()

import contextlib
import warnings

import dataclasses

@dataclasses.dataclass
class DebuggableHook(object):
    dump_src_dir: str
    type_name: str

    def __call__(self, code, new_code):
        try:
            src = Decompiler(new_code).decompile(overwite_fn_name=self.type_name)
            full_hash = structure_hash(src)
            filename = os.path.join(self.dump_src_dir, f"{self.type_name}_{full_hash}.py")
            if not os.path.exists(filename):
                with open(filename, "w") as f:
                    f.write(src)
            compiled_code = compile(src, filename=filename, mode="exec")
            scope = {}
            exec(compiled_code, scope)
            func = scope[self.type_name]
            return func.__code__
        except Exception:
            pass

@contextlib.contextmanager
def prepare_debug(func, dump_src_dir):
    import os

    warnings.warn((
        "You are trying to debug `torch.compile`. Please make sure the code "
        "runs multiple times to cover all the possible branches."
    ))

    if not os.path.exists(dump_src_dir):
        os.makedirs(dump_src_dir)

    import torch
    handle = torch._dynamo.convert_frame.register_bytecode_hook(DebuggableHook(dump_src_dir, "compiled_code"))

    try:
        yield
    finally:
        handle.remove()
        from depyf.explain import dump_src
        full_src = dump_src(func)
        filename = os.path.join(dump_src_dir, f"full_code.py")
        with open(filename, "w") as f:
            f.write(full_src)
        input(f"Please check the full source code in {filename}, and set breakpoints for functions in {dump_src_dir} according to the hash value. Then press enter to continue.")

@contextlib.contextmanager
def debug():
    import torch
    callback = torch._dynamo.eval_frame.set_eval_frame(False)
    try:
        yield
    finally:
        torch._dynamo.eval_frame.set_eval_frame(callback)
