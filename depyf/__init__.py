from typing import List, Tuple, Dict, Union, Callable, Optional, Any
from types import CodeType

import functools

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
import itertools

@dataclasses.dataclass
class DebuggableHook(object):
    dump_src_dir: str
    type_name: str
    code_counter: Any = dataclasses.field(default_factory=lambda: itertools.count(start=0))

    def __call__(self, code, new_code):
        try:
            n = next(self.code_counter)
            filename = os.path.join(self.dump_src_dir, f"{self.type_name}_{n}.py")
            func_name = f"{self.type_name}_{n}"
            src = Decompiler(new_code).decompile(overwite_fn_name=func_name)
            with open(filename, "w") as f:
                f.write(src)
            compiled_code = compile(src, filename=filename, mode="exec")
            scope = {}
            exec(compiled_code, scope)
            func = scope[func_name]
            return func.__code__
        except Exception:
            pass

def extract_underlying_func(fn: Callable):
    """Functions like bounded method also have `__code__` attribute, but it is read-only. We want to get the underlying function so that we can change its code object.
    """
    if hasattr(fn, "__func__"):
        # deal with bounded function
        fn = fn.__func__
    elif hasattr(fn, "__wrapped__"):
        # deal with lru_cache or other decorators
        fn = fn.__wrapped__
    elif isinstance(fn, functools.partial):
        # deal with partial function
        fn = fn.func
    elif hasattr(fn, "__call__") and hasattr(fn.__call__, "__func__"):
        # deal with callable object
        fn = fn.__call__.__func__
    else:
        return fn
    return extract_underlying_func(fn)

@dataclasses.dataclass
class CompiledSubgraphHook(object):
    dump_src_dir: str
    type_name: str

    def __call__(self, name, fn):
        try:
            import torch
            import types
            fn = torch._dynamo.eval_frame.innermost_fn(fn)
            fn = extract_underlying_func(fn)
            src = Decompiler(fn).decompile(overwite_fn_name=self.type_name)
            full_hash = structure_hash(src)
            filename = os.path.join(self.dump_src_dir, f"{name}.py")
            with open(filename, "w") as f:
                f.write(src)
            compiled_code = compile(src, filename=filename, mode="exec")
            scope = {}
            exec(compiled_code, scope)
            func = scope[self.type_name]
            fn.__code__ = func.__code__

        except Exception as e:
            print(str(e))


code = """
def _exec_with_source(src: str, globals, co_fields=None):
    key = _loader.cache(src, globals, co_fields)
    import hashlib
    import os
    hash_value = hashlib.md5(src.encode()).hexdigest()
    src = "#" + key + src
    count = 0
    while True:
        filename = "{filedir}/compiled_code_" + hash_value + "_" + str(count) + ".py"
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                f.write(src)
            break
        # might be a hash collision
        existing_code = open(filename).read()
        if existing_code == src:
            break
        count += 1
    exec(compile(src, filename, "exec"), globals)
"""

@contextlib.contextmanager
def prepare_debug(func, dump_src_dir, pause=True):
    """
    pause: whether to wait for users to set breakpoints. set it to False in testing.
    """
    import os

    warnings.warn((
        "You are trying to debug `torch.compile`. Please make sure the code "
        "runs multiple times to cover all the possible branches."
    ))

    if not os.path.exists(dump_src_dir):
        os.makedirs(dump_src_dir)

    import torch
    compiled_code_handle = torch._dynamo.convert_frame.register_bytecode_hook(DebuggableHook(dump_src_dir, "compiled_code"))

    scope = {}
    exec(compile(code.format(filedir=dump_src_dir), "noname", "exec"), scope)
    old_code = torch.fx.graph_module._exec_with_source.__code__
    torch.fx.graph_module._exec_with_source.__code__ = scope["_exec_with_source"].__code__

    try:
        yield
    finally:
        compiled_code_handle.remove()
        from depyf.explain import dump_src, _extract_artifacts, _collect_compiled_subgraphs
        full_src = dump_src(func)
        filename = os.path.join(dump_src_dir, f"full_code.py")
        with open(filename, "w") as f:
            f.write(full_src)
        torch.fx.graph_module._exec_with_source.__code__ = old_code
        if pause:
            input(f"Please check the full source code in {filename}, and set breakpoints for functions in {dump_src_dir} according to the hash value. Then press enter to continue.")

@contextlib.contextmanager
def debug():
    import torch
    callback = torch._dynamo.eval_frame.set_eval_frame(False)
    try:
        yield
    finally:
        torch._dynamo.eval_frame.set_eval_frame(callback)
