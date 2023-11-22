from typing import List, Tuple, Dict, Union, Callable, Optional, Any

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
            transformed_code = compile(src, filename=filename, mode="exec")
            scope = {}
            exec(transformed_code, scope)
            func = scope[func_name]
            return func.__code__
        except Exception:
            pass

from .patched_boxed_run import patched_boxed_run
from .patched__exec_with_source import patched__exec_with_source
from .patched_load_by_key_path import patched_load_by_key_path
from .patched_lazy_format_graph_code import patched_lazy_format_graph_code


@contextlib.contextmanager
def patch(parent, name, value):
    old_value = getattr(parent, name)
    setattr(parent, name, value)
    try:
        yield
    finally:
        setattr(parent, name, old_value)

@contextlib.contextmanager
def enable_bytecode_hook(hook):
    import torch
    handle = torch._dynamo.convert_frame.register_bytecode_hook(hook)
    try:
        yield
    finally:
        handle.remove()

@contextlib.contextmanager
def prepare_debug(func, dump_src_dir, clean_wild_fx_code=True):
    """
    clean_wild_fx_code: whether to clean the wild fx code that are not recognized for parts of compiled functions. They are usually used by PyTorch internally.
    """
    import os
    import torch

    warnings.warn((
        "You are trying to debug `torch.compile`. Please make sure the code "
        "runs multiple times to cover all the possible branches."
    ))

    if not os.path.exists(dump_src_dir):
        os.makedirs(dump_src_dir)

    dump_src_dir = os.path.abspath(dump_src_dir)

    from .global_variables import data

    data["dump_src_dir"] = dump_src_dir
    data["unpatched__exec_with_source"] = torch.fx.graph_module._exec_with_source
    data["unpatched_load_by_key_path"] = torch._inductor.codecache.PyCodeCache.load_by_key_path

    bytecode_hook = DebuggableHook(dump_src_dir, "transformed_code")

    # patch some functions
    with patch(torch.fx.graph_module, "_exec_with_source", patched__exec_with_source), \
        patch(torch._inductor.codecache.PyCodeCache, "load_by_key_path", patched_load_by_key_path), \
        patch(torch._dynamo.utils.lazy_format_graph_code, "__code__", patched_lazy_format_graph_code.__code__):
        # we have to directly manipulate the code object, since the function has been imported in many places.
        # simply replacing torch._dynamo.utils.lazy_format_graph_code does not work for those functions.
        # Note: `unitest.mock.patch` does not work here, since it will not patch the code object. (it will try to delete the code object and then set a new code object. The `delattr` will raise an error.)

        # enable bytecode hook
        with enable_bytecode_hook(bytecode_hook):
            try:
                yield
            finally:
                from depyf.explain import dump_src, _extract_artifacts, _collect_compiled_subgraphs
                full_src = dump_src(func)
                filename = os.path.join(dump_src_dir, f"full_code.py")
                with open(filename, "w") as f:
                    f.write(full_src)
                if clean_wild_fx_code:
                    for file in os.listdir(dump_src_dir):
                        if file.split(os.path.sep)[-1].startswith("fx_graph_code"):
                            os.remove(os.path.join(dump_src_dir, file))
                input(f"Please check the full source code in {filename}, and set breakpoints for functions in {dump_src_dir} according to the hash value. Then press enter to continue.")

@contextlib.contextmanager
def debug():
    import torch
    callback = torch._dynamo.eval_frame.set_eval_frame(False)
    # sometimes pytorch use Interpreter to run node by node. This cannot be debugged.
    # we patch this function to run the graph function directly.
    with patch(torch.fx.Interpreter.boxed_run, "__code__", patched_boxed_run.__code__):
        try:
            yield
        finally:
            torch._dynamo.eval_frame.set_eval_frame(callback)
