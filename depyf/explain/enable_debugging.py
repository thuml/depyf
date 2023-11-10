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
            compiled_code = compile(src, filename=filename, mode="exec")
            scope = {}
            exec(compiled_code, scope)
            func = scope[func_name]
            return func.__code__
        except Exception:
            pass


from .lazy_format_graph_code import lazy_format_graph_code
from .patch_run import boxed_run

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

    dump_src_dir = os.path.abspath(dump_src_dir)

    import torch
    compiled_code_handle = torch._dynamo.convert_frame.register_bytecode_hook(DebuggableHook(dump_src_dir, "compiled_code"))

    old_func = torch.fx.graph_module._exec_with_source
    def _exec_with_source(src: str, globals, co_fields=None):
        old_func(src, globals, co_fields)
        import inspect
        key = inspect.getsourcefile(globals["forward"])
        import hashlib
        import os
        hash_value = hashlib.md5(src.encode()).hexdigest()
        src = "# " + key + src
        count = 0
        while True:
            filename = f"{dump_src_dir}/fx_graph_code_" + hash_value + "_" + str(count) + ".py"
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

    torch.fx.graph_module._exec_with_source = _exec_with_source
    # we have to directly manipulate the code object, since the function has been imported in many places.
    # simply replacing torch._dynamo.utils.lazy_format_graph_code does not work for those functions.
    old_code = torch._dynamo.utils.lazy_format_graph_code.__code__
    torch._dynamo.utils.lazy_format_graph_code.__code__ = lazy_format_graph_code.__code__

    from torch._inductor.codecache import PyCodeCache
    old_inductor_fn = PyCodeCache.load_by_key_path

    def new_inductor_fn(
            key: str,
            path: str,
            linemap,
            attrs,
        ):
        # hack the path to our dump_src_dir
        src = open(path).read()
        os.remove(path)

        from torch._dynamo.bytecode_transformation import _unique_id_counter
        from copy import copy
        # torch.compile already called the next, we should add minus 1 to get the correct name
        current_count = next(copy(_unique_id_counter)) - 1
        func_name = "__compiled_fn_" + str(current_count)
        count = 0
        while True:
            new_filepath = os.path.join(dump_src_dir, func_name + " kernel " + str(count) + ".py")
            if not os.path.exists(new_filepath):
                with open(new_filepath, "w") as f:
                    f.write(src)
                break
            # might be a hash collision
            existing_code = open(new_filepath).read()
            if existing_code == src:
                break
            count += 1

        path = new_filepath
        return old_inductor_fn(key, path, linemap, attrs)

    PyCodeCache.load_by_key_path = new_inductor_fn
    try:
        yield
    finally:
        compiled_code_handle.remove()
        from depyf.explain import dump_src, _extract_artifacts, _collect_compiled_subgraphs
        full_src = dump_src(func)
        filename = os.path.join(dump_src_dir, f"full_code.py")
        with open(filename, "w") as f:
            f.write(full_src)
        torch.fx.graph_module._exec_with_source = old_func
        torch._dynamo.utils.lazy_format_graph_code.__code__ = old_code
        PyCodeCache.load_by_key_path = old_inductor_fn
        if pause:
            input(f"Please check the full source code in {filename}, and set breakpoints for functions in {dump_src_dir} according to the hash value. Then press enter to continue.")

@contextlib.contextmanager
def debug():
    import torch
    callback = torch._dynamo.eval_frame.set_eval_frame(False)
    # sometimes pytorch use Interpreter to run node by node. This cannot be debugged.
    # we patch this function to run the graph function directly.
    old_run_code = torch.fx.Interpreter.boxed_run.__code__
    torch.fx.Interpreter.boxed_run.__code__ = boxed_run.__code__
    try:
        yield
    finally:
        torch._dynamo.eval_frame.set_eval_frame(callback)
        torch.fx.Interpreter.boxed_run.__code__ = old_run_code
