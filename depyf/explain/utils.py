import torch
from torch._dynamo.eval_frame import innermost_fn
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
import inspect
from importlib import import_module

import dis
from types import CodeType
from typing import List, Callable, Dict, Union, Set
from dataclasses import dataclass
import contextlib

import depyf
from depyf.decompiler import DecompilationError


def decompile_ensure(fn, overwite_fn_name=None):
    try:
        decompiled_source_code = depyf.Decompiler(
            fn).decompile(overwite_fn_name=overwite_fn_name)
    except DecompilationError as e:
        print(str(e))
        decompiled_source_code = "'Failed to decompile.'\n"
    return decompiled_source_code


class CodeProxy:
    instances: Dict[str, "CodeProxy"] = {}
    used_instances: Set[str] = set()

    @staticmethod
    def get_new_name(name: str):
        i = 0
        new_name = name
        if new_name.endswith(":"):
            name = name[:-1]
            while True:
                new_name = f"{name}_{i}"
                if new_name not in CodeProxy.instances:
                    break
                i += 1
        return new_name

    @staticmethod
    def consume_new_name(name: str):
        new_name = CodeProxy.get_new_name(name)
        CodeProxy.instances[new_name] = None
        return new_name

    @staticmethod
    def decompile_with_name(code: CodeType, name: str):
        if hasattr(code, "__code__"):
            code = code.__code__
        if code.co_name.startswith("transformed_code_") or code.co_name.startswith("__transformed_code_"):
            src = open(code.co_filename).read()
            new_name = code.co_name
        else:
            new_name = CodeProxy.get_new_name(name)
            src = decompile_ensure(code, new_name)
        self = CodeProxy(src)
        self.name = new_name
        self.code = f"""<details>
  <summary>{self.name}</summary>

  ```python
{self.raw_code}
  ```
</details>
"""
        CodeProxy.instances[self.name] = self
        return self

    def __init__(self, code: str):
        # Don't directly use this constructor. Use decompile_with_name instead.
        self.raw_code = "".join(
            ["  " + line + "\n" for line in code.splitlines() if line.strip() != ""])

    def __str__(self):
        CodeProxy.used_instances.add(self.name)
        return self.name

    @contextlib.contextmanager
    @staticmethod
    def record():
        CodeProxy.used_instances = set()
        yield CodeProxy.used_instances


def display_func(self):
    from IPython.display import display, JSON, Markdown, Code
    display(Markdown("# transformed source code:"))
    with CodeProxy.record() as instances:
        data = self.to_data()
    display(JSON(data))
    display(Markdown("# source code of referenced function:"))
    markdown = ""
    for instance in instances:
        markdown += CodeProxy.instances[instance].code
    display(Markdown(markdown))


@dataclass
class CacheResult:
    code: CodeType
    guard: List[str]
    compiled_subgraph: Callable
    compiled_subgraph_proxy: CodeProxy
    transformed_code: CodeType
    transformed_code_proxy: CodeProxy
    referenced_global_functions: Dict[str, "DynamoOptimizationResult"]

    def __init__(self, fn, cache):
        guard = cache.check_fn.code_parts
        code = cache.code
        compiled_subgraphs = [
            name for name in code.co_names if name.startswith("__compiled")]
        assert len(compiled_subgraphs) == 1
        module = import_module(fn.__module__)
        # deal with compiled_subgraph
        self.compiled_subgraph = innermost_fn(
            getattr(module, compiled_subgraphs[0]))
        self.compiled_subgraph_proxy = CodeProxy.decompile_with_name(
            self.compiled_subgraph, compiled_subgraphs[0])
        # deal with transformed_code
        self.transformed_code = code
        self.transformed_code_proxy = CodeProxy.decompile_with_name(
            self.transformed_code, "transformed_code:")
        resume_fns = [
            name for name in code.co_names if name.startswith("__resume")]
        self.referenced_global_functions = {
            name: DynamoOptimizationResult(
                getattr(
                    module,
                    name),
                name) for name in resume_fns}
        self.code = code
        self.guard = guard

    def to_data(self):
        return {
            "guard": self.guard,
            "transformed_code": str(
                self.transformed_code_proxy),
            "compiled_subgraph": str(
                self.compiled_subgraph_proxy),
            "referenced_global_functions": {
                name: fn.to_data() for name,
                fn in self.referenced_global_functions.items()}}

    _ipython_display_ = display_func


@dataclass
class DynamoOptimizationResult:
    name: str
    fn: Callable
    code: CodeType
    source_code_proxy: CodeProxy
    transformed_code_entries: List[CacheResult]

    def __init__(self, fn, name=None):
        self.fn = fn
        if name is None:
            self.name = fn.__name__
        else:
            self.name = name
        caches = _debug_get_cache_entry_list(fn.__code__)
        self.transformed_code_entries = [
            CacheResult(fn, cache) for cache in caches]
        self.code = fn.__code__
        self.source_code_proxy = CodeProxy.decompile_with_name(
            self.code, self.name)

    def to_data(self):
        data = {
            "name": self.name,
            "source_code": str(
                self.source_code_proxy),
            "transformed_code_entries": [
                entry.to_data() for entry in self.transformed_code_entries]}
        return data

    def to_src(self):
        raw_code = self.source_code_proxy.raw_code

        # prepare function signature, from `def toy_example(a, b)` to `def
        # transformed_toy_example(a, b)`
        signature = raw_code.splitlines()[0].replace(
            "def ", "def transformed_", 1)
        code = signature.strip()

        # prepare args for guards, like `L = {"a": a, "b": b}`
        code_obj = self.fn.__code__
        normal_arg_count = code_obj.co_argcount + code_obj.co_kwonlyargcount
        arg_names = code_obj.co_varnames[:normal_arg_count]
        arg_dict = "L = {" + \
            ", ".join([f'"{name}": {name}' for name in arg_names]) + "}"
        code += "\n" + " " * 4 + arg_dict

        additional_code = ""

        for entry in self.transformed_code_entries:

            # prepare guards, like `def guard_0(L):\n    return a > 0 and b >
            # 0`
            guard = (" \\\n" + " " * 8 +
                     "and ").join(["(" + x + ")" for x in entry.guard])
            if entry.transformed_code_proxy.name.startswith("__transformed_code_"):
                guard_func_name = entry.transformed_code_proxy.name.replace("__transformed_code_", "__guard_")
            else:
                guard_func_name = CodeProxy.consume_new_name("guard:")
            additional_code += f"\ndef {guard_func_name}(L):\n" + \
                " " * 4 + "return " + guard + "\n"

            # prepare compiled subgraph, like `__compiled_fn_0`
            subgraph_name = entry.compiled_subgraph_proxy.name
            additional_code += "\n"
            additional_code += f"# Note: please refer to the graph code in {subgraph_name}*.py.\n"
            additional_code += f"# Captured Graph: Dynamo generated graph (debuggable when using eager backend).\n"
            additional_code += f"# Joint graph: joint forward+backward graph from aot autograd.\n"
            additional_code += f"# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).\n"
            additional_code += f"# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).\n"
            additional_code += f"# AFTER XXX: graph processed by inductor (not debuggable).\n"
            additional_code += f"def {subgraph_name}(*args, **kwargs):\n    pass\n"

            # prepare transformed code, like `transformed_code_0`
            additional_code += "\n" + \
                remove_indentation(entry.transformed_code_proxy.raw_code) + "\n"

            for name, func in entry.referenced_global_functions.items():
                additional_code = func.to_src() + additional_code

            code += "\n" + " " * 4 + \
                f"if {guard_func_name}(L):\n" + " " * 8 + f"return {entry.transformed_code_proxy.name}({', '.join(arg_names)})"

        additional_code += "\n" + "# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.\n" + \
            remove_indentation(self.source_code_proxy.raw_code) + "\n"

        code += "\n" + " " * 4 + "# Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.\n" + \
            " " * 4 + f"return {self.source_code_proxy.name}({', '.join(arg_names)})"
        return additional_code + code + \
            f"\n\n#============ end of {self.name} ============#\n"

    _ipython_display_ = display_func


def remove_indentation(code: str):
    lines = code.splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "".join([line[indent:] + "\n" for line in lines])

from contextlib import contextmanager

@contextmanager
def lock_on_file(lock_path):
    from filelock import FileLock
    import os
    lock = FileLock(lock_path)
    try:
        with lock:
            yield
    finally:
        pass


def write_code_to_file_template(src, path_template):
    lock_path = path_template % "lock"

    with lock_on_file(lock_path):
        import os
        count = 0
        while True:
            new_filepath = path_template % str(count)
            if not os.path.exists(new_filepath):
                with open(new_filepath, "w") as f:
                    f.write(src)
                break
            # might be a hash collision
            existing_code = open(new_filepath).read()
            if existing_code == src:
                break
            count += 1
        return new_filepath


def get_code_owner(fn):
    """A callable object `fn` might have a __code__ attribute, which is a code object.
    However, `fn` might not be the owner of the code object. Only the code owner can change the code object.
    This function returns the owner of the code object.
    An example:
    class A:
        def func(self):
            return 1
    a = A()
    `a.func.__code__` is read-only. `A.func.__code__` is writable.
    We can change the code object via `a.func.__func__.__code__`.
    """
    import functools
    while True:
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
            break
    return fn


def get_current_compiled_fn_name():
    import torch
    from torch._dynamo.bytecode_transformation import _unique_id_counter
    from copy import copy
    # torch.compile already called the next, we should add minus 1 to get the
    # correct name
    current_count = next(copy(_unique_id_counter)) - 1
    return "__compiled_fn_" + str(current_count)
