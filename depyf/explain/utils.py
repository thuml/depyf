import torch
from torch._dynamo.eval_frame import innermost_fn
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
import inspect

import dis
from types import CodeType
from typing import List, Callable, Dict, Union, Set
from dataclasses import dataclass
import contextlib

import depyf
from depyf.decompiler import DecompilationError
from depyf.utils import get_function_signature


def decompile_ensure(fn, overwite_fn_name=None):
    try:
        decompiled_source_code = depyf.Decompiler(
            fn).decompile(overwite_fn_name=overwite_fn_name)
    except DecompilationError as e:
        header = get_function_signature(fn, overwite_fn_name=overwite_fn_name)
        decompiled_source_code = header + "    'Failed to decompile.'\n"
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
    def decompile_with_name(code: CodeType, name: str, skip_decompile=False):
        if hasattr(code, "__code__"):
            code = code.__code__
        if code.co_name.startswith("transformed_code_") or code.co_name.startswith("__transformed_code_"):
            src = open(code.co_filename).read()
            new_name = code.co_name
        else:
            new_name = CodeProxy.get_new_name(name)
            if not skip_decompile:
                src = decompile_ensure(code, new_name)
            else:
                src = ""
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


@dataclass
class CacheResult:
    original_code: CodeType
    transformed_code: CodeType
    guard: List[str]
    compiled_subgraph: Callable
    compiled_subgraph_proxy: CodeProxy
    transformed_code_proxy: CodeProxy
    referenced_global_functions: Dict[str, "DynamoOptimizationResult"]

    def __init__(self, original_code, module, cache):
        self.original_code = original_code

        cpp_guard = False

        try:
            from torch._dynamo.guards import GuardManager
            cpp_guard = isinstance(cache.check_fn, GuardManager)
        except Exception:
            pass

        if not cpp_guard:
            guard = cache.check_fn.code_parts
            freevar_names = cache.check_fn.__code__.co_freevars
            freevar_values = [x.cell_contents for x in cache.check_fn.__closure__]
        else:
            # keep the logic synced with
            # https://github.com/pytorch/pytorch/blob/7b6b10417d8616ebd7a42b06528c5c2b2fded55a/torch/_dynamo/guards.py#L262
            tensor_aliasing_guard_seen = False
            def visit(root, ans):
                nonlocal tensor_aliasing_guard_seen
                for x in root.get_leaf_guards():
                    if isinstance(guard, torch._C._dynamo.guards.NO_TENSOR_ALIASING):
                        if not tensor_aliasing_guard_seen:
                            tensor_aliasing_guard_seen = True
                        else:
                            continue
                    for verbose_str in x.verbose_code_parts():
                        verbose_str = verbose_str.strip()
                        ans.append(verbose_str)
                for child in root.get_child_managers():
                    visit(child, ans)
            guard = []
            root = cache.check_fn.root
            visit(root, guard)
            if cache.check_fn.closure_vars is None:
                freevar_names = tuple()
                freevar_values = []
            else:
                freevar_names = tuple(cache.check_fn.closure_vars.keys())
                freevar_values = list(cache.check_fn.closure_vars.values())

        self.guard = guard
        self.freevars = {name: value for name, value in zip(freevar_names, freevar_values)}
        code = cache.code

        compiled_subgraphs = [
            name for name in code.co_names if name.startswith("__compiled")]
        assert len(compiled_subgraphs) <= 1

        if compiled_subgraphs:
            # deal with compiled_subgraph
            self.compiled_subgraph = innermost_fn(module[compiled_subgraphs[0]])
            # subgraph does not need decompile
            self.compiled_subgraph_proxy = CodeProxy.decompile_with_name(
                self.compiled_subgraph, compiled_subgraphs[0], skip_decompile=True)
        else:
            self.compiled_subgraph = None
            self.compiled_subgraph_proxy = None
        # deal with transformed_code
        self.transformed_code = code
        self.transformed_code_proxy = CodeProxy.decompile_with_name(
            self.transformed_code, "transformed_code:")
        resume_fns = [
            name for name in code.co_names if name.startswith("__resume")]
        self.referenced_global_functions = {}
        for name in resume_fns:
            self.referenced_global_functions[name] = DynamoOptimizationResult(
                original_code=module[name].__code__,
                function_name=name,
                module=module)

    def to_data(self):
        return {
            "guard": self.guard,
            "transformed_code": str(
                self.transformed_code_proxy),
            "compiled_subgraph": str(
                self.compiled_subgraph_proxy) if self.compiled_subgraph_proxy is not None else '"No compiled subgraph."',
            "referenced_global_functions": {
                name: fn.to_data() for name,
                fn in self.referenced_global_functions.items()}}


@dataclass
class DynamoOptimizationResult:
    function_name: str
    module: dict
    original_code: CodeType
    source_code_proxy: CodeProxy
    transformed_code_entries: List[CacheResult]

    def __init__(self, original_code, function_name=None, module=None):
        self.original_code = original_code
        if function_name is None:
            self.function_name = original_code.co_name
        else:
            self.function_name = function_name
        self.module = module
        caches = _debug_get_cache_entry_list(original_code)
        self.transformed_code_entries = [
            CacheResult(original_code, module, cache) for cache in caches]
        self.source_code_proxy = CodeProxy.decompile_with_name(
            self.original_code, self.function_name)

    def to_data(self):
        data = {
            "function_name": self.function_name,
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
        code_obj = self.original_code
        normal_arg_count = code_obj.co_argcount + code_obj.co_kwonlyargcount
        arg_names = code_obj.co_varnames[:normal_arg_count]
        arg_dict = "__local_dict = {" + \
            ", ".join([f'"{name}": {name}' for name in arg_names]) + "}"
        code += "\n" + " " * 4 + arg_dict
        code += "\n" + " " * 4 + "__global_dict = globals()"

        additional_code = ""

        for entry in self.transformed_code_entries:

            # prepare guards, like `def guard_0(L):\n    return a > 0 and b >
            # 0`
            freevars = "".join([f"{name} = '''{value}'''\n" for name, value in entry.freevars.items() if name not in ["__builtins__"]])
            if freevars:
                freevars = "# Note: the following variables are used inside the guard function.\n" + freevars
            guard_lines = [" " * 4 + "__guard_hit = True\n"]
            for x in entry.guard:
                guard_lines.append(" " * 4 + f"__guard_hit = __guard_hit and {x}\n")
            guard_lines.append(" " * 4 + "return __guard_hit\n")
            guard = "".join(guard_lines)
            if entry.transformed_code_proxy.name.startswith("__transformed_code_"):
                guard_func_name = entry.transformed_code_proxy.name.replace("__transformed_code_", "__guard_")
            else:
                guard_func_name = CodeProxy.consume_new_name("guard:")
            additional_code += "\n" + freevars + f"def {guard_func_name}(L, G, **___kwargs_ignored):\n" + guard

            if entry.compiled_subgraph_proxy is not None:
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
                f"if {guard_func_name}(__local_dict, __global_dict):\n" + " " * 8 + f"return {entry.transformed_code_proxy.name}({', '.join(arg_names)})"

        additional_code += "\n" + "# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.\n" + \
            remove_indentation(self.source_code_proxy.raw_code) + "\n"

        code += "\n" + " " * 4 + "# Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.\n" + \
            " " * 4 + f"return {self.source_code_proxy.name}({', '.join(arg_names)})"
        return additional_code + code + \
            f"\n\n#============ end of {self.function_name} ============#\n"


def remove_indentation(code: str):
    lines = code.splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "".join([line[indent:] + "\n" for line in lines])

from contextlib import contextmanager

@contextmanager
def lock_on_file(path_template):
    lock_path = path_template + ".lock"
    from filelock import FileLock
    import os
    lock = FileLock(lock_path)
    try:
        with lock:
            yield
    finally:
        pass


def write_code_to_file_template(src, path_template):
    with lock_on_file(path_template):
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


def get_current_compiled_fn_name():
    import torch
    from torch._dynamo.bytecode_transformation import _unique_id_counter
    from copy import copy
    # torch.compile already called the next, we should add minus 1 to get the
    # correct name
    current_count = next(copy(_unique_id_counter)) - 1
    return "__compiled_fn_" + str(current_count)
