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

from IPython.display import display, JSON, Markdown, Code

import depyf

class CodeProxy:
    instances: Dict[str, "CodeProxy"] = {}
    used_instances: Set[str] = set()
    def __init__(self, code: str, name: str):
        i = 0
        new_name = name
        if new_name.endswith(":"):
            name = name[:-1]
            while True:
                new_name = f"{name}_{i}"
                if new_name not in CodeProxy.instances:
                    break
                i += 1
        self.name = new_name
        code = "".join(["  " + line + "\n" for line in code.splitlines() if line.strip() != ""])
        self.raw_code = code
        self.code = f"""<details>
  <summary>{self.name}</summary>

  ```python
{code}
  ```
</details>
"""
        CodeProxy.instances[self.name] = self
    
    def __str__(self):
        CodeProxy.used_instances.add(self.name)
        return self.name

    @contextlib.contextmanager
    @staticmethod
    def record():
        CodeProxy.used_instances = set()
        yield CodeProxy.used_instances

def display_func(self):
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
    compiled_code: str
    compiled_code_proxy: CodeProxy
    referenced_global_functions: Dict[str, "DynamoOptimizationResult"]

    def __init__(self, fn, cache):
        guard = cache.check_fn.code_parts
        code = cache.code
        compiled_subgraphs = [name for name in code.co_names if name.startswith("__compiled")]
        assert len(compiled_subgraphs) == 1
        module = import_module(fn.__module__)
        self.compiled_subgraph = innermost_fn(getattr(module, compiled_subgraphs[0]))
        self.compiled_subgraph_proxy = CodeProxy(innermost_fn(self.compiled_subgraph).__self__.code, compiled_subgraphs[0])
        try:
            compiled_code = depyf.decompile(code)
        except Exception as e:
            print(str(e))
            compiled_code = "'Failed to decompile.'\n"
        self.compiled_code = compiled_code
        self.compiled_code_proxy = CodeProxy(compiled_code, "cache_code:")
        resume_fns = [name for name in code.co_names if name.startswith("__resume")]
        self.referenced_global_functions = {name: DynamoOptimizationResult(getattr(module, name), name) for name in resume_fns}
        self.code = code
        self.guard = guard

    def to_data(self):
        return {
            "guard": self.guard,
            "compiled_code": str(self.compiled_code_proxy),
            "compiled_subgraph": str(self.compiled_subgraph_proxy),
            "referenced_global_functions": {name: fn.to_data() for name, fn in self.referenced_global_functions.items()}
        }

    _ipython_display_ = display_func

@dataclass
class DynamoOptimizationResult:
    name: str
    fn: Callable
    code: CodeType
    source_code: str
    source_code_proxy: CodeProxy
    compiled_code_entries: List[CacheResult]

    def __init__(self, fn, name=None):
        self.fn = fn
        if name is None:
            self.name = fn.__name__
        else:
            self.name = name
        caches = _debug_get_cache_entry_list(fn.__code__)
        self.compiled_code_entries = [CacheResult(fn, cache) for cache in caches]
        self.code = fn.__code__
        try:
            decompiled_source_code = depyf.Decompiler(self.code).decompile(overwite_fn_name=self.name)
        except Exception as e:
            print(str(e))
            decompiled_source_code = "'Failed to decompile.'\n"
        self.source_code = decompiled_source_code
        self.source_code_proxy = CodeProxy(self.source_code, self.name)
    
    def to_data(self):
        data = {
            "name": self.name,
            "source_code": str(self.source_code_proxy),
            "compiled_code_entries": [entry.to_data() for entry in self.compiled_code_entries]
        }
        return data

    def to_src(self):
        raw_code = self.source_code_proxy.raw_code
        signature = raw_code.splitlines()[0].replace("def ", "def compiled_", 1)
        code = signature.strip()
        code_obj = self.fn.__code__
        normal_arg_count = code_obj.co_argcount + code_obj.co_kwonlyargcount
        arg_names = code_obj.co_varnames[:normal_arg_count]
        arg_dict = "L = {" + ", ".join([f'"{name}": {name}' for name in arg_names]) + "}"
        code += "\n" + " " * 4 + arg_dict

        additional_code = ""

        for entry in self.compiled_code_entries:
            guard = (" \\\n" + " " * 8 + "and ").join(["(" + x + ")" for x in entry.guard])
            guard_func_name = CodeProxy(guard, "guard:").name
            additional_code += f"\ndef {guard_func_name}(L):\n" + " " * 4 + "return " + guard + "\n"

            compiled_subgraph_name = entry.compiled_subgraph_proxy.name
            compiled_subgraph_code = entry.compiled_subgraph_proxy.raw_code
            compiled_subgraph_code = f"def {compiled_subgraph_name}(" + compiled_subgraph_code[compiled_subgraph_code.index("(self,") + 6:].lstrip()
            additional_code += "\n" + remove_indentation(compiled_subgraph_code) + "\n"

            func_name = entry.compiled_code_proxy.name
            compiled_code = entry.compiled_code_proxy.raw_code
            compiled_code = f"def {func_name}" + compiled_code[compiled_code.index("("):]
            additional_code += "\n" + remove_indentation(compiled_code) + "\n"

            for name, func in entry.referenced_global_functions.items():
                additional_code = func.to_src() + f"\n\n#============ separator for {name} ============#\n" + additional_code

            code += "\n" + " " * 4 + f"if {guard_func_name}(L):\n" + " " * 8 + f"return {func_name}({', '.join(arg_names)})"
        
        additional_code += "\n" + remove_indentation(self.source_code_proxy.raw_code) + "\n"
        original_func_name = self.source_code_proxy.name
        code += "\n" + " " * 4 + "# Note: this function might be compiled again, i.e. adding one more guard and compiled code. It might well not be executed directly.\n" + " " * 4 + f"return {original_func_name}({', '.join(arg_names)})"
        return additional_code + code

    _ipython_display_ = display_func

def explain(fn: Callable):
    if hasattr(fn, "_torchdynamo_orig_callable"):
        inner_fn = innermost_fn(fn)
    else:
        inner_fn = fn
    result = DynamoOptimizationResult(inner_fn)
    return result

def remove_indentation(code: str):
    lines = code.splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "".join([line[indent:] + "\n" for line in lines])

def dump_src(fn: Callable):
    result = explain(fn)
    return result.to_src()
