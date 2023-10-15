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

from IPython.display import display, JSON, Markdown

import depyf

class CodeProxy:
    instances: Dict[str, "CodeProxy"] = {}
    used_instances: Set[str] = set()
    def __init__(self, code: str, name: str):
        i = 0
        new_name = name
        if new_name.endswith(":"):
            while True:
                new_name = f"{name}{i}"
                if new_name not in CodeProxy.instances:
                    break
                i += 1
        self.name = new_name
        code = "".join(["  " + line + "\n" for line in code.splitlines() if line.strip() != ""])
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
            decompiled_source_code = depyf.decompile(self.code)
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

    _ipython_display_ = display_func

def explain(fn: Callable):
    if hasattr(fn, "_torchdynamo_orig_callable"):
        inner_fn = innermost_fn(fn)
    else:
        inner_fn = fn
    result = DynamoOptimizationResult(inner_fn)
    return result
