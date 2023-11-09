import torch
from typing import List, Tuple, Dict, Union, Callable, Optional, Any
import torch
from torch._functorch.aot_autograd import aot_function
from torch._functorch.partitioners import default_partition
from torch._dynamo.bytecode_transformation import _unique_id_counter
from copy import copy
from collections import defaultdict
import inspect
import functools

def _get_current_count():
    # torch.compile already called the next, we should add minus 1 to get the correct name
    return next(copy(_unique_id_counter)) - 1

def _get_current_name():
    return f"__compiled_fn_{_get_current_count()}"

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

def update_fn_name(fn, name):
    import os
    fn = extract_underlying_func(fn)
    filepath = inspect.getsourcefile(fn)
    src = open(filepath).read()
    new_filepath = os.path.dirname(filepath) + "/" + name + ".py"
    os.rename(filepath, new_filepath)
    scope = fn.__globals__
    exec(compile(src, filename=new_filepath, mode="exec"), scope)
    fn.__code__ = scope[fn.__name__].__code__
    del scope[fn.__name__]

def convert_eager_backend(name):
    original_backend = torch._dynamo.backends.registry.lookup_backend(name)

    def new_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        func = original_backend(gm, example_inputs)
        update_fn_name(func.forward, _get_current_name() + "_captured_graph")
        return func  # return a python callable

    return new_backend

eager = convert_eager_backend("eager")

def convert_aot_backend(name):
    original_backend = torch._dynamo.backends.registry.lookup_backend(name)
    if "partition_fn" in original_backend.__closure__[0].cell_contents:
        partition_fn = original_backend.__closure__[0].cell_contents["partition_fn"]
    else:
        partition_fn = torch._functorch.aot_autograd.aot_module_simplified.__defaults__[1]
    def new_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        name = _get_current_name()
        update_fn_name(gm.forward, name + "_captured_graph")
        def my_partition(*args, **kwargs):
            fw_graph, bw_graph = partition_fn(*args, **kwargs)
            update_fn_name(fw_graph.forward, name + "_forward_graph")
            update_fn_name(bw_graph.forward, name + "_backward_graph")
            update_fn_name(args[0].forward, name + "_joint_graph")
            return fw_graph, bw_graph
        
        # mock the partition_fn
        if "partition_fn" in original_backend.__closure__[0].cell_contents:
            original_backend.__closure__[0].cell_contents["partition_fn"] = my_partition

            output = original_backend(gm, example_inputs)

            original_backend.__closure__[0].cell_contents["partition_fn"] = partition_fn
        else:
            defaults = torch._functorch.aot_autograd.aot_module_simplified.__defaults__
            torch._functorch.aot_autograd.aot_module_simplified.__defaults__ = (torch._functorch.aot_autograd.aot_module_simplified.__defaults__[0], my_partition, *torch._functorch.aot_autograd.aot_module_simplified.__defaults__[2:])

            output = original_backend(gm, example_inputs)

            torch._functorch.aot_autograd.aot_module_simplified.__defaults__ = defaults
        return output
    
    return new_backend

aot_eager = convert_aot_backend("aot_eager")

# not tested
aot_eager_decomp_partition = convert_aot_backend("aot_eager_decomp_partition")
aot_eager_default_partitioner = convert_aot_backend("aot_eager_default_partitioner")
aot_ts = convert_aot_backend("aot_ts")

__all__ = [
    "eager",
    "aot_eager",
    # "aot_eager_decomp_partition",
    # "aot_eager_default_partitioner",
    # "aot_ts",
]
