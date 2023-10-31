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

def eager(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    update_fn_name(gm.forward, _get_current_name() + "_captured_graph")
    return gm.forward  # return a python callable

def aot_eager(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    name = _get_current_name()
    update_fn_name(gm.forward, name + "_captured_graph")
    def my_partition(*args, **kwargs):
        fw_graph, bw_graph = default_partition(*args, **kwargs)
        update_fn_name(fw_graph.forward, name + "_forward_graph")
        update_fn_name(bw_graph.forward, name + "_backward_graph")
        update_fn_name(args[0].forward, name + "_joint_graph")
        return fw_graph, bw_graph
    def fwd_compiler(gm, example_inputs):
        return gm.forward
    # bwd_compiler is called lazily. we cannot rely on that.
    return aot_function(gm, fwd_compiler, partition_fn=my_partition)  # return a python callable
