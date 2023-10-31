import torch
from typing import List
import torch
from torch._functorch.aot_autograd import aot_function
from torch._functorch.partitioners import default_partition
from torch._dynamo.bytecode_transformation import _unique_id_counter
from copy import copy
from collections import defaultdict
import inspect

subgraph_name_to_src_files = defaultdict(list)
subgraph_name_to_joint_graph_files = defaultdict(list)
subgraph_name_to_fwd_files = defaultdict(list)
subgraph_name_to_bwd_files = defaultdict(list)

def _get_current_count():
    # torch.compile already called the next, we should add minus 1 to get the correct name
    return next(copy(_unique_id_counter)) - 1

def _get_current_name():
    return f"__compiled_fn_{_get_current_count()}"

def eager(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    subgraph_name_to_src_files[_get_current_name()].append(inspect.getsourcefile(gm.forward))
    return gm.forward  # return a python callable

def aot_eager(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    name = _get_current_name()
    subgraph_name_to_src_files[name].append(inspect.getsourcefile(gm.forward))
    def my_partition(*args, **kwargs):
        fw_graph, bw_graph = default_partition(*args, **kwargs)
        subgraph_name_to_fwd_files[name].append(inspect.getsourcefile(fw_graph.forward))
        subgraph_name_to_bwd_files[name].append(inspect.getsourcefile(bw_graph.forward))
        subgraph_name_to_joint_graph_files[name].append(inspect.getsourcefile(args[0].forward))
        return fw_graph, bw_graph
    def fwd_compiler(gm, example_inputs):
        return gm.forward
    # bwd_compiler is called lazily. we cannot rely on that.
    return aot_function(gm, fwd_compiler, partition_fn=my_partition)  # return a python callable
