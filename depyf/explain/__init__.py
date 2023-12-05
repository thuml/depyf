from depyf.explain.utils import DynamoOptimizationResult

from torch._dynamo.eval_frame import innermost_fn

from typing import List, Callable, Dict, Union, Set


def _extract_artifacts(fn: Callable):
    if hasattr(fn, "_torchdynamo_orig_callable"):
        # this can deal with various types of callable objects, including
        # `nn.Module`.
        inner_fn = innermost_fn(fn)
    else:
        inner_fn = fn
    result = DynamoOptimizationResult(inner_fn)
    return result


def _collect_compiled_subgraphs(result: DynamoOptimizationResult):
    compiled_subgraphs = {
        entry.compiled_subgraph_proxy.name: entry.compiled_subgraph for entry in result.compiled_code_entries}
    for entry in result.compiled_code_entries:
        for func in entry.referenced_global_functions.values():
            ans = _collect_compiled_subgraphs(func)
            compiled_subgraphs.update(ans)
    return compiled_subgraphs


def interactive_explain(fn: Callable):
    artifacts = _extract_artifacts(fn)
    return artifacts


def dump_src(fn: Callable):
    from depyf.explain.global_variables import data
    assert data["is_inside_prepare_debug"], "`dump_src` must be used inside `depyf.prepare_debug`."
    artifacts = _extract_artifacts(fn)
    return artifacts.to_src()
