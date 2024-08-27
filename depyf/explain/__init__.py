from depyf.explain.utils import DynamoOptimizationResult

from torch._dynamo.eval_frame import innermost_fn

from typing import List, Callable, Dict, Union, Set
from types import CodeType


def _extract_artifacts(original_code: CodeType, module):
    result = DynamoOptimizationResult(original_code, None, module)
    return result


def _collect_compiled_subgraphs(result: DynamoOptimizationResult):
    compiled_subgraphs = {
        entry.compiled_subgraph_proxy.name: entry.compiled_subgraph for entry in result.compiled_code_entries}
    for entry in result.compiled_code_entries:
        for func in entry.referenced_global_functions.values():
            ans = _collect_compiled_subgraphs(func)
            compiled_subgraphs.update(ans)
    return compiled_subgraphs

def dump_src(original_code: CodeType, module):
    from depyf.explain.global_variables import data
    assert data["is_inside_prepare_debug"], "`dump_src` must be used inside `depyf.prepare_debug`."
    artifacts = _extract_artifacts(original_code, module)
    return artifacts.to_src()
