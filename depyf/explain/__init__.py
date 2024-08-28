from depyf.explain.utils import DynamoOptimizationResult

from torch._dynamo.eval_frame import innermost_fn

from typing import List, Callable, Dict, Union, Set
from types import CodeType


def _extract_artifacts(original_code: CodeType, module):
    result = DynamoOptimizationResult(original_code, None, module)
    return result

def dump_src(original_code: CodeType, module):
    from depyf.explain.global_variables import data
    assert data["is_inside_prepare_debug"], "`dump_src` must be used inside `depyf.prepare_debug`."
    artifacts = _extract_artifacts(original_code, module)
    return artifacts.to_src()
