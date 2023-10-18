from depyf.explain.utils import DynamoOptimizationResult

from torch._dynamo.eval_frame import innermost_fn

from typing import List, Callable, Dict, Union, Set

def _extract_artifacts(fn: Callable):
    if hasattr(fn, "_torchdynamo_orig_callable"):
        inner_fn = innermost_fn(fn)
    else:
        inner_fn = fn
    result = DynamoOptimizationResult(inner_fn)
    return result

def interactive_explain(fn: Callable):
    artifacts = _extract_artifacts(fn)
    return artifacts

def dump_src(fn: Callable):
    artifacts = _extract_artifacts(fn)
    return artifacts.to_src()
