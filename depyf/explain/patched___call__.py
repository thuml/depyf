def patched___call__(self, fn):
    from depyf.explain.global_variables import data
    from depyf.explain.utils import get_code_owner
    import torch
    unpatched___call__ = data["unpatched___call__"]
    optimized_functions = data["optimized_functions"]
    from torch._dynamo.eval_frame import innermost_fn, OptimizeContext
    inner_fn = innermost_fn(fn)
    code_owner = get_code_owner(inner_fn)
    if isinstance(self, OptimizeContext):
        optimized_functions.add(code_owner)
    
    return unpatched___call__(self, fn)