def patched___call__(self, fn):
    from depyf.explain.global_variables import data
    import torch
    unpatched___call__ = data["unpatched___call__"]
    optimized_functions = data["optimized_functions"]
    from torch._dynamo.eval_frame import innermost_fn, OptimizeContext
    inner_fn = innermost_fn(fn)
    if isinstance(self, OptimizeContext):
        optimized_functions.add(inner_fn)
    
    return unpatched___call__(self, fn)