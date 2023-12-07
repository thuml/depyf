def patched___call__(self, code, check_fn):
    from depyf.explain.global_variables import data
    from depyf.explain.utils import get_code_owner
    import torch
    unpatched___call__ = data["unpatched___call__"]
    optimized_functions = data["optimized_functions"]
    optimized_functions.add(code)
    
    return unpatched___call__(self, code, check_fn)