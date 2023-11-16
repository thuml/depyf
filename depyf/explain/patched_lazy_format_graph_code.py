def patched_lazy_format_graph_code(name, gm, maybe_id=None):
    from torch._dynamo.bytecode_transformation import _unique_id_counter
    from copy import copy
    # torch.compile already called the next, we should add minus 1 to get the correct name
    current_count = next(copy(_unique_id_counter)) - 1
    func_name = "__compiled_fn_" + str(current_count)
    file_name = name if name != func_name else "Captured Graph"
    file_name = func_name + " " + file_name
    import inspect, os
    fn = gm.forward

    while True:
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
            break

    # update file path
    filepath = inspect.getsourcefile(fn)
    # try to use verbose code with type and shape annotations
    src = "from __future__ import annotations\n" + gm._graph.python_code(root_module="self", verbose=True).src
    try:
        compile(src, "noname", "exec")
    except Exception as e:
        # the pytorch version is before this PR: https://github.com/pytorch/pytorch/pull/113345
        # Verbose code contains syntax error, it is recommended to use new version of PyTorch to get runnable code with shape and type annotations.
        simple_code = open(filepath).read()
        commented_src = "\n# code below is commented out due to syntax error. You can refer to the code for shape and dtype annotation.\n"
        commented_src += "".join(["# " + line + "\n" for line in src.splitlines()])
        src = simple_code + commented_src
    os.remove(filepath)
    count = 0
    while True:
        new_filepath = os.path.dirname(filepath) + "/" + file_name + " " + str(count) + ".py"
        if not os.path.exists(new_filepath):
            with open(new_filepath, "w") as f:
                f.write(src)
            break
        # might be a hash collision
        existing_code = open(new_filepath).read()
        if existing_code == src:
            # really the same code
            break
        count += 1
    scope = fn.__globals__
    exec(compile(src, filename=new_filepath, mode="exec"), scope)
    fn.__code__ = scope[fn.__name__].__code__
    del scope[fn.__name__]
