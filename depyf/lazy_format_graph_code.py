def lazy_format_graph_code(name, gm, maybe_id=None):
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
    src = open(filepath).read()
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
