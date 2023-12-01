def patched_lazy_format_graph_code(name, gm, maybe_id=None):
    from depyf.explain.utils import get_current_compiled_fn_name, get_code_owner, write_code_to_file_template
    func_name = get_current_compiled_fn_name()
    file_name = name if name != func_name else "Captured Graph"
    file_name = func_name + " " + file_name
    import inspect
    import os
    fn = gm.forward

    fn = get_code_owner(fn)

    # update file path
    filepath = inspect.getsourcefile(fn)
    # try to use verbose code with type and shape annotations
    src = "from __future__ import annotations\n" + \
        gm._graph.python_code(root_module="self", verbose=True).src
    try:
        compile(src, "noname", "exec")
    except Exception as e:
        # the pytorch version is before this PR: https://github.com/pytorch/pytorch/pull/113345
        # Verbose code contains syntax error, it is recommended to use new
        # version of PyTorch to get runnable code with shape and type
        # annotations.
        simple_code = gm._graph.python_code(root_module="self", verbose=False).src
        commented_src = "\n# code below is commented out due to syntax error. You can refer to the code for shape and dtype annotation.\n"
        commented_src += "".join(["# " + line +
                                 "\n" for line in src.splitlines()])
        src = simple_code + commented_src
    if filepath is not None:
        new_filepath = write_code_to_file_template(
            src, os.path.dirname(filepath) + "/" + file_name + " " + "%s" + ".py")
        scope = fn.__globals__
        exec(compile(src, filename=new_filepath, mode="exec"), scope)
        fn.__code__ = scope[fn.__name__].__code__
        del scope[fn.__name__]

    # =========================================
    # original code of `lazy_format_graph_code`
    def format_name():
        if maybe_id is not None:
            return f"{name} {maybe_id}"
        else:
            return name

    return LazyString(
        lambda: _format_graph_code(
            f"===== {format_name()} =====\n",
            gm.forward.__code__.co_filename,
            gm.print_readable(print_output=False),
        )
    )
