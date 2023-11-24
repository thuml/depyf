def patched__exec_with_source(src: str, globals, co_fields=None):
    from depyf.explain.global_variables import data
    from depyf.explain.utils import write_code_to_file_template
    dump_src_dir = data["dump_src_dir"]
    unpatched__exec_with_source = data["unpatched__exec_with_source"]
    unpatched__exec_with_source(src, globals, co_fields)
    import inspect
    key = inspect.getsourcefile(globals["forward"])
    import hashlib
    import os
    hash_value = hashlib.md5(src.encode()).hexdigest()
    src = "# " + key + src
    filename = write_code_to_file_template(
        src,
        f"{dump_src_dir}/fx_graph_code_" +
        hash_value +
        "_" +
        "%s" +
        ".py")
    exec(compile(src, filename, "exec"), globals)
