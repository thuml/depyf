def patched__exec_with_source(src: str, globals, co_fields=None):
    from depyf.explain.global_variables import data
    dump_src_dir = data["dump_src_dir"]
    unpatched__exec_with_source = data["unpatched__exec_with_source"]
    unpatched__exec_with_source(src, globals, co_fields)
    import inspect
    key = inspect.getsourcefile(globals["forward"])
    import hashlib
    import os
    hash_value = hashlib.md5(src.encode()).hexdigest()
    src = "# " + key + src
    count = 0
    while True:
        filename = f"{dump_src_dir}/fx_graph_code_" + hash_value + "_" + str(count) + ".py"
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                f.write(src)
            break
        # might be a hash collision
        existing_code = open(filename).read()
        if existing_code == src:
            break
        count += 1
    exec(compile(src, filename, "exec"), globals)
