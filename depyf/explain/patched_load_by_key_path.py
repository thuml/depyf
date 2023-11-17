def patched_load_by_key_path(
        key: str,
        path: str,
        linemap,
        attrs,
    ):
    from depyf.explain.global_variables import data
    from depyf.explain.utils import write_code_to_file_template, get_current_compiled_fn_name
    dump_src_dir = data["dump_src_dir"]
    unpatched_load_by_key_path = data["unpatched_load_by_key_path"]
    import os
    # hack the path to our dump_src_dir
    src = open(path).read()
    os.remove(path)

    func_name = get_current_compiled_fn_name()
    new_filepath = write_code_to_file_template(src, os.path.join(dump_src_dir, func_name + " kernel " + "%s" + ".py"))
    path = new_filepath
    return unpatched_load_by_key_path(key, path, linemap, attrs)
