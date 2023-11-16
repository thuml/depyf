def patched_load_by_key_path(
        key: str,
        path: str,
        linemap,
        attrs,
    ):
    from depyf.explain.global_variables import data
    from depyf.explain.utils import write_code_to_file_template
    dump_src_dir = data["dump_src_dir"]
    unpatched_load_by_key_path = data["unpatched_load_by_key_path"]
    import os
    # hack the path to our dump_src_dir
    src = open(path).read()
    os.remove(path)

    from torch._dynamo.bytecode_transformation import _unique_id_counter
    from copy import copy
    # torch.compile already called the next, we should add minus 1 to get the correct name
    current_count = next(copy(_unique_id_counter)) - 1
    func_name = "__compiled_fn_" + str(current_count)
    new_filepath = write_code_to_file_template(src, os.path.join(dump_src_dir, func_name + " kernel " + "%s" + ".py"))
    path = new_filepath
    return unpatched_load_by_key_path(key, path, linemap, attrs)
