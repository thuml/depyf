from .patched_boxed_run import patched_boxed_run
from .patched_lazy_format_graph_code import patched_lazy_format_graph_code
from .patched_load_by_key_path import patched_load_by_key_path
from .patched__exec_with_source import patched__exec_with_source
from typing import List, Tuple, Dict, Union, Callable, Optional, Any

import contextlib
import warnings

import dataclasses
import itertools


@dataclasses.dataclass
class DebuggableHook(object):
    dump_src_dir: str

    def __call__(self, code, new_code):
        from depyf.decompiler import DecompilationError
        try:
            import os
            # replace " "/"<"/">"/"." with "_"
            func_name = code.co_name.replace(".", "_").replace("<", "_").replace(">", "_").replace(" ", "_")
            filepath_template = os.path.join(
                self.dump_src_dir,
                f"__transformed_code_%s_for_{func_name}.py")

            from depyf.explain.utils import lock_on_file
            from depyf.decompiler import Decompiler

            # function name and file name are related.
            with lock_on_file(filepath_template % "lock"):
                # we first try to find an existing file with the same code body.
                src = Decompiler(new_code).decompile(overwite_fn_name="place_holder")
                src_body = src[src.find("("):]

                count = 0
                while True:
                    filename = filepath_template % count
                    if os.path.exists(filename):
                        existing_code = open(filename, "r").read()
                        existing_code_body = existing_code[existing_code.find("("):]
                        if src_body == existing_code_body:
                            # the same code body is found, we do not need to dump the code again.
                            src = existing_code
                            break
                        else:
                            count += 1
                    else:
                        func_name = filename.split(os.path.sep)[-1].split(".")[0]
                        src = f"def {func_name}" + src_body
                        with open(filename, "w") as f:
                            f.write(src)
                        break

            transformed_code = compile(src, filename=filename, mode="exec")
            scope = {}
            exec(transformed_code, scope)
            func_name = filename.split(os.path.sep)[-1].split(".")[0]
            func = scope[func_name]

            # this fix is used for PyTorch prior to PR https://github.com/pytorch/pytorch/pull/114487
            from torch._dynamo.utils import orig_code_map
            from torch._dynamo.convert_frame import output_codes
            output_codes.add(func.__code__)
            orig_code_map[func.__code__] = code

            return func.__code__
        except (DecompilationError, SyntaxError) as e:
            from io import StringIO
            string_io = StringIO()
            import dis
            print("There is a problem when decompiling and compiling the following code:", file=string_io)
            dis.dis(new_code, file=string_io)
            print("Please consider submitting an issue to https://github.com/thuml/depyf .", file=string_io)
            # do not stop the program for decompilation error and compile error
            warnings.warn(string_io.getvalue())

@contextlib.contextmanager
def patch(parent, name, value):
    old_value = getattr(parent, name)
    setattr(parent, name, value)
    try:
        yield
    finally:
        setattr(parent, name, old_value)


@contextlib.contextmanager
def enable_bytecode_hook(hook):
    import torch
    handle = torch._dynamo.convert_frame.register_bytecode_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def prepare_debug(func, dump_src_dir, clean_wild_fx_code=True, pause=True):
    """
    Args:
        func: the function to debug, can be `None`. If it is `None`, do not dump all the source code in `full_code.py`.
        clean_wild_fx_code: whether to clean the wild fx code that are not recognized for parts of compiled functions. They are usually used by PyTorch internally.
        pause: whether to pause the program after the source code is dumped.
    """
    import os
    import torch

    warnings.warn((
        "You are trying to debug `torch.compile`. Please make sure the code "
        "runs multiple times to cover all the possible branches."
    ))

    if not os.path.exists(dump_src_dir):
        os.makedirs(dump_src_dir)

    dump_src_dir = os.path.abspath(dump_src_dir)

    from .global_variables import data

    data["dump_src_dir"] = dump_src_dir
    data["unpatched__exec_with_source"] = torch.fx.graph_module._exec_with_source
    data["unpatched_load_by_key_path"] = torch._inductor.codecache.PyCodeCache.load_by_key_path

    bytecode_hook = DebuggableHook(dump_src_dir)

    # patch some functions
    with patch(torch.fx.graph_module, "_exec_with_source", patched__exec_with_source), \
            patch(torch._inductor.codecache.PyCodeCache, "load_by_key_path", patched_load_by_key_path), \
            patch(torch._dynamo.utils.lazy_format_graph_code, "__code__", patched_lazy_format_graph_code.__code__):
        # we have to directly manipulate the code object, since the function has been imported in many places.
        # simply replacing torch._dynamo.utils.lazy_format_graph_code does not work for those functions.
        # Note: `unitest.mock.patch` does not work here, since it will not
        # patch the code object. (it will try to delete the code object and
        # then set a new code object. The `delattr` will raise an error.)

        # enable bytecode hook
        with enable_bytecode_hook(bytecode_hook):
            try:
                yield
            finally:
                if clean_wild_fx_code:
                    for file in os.listdir(dump_src_dir):
                        if file.split(
                                os.path.sep)[-1].startswith("fx_graph_code"):
                            os.remove(os.path.join(dump_src_dir, file))

                if func is None:
                    if pause:
                        input(
                            f"Please set breakpoints in {dump_src_dir}. Then press enter to continue.")
                else:
                    from depyf.explain import dump_src
                    from depyf.explain.utils import write_code_to_file_template
                    from torch._dynamo.eval_frame import innermost_fn
                    func = innermost_fn(func)
                    full_src = dump_src(func)
                    filepath_template = os.path.join(dump_src_dir, f"full_code_for_{func.__code__.co_name}_%s.py")
                    filename = write_code_to_file_template(full_src, filepath_template)

                    if pause:
                        input(
                            f"Please check the full source code in {filename}, and set breakpoints for functions in {dump_src_dir} according to the function name. Then press enter to continue.")


@contextlib.contextmanager
def debug():
    import torch
    callback = torch._dynamo.eval_frame.set_eval_frame(False)
    # sometimes pytorch use Interpreter to run node by node. This cannot be debugged.
    # we patch this function to run the graph function directly.
    with patch(torch.fx.Interpreter.boxed_run, "__code__", patched_boxed_run.__code__):
        try:
            yield
        finally:
            torch._dynamo.eval_frame.set_eval_frame(callback)
