from .patched_boxed_run import patched_boxed_run
from .patched_lazy_format_graph_code import patched_lazy_format_graph_code
from .patched_load_by_key_path import patched_load_by_key_path
from .patched__exec_with_source import patched__exec_with_source
from typing import List, Tuple, Dict, Union, Callable, Optional, Any

import contextlib
import warnings
import traceback

import dataclasses
import itertools
import sys
import os
import inspect


@dataclasses.dataclass
class DebuggableHook(object):
    dump_src_dir: str
    log_bytecode: bool
    optimized_code_and_module: List =dataclasses.field(default_factory=list, init=False)

    def __call__(self, code, new_code):
        frame = sys._getframe()
        import os
        while True:
            frame = frame.f_back
            code_name = frame.f_code.co_name
            file_name = frame.f_code.co_filename.split(os.path.sep)[-1]
            if code_name == "_compile" and file_name == "convert_frame.py":
                break
        frame = frame.f_locals["frame"]
        assert frame.f_code == code
        self.optimized_code_and_module.append([code, frame.f_globals])
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
            with lock_on_file(filepath_template):
                decompiled_and_compiled_back_code = Decompiler.decompile_and_compile_like(code_to_decompile=new_code, reference_code=code, filepath_template=filepath_template)
                filename = decompiled_and_compiled_back_code.co_filename
            if self.log_bytecode:
                with lock_on_file(filename):
                    import dill
                    # code object, especially `new_code` constructed by Dynamo, may not be able to be dumped using `marshal`.
                    # see https://github.com/pytorch/pytorch/issues/116013 for more details.
                    with contextlib.suppress(Exception):
                        dill.dump(code, open(filename + ".original_bytecode", "wb"))

                    with contextlib.suppress(Exception):
                        dill.dump(new_code, open(filename + ".transformed_bytecode", "wb"))

                    with contextlib.suppress(Exception):
                        dill.dump(decompiled_and_compiled_back_code, open(filename + ".decompiled_and_compiled_back_bytecode", "wb"))

            # this fix is used for PyTorch prior to PR https://github.com/pytorch/pytorch/pull/114487
            from torch._dynamo.utils import orig_code_map
            from torch._dynamo.convert_frame import output_codes
            output_codes.add(decompiled_and_compiled_back_code)
            orig_code_map[decompiled_and_compiled_back_code] = code

            return decompiled_and_compiled_back_code
        except (DecompilationError, SyntaxError) as e:
            from io import StringIO
            string_io = StringIO()
            import dis
            print("There is a problem when decompiling and compiling the following code:", file=string_io)
            dis.dis(new_code, file=string_io)
            print("Please consider submitting an issue to https://github.com/thuml/depyf .", file=string_io)
            # do not stop the program for decompilation error and compile error
            warnings.warn(string_io.getvalue())
            traceback.print_exc()

@contextlib.contextmanager
def patch(parent, name, value):
    old_value = getattr(parent, name, None)
    if old_value is not None:
        setattr(parent, name, value)
    try:
        yield
    finally:
        if old_value is not None:
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
def prepare_debug(dump_src_dir, clean_wild_fx_code=True, log_bytecode=False):
    """
    A context manager to dump debugging information for torch.compile.
    It should wrap the code that actually triggers the compilation, rather than
    the code that applies ``torch.compile``.

    Example:

    .. code-block:: python

        import torch

        @torch.compile
        def toy_example(a, b):
            x = a / (torch.abs(a) + 1)
            if b.sum() < 0:
                b = b * -1
            return x * b

        def main():
            for _ in range(100):
                toy_example(torch.randn(10), torch.randn(10))

        if __name__ == "__main__":
            # main()
            # surround the code you want to run inside `with depyf.prepare_debug`
            import depyf
            with depyf.prepare_debug("./dump_src_dir"):
                main()

    After running the code, you will find the dumped information in the directory ``dump_src_dir``. The details are organized into the following:

    - ``full_code_for_xxx.py`` for each function using torch.compile
    - ``__transformed_code_for_xxx.py`` for Python code associated with each graph.
    - ``__transformed_code_for_xxx.py.xxx_bytecode`` for Python bytecode, dumped code object, can be loaded via ``dill.load(open("/path/to/file", "wb"))``. Note that the load function might import some modules like transformers. Make sure you have these modules installed.
    - ``__compiled_fn_xxx.py`` for each computation graph and its optimization:
        - ``Captured Graph``: a plain forward computation graph
        - ``Joint Graph``: joint forward-backward graph from AOTAutograd
        - ``Forward Graph``: forward graph from AOTAutograd
        - ``Backward Graph``: backward graph from AOTAutograd
        - ``kernel xxx``: compiled CPU/GPU kernel wrapper from Inductor.

    Arguments:

    - ``dump_src_dir``: the directory to dump the source code.
    - ``clean_wild_fx_code``: whether to clean the wild fx code that are not recognized for parts of compiled functions. They are usually used by PyTorch internally.
    - ``log_bytecode``: whether to log bytecode (original bytecode, transformed bytecode from Dynamo, and decompiled_and_compiled_back_code).
    """

    if not isinstance(dump_src_dir, str):
        raise RuntimeError('''You are using an obsolete usage style`depyf.prepare_debug(func=function, dump_src_dir="/path")`. Please use `depyf.prepare_debug(dump_src_dir="/path")` instead, which will automatically capture all compiled functions.''')

    import os
    import torch

    current_line_number = inspect.currentframe().f_lineno + 1
    warnings.warn_explicit(f"{__file__}:{current_line_number}: You are trying to debug `torch.compile`. Please make sure the code runs multiple times to cover all the possible branches.", UserWarning, "", 0)

    from depyf.utils import safe_create_directory

    if not os.path.exists(dump_src_dir):
        safe_create_directory(dump_src_dir)

    dump_src_dir = os.path.abspath(dump_src_dir)

    from .global_variables import data

    data["dump_src_dir"] = dump_src_dir
    data["unpatched__exec_with_source"] = torch.fx.graph_module._exec_with_source
    data["unpatched_load_by_key_path"] = torch._inductor.codecache.PyCodeCache.load_by_key_path
    data["unpatched___call__"] = torch._dynamo.eval_frame.OptimizeContext.__call__
    data["is_inside_prepare_debug"] = True

    bytecode_hook = DebuggableHook(dump_src_dir, log_bytecode)

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

                code_names = {x[0].co_name for x in bytecode_hook.optimized_code_and_module}
                for code, module in bytecode_hook.optimized_code_and_module:
                    if code.co_name.startswith("resume_in_") and any(f"resume_in_{name}" in code.co_name for name in code_names):
                        continue
                    # https://github.com/pytorch/pytorch/pull/118201 introduces `torch_dynamo_resume_in_` names.
                    if code.co_name.startswith("torch_dynamo_resume_in_") and any(f"torch_dynamo_resume_in_{name}" in code.co_name for name in code_names):
                        continue
                    from depyf.explain import dump_src
                    from depyf.explain.utils import write_code_to_file_template
                    from torch._dynamo.eval_frame import innermost_fn, _debug_get_cache_entry_list
                    entries = _debug_get_cache_entry_list(code)
                    if not entries:
                        current_line_number = inspect.currentframe().f_lineno + 1
                        warnings.warn_explicit(f"{__file__}:{current_line_number}: Code object {code} is compiled but does not have any compiled cache entries. Probably some torch.nn.Module instances are destroyed too early. It is recommended to make sure the torch.nn.Module instances exist after `with depyf.prepare_debug`.", UserWarning, "", 0)
                    full_src = dump_src(code, module)
                    filepath_template = os.path.join(dump_src_dir, f"full_code_for_{code.co_name}_%s.py")
                    full_code_path = write_code_to_file_template(full_src, filepath_template)

                for file in os.listdir(dump_src_dir):
                    name = file.split(os.path.sep)[-1]
                    # remove *.lock file and possibly fx_graph_code* file
                    if (clean_wild_fx_code and name.startswith("fx_graph_code")) or name.endswith(".lock"):
                        try:
                            # multiple processes may try to remove the same file.
                            os.remove(os.path.join(dump_src_dir, file))
                        except OSError:
                            pass

                data["is_inside_prepare_debug"] = False

@contextlib.contextmanager
def debug():
    """
    A context manager to debug the compiled code. Essentially, it sets a breakpoint to pause the program and allows you to check the full source code in files with prefix ``full_code_for_`` in the ``dump_src_dir`` argument of :func:`depyf.prepare_debug`, and set breakpoints in their separate ``__transformed_code_`` files according to the function name. Then continue your debugging.
    """
    from .global_variables import data
    if data["is_inside_prepare_debug"]:
        raise RuntimeError("You cannot use `depyf.debug` inside `depyf.prepare_debug`.")
    dump_src_dir = data["dump_src_dir"]
    import torch
    # after https://github.com/pytorch/pytorch/pull/131258
    # torch._dynamo.eval_frame.set_eval_frame is not available in the module
    # we need to directly access it from the `_C` extension.
    callback = torch._C._dynamo.eval_frame.set_eval_frame(False)
    # sometimes pytorch use Interpreter to run node by node. This cannot be debugged.
    # we patch this function to run the graph function directly.
    with patch(torch.fx.Interpreter.boxed_run, "__code__", patched_boxed_run.__code__):
        try:
            msg = f"`depyf` places a breakpoint here to pause the program. You can check the full source code in files with prefix `full_code_for_` in {dump_src_dir} first, and set breakpoints in their separate files according to the function name. Then continue your debugging."
            print(msg)
            breakpoint()
            yield
        finally:
            torch._C._dynamo.eval_frame.set_eval_frame(callback)
