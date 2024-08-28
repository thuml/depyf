import dis
from typing import List, Tuple, Union, Optional, Callable, Any, Dict, Set
from types import CodeType


def get_function_signature(code_obj: CodeType,
                           overwite_fn_name: Optional[str] = None) -> str:
    # Extract all required details from the code object
    # Sometimes the code object does not have a name, e.g. when it is a lambda
    # function, so we can overwrite it to be a valid name
    normal_arg_count = code_obj.co_argcount + code_obj.co_kwonlyargcount
    arg_names = code_obj.co_varnames[:normal_arg_count]
    arg_names = [
        x if not x.startswith(".") else x.replace(
            ".", "comp_arg_") for x in arg_names]

    import inspect
    if code_obj.co_flags & inspect.CO_VARARGS:
        arg_names.append('*' + code_obj.co_varnames[normal_arg_count])
        normal_arg_count += 1
    if code_obj.co_flags & inspect.CO_VARKEYWORDS:
        arg_names.append('**' + code_obj.co_varnames[normal_arg_count])
        normal_arg_count += 1
    args_str = ', '.join(arg_names)
    fn_name = overwite_fn_name if overwite_fn_name is not None else code_obj.co_name
    header = f"def {fn_name}({args_str}):\n"
    return header


def collect_all_code_objects(code: CodeType) -> List[CodeType]:
    code_objects = [code]
    for const in code.co_consts:
        if isinstance(const, type(code)):
            code_objects.extend(collect_all_code_objects(const))
    return code_objects


def safe_create_directory(path):
    # allow multiple processes to create the same directory
    import os
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if not os.path.isdir(path):
            raise



def get_code_owner(fn):
    """A callable object `fn` might have a __code__ attribute, which is a code object.
    However, `fn` might not be the owner of the code object. Only the code owner can change the code object.
    This function returns the owner of the code object.
    An example:
    class A:
        def func(self):
            return 1
    a = A()
    `a.func.__code__` is read-only. `A.func.__code__` is writable.
    We can change the code object via `a.func.__func__.__code__`.
    """
    import functools
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
    return fn



def decompile_ensure(fn: CodeType, overwite_fn_name=None):
    import depyf
    from depyf.decompiler import DecompilationError
    try:
        decompiled_source_code = depyf.Decompiler(
            fn).decompile(overwite_fn_name=overwite_fn_name)
    except DecompilationError as e:
        header = get_function_signature(fn, overwite_fn_name=overwite_fn_name)
        decompiled_source_code = header + "    'Failed to decompile.'\n"
    return decompiled_source_code
