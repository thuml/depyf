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
