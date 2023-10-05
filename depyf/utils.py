import dis
from typing import List, Tuple, Union, Optional, Callable, Any, Dict, Set
from types import CodeType

def escape_html(s: str) -> str:
    """Escape string for use in HTML."""
    return (s.replace("&", "&amp;")
            .replace(" ", "&nbsp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))

def generate_dot_table(header: str, rows: List[List[str]]) -> str:
    """
    Generate an HTML table string with a header spanning n columns and given rows.

    Args:
    - header (str): The header string.
    - rows (list of list of str): A list of rows, where each row is a list of n strings.

    Returns:
    - str: The generated HTML table string.
    """

    # Start the table
    html_str = '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'

    n = len(rows[0])
    # Add the header spanning n columns
    html_str += f'<TR><TD  ALIGN="LEFT">{header}</TD>' + '<TD  ALIGN="LEFT"></TD>' * (n - 1) + '</TR>'

    # Add each row
    for row in rows:
        if len(row) != n:
            raise ValueError("Inconsistent number of columns.")
        html_str += '<TR>' + ''.join([f'<TD  ALIGN="LEFT">{escape_html(cell)}</TD>' for cell in row]) + '</TR>'

    # Close the table
    html_str += '</TABLE>'
    
    return "<\n" + html_str + "\n>"

def get_function_signature(code_obj: CodeType, overwite_fn_name: Optional[str]=None) -> str:
    # Extract all required details from the code object
    # Sometimes the code object does not have a name, e.g. when it is a lambda function, so we can overwrite it to be a valid name
    normal_arg_count = code_obj.co_argcount + code_obj.co_kwonlyargcount
    arg_names = code_obj.co_varnames[:normal_arg_count]
    arg_names = [x if not x.startswith(".") else x.replace(".", "comp_arg_") for x in arg_names]
    args_str = ', '.join(arg_names)
    import inspect
    if code_obj.co_flags & inspect.CO_VARARGS:
        args_str += ', *' + code_obj.co_varnames[normal_arg_count]
        normal_arg_count += 1
    if code_obj.co_flags & inspect.CO_VARKEYWORDS:
        args_str += ', **' + code_obj.co_varnames[normal_arg_count]
        normal_arg_count += 1
    fn_name = overwite_fn_name if overwite_fn_name is not None else code_obj.co_name
    header = f"def {fn_name}({args_str}):\n"
    return header
