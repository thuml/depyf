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
    html_str += f'<TR><TD>{header}</TD>' + '<TD></TD>' * (n - 1) + '</TR>'

    # Add each row
    for row in rows:
        if len(row) != n:
            raise ValueError("Inconsistent number of columns.")
        html_str += '<TR>' + ''.join([f'<TD>{escape_html(cell)}</TD>' for cell in row]) + '</TR>'

    # Close the table
    html_str += '</TABLE>'
    
    return "<\n" + html_str + "\n>"

def get_function_signature(code_obj: CodeType) -> str:
    # Extract all required details from the code object
    arg_names = code_obj.co_varnames[:code_obj.co_argcount]
    args_str = ', '.join(arg_names)
    header = f"def {code_obj.co_name}({args_str}):\n"
    return header
