import dis
from typing import List, Tuple, Union, Optional, Callable, Any, Dict, Set
from types import CodeType

def nop_unreachable_bytecode(instructions: List[dis.Instruction]) -> List[dis.Instruction]:
    """Mark unreachable bytecode as NOP."""
    jumps = set(dis.hasjabs) | set(dis.hasjrel)

    reachable = [False for x in instructions]
    reachable[0] = True
    # each instruction marks the instruction after it
    for i, inst in enumerate(instructions):
        # the last instruction does not need to mark any following instructions
        if i == len(instructions) - 1:
            break
        if inst.is_jump_target:
            # the instruction is the target of a jump
            reachable[i] = True
        # this instruction is not reachable, nothing to do
        if not reachable[i]:
            continue
        # this instruction is reachable
        # the following instruction is reachable if it is sequential op or conditional jump
        if inst.opname in ["RETURN_VALUE", "BREAK_LOOP"]:
            # the instruction after the return is unreachable
            pass
        elif inst.opcode in jumps:
            if inst.opcode in dis.hasjrel and inst.get_jump_target() == inst.offset:
                # this is a jump to itself, it is regarded as a NOP, per the documentation at
                # https://devguide.python.org/internals/interpreter/#jumps
                reachable[i] = False
                reachable[i + 1] = True
                continue
            if "IF" in inst.opname:
                # the fallback block is always reachable for conditional jumps
                reachable[i + 1] = True
            else:
                # this is a direct jump, the target is reachable
                # we further check if any other in-between instructions are jump targets
                # if not, we can mark this instruction as unreachable, too
                # later, in-between instructions will be marked as unreachable (NOP)
                # and the interpreter will slide through all the NOP directly to the target
                jump_forwards = [j for j, instruct in enumerate(instructions) if instruct.offset >= inst.get_jump_target()]
                if len(jump_forwards):
                    j = jump_forwards[0]
                    if j > i and all(not instruct.is_jump_target for instruct in instructions[i + 1:j]):
                        reachable[i] = False
        else:
            reachable[i + 1] = True
    
    # mark unreachable instructions as NOP
    new_instructions = [inst if flag else dis.Instruction(
                    opname="NOP",
                    opcode=dis.opmap["NOP"],
                    arg=0,
                    argval=0,
                    argrepr="",
                    offset=inst.offset,
                    starts_line=inst.starts_line,
                    is_jump_target=False,
                ) for inst, flag in zip(instructions, reachable)]
    return new_instructions


def add_indentation(code: str, indentation: int = 4) -> str:
    """Add indentation to code."""
    return "".join(" " * indentation + line + "\n" for line in code.splitlines())

def remove_indentation(code: str, indentation: int = 4) -> str:
    """Remove indentation from code."""
    return "".join(line[indentation:] + "\n" for line in code.splitlines())

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
