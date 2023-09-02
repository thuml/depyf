import dis
from typing import List, Tuple, Union, Optional, Callable, Any, Dict, Set
from types import CodeType
import ast
import astor
from collections import defaultdict


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
            if "IF" in inst.opname or "FOR_ITER" in inst.opname:
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


class RemoveAssignmentTransformer(ast.NodeTransformer):
    def __init__(self, temp_name: str, temp_occurrences: Dict[str, List[ast.Name]]):
        # optimize one temp_name at a time
        self.temp_name = temp_name
        self.temp_occurrences = temp_occurrences
    def visit_Assign(self, node):
        # single assimngment like `temp = xxx`
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            # the assignment is like `temp = xxx`
            if name == self.temp_name:
                if len(self.temp_occurrences[name]) == 1:
                    return ast.Expr(value=node.value)
                elif len(self.temp_occurrences[name]) == 2:
                    # we save the `xxx` here
                    self.temp_occurrences[name][0] = node.value
                    return None
        return node

class RemoveAssignment2Transformer(ast.NodeTransformer):
    def __init__(self, temp_name: str, temp_occurrences: Dict[str, List[ast.Name]]):
        # optimize one temp_name at a time
        self.temp_name = temp_name
        self.temp_occurrences = temp_occurrences
    def visit_Name(self, node):
        name = node.id
        if name == self.temp_name and len(self.temp_occurrences[name]) == 2:
            return self.temp_occurrences[name][0]
        return node


def remove_some_temp(source_code: str, temp_prefix:str, indentation: int=4) -> str:
    tree = ast.parse(source_code)

    temp_occurrences = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith(temp_prefix):
            temp_occurrences[node.id].append(node)

    for key in temp_occurrences:
        tree = RemoveAssignmentTransformer(key, temp_occurrences).visit(tree)
        tree = RemoveAssignment2Transformer(key, temp_occurrences).visit(tree)

    reconstructed_code = astor.to_source(tree, indent_with=" " * indentation)
    return reconstructed_code
