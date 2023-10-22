import dis
from typing import List, Tuple, Union, Optional, Callable, Any, Dict, Set
from types import CodeType
import ast
import astor
from collections import defaultdict
import dataclasses
import sys
import hashlib

py311 = sys.version_info >= (3, 11)
all_jump_opcode_set = set(dis.hasjabs) | set(dis.hasjrel)


@dataclasses.dataclass
class Instruction:
    """A mutable version of dis.Instruction"""

    opcode: int
    opname: str
    arg: Optional[int]
    argval: Any
    argrepr: str
    offset: Optional[int] = None
    starts_line: Optional[int] = None
    is_jump_target: bool = False

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def short_inst_repr(self):
        return f"Instruction(opname={self.opname}, offset={self.offset})"

    def is_jump(self):
        return self.opcode in all_jump_opcode_set

    def get_jump_target(self: "Instruction"):
        if self.is_jump() and "to " in self.argrepr:
            return int(self.argrepr.replace("to ", "").strip())
        # seems like a bug, "FOR_ITER" is in `dis.hasjrel`, but its `argval` is an absolute offset
        if self.opcode in dis.hasjabs:
            return self.argval
        elif self.opcode in dis.hasjrel:
            return self.offset + self.argval if not py311 else self.argval
        else:
            raise ValueError(f"Instruction {self.opname} does not have jump target")


def convert_instruction(i: dis.Instruction) -> Instruction:
    return Instruction(
        i.opcode,
        i.opname,
        i.arg,
        i.argval,
        i.argrepr,
        i.offset,
        i.starts_line,
        i.is_jump_target,
    )


def nop_instruction(inst: Instruction):
    """Inplace modify an instruction as nop."""
    inst.opname = "NOP"
    inst.opcode = dis.opmap["NOP"]
    inst.arg = 0
    inst.argval = 0
    inst.argrepr = ""
    inst.offset
    inst.starts_line
    inst.is_jump_target = False
    return inst


def propagate_line_nums(instructions: List[Instruction]):
    """Ensure every instruction has line number set in case some are removed"""
    cur_line_no = None

    def populate_line_num(inst):
        nonlocal cur_line_no
        if inst.starts_line:
            cur_line_no = inst.starts_line

        inst.starts_line = cur_line_no

    for inst in instructions:
        populate_line_num(inst)


def simplify_with_statement(instructions: List[Instruction]):
    """Simplify with statement.
    3.10 with statement:
    SETUP_WITH
    with body
    POP_BLOCK
    some extra code (starts_line == with)
    """
    for i, inst in enumerate(instructions):
        if inst.opname == "SETUP_WITH":
            line_no = inst.starts_line
            pop_blocks = [j for j, _inst in enumerate(instructions) if j > i and _inst.opname == "POP_BLOCK" and instructions[j + 1].starts_line == line_no]
            if pop_blocks:
                pop_block_index = pop_blocks[0]
                nop_instruction(instructions[pop_block_index])
                for _inst in instructions[pop_block_index:]:
                    if _inst.starts_line == line_no:
                        nop_instruction(_inst)


def simplify_finally_statement(instructions: List[Instruction]):
    """Simplify finally statement.
    3.10 finally statement:
    SETUP_FINALLY
    body
    POP_BLOCK
    finally code
    Exception code
    RERAISE
    """
    for i, inst in enumerate(instructions):
        if inst.opname == "SETUP_FINALLY":
            finally_target = inst.get_jump_target()
            reraise_idx = [j for j, _inst in enumerate(instructions) if _inst.offset >= finally_target and _inst.opname == "RERAISE"]
            if reraise_idx:
                reraise_index = reraise_idx[0]
                for j, _inst in enumerate(instructions):
                    if _inst.offset >= finally_target and j <= reraise_index:
                        nop_instruction(_inst)


def nop_unreachable_bytecode(instructions: List[dis.Instruction]) -> List[dis.Instruction]:
    """Mark unreachable bytecode as NOP."""
    jumps = set(dis.hasjabs) | set(dis.hasjrel)

    reachable = [False for x in instructions]
    reachable[0] = True
    # each instruction marks the instruction after it
    for i, inst in enumerate(instructions):
        if inst.is_jump_target:
            # the instruction is the target of a jump
            reachable[i] = True
        # the last instruction does not need to mark any following instructions
        if i == len(instructions) - 1:
            break
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
            if "IF" in inst.opname or "FOR_ITER" in inst.opname or "SETUP_LOOP" in inst.opname:
                # the fallback block is always reachable for conditional jumps
                reachable[i + 1] = True
            elif inst.opname == "SETUP_FINALLY":
                # the finally block is always reachable
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
    for inst, flag in zip(instructions, reachable):
        if not flag:
            nop_instruction(inst)

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
                elif len(self.temp_occurrences[name]) == 3 and isinstance(self.temp_occurrences[name][-1], bool):
                    # we save the `xxx` here
                    self.temp_occurrences[name].append(node.value)
                    if self.temp_occurrences[name][-2]:
                        return None
        return node

class RemoveAssignment2Transformer(ast.NodeTransformer):
    def __init__(self, temp_name: str, temp_occurrences: Dict[str, List[ast.Name]]):
        # optimize one temp_name at a time
        self.temp_name = temp_name
        self.temp_occurrences = temp_occurrences
    def visit_Name(self, node):
        name = node.id
        if name == self.temp_name and len(self.temp_occurrences[name]) == 4 and isinstance(self.temp_occurrences[name][-2], bool):
            if self.temp_occurrences[name][-2]:
                return self.temp_occurrences[name][-1]
        return node

def get_parents(node):
    """Collect all parent nodes of a given node."""
    parents = []
    while node:
        parents.append(node)
        node = getattr(node, "parent", None)
    return parents

def set_parents(node, parent=None):
    """Recursively set the parent attribute for each node."""
    for child in ast.iter_child_nodes(node):
        child.parent = parent
        set_parents(child, child)

def lowest_common_parent(node1, node2):
    """Get the lowest common parent for two nodes."""
    parents1 = get_parents(node1)
    parents2 = get_parents(node2)
    
    # Reverse the parents list to start comparing from the root.
    parents1.reverse()
    parents2.reverse()
    
    last_common = None
    for p1, p2 in zip(parents1, parents2):
        if p1 is p2:
            last_common = p1
        else:
            break
    return last_common, p1, p2

def remove_some_temp(source_code: str, temp_prefix:str, indentation: int=4) -> str:
    tree = ast.parse(source_code)
    set_parents(tree)

    temp_occurrences = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith(temp_prefix):
            temp_occurrences[node.id].append(node)

    for key in temp_occurrences:
        if len(temp_occurrences[key]) == 2:
            node1 = temp_occurrences[key][0]
            node2 = temp_occurrences[key][1]
            parent, parent1, parent2 = lowest_common_parent(node1, node2)
            assignment_node = node1 if isinstance(node1.parent, ast.Assign) else node2
            assignment_parent = parent1 if isinstance(node1.parent, ast.Assign) else parent2
            indentation_nodes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.For, ast.AsyncFor, ast.While, ast.If, ast.Try, ast.With, ast.AsyncWith, ast.ClassDef)
            # we cannot remove the assignment if the assignment `temp=xxx` is in an indentation block while the usage of `temp` is not
            can_merge = not isinstance(assignment_parent, indentation_nodes)
            temp_occurrences[key].append(can_merge)
        tree = RemoveAssignmentTransformer(key, temp_occurrences).visit(tree)
        tree = RemoveAssignment2Transformer(key, temp_occurrences).visit(tree)

    reconstructed_code = astor.to_source(tree, indent_with=" " * indentation)
    return reconstructed_code

class IdentifierReplacer(ast.NodeTransformer):

    # def visit_Name(self, node):
    #     return ast.copy_location(ast.Name(id='PLACEHOLDER', ctx=node.ctx), node)

    def visit_FunctionDef(self, node):
        node.name = 'PLACEHOLDER'
        return self.generic_visit(node)

    # def visit_AsyncFunctionDef(self, node):
    #     node.name = 'PLACEHOLDER'
    #     return self.generic_visit(node)

    # def visit_ClassDef(self, node):
    #     node.name = 'PLACEHOLDER'
    #     return self.generic_visit(node)

    # def visit_Attribute(self, node):
    #     node.attr = 'PLACEHOLDER'
    #     return self.generic_visit(node)

def structure_hash(source_code: str) -> str:
    """Compute the hash of code structure, ignore the function name difference.
    This is because PyTorch dynamically generates function names.
    """
    tree = ast.parse(source_code)
    tree = IdentifierReplacer().visit(tree)
    modified_code = astor.to_source(tree)
    hash_value = hashlib.md5(modified_code.encode()).hexdigest()
    return hash_value
