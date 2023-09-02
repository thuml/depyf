"""A simple program to transform bytecode into more readable source code."""

import ast
import sys
import os
import dis
from types import CodeType
from typing import List, Tuple, Dict, Union, Callable, Optional
import dataclasses
import inspect
import functools
from collections import defaultdict

import astor

from .patch import *
from .utils import (
    nop_unreachable_bytecode,
    add_indentation,
    remove_indentation,
    generate_dot_table,
)


@functools.lru_cache(maxsize=None)
def get_supported_opnames(code: CodeType):
    args = code.co_consts
    opnames = []
    for arg in args:
        if isinstance(arg, str):
            opnames.append(arg)
        elif isinstance(arg, tuple):
            opnames += list(arg)
    opnames = set(opnames) & set(dis.opmap.keys())
    return list(opnames)


@dataclasses.dataclass
class BasicBlock:
    """A basic block without internal control flow. The bytecode in this block is executed sequentially.
    The block ends with a jump instruction or a return instruction."""
    instructions: List[dis.Instruction]
    to_blocks: List['BasicBlock'] = dataclasses.field(default_factory=list)
    from_blocks: List['BasicBlock'] = dataclasses.field(default_factory=list)

    def __init__(self, instructions: List[dis.Instruction]):
        self.instructions = instructions
        self.to_blocks = []
        self.from_blocks = []

    def code_range(self):
        return (self.code_start(), self.code_end())

    def code_start(self) -> int:
        return self.instructions[0].offset

    def code_end(self) -> int:
        return self.instructions[-1].offset + 2

    def __str__(self):
        return f"{self.code_range()}"

    def __repr__(self):
        lines = []
        for inst in self.instructions:
            line = [">>" if inst.is_jump_target else "  ", str(inst.offset), inst.opname, str(inst.argval), f"({inst.argrepr})"]
            lines.append(line)
        return generate_dot_table(f"{self.code_range()}", lines)

    def __eq__(self, other):
        return self.code_range() == other.code_range()

    def jump_to_block(self, offset: int) -> 'BasicBlock':
        return [b for b in self.to_blocks if b.code_start() == offset][0]

    @staticmethod
    def find_the_first_block(blocks: List['BasicBlock'], offset: int) -> Optional['BasicBlock']:
        candidates = [b for b in blocks if b.code_start() >= offset]
        if not candidates:
            return None
        return min(candidates, key=BasicBlock.code_start)

    @staticmethod
    def decompose_basic_blocks(insts: List[dis.Instruction]) -> List['BasicBlock']:
        """Decompose a list of instructions into basic blocks without internal control flow."""
        block_starts = {0, insts[-1].offset + 2}
        jumps = set(dis.hasjabs) | set(dis.hasjrel)
        for i, inst in enumerate(insts):
            if inst.opcode in jumps:
                # both jump target and the instruction after the jump are block starts
                block_starts.add(inst.get_jump_target())
                block_starts.add(inst.offset + 2)
            elif inst.opname == "RETURN_VALUE":
                # the instruction after the return is a block start
                block_starts.add(inst.offset + 2)
            # the instruction is the target of a jump
            if inst.is_jump_target:
                block_starts.add(inst.offset)
        block_starts = sorted(block_starts)
        # split into basic blocks
        blocks = []
        for start, end in zip(block_starts[:-1], block_starts[1:]):
            block_insts = [inst for inst in insts if start <= inst.offset and inst.offset < end]
            blocks.append(BasicBlock(block_insts))
        # connect basic blocks
        for block in blocks:
            last_inst = block.instructions[-1]
            if last_inst.opcode in jumps:
                to_block = BasicBlock.find_the_first_block(blocks, last_inst.get_jump_target())
                if to_block:
                    block.to_blocks.append(to_block)
                    to_block.from_blocks.append(block)
                # this is a conditional jump, the fallthrough block is also reachable
                if "IF" in last_inst.opname:
                    fallthrough_block = BasicBlock.find_the_first_block(blocks, last_inst.offset + 2)
                    if not fallthrough_block:
                        # this is a jump to the end of the function, we don't need to connect it
                        continue
                    block.to_blocks.append(fallthrough_block)
                    fallthrough_block.from_blocks.append(block)
            elif last_inst.opname != "RETURN_VALUE":
                fallthrough_block = BasicBlock.find_the_first_block(blocks, last_inst.offset + 2)
                if not fallthrough_block:
                    # this is a jump to the end of the function, we don't need to connect it
                    continue
                block.to_blocks.append(fallthrough_block)
                fallthrough_block.from_blocks.append(block)
        for block in blocks:
            block.to_blocks.sort(key=BasicBlock.code_start)
            block.from_blocks.sort(key=BasicBlock.code_start)
        return blocks


@dataclasses.dataclass(frozen=True)
class IndentationBlock:
    """An indentation block consists several basic blocks. It represents any block that is indented,
     e.g. if-else, while, for, etc."""
    blocks: List[BasicBlock]

    def __bool__(self):
        return bool(self.blocks)

    @property
    def start(self) -> Optional[int]:
        if not self.blocks:
            return None
        return self.blocks[0].code_start()

    @property
    def end(self) -> Optional[int]:
        if not self.blocks:
            return None
        return self.blocks[-1].code_end()


@dataclasses.dataclass
class Decompiler:
    """A decompiler for a code object."""
    code: CodeType
    temp_count: int = 0
    temp_prefix: str = "__temp_"
    blocks: List[BasicBlock] = dataclasses.field(default_factory=list)
    blocks_map: Dict[str, BasicBlock] = dataclasses.field(default_factory=dict)
    blocks_decompiled: Dict[str, bool] = dataclasses.field(default_factory=dict)

    def __init__(self, code: Union[CodeType, Callable]):
        if callable(code):
            code = code.__code__
        self.code = code
        instructions = list(dis.get_instructions(code))
        self.instructions = nop_unreachable_bytecode(instructions)
        supported_opnames = self.supported_opnames()
        for inst in self.instructions:
            if inst.opname not in supported_opnames:
                raise NotImplementedError(f"Unsupported instruction: {inst.opname}")
        self.blocks = BasicBlock.decompose_basic_blocks(self.instructions)
        self.blocks_map = {str(block): block for block in self.blocks}
        self.blocks_decompiled = {str(block): False for block in self.blocks}

    def visualize_cfg(self):
        import networkx as nx

        cfg = nx.DiGraph()

        for block in self.blocks:
            cfg.add_node(str(block), label=repr(block), shape="none")
        for blocka, blockb in zip(self.blocks[:-1], self.blocks[1:]):
            if blockb not in blocka.to_blocks:
                cfg.add_edge(str(blocka), str(blockb), weight=100, style="invis")
        for block in self.blocks:
            for to_block in block.to_blocks:
                cfg.add_edge(str(block), str(to_block))
        import pygraphviz as pgv
        cfg = nx.nx_agraph.to_agraph(cfg)
        cfg.node_attr['style'] = 'rounded'
        # cfg.node_attr['fillcolor'] = '#c0e4f0'
        cfg.node_attr['halign'] = 'left'
        cfg.layout(prog="dot")  # Use dot layout
        cfg.draw("output.png")  # Save to a file
        from matplotlib import pyplot as plt
        plt.imshow(plt.imread("output.png"))
        plt.axis('off')
        plt.show()

    def get_function_signature(self) -> str:
        code_obj: CodeType = self.code
        # Extract all required details from the code object
        arg_names = code_obj.co_varnames[:code_obj.co_argcount]
        args_str = ', '.join(arg_names)
        header = f"def {code_obj.co_name}({args_str}):\n"
        return header

    def get_loop_body(self, starting_block: BasicBlock) -> IndentationBlock:
        end_blocks = [block for block in starting_block.from_blocks if block.code_end() >= starting_block.code_end()]
        if not end_blocks:
            # not a loop back edge
            return IndentationBlock([])
        # loop end block is the largest block looping back to the starting block
        end_block = max(end_blocks, key=BasicBlock.code_end)
        loop_body_blocks = [block for block in self.blocks if starting_block.code_start() <= block.code_start() and block.code_end() <= end_block.code_end()]
        return IndentationBlock(blocks=loop_body_blocks)


    def get_temp_name(self):
        self.temp_count += 1
        return f"{self.temp_prefix}{self.temp_count}"

    @staticmethod
    def supported_opnames():
        return get_supported_opnames(Decompiler.decompile_block.__code__)

    def simplify_code(self, source_code: str, indentation: int=4) -> str:
        tree = ast.parse(source_code)

        temp_occurrences = defaultdict(list)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id.startswith(self.temp_prefix):
                temp_occurrences[node.id].append(node)
        
        class RemoveAssignmentTransformer(ast.NodeTransformer):
            def __init__(self, temp_name: str):
                # optimize one temp_name at a time
                self.temp_name = temp_name
            def visit_Assign(self, node):
                # single assimngment like `temp = xxx`
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    # the assignment is like `temp = xxx`
                    if name == self.temp_name:
                        if len(temp_occurrences[name]) == 1:
                            return ast.Expr(value=node.value)
                        elif len(temp_occurrences[name]) == 2:
                            # we save the `xxx` here
                            temp_occurrences[name][0] = node.value
                            return None
                return node

        class RemoveAssignment2Transformer(ast.NodeTransformer):
            def __init__(self, temp_name: str):
                # optimize one temp_name at a time
                self.temp_name = temp_name
            def visit_Name(self, node):
                name = node.id
                if name == self.temp_name and len(temp_occurrences[name]) == 2:
                    return temp_occurrences[name][0]
                return node

        for key in temp_occurrences:
            tree = RemoveAssignmentTransformer(key).visit(tree)
            tree = RemoveAssignment2Transformer(key).visit(tree)

        reconstructed_code = astor.to_source(tree, indent_with=" " * indentation)
        return reconstructed_code

    @functools.lru_cache(maxsize=None)
    def decompile(self, indentation=4, temp_prefix: str="__temp_"):
        self.temp_prefix = temp_prefix
        header = self.get_function_signature()
        source_code = ""
        for block in self.blocks:
            if self.blocks_decompiled[str(block)]:
                continue
            self.blocks_decompiled[str(block)] = True
            source_code += self.decompile_block(block, [], indentation, self.get_loop_body(block))
        source_code = remove_indentation(source_code, indentation)
        source_code = self.simplify_code(source_code, indentation)
        # the header might have invalid function name in torchdynamo. only optimize the function body.
        source_code = header + add_indentation(source_code, indentation)
        return source_code

    def __hash__(self):
        return hash(self.code)

    def decompile_block(
            self,
            block: BasicBlock,
            stack: List[str],
            indentation: int=4,
            loop: Optional[IndentationBlock]=None,
        ) -> str:
        """Decompile a basic block into source code.
        The `stack` holds expressions in string, like "3 + 4".
        The `loop` indicates which loop structure the block is in, so that it can decompile jump instructions.
        This function returns the source code of the block, which is already indented..
        """
        loopbody = self.get_loop_body(block)

        source_code = ""
        for inst in block.instructions:
            # ==================== Load Instructions =============================
            if inst.opname in ["LOAD_CONST"]:
                # `inst.argval` is the constant value, we have to use `repr` to get the source code
                stack.append(repr(inst.argval))
            elif inst.opname in ["LOAD_FAST", "LOAD_GLOBAL", "LOAD_DEREF", "LOAD_CLOSURE", "LOAD_NAME", "LOAD_CLASSDEREF"]:
                # `inst.argval` is the variable name, in string
                if "NULL + " in inst.argrepr:
                    # Python 3.11 support
                    stack.append(None)
                stack.append(inst.argval)
            elif inst.opname in ["COPY_FREE_VARS"]:
                # this opcode is used to copy free variables from the outer scope to the closure
                # it affects the frame, but not the stack or the source code
                pass
            elif inst.opname in ["LOAD_ATTR"]:
                stack.append(f"getattr({stack.pop()}, {repr(inst.argval)})")
            elif inst.opname in ["LOAD_METHOD"]:
                stack.append(f"{stack.pop()}.{inst.argval}")
            elif inst.opname in ["LOAD_ASSERTION_ERROR"]:
                stack.append("AssertionError")
            elif inst.opname in ["PUSH_NULL"]:
                # the `None` object is used to represent `NULL` in python bytecode
                stack.append(None)
            # ==================== Store Instructions =============================
            elif inst.opname in ["STORE_FAST", "STORE_GLOBAL", "STORE_DEREF", "STORE_NAME"]:
                source_code += f"{inst.argval} = {stack.pop()}\n"
            elif inst.opname in ["STORE_SUBSCR"]:
                index = stack.pop()
                x = stack.pop()
                value = stack.pop()
                source_code += f"{x}[{index}] = {value}\n"
            elif inst.opname in ["STORE_ATTR"]:
                x = stack.pop()
                value = stack.pop()
                source_code += f"{x}.{inst.argval} = {value}\n"
            # ==================== Del Instructions =============================
            elif inst.opname in ["DELETE_SUBSCR"]:
                index = stack.pop()
                x = stack.pop()
                source_code += f"del {x}[{index}]\n"
            elif inst.opname in ["DELETE_NAME", "DELETE_FAST", "DELETE_GLOBAL", "DELETE_DEREF"]:
                source_code += f"del {inst.argval}\n"
            elif inst.opname in ["DELETE_ATTR"]:
                x = stack.pop()
                source_code += f"del {x}.{inst.argval}\n"
            # ==================== Import Instructions =============================
            elif inst.opname in ["IMPORT_NAME"]:
                # TODO: check multi-level import, e.g. `import a.b.c`
                name = inst.argval.split(".")[0]
                fromlist = stack.pop()
                level = stack.pop()
                source_code += f"{name} = __import__({repr(inst.argval)}, fromlist={fromlist}, level={level})\n"
                stack.append(name)
            elif inst.opname in ["IMPORT_FROM"]:
                name = inst.argval
                module = stack[-1]
                source_code += f"{name} = {module}.{name}\n"
                stack.append(name)
            # ==================== Unary Instructions =============================
            elif inst.opname in ["UNARY_NEGATIVE", "UNARY_POSITIVE", "UNARY_INVERT", "UNARY_NOT"]:
                op = {
                    "UNARY_NEGATIVE": "-",
                    "UNARY_POSITIVE": "+",
                    "UNARY_INVERT": "~",
                    "UNARY_NOT": "not",
                }[inst.opname]
                stack.append(f"({op} {stack.pop()})")
            elif inst.opname in ["GET_LEN"]:
                stack.append(f"len({stack[-1]})")
            # ==================== Binary Instructions =============================
            elif inst.opname in ["BINARY_MULTIPLY", "BINARY_ADD", "BINARY_SUBTRACT", "BINARY_TRUE_DIVIDE", "BINARY_FLOOR_DIVIDE", "BINARY_MODULO", "BINARY_POWER", "BINARY_AND", "BINARY_OR", "BINARY_XOR", "BINARY_LSHIFT", "BINARY_RSHIFT", "BINARY_MATRIX_MULTIPLY"]:
                rhs = stack.pop()
                lhs = stack.pop()
                op = {
                    "BINARY_MULTIPLY": "*",
                    "BINARY_ADD": "+",
                    "BINARY_SUBTRACT": "-",
                    "BINARY_TRUE_DIVIDE": "/",
                    "BINARY_FLOOR_DIVIDE": "//",
                    "BINARY_MODULO": "%",
                    "BINARY_POWER": "**",
                    "BINARY_AND": "&",
                    "BINARY_OR": "|",
                    "BINARY_XOR": "^",
                    "BINARY_LSHIFT": "<<",
                    "BINARY_RSHIFT": ">>",
                    "BINARY_MATRIX_MULTIPLY": "@",
                }[inst.opname]
                stack.append(f"({lhs} {op} {rhs})")
            elif inst.opname in ["BINARY_SUBSCR"]:
                rhs = stack.pop()
                lhs = stack.pop()
                stack.append(f"{lhs}[{rhs}]")
            # ==================== Binary Inplace Instructions =============================
            elif inst.opname in ["INPLACE_MULTIPLY", "INPLACE_ADD", "INPLACE_SUBTRACT", "INPLACE_TRUE_DIVIDE", "INPLACE_FLOOR_DIVIDE", "INPLACE_MODULO", "INPLACE_POWER", "INPLACE_AND", "INPLACE_OR", "INPLACE_XOR", "INPLACE_LSHIFT", "INPLACE_RSHIFT", "INPLACE_MATRIX_MULTIPLY"]:
                rhs = stack.pop()
                lhs = stack.pop()
                op = {
                    "INPLACE_MULTIPLY": "*",
                    "INPLACE_ADD": "+",
                    "INPLACE_SUBTRACT": "-",
                    "INPLACE_TRUE_DIVIDE": "/",
                    "INPLACE_FLOOR_DIVIDE": "//",
                    "INPLACE_MODULO": "%",
                    "INPLACE_POWER": "**",
                    "INPLACE_AND": "&",
                    "INPLACE_OR": "|",
                    "INPLACE_XOR": "^",
                    "INPLACE_LSHIFT": "<<",
                    "INPLACE_RSHIFT": ">>",
                    "INPLACE_MATRIX_MULTIPLY": "@",
                }[inst.opname]
                source_code += f"{lhs} {op}= {rhs}\n"
                stack.append(lhs)
            elif inst.opname in ["BINARY_OP"]:
                rhs = stack.pop()
                lhs = stack.pop()
                if "=" in inst.argrepr:
                    source_code += f"{lhs} {inst.argrepr} {rhs}\n"
                    stack.append(lhs)
                else:
                    stack.append(f"({lhs} {inst.argrepr} {rhs})")
            # ==================== Conditional Test Instructions =============================
            elif inst.opname in ["COMPARE_OP"]:
                rhs = stack.pop()
                lhs = stack.pop()
                stack.append(f"({lhs} {inst.argval} {rhs})")
            elif inst.opname in ["IS_OP"]:
                rhs = stack.pop()
                lhs = stack.pop()
                op = "is" if inst.argval == 0 else "is not"
                stack.append(f"({lhs} {op} {rhs})")
            elif inst.opname in ["CONTAINS_OP"]:
                rhs = stack.pop()
                lhs = stack.pop()
                op = "in" if inst.argval == 0 else "not in"
                stack.append(f"({lhs} {op} {rhs})")
            # ==================== Control Flow Instructions =============================
            # "POP_EXCEPT"/"RERAISE"/"WITH_EXCEPT_START"/"JUMP_IF_NOT_EXC_MATCH"/"SETUP_FINALLY"/"CHECK_EG_MATCH"/"PUSH_EXC_INFO"/"PREP_RERAISE_STAR"/"BEGIN_FINALLY"/"END_FINALLY"/"WITH_CLEANUP_FINISH"/"CALL_FINALLY"/"POP_FINALLY"/"WITH_CLEANUP_START"/"SETUP_EXCEPT"/"CHECK_EXC_MATCH" is unsupported, this means we don't support try-except/try-finally
            # "FOR_ITER"/"GET_ITER"/"CONTINUE_LOOP"/ is unsupported, this means we don't support for loop
            # "GET_AWAITABLE"/"GET_AITER"/"GET_ANEXT"/"END_ASYNC_FOR"/"BEFORE_ASYNC_WITH"/"SETUP_ASYNC_WITH"/"SEND"/"ASYNC_GEN_WRAP" are unsupported, this means we don't support async/await
            elif inst.opname in [
                "POP_JUMP_IF_TRUE", "POP_JUMP_FORWARD_IF_TRUE", "POP_JUMP_BACKWARD_IF_TRUE",
                "POP_JUMP_IF_FALSE", "POP_JUMP_FORWARD_IF_FALSE", "POP_JUMP_BACKWARD_IF_FALSE",
                "POP_JUMP_FORWARD_IF_NOT_NONE", "POP_JUMP_BACKWARD_IF_NOT_NONE",
                "POP_JUMP_FORWARD_IF_NONE", "POP_JUMP_BACKWARD_IF_NONE",
                "JUMP_IF_TRUE_OR_POP", "JUMP_IF_FALSE_OR_POP"
                ]:
                jump_offset = inst.get_jump_target()
                fallthrough_offset = inst.offset + 2
                jump_block = block.jump_to_block(jump_offset)
                fallthrough_block = block.jump_to_block(fallthrough_offset)
                cond = stack[-1]
                fallthrough_stack = stack.copy()[:-1]

                # POP_AND_JUMP / JUMP_OR_POP
                if "POP_JUMP" in inst.opname:
                    jump_stack = stack.copy()[:-1]
                elif "OR_POP" in inst.opname:
                    jump_stack = stack.copy()

                if self.blocks_decompiled[str(jump_block)] and self.blocks_decompiled[str(fallthrough_block)]:
                    # both blocks are already decompiled
                    continue

                # JUMP_IF_X, so fallthrough if not X
                if "IF_FALSE" in inst.opname:
                    source_code += f"if {cond}:\n"
                elif "IF_TRUE" in inst.opname:
                    source_code += f"if not {cond}:\n"
                elif "IF_NOT_NONE" in inst.opname:
                    source_code += f"if {cond} is None:\n"
                elif "IF_NONE" in inst.opname:
                    source_code += f"if {cond} is not None:\n"
                
                source_code += self.decompile_block(fallthrough_block, fallthrough_stack, indentation, loopbody if loopbody else loop)
                self.blocks_decompiled[str(fallthrough_block)] = True

                if fallthrough_block.instructions[-1].opcode in (dis.hasjabs + dis.hasjabs):
                    source_code += "else:\n"
                    if not loopbody or jump_block.code_end()  <= loopbody.end:
                        source_code += self.decompile_block(jump_block, jump_stack, indentation, loopbody if loopbody else loop)
                        self.blocks_decompiled[str(jump_block)] = True
                    else:
                        source_code += add_indentation("break\n", indentation)
                else:
                        code = self.decompile_block(jump_block, jump_stack, indentation, loopbody if loopbody else loop)
                        self.blocks_decompiled[str(jump_block)] = True
                        source_code += remove_indentation(code, indentation)

                if loopbody and loopbody.start == block.code_start():
                    source_code = "while True:\n" + add_indentation(source_code, indentation)
            elif inst.opname in ["BREAK_LOOP"]:
                source_code += "break\n"
            elif inst.opname in ["JUMP_FORWARD", "JUMP_ABSOLUTE", "JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"]:
                jump_offset = inst.get_jump_target()
                if loop.start is not None and jump_offset == loop.start:
                    source_code += "continue\n"
                elif loop.end is not None and jump_offset >= loop.end:
                    source_code += "break\n"
                else:
                    if loopbody and jump_offset == loopbody.start:
                        source_code = "while True:\n" + add_indentation(source_code, indentation)
                        return source_code
                    if jump_offset > block.code_start():
                        jump_block = block.jump_to_block(jump_offset)
                        source_code += self.decompile_block(jump_block, stack.copy(), indentation, loopbody if loopbody else loop)
                        self.blocks_decompiled[str(jump_block)] = True
                    else:
                        # this code should be decompiled elsewhere?
                        pass
                        # raise NotImplementedError(f"Unsupported jump backward")
            elif inst.opname in ["RETURN_VALUE"]:
                source_code += f"return {stack[-1]}\n"
            elif inst.opname in ["YIELD_VALUE"]:
                source_code += f"yield {stack[-1]}\n"
            elif inst.opname in ["RETURN_GENERATOR"]:
                # we don't handle generator/coroutine, add this to support simple yield
                stack.append(None)
            elif inst.opname in ["GEN_START"]:
                # stack.pop()
                assert inst.argval == 0, "Only generator expression is supported"
            # ==================== Function Call Instructions =============================
            elif inst.opname in ["KW_NAMES"]:
                names = self.code.co_consts[inst.arg]
                stack.append(repr(names))
            elif inst.opname in ["CALL"]:
                last_inst = [x for x in block.instructions if x.offset < inst.offset][-1]
                has_kw_names = last_inst.opname == "KW_NAMES"
                kw_names = tuple()
                if has_kw_names:
                    kw_names = eval(stack.pop())
                args = [(stack.pop()) for _ in range(inst.argval)]
                args = args[::-1]
                pos_args = args[:len(args) - len(kw_names)]
                kwargs = args[len(args) - len(kw_names):]
                kwcalls = []
                for name, value in zip(kwargs, kw_names):
                    kwcalls.append(f"{name}={value}")
                func = stack.pop()
                if stack and stack[-1] is None:
                    stack.pop()
                temp = self.get_temp_name()
                source_code += f"{temp} = {func}({', '.join(pos_args + kwcalls)})\n"
                stack.append(temp)
            elif inst.opname in ["CALL_FUNCTION", "CALL_METHOD"]:
                args = [(stack.pop()) for _ in range(inst.argval)]
                args = args[::-1]
                func = stack.pop()
                temp = self.get_temp_name()
                source_code += f"{temp} = {func}({', '.join(args)})\n"
                stack.append(temp)
            elif inst.opname in ["CALL_FUNCTION_KW"]:
                kw_args = eval(stack.pop())
                kwcalls = []
                for name in kw_args:
                    kwcalls.append(f"{name}={stack.pop()}")
                pos_args = [(stack.pop()) for _ in range(inst.argval - len(kw_args))]
                pos_args = pos_args[::-1]
                func = stack.pop()
                temp = self.get_temp_name()
                source_code += f"{temp} = {func}({', '.join(pos_args + kwcalls)})\n"
                stack.append(temp)
            elif inst.opname in ["CALL_FUNCTION_EX"]:
                if inst.argval == 0:
                    args = stack.pop()
                    func = stack.pop()
                    temp = self.get_temp_name()
                    source_code += f"{temp} = {func}(*{args})\n"
                    stack.append(temp)
                elif inst.argval == 1:
                    kw_args = stack.pop()
                    args = stack.pop()
                    func = stack.pop()
                    temp = self.get_temp_name()
                    source_code += f"{temp} = {func}(*{args}, **{kw_args})\n"
                    stack.append(temp)
            # ==================== Container Related Instructions (tuple, list, set, dict) =============================
            # "SET_ADD"/"MAP_ADD"/"LIST_APPEND" are unsupported, this means we cannot use list/set/map comprehension
            elif inst.opname in ["UNPACK_SEQUENCE"]:
                varname = stack.pop()
                for i in range(inst.argval):
                    stack.append(f"{varname}[{inst.argval - 1 - i}]")
            elif inst.opname in ["UNPACK_EX"]:
                varname = stack.pop()
                stack.append(f"list({varname}[{inst.argval}:])")
                for i in range(inst.argval):
                    stack.append(f"{varname}[{inst.argval - 1 - i}]")
            elif inst.opname in ["BUILD_SLICE"]:
                tos = stack.pop()
                tos1 = stack.pop()
                if inst.argval == 2:
                    stack.append(f"slice({tos1}, {tos})")
                elif inst.argval == 3:
                    tos2 = stack.pop()
                    stack.append(f"slice({tos2}, {tos1}, {tos})")
            elif inst.opname in ["BUILD_TUPLE", "BUILD_TUPLE_UNPACK", "BUILD_TUPLE_UNPACK_WITH_CALL"]:
                args = [stack.pop() for _ in range(inst.argval)]
                args = args[::-1]
                if "UNPACK" in inst.opname:
                    args = [f"*{arg}" for arg in args]
                if inst.argval == 1:
                    stack.append(f"({args[0]},)")
                else:
                    stack.append(f"({', '.join(args)})")
            elif inst.opname in ["BUILD_LIST", "BUILD_LIST_UNPACK"]:
                args = [stack.pop() for _ in range(inst.argval)]
                args = args[::-1]
                if "UNPACK" in inst.opname:
                    args = [f"*{arg}" for arg in args]
                stack.append(f"[{', '.join(args)}]")
            elif inst.opname in ["BUILD_SET", "BUILD_SET_UNPACK"]:
                if inst.argval == 0:
                    stack.append("set()")
                else:
                    args = [stack.pop() for _ in range(inst.argval)]
                    args = args[::-1]
                    if "UNPACK" in inst.opname:
                        args = [f"*{arg}" for arg in args]
                    stack.append(f"{{{', '.join(args)}}}")
            elif inst.opname in ["BUILD_MAP_UNPACK", "BUILD_MAP_UNPACK_WITH_CALL"]:
                if inst.argval == 0:
                    stack.append("dict()")
                else:
                    args = [stack.pop() for _ in range(inst.argval)]
                    args = args[::-1]
                    args = [f"**{arg}" for arg in args]
                    stack.append(f"{{{', '.join(args)}}}")
            elif inst.opname in ["BUILD_MAP"]:
                args = [stack.pop() for _ in range(inst.argval * 2)]
                args = args[::-1]
                keys = args[::2]
                values = args[1::2]
                stack.append(f"{{{', '.join([f'{k}: {v}' for k, v in zip(keys, values)])}}}")
            elif inst.opname in ["BUILD_CONST_KEY_MAP"]:
                keys = eval(stack.pop())
                args = [stack.pop() for _ in range(inst.argval)]
                values = args[::-1]
                stack.append(f"{{{', '.join([f'{k}: {v}' for k, v in zip(keys, values)])}}}")
            elif inst.opname in ["BUILD_STRING"]:
                args = [stack.pop() for _ in range(inst.argval)]
                args = args[::-1]
                values = " + ".join(args)
                stack.append(values)
            elif inst.opname in ["LIST_TO_TUPLE"]:
                item = stack.pop()
                stack.append(f"tuple({item})")
            elif inst.opname in ["LIST_EXTEND"]:
                assert inst.argval == 1, "Only tested for argval==1"
                values = stack.pop()
                temp = self.get_temp_name()
                x = stack.pop()
                source_code += f"{temp} = {x}\n"
                source_code += f"{temp}.extend({values})\n"
                stack.append(temp)
            elif inst.opname in ["SET_UPDATE", "DICT_UPDATE", "DICT_MERGE"]:
                assert inst.argval == 1, "Only tested for argval==1"
                values = stack.pop()
                temp = self.get_temp_name()
                x = stack.pop()
                source_code += f"{temp} = {x}\n"
                source_code += f"{temp}.update({values})\n"
                stack.append(temp)
            # ==================== Misc Instructions =============================
            elif inst.opname in ["RAISE_VARARGS"]:
                if inst.argval == 0:
                    source_code += "raise\n"
                elif inst.argval == 1:
                    source_code += f"raise {stack.pop()}\n"
                elif inst.argval == 2:
                    tos = stack.pop()
                    tos1 = stack.pop()
                    source_code += f"raise {tos1} from {tos}\n"
            elif inst.opname in ["GET_ITER"]:
                raise NotImplementedError(f"Unsupported instruction: {inst.opname}")
                # "GET_YIELD_FROM_ITER" is not supported
                stack.append(f"iter({stack.pop()})")
            elif inst.opname in ["FORMAT_VALUE"]:
                func, spec = inst.argval
                if spec:
                    form_spec = stack.pop()
                    value = stack.pop()
                    stack.append(f"format({value}, {form_spec})")
                else:
                    value = stack.pop()
                    func = str if func is None else func
                    stack.append(f"{func.__name__}({value})")
            elif inst.opname in ["ROT_N", "ROT_TWO", "ROT_THREE", "ROT_FOUR"]:
                if inst.opname == "ROT_N":
                    n = inst.argval
                else:
                    n = {
                        "ROT_TWO": 2,
                        "ROT_THREE": 3,
                        "ROT_FOUR": 4,
                    }[inst.opname]
                values = stack[-n:]
                values = [values[-1]] + values[:-1]
                stack[-n:] = values
            elif inst.opname in ["SWAP"]:
                # not tested, don't know how to generate this instruction
                n = inst.argval
                tos = stack[-1]
                value = stack[-1 - n]
                tos, value = value, tos
                stack[-1] = tos
                stack[-1 - n] = value
            elif inst.opname in ["COPY"]:
                # not tested, don't know how to generate this instruction
                n = inst.argval
                value = stack[-1 - n]
                stack.append(value)
            elif inst.opname in ["NOP", "RESUME", "SETUP_LOOP", "POP_BLOCK", "PRECALL", "EXTENDED_ARG"]:
                # "EXTENDED_ARG" is treated as NOP here, because it has been handled by `dis.get_instructions`.
                # The extended args are already merged into the following instruction's `inst.argval`.
                continue
            elif inst.opname in ["POP_TOP"]:
                stack.pop()
            elif inst.opname in ["DUP_TOP"]:
                # not tested
                stack.append(stack[-1])
            elif inst.opname in ["DUP_TOP_TWO"]:
                # not tested
                tos = stack[-1]
                tos1 = stack[-2]
                stack.append(tos1)
                stack.append(tos)
            # ==================== Unsupported Misc Instructions =============================
            # "CACHE" is unsupported
            # "MAKE_CELL" is unsupported
            # "MAKE_FUNCTION" is unsupported
            # "PRINT_EXPR"/"COPY_DICT_WITHOUT_KEYS" is I don't know
            # "YIELD_FROM"/"SETUP_ANNOTATIONS" is unsupported
            # "IMPORT_STAR" is unsupported, because we only support bytecode for functions
            # "LOAD_BUILD_CLASS"/"SETUP_WITH"/"BEFORE_WITH" is unsupported
            # "MATCH_MAPPING"/"MATCH_SEQUENCE"/"MATCH_KEYS"/"MATCH_CLASS" is unsupported
            else:
                raise NotImplementedError(f"Unsupported instruction: {inst.opname}")

        source_code = add_indentation(source_code, indentation)
        return source_code
