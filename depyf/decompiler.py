"""A simple program to transform bytecode into more readable source code."""

import sys
import os
import dis
from types import CodeType
from typing import List, Tuple, Dict, Union, Callable, Optional
import dataclasses
import inspect
import functools

from .patch import *


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
        return f"BasicBlock({self.code_range()})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.code_range() == other.code_range()

    def jump_to_block(self, offset: int) -> 'BasicBlock':
        return [b for b in self.to_blocks if b.code_start() == offset][0]

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
                jump_offset = last_inst.get_jump_target()
                fallthrough_offset = last_inst.offset + 2
                to_block = [b for b in blocks if b.code_start() == jump_offset][0]
                fallthrough_block = [b for b in blocks if b.code_start() == fallthrough_offset][0]
                block.to_blocks += [to_block, fallthrough_block]
                to_block.from_blocks.append(block)
                fallthrough_block.from_blocks.append(block)
        return blocks


@dataclasses.dataclass(frozen=True)
class LoopBody:
    """A loop body, the final block will jump back to the first block, with conditions."""
    blocks: List[BasicBlock]

    def __bool__(self):
        return bool(self.blocks)

    @property
    def loop_start(self) -> Optional[int]:
        if not self.blocks:
            return None
        return self.blocks[0].code_start()

    @property
    def loop_end(self) -> Optional[int]:
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

    def __init__(self, code: Union[CodeType, Callable], temp_prefix: str="__temp_"):
        if callable(code):
            code = code.__code__
        self.code = code
        self.temp_prefix = temp_prefix
        self.instructions = list(dis.get_instructions(code))
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
            cfg.add_node(str(block))
        for block in self.blocks:
            for to_block in block.to_blocks:
                cfg.add_edge(str(block), str(to_block))
            for from_block in block.from_blocks:
                cfg.add_edge(str(from_block), str(block))

        pos = nx.spring_layout(cfg)
        nx.draw_networkx_nodes(cfg, pos, node_size=1000)
        nx.draw_networkx_edges(cfg, pos, node_size=1000)
        nx.draw_networkx_labels(cfg, pos)
        from matplotlib import pyplot as plt
        plt.show()

    def get_function_signature(self) -> str:
        code_obj: CodeType = self.code
        # Extract all required details from the code object
        arg_names = code_obj.co_varnames[:code_obj.co_argcount]
        args_str = ', '.join(arg_names)
        header = f"def {code_obj.co_name}({args_str}):\n"
        return header

    def get_loop_body(self, starting_block: BasicBlock) -> LoopBody:
        end_blocks = [block for block in starting_block.from_blocks if block.code_end() >= starting_block.code_end()]
        if not end_blocks:
            # not a loop back edge
            return LoopBody([])
        # loop end block is the largest block looping back to the starting block
        loop_end_block = max(end_blocks, key=BasicBlock.code_end)
        loop_body_blocks = [block for block in self.blocks if starting_block.code_start() <= block.code_start() and block.code_end() <= loop_end_block.code_end()]
        return LoopBody(blocks=loop_body_blocks)


    def get_temp_name(self):
        self.temp_count += 1
        return f"{self.temp_prefix}{self.temp_count}"

    @staticmethod
    def supported_opnames():
        return get_supported_opnames(Decompiler.decompile_block.__code__)

    @functools.lru_cache(maxsize=None)
    def decompile(self, indentation=4):
        header = self.get_function_signature()
        source_code = ""
        for block in self.blocks:
            if self.blocks_decompiled[str(block)]:
                continue
            self.blocks_decompiled[str(block)] = True
            source_code += self.decompile_block(block, [], indentation, self.get_loop_body(block))
        source_code = header + source_code
        return source_code

    def __hash__(self):
        return hash(self.code) + hash(self.temp_prefix)

    def decompile_block(
            self,
            block: BasicBlock,
            stack: List[str],
            indentation: int=4,
            loop: Optional[LoopBody]=None,
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
                stack.append(inst.argval)
            elif inst.opname in ["LOAD_ATTR"]:
                stack.append(f"getattr({stack.pop()}, {repr(inst.argval)})")
            elif inst.opname in ["LOAD_METHOD"]:
                stack.append(f"{stack.pop()}.{inst.argval}")
            elif inst.opname in ["LOAD_ASSERTION_ERROR"]:
                stack.append("AssertionError")
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
            # "POP_BLOCK"/"POP_EXCEPT"/"RERAISE"/"WITH_EXCEPT_START"/"JUMP_IF_NOT_EXC_MATCH"/"SETUP_FINALLY" is unsupported, this means we don't support try-except/try-finally
            # "FOR_ITER"/"GET_ITER" is unsupported, this means we don't support for loop
            # "GET_AWAITABLE"/"GET_AITER"/"GET_ANEXT"/"END_ASYNC_FOR"/"BEFORE_ASYNC_WITH"/"SETUP_ASYNC_WITH" are unsupported, this means we don't support async/await
            elif inst.opname in ["POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE", "JUMP_IF_TRUE_OR_POP", "JUMP_IF_FALSE_OR_POP"]:
                jump_offset = inst.get_jump_target()
                fallthrough_offset = inst.offset + 2
                jump_block = block.jump_to_block(jump_offset)
                fallthrough_block = block.jump_to_block(fallthrough_offset)
                cond = stack[-1]
                fallthrough_stack = stack.copy()[:-1]

                # POP_AND_JUMP / JUMP_OR_POP
                if inst.opname in ["POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE"]:
                    jump_stack = stack.copy()[:-1]
                elif inst.opname in ["JUMP_IF_FALSE_OR_POP", "JUMP_IF_TRUE_OR_POP"]:
                    jump_stack = stack.copy()

                assert not self.blocks_decompiled[str(jump_block)] and not self.blocks_decompiled[str(fallthrough_block)], "Blocks are already decompiled"

                # JUMP_IF_X, so fallthrough if not X
                if inst.opname in ["POP_JUMP_IF_FALSE", "JUMP_IF_FALSE_OR_POP"]:
                    source_code += f"if {cond}:\n"
                elif inst.opname in ["POP_JUMP_IF_TRUE", "JUMP_IF_TRUE_OR_POP"]:
                    source_code += f"if not {cond}:\n"
                
                source_code += self.decompile_block(fallthrough_block, fallthrough_stack, indentation, loopbody if loopbody else loop)
                self.blocks_decompiled[str(fallthrough_block)] = True

                source_code += "else:\n"

                if not loopbody or jump_block.code_end()  <= loopbody.loop_end:
                    source_code += self.decompile_block(jump_block, jump_stack, indentation, loopbody if loopbody else loop)
                    self.blocks_decompiled[str(jump_block)] = True
                else:
                    source_code += " " * indentation + "break\n"

                if loopbody and loopbody.loop_start == block.code_start():
                    source_code = "while True:\n" + "".join([" " * indentation + line + "\n" for line in source_code.splitlines()])

            elif inst.opname in ["JUMP_FORWARD", "JUMP_ABSOLUTE"]:
                jump_offset = inst.get_jump_target()
                if loop.loop_start is not None and jump_offset == loop.loop_start:
                    source_code += "continue\n"
                elif loop.loop_end is not None and jump_offset >= loop.loop_end:
                    source_code += "break\n"
                else:
                    if loopbody and jump_offset == loopbody.loop_start:
                        source_code = "while True:\n" + "".join([" " * indentation + line + "\n" for line in source_code.splitlines()])
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
            elif inst.opname in ["GEN_START"]:
                # stack.pop()
                assert inst.argval == 0, "Only generator expression is supported"
            # ==================== Function Call Instructions =============================
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
            elif inst.opname in ["BUILD_TUPLE"]:
                args = [stack.pop() for _ in range(inst.argval)]
                args = args[::-1]
                if inst.argval == 1:
                    stack.append(f"({args[0]},)")
                else:
                    stack.append(f"({', '.join(args)})")
            elif inst.opname in ["BUILD_LIST"]:
                args = [stack.pop() for _ in range(inst.argval)]
                args = args[::-1]
                stack.append(f"[{', '.join(args)}]")
            elif inst.opname in ["BUILD_SET"]:
                if inst.argval == 0:
                    stack.append("set()")
                else:
                    args = [stack.pop() for _ in range(inst.argval)]
                    args = args[::-1]
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
            elif inst.opname in ["NOP"]:
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
            # "EXTENDED_ARG" is unsupported
            # "MAKE_FUNCTION" is unsupported
            # "PRINT_EXPR"/"COPY_DICT_WITHOUT_KEYS" is I don't know
            # "YIELD_FROM"/"SETUP_ANNOTATIONS" is unsupported
            # "IMPORT_STAR" is unsupported, because we only support bytecode for functions
            # "LOAD_BUILD_CLASS"/"SETUP_WITH" is unsupported
            # "MATCH_MAPPING"/"MATCH_SEQUENCE"/"MATCH_KEYS"/"MATCH_CLASS" is unsupported
            else:
                raise NotImplementedError(f"Unsupported instruction: {inst.opname}")

        source_code = "".join([" " * indentation + line + "\n" for line in source_code.splitlines()])
        return source_code
