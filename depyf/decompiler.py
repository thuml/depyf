"""A simple program to transform bytecode into more readable source code."""

import sys
import os
import dis
from types import CodeType
from typing import List, Tuple, Dict, Union, Callable, Optional
import dataclasses
import inspect
import functools
from collections import defaultdict


from .block import BasicBlock, IndentationBlock
from .patch import *
from .code_transform import (
    nop_unreachable_bytecode,
    add_indentation,
    remove_indentation,
    remove_some_temp,
)
from .utils import (
    get_function_signature,
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
class Decompiler:
    """A decompiler for a code object."""
    code: CodeType
    temp_count: int = 0
    temp_prefix: str = "__temp_"
    blocks: List[BasicBlock] = dataclasses.field(default_factory=list)
    blocks_index_map: Dict[str, int] = dataclasses.field(default_factory=dict)
    blocks_decompiled: Dict[str, bool] = dataclasses.field(default_factory=dict)

    def __init__(self, code: Union[CodeType, Callable]):
        if callable(code):
            code = code.__code__
        self.code = code
        instructions = list(dis.get_instructions(code))
        self.instructions = nop_unreachable_bytecode(instructions)
        # supported_opnames = self.supported_opnames()
        # for inst in self.instructions:
        #     if inst.opname not in supported_opnames:
        #         raise NotImplementedError(f"Unsupported instruction: {inst.opname}")
        self.blocks = BasicBlock.decompose_basic_blocks(self.instructions)
        self.blocks_index_map = {str(block): idx for idx, block in enumerate(self.blocks)}
        self.blocks_decompiled = {str(block): False for block in self.blocks}

    def visualize_cfg(self, filepath: str="cfg.png"):
        BasicBlock.to_graph(self.blocks, filepath=filepath)

    def get_indentation_block(self, starting_block: BasicBlock) -> IndentationBlock:
        """Get the indentation block that contains the starting block.
        An indentation block is a block that is indented, e.g. if-else, while, for, etc.
        Basic blocks in this indentation block are never targeted from outside blocks, and they can only jump to internal blocks or the next basic block only.
        """
        start_index = self.blocks_index_map[str(starting_block)]
        running_index = start_index + 1
        while running_index < len(self.blocks):
            left_blocks = self.blocks[start_index: running_index]
            next_index = running_index
            for left_block in left_blocks:
                if left_block.to_blocks:
                    # this is jmp instruction
                    to_block = left_block.jump_to_block
                    block_index = self.blocks_index_map[str(to_block)]
                    next_index = max(next_index, block_index + 1)
                    if left_block.end_with_if_jmp:
                        # this is a conditional jmp, we also need to consider the fallthrough block
                        fallthrough_block = left_block.fallthrough_block
                        block_index = self.blocks_index_map[str(fallthrough_block)]
                        next_index = max(next_index, block_index + 1)
                if left_block.from_blocks:
                    for from_block in left_block.from_blocks:
                        block_index = self.blocks_index_map[str(from_block)]
                        next_index = max(next_index, block_index + 1)
            if next_index == running_index:
                break
            running_index = next_index
        return IndentationBlock(blocks=self.blocks[start_index: running_index])

    def get_temp_name(self):
        self.temp_count += 1
        return f"{self.temp_prefix}{self.temp_count}"

    @staticmethod
    def supported_opnames():
        return get_supported_opnames(Decompiler.decompile_block.__code__)

    @functools.lru_cache(maxsize=None)
    def decompile(self, indentation=4, temp_prefix: str="__temp_"):
        self.temp_prefix = temp_prefix
        header = get_function_signature(self.code)
        source_code, stack = self.decompile_blocks(self.blocks, [], indentation)
        # source_code = remove_indentation(source_code, indentation)
        source_code = remove_some_temp(source_code, self.temp_prefix, indentation)
        # the header might have invalid function name in torchdynamo. only optimize the function body.
        source_code = header + add_indentation(source_code, indentation)
        return source_code

    def decompile_blocks(
            self,
            blocks: List[BasicBlock],
            stack: List[str],
            indentation: int=4,
        ) -> str:
        indentation_blocks = []
        start_index = self.blocks_index_map[str(blocks[0])]
        end_index = self.blocks_index_map[str(blocks[-1])] + 1
        while start_index < end_index:
            indentation_block = self.get_indentation_block(self.blocks[start_index])
            indentation_blocks.append(indentation_block)
            start_index = self.blocks_index_map[str(indentation_block.blocks[-1])] + 1

        source_code = ""
        for indentation_block in indentation_blocks:
            block_code, stack = self.decompile_indentation_block(indentation_block, stack.copy(), indentation)
            source_code += block_code
        return source_code, stack

    def decompile_indentation_block(
            self,
            indentation_block: IndentationBlock,
            stack: List[str],
            indentation: int=4,
        ) -> Tuple[str, List[str]]:
        """Decompile an indentation block into source code.
        This function is responsible to handle if-else, while, for, etc."""
        source_code, stack = self.decompile_block(indentation_block.blocks[0], stack.copy(), indentation, indentation_block)
        has_loop = False
        for block in indentation_block.blocks[0].from_blocks:
            block_index = self.blocks_index_map[str(block)]
            if block_index >= self.blocks_index_map[str(indentation_block.blocks[0])] and block_index <= self.blocks_index_map[str(indentation_block.blocks[-1])]:
                # this is a loop
                has_loop = True
                break

        # split blocks into if-else
        has_if = "IF" in indentation_block.blocks[0].instructions[-1].opname
        new_stack = stack.copy()
        if has_if:
            jump_to_block = max(indentation_block.blocks[0].to_blocks, key=lambda x: x.code_start)
            if_blocks = [block for block in indentation_block.blocks[1:] if block.code_start < jump_to_block.code_start]
            else_blocks = [block for block in indentation_block.blocks[1:] if block.code_start >= jump_to_block.code_start]
            block_code, new_stack = self.decompile_blocks(if_blocks, stack.copy(), indentation)
            source_code += add_indentation(block_code, indentation)
            if else_blocks:
                block_code, _ = self.decompile_blocks(else_blocks, stack.copy(), indentation)
                source_code += f"else:\n" + add_indentation(block_code, indentation)
            else:
                source_code += f"else:\n" + add_indentation("pass\n" if not has_loop else "break\n", indentation)
        if has_loop:
            source_code = "while True:\n" + add_indentation(source_code, indentation)

        return source_code, new_stack

    def __hash__(self):
        return hash(self.code)

    def decompile_block(
            self,
            block: BasicBlock,
            stack: List[str],
            indentation: int=4,
            indentation_block: Optional[IndentationBlock] = None,
        ) -> str:
        """Decompile a basic block into source code.
        The `stack` holds expressions in string, like "3 + 4".
        """
        source_code = ""
        # source_code += "=" * 40 + "Basic Block Start" + "=" * 40 + "\n"
        # for inst in block.instructions:
        #     source_code += f"{inst.offset} {inst.opname} {inst.argval} ({inst.argrepr})\n"
        # source_code += "=" * 40 + "Basic Block End" + "=" * 40 + "\n"
        # return source_code, stack

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
                jump_block = block.jump_to_block
                fallthrough_block = block.fallthrough_block
                cond = stack[-1]
                fallthrough_stack = stack.copy()[:-1]

                # POP_AND_JUMP / JUMP_OR_POP
                if "POP_JUMP" in inst.opname:
                    jump_stack = stack.copy()[:-1]
                elif "OR_POP" in inst.opname:
                    jump_stack = stack.copy()

                # JUMP_IF_X, so fallthrough if not X
                if "IF_FALSE" in inst.opname:
                    source_code += f"if {cond}:\n"
                elif "IF_TRUE" in inst.opname:
                    source_code += f"if not {cond}:\n"
                elif "IF_NOT_NONE" in inst.opname:
                    source_code += f"if {cond} is None:\n"
                elif "IF_NONE" in inst.opname:
                    source_code += f"if {cond} is not None:\n"
                
                stack = jump_stack
            elif inst.opname in ["BREAK_LOOP"]:
                source_code += "break\n"
            elif inst.opname in ["JUMP_FORWARD", "JUMP_ABSOLUTE", "JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"]:
                jump_offset = inst.get_jump_target()
                if jump_offset > inst.offset:
                    source_code += "break\n"
                else:
                    source_code += "continue\n"
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

        return source_code, stack
