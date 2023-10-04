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
import contextlib

from .block import BasicBlock, IndentationBlock
from .code_transform import (
    nop_unreachable_bytecode,
    add_indentation,
    remove_indentation,
    remove_some_temp,
    propagate_line_nums,
    convert_instruction,
    simplify_with_statement,
    Instruction,
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
class BlockResult:
    """The result of a decompiled block.
    The fallthrough stack is the stack after the block is executed.
    The jump stack is the stack before the jump instruction is executed.
    """
    source_code: str
    fallthrough_stack: List[str]
    jump_stack: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class DecompilerState:
    """State of decompiler, keep track of the evaluation stack, as well as the decompiled source code."""
    source_code: str
    stack: list


@dataclasses.dataclass
class Decompiler:
    """A decompiler for a code object."""
    code: CodeType
    temp_count: int = 0
    temp_prefix: str = "__temp_"
    blocks: List[BasicBlock] = dataclasses.field(default_factory=list)
    blocks_index_map: Dict[str, int] = dataclasses.field(default_factory=dict)
    state: DecompilerState = dataclasses.field(default_factory=lambda: DecompilerState(source_code="", stack=[]))
    indentation: int = 4

    @contextlib.contextmanager
    def new_state(self, stack):
        """Create a new state for decompiler."""
        state = DecompilerState(source_code="", stack=stack)
        old_state = self.state
        self.state = state
        yield
        self.state = old_state

# ==================== Load Instructions =============================

    def LOAD_CONST(self, inst: Instruction):
        """Push a constant onto the stack.
        `inst.argval` is the constant value, we have to use `repr` to get the source code
        """
        can_repr = False
        try:
            can_repr = eval(repr(inst.argval)) == inst.argval
        except:
            pass
        if can_repr:
            self.state.stack.append(repr(inst.argval))
        else:
            self.state.stack.append(inst.argval)

    def generic_load(self, inst: Instruction):
        """`inst.argval` is the variable name, in string"""
        if "NULL + " in inst.argrepr:
            # Python 3.11 support
            self.state.stack.append(None)
        self.state.stack.append(inst.argval)

    LOAD_FAST = LOAD_GLOBAL = LOAD_DEREF = LOAD_NAME = LOAD_CLASSDEREF = LOAD_CLOSURE = generic_load

    def MAKE_FUNCTION(self, inst: Instruction):
        qual_name = self.state.stack.pop()
        code = self.state.stack.pop()
        if inst.argval & 0x08:
            # has closure
            self.state.stack.pop()
        if inst.argval & 0x04:
            # has annotations
            self.state.stack.pop()
        kw_defaults = self.state.stack.pop() if inst.argval & 0x02 else {}
        defaults = self.state.stack.pop() if inst.argval & 0x01 else ()
        if len(kw_defaults) or len(defaults):
            print("Function with default arguments is not supported, ignore the default arguments")
        this_index = self.index_of(inst.offset)
        func_name = qual_name
        immediately_used = False
        if self.instructions[this_index + 1].opname == "STORE_FAST":
            # the function is immediately stored in a variable, use that variable name
            func_name = self.instructions[this_index + 1].argval
            immediately_used = True
        if "<" in func_name:
            self.state.source_code += f'"original name {qual_name} is illegal, use a temp name."\n'
            func_name = self.get_temp_name()
        inner_func = Decompiler(code).decompile(overwite_fn_name=func_name)
        self.state.source_code += inner_func
        if not immediately_used:
            self.state.stack.append(func_name)
        else:
            # skip one instruction
            return this_index + 2

    def COPY_FREE_VARS(self, inst: Instruction):
        # this opcode is used to copy free variables from the outer scope to the closure
        # it affects the frame, but not the stack or the source code
        pass

    def LOAD_ATTR(self, inst: Instruction):
        self.state.stack.append(f"getattr({self.state.stack.pop()}, {repr(inst.argval)})")

    def LOAD_METHOD(self, inst: Instruction):
        self.state.stack.append(f"{self.state.stack.pop()}.{inst.argval}")

    def LOAD_ASSERTION_ERROR(self, inst: Instruction):
        self.state.stack.append("AssertionError")

    def PUSH_NULL(self, inst: Instruction):
        # the `None` object is used to represent `NULL` in python bytecode
        self.state.stack.append(None)

# ==================== Store Instructions =============================

    def generic_store(self, inst: Instruction):
        left = inst.argval
        right = self.state.stack.pop()
        if left != right:
            # Inplace operations like `+=` will pop the variable name from the stack, and push the result back to the stack
            # leading to a source code like `x = x`. We need to avoid this.
            self.state.source_code += f"{left} = {right}\n"

    STORE_FAST = STORE_GLOBAL = STORE_DEREF = STORE_NAME = generic_store

    def STORE_SUBSCR(self, inst: Instruction):
        index = self.state.stack.pop()
        x = self.state.stack.pop()
        value = self.state.stack.pop()
        self.state.source_code += f"{x}[{index}] = {value}\n"

    def STORE_ATTR(self, inst: Instruction):
        x = self.state.stack.pop()
        value = self.state.stack.pop()
        self.state.source_code += f"{x}.{inst.argval} = {value}\n"

# ==================== Del Instructions =============================

    def DELETE_SUBSCR(self, inst: Instruction):
        index = self.state.stack.pop()
        x = self.state.stack.pop()
        self.state.source_code += f"del {x}[{index}]\n"

    def generic_delete(self, inst: Instruction):
        self.state.source_code += f"del {inst.argval}\n"
    
    DELETE_NAME = DELETE_FAST = DELETE_GLOBAL = DELETE_DEREF = generic_delete

    def DELETE_ATTR(self, inst: Instruction):
        x = self.state.stack.pop()
        self.state.source_code += f"del {x}.{inst.argval}\n"

# ==================== Import Instructions =============================
    def IMPORT_NAME(self, inst: Instruction):
        # TODO: check multi-level import, e.g. `import a.b.c`
        name = inst.argval.split(".")[0]
        fromlist = self.state.stack.pop()
        level = self.state.stack.pop()
        self.state.source_code += f"{name} = __import__({repr(inst.argval)}, fromlist={fromlist}, level={level})\n"
        self.state.stack.append(name)
    
    def IMPORT_FROM(self, inst: Instruction):
        name = inst.argval
        module = self.state.stack[-1]
        self.state.source_code += f"{name} = {module}.{name}\n"
        self.state.stack.append(name)

# ==================== Unary Instructions =============================

    def generic_unary(self, inst: Instruction):
        op = {
            "UNARY_NEGATIVE": "-",
            "UNARY_POSITIVE": "+",
            "UNARY_INVERT": "~",
            "UNARY_NOT": "not",
        }[inst.opname]
        self.state.stack.append(f"({op} {self.state.stack.pop()})")
    
    UNARY_NEGATIVE = UNARY_POSITIVE = UNARY_INVERT = UNARY_NOT = generic_unary

    def GET_LEN(self, inst: Instruction):
        self.state.stack.append(f"len({self.state.stack[-1]})")

# ==================== Binary Instructions =============================
    def generic_binary(self, inst: Instruction):
        rhs = self.state.stack.pop()
        lhs = self.state.stack.pop()
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
        self.state.stack.append(f"({lhs} {op} {rhs})")

    BINARY_MULTIPLY = BINARY_ADD = BINARY_SUBTRACT = BINARY_TRUE_DIVIDE = BINARY_FLOOR_DIVIDE = BINARY_MODULO = BINARY_POWER = BINARY_AND = BINARY_OR = BINARY_XOR = BINARY_LSHIFT = BINARY_RSHIFT = BINARY_MATRIX_MULTIPLY = generic_binary

    def BINARY_SUBSCR(self, inst: Instruction):
        rhs = self.state.stack.pop()
        lhs = self.state.stack.pop()
        self.state.stack.append(f"{lhs}[{rhs}]")

# ==================== Binary Inplace Instructions =============================
    def generic_inplace_binary(self, inst: Instruction):
        rhs = self.state.stack.pop()
        lhs = self.state.stack.pop()
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
        self.state.source_code += f"{lhs} {op}= {rhs}\n"
        self.state.stack.append(lhs)
    
    INPLACE_MULTIPLY = INPLACE_ADD = INPLACE_SUBTRACT = INPLACE_TRUE_DIVIDE = INPLACE_FLOOR_DIVIDE = INPLACE_MODULO = INPLACE_POWER = INPLACE_AND = INPLACE_OR = INPLACE_XOR = INPLACE_LSHIFT = INPLACE_RSHIFT = INPLACE_MATRIX_MULTIPLY = generic_inplace_binary

    def BINARY_OP(self, inst: Instruction):
        rhs = self.state.stack.pop()
        lhs = self.state.stack.pop()
        if "=" in inst.argrepr:
            self.state.source_code += f"{lhs} {inst.argrepr} {rhs}\n"
            self.state.stack.append(lhs)
        else:
            self.state.stack.append(f"({lhs} {inst.argrepr} {rhs})")

# ==================== Conditional Test Instructions =============================
    def COMPARE_OP(self, inst: Instruction):
        rhs = self.state.stack.pop()
        lhs = self.state.stack.pop()
        self.state.stack.append(f"({lhs} {inst.argval} {rhs})")

    def IS_OP(self, inst: Instruction):
        rhs = self.state.stack.pop()
        lhs = self.state.stack.pop()
        op = "is" if inst.argval == 0 else "is not"
        self.state.stack.append(f"({lhs} {op} {rhs})")

    def CONTAINS_OP(self, inst: Instruction):
        rhs = self.state.stack.pop()
        lhs = self.state.stack.pop()
        op = "in" if inst.argval == 0 else "not in"
        self.state.stack.append(f"({lhs} {op} {rhs})")

# ==================== Control Flow Instructions =============================

    def BREAK_LOOP(self, inst: Instruction):
        self.state.source_code += "break\n"
    
    def generic_abs_jump(self, inst: Instruction):
        jump_offset = inst.get_jump_target()
        if jump_offset > inst.offset:
            self.state.source_code += "break\n"
        else:
            self.state.source_code += "continue\n"
    
    JUMP_ABSOLUTE = JUMP_FORWARD = JUMP_BACKWARD = JUMP_BACKWARD_NO_INTERRUPT = generic_abs_jump

    def RETURN_VALUE(self, inst: Instruction):
        self.state.source_code += f"return {self.state.stack[-1]}\n"

    def YIELD_VALUE(self, inst: Instruction):
        self.state.source_code += f"yield {self.state.stack[-1]}\n"

    def RETURN_GENERATOR(self, inst: Instruction):
        # we don't handle generator/coroutine, add this to support simple yield
        self.state.stack.append(None)
    def GEN_START(self, inst: Instruction):
        # self.state.stack.pop()
        assert inst.argval == 0, "Only generator expression is supported"

    def generic_jump_if(self, inst: Instruction):
        jump_offset = inst.get_jump_target()
        jump_index = self.index_of(jump_offset)
        cond = self.state.stack[-1]
        fallthrough_stack = self.state.stack.copy()[:-1]

        # POP_AND_JUMP / JUMP_OR_POP
        if "POP_JUMP" in inst.opname:
            jump_stack = self.state.stack.copy()[:-1]
        elif "OR_POP" in inst.opname:
            jump_stack = self.state.stack.copy()

        # JUMP_IF_X, so fallthrough if not X
        if "IF_FALSE" in inst.opname:
            self.state.source_code += f"if {cond}:\n"
        elif "IF_TRUE" in inst.opname:
            self.state.source_code += f"if not {cond}:\n"
        elif "IF_NOT_NONE" in inst.opname:
            self.state.source_code += f"if {cond} is None:\n"
        elif "IF_NONE" in inst.opname:
            self.state.source_code += f"if {cond} is not None:\n"

        if_body_start = self.index_of(inst.offset)

        jump_targets = [i.get_jump_target() for i in self.instructions[if_body_start + 1: jump_index] if i.is_jump() and i.get_jump_target() > jump_offset]
        else_code = ""
        if jump_targets:
            # has "else" branch
            max_jump = max(jump_targets)
            max_jump_index = self.index_of(max_jump)
            else_code += "else:\n"
            with self.new_state(jump_stack):
                self.decompile_range(jump_index, max_jump_index)
                source_code = self.state.source_code
            else_code += add_indentation(source_code, self.indentation)

        with self.new_state(fallthrough_stack):
            if_body_end = jump_index
            if else_code and self.instructions[if_body_end - 1].is_jump():
                # the last instruction is a jump, so it is not part of the if body, but the jump out of the if-else block
                if_body_end -= 1
            self.decompile_range(if_body_start + 1, if_body_end)
            if_code = self.state.source_code
            if_code = add_indentation(if_code, self.indentation)
        
        self.state.source_code += if_code + else_code

        return max_jump_index if else_code else jump_index

    POP_JUMP_IF_TRUE = POP_JUMP_IF_FALSE = generic_jump_if
    POP_JUMP_FORWARD_IF_TRUE = POP_JUMP_FORWARD_IF_FALSE = generic_jump_if
    POP_JUMP_BACKWARD_IF_TRUE = POP_JUMP_BACKWARD_IF_FALSE = generic_jump_if
    POP_JUMP_FORWARD_IF_NONE = POP_JUMP_FORWARD_IF_NOT_NONE = generic_jump_if
    POP_JUMP_BACKWARD_IF_NONE = POP_JUMP_BACKWARD_IF_NOT_NONE = generic_jump_if
    JUMP_IF_TRUE_OR_POP = JUMP_IF_FALSE_OR_POP = generic_jump_if

# ==================== Stack Manipulation Instructions =============================
    def rot_n(self, inst: Instruction):
        if inst.opname == "ROT_N":
            n = inst.argval
        else:
            n = {
                "ROT_TWO": 2,
                "ROT_THREE": 3,
                "ROT_FOUR": 4,
            }[inst.opname]
        values = self.state.stack[-n:]
        values = [values[-1]] + values[:-1]
        self.state.stack[-n:] = values
        
    ROT_N = ROT_TWO = ROT_THREE = ROT_FOUR = rot_n

    def SWAP(self, inst: Instruction):
        # not tested, don't know how to generate this instruction
        n = inst.argval
        tos = self.state.stack[-1]
        value = self.state.stack[-1 - n]
        tos, value = value, tos
        self.state.stack[-1] = tos
        self.state.stack[-1 - n] = value
    
    def COPY(self, inst: Instruction):
        # not tested, don't know how to generate this instruction
        n = inst.argval
        value = self.state.stack[-1 - n]
        self.state.stack.append(value)
    
    def POP_TOP(self, inst: Instruction):
        self.state.stack.pop()
    
    def DUP_TOP(self, inst: Instruction):
        # not tested
        self.state.stack.append(self.state.stack[-1])
    
    def DUP_TOP_TWO(self, inst: Instruction):
        # not tested
        tos = self.state.stack[-1]
        tos1 = self.state.stack[-2]
        self.state.stack.append(tos1)
        self.state.stack.append(tos)

# ==================== Function Call Instructions =============================
    def KW_NAMES(self, inst: Instruction):
        names = self.code.co_consts[inst.arg]
        self.state.stack.append(repr(names))

    def CALL(self, inst: Instruction):
        last_inst = [x for x in block.instructions if x.offset < inst.offset][-1]
        has_kw_names = last_inst.opname == "KW_NAMES"
        kw_names = tuple()
        if has_kw_names:
            kw_names = eval(self.state.stack.pop())
        args = [(self.state.stack.pop()) for _ in range(inst.argval)]
        args = args[::-1]
        pos_args = args[:len(args) - len(kw_names)]
        kwargs = args[len(args) - len(kw_names):]
        kwcalls = []
        for name, value in zip(kwargs, kw_names):
            kwcalls.append(f"{name}={value}")
        func = self.state.stack.pop()
        if self.state.stack and self.state.stack[-1] is None:
            self.state.stack.pop()
        temp = self.get_temp_name()
        self.state.source_code += f"{temp} = {func}({', '.join(pos_args + kwcalls)})\n"
        self.state.stack.append(temp)

    def generic_call(self, inst: Instruction):
        args = [(self.state.stack.pop()) for _ in range(inst.argval)]
        args = args[::-1]
        func = self.state.stack.pop()
        temp = self.get_temp_name()
        self.state.source_code += f"{temp} = {func}({', '.join(args)})\n"
        self.state.stack.append(temp)
    
    CALL_FUNCTION = CALL_METHOD = generic_call

    def CALL_FUNCTION_KW(self, inst: Instruction):
        kw_args = eval(self.state.stack.pop())
        kwcalls = []
        for name in kw_args:
            kwcalls.append(f"{name}={self.state.stack.pop()}")
        pos_args = [(self.state.stack.pop()) for _ in range(inst.argval - len(kw_args))]
        pos_args = pos_args[::-1]
        func = self.state.stack.pop()
        temp = self.get_temp_name()
        self.state.source_code += f"{temp} = {func}({', '.join(pos_args + kwcalls)})\n"
        self.state.stack.append(temp)

    def CALL_FUNCTION_EX(self, inst: Instruction):
        if inst.argval == 0:
            args = self.state.stack.pop()
            func = self.state.stack.pop()
            temp = self.get_temp_name()
            self.state.source_code += f"{temp} = {func}(*{args})\n"
            self.state.stack.append(temp)
        elif inst.argval == 1:
            kw_args = self.state.stack.pop()
            args = self.state.stack.pop()
            func = self.state.stack.pop()
            temp = self.get_temp_name()
            self.state.source_code += f"{temp} = {func}(*{args}, **{kw_args})\n"
            self.state.stack.append(temp)

# ==================== Container Related Instructions (tuple, list, set, dict) =============================
    def UNPACK_SEQUENCE(self, inst: Instruction):
        varname = self.state.stack.pop()
        for i in range(inst.argval):
            self.state.stack.append(f"{varname}[{inst.argval - 1 - i}]")
    
    def UNPACK_EX(self, inst: Instruction):
        varname = self.state.stack.pop()
        self.state.stack.append(f"list({varname}[{inst.argval}:])")
        for i in range(inst.argval):
            self.state.stack.append(f"{varname}[{inst.argval - 1 - i}]")

    def BUILD_SLICE(self, inst: Instruction):
        tos = self.state.stack.pop()
        tos1 = self.state.stack.pop()
        if inst.argval == 2:
            self.state.stack.append(f"slice({tos1}, {tos})")
        elif inst.argval == 3:
            tos2 = self.state.stack.pop()
            self.state.stack.append(f"slice({tos2}, {tos1}, {tos})")

    def build_tuple(self, inst: Instruction):
        args = [self.state.stack.pop() for _ in range(inst.argval)]
        args = args[::-1]
        if "UNPACK" in inst.opname:
            args = [f"*{arg}" for arg in args]
        if inst.argval == 1:
            self.state.stack.append(f"({args[0]},)")
        else:
            self.state.stack.append(f"({', '.join(args)})")
    
    BUILD_TUPLE = BUILD_TUPLE_UNPACK = BUILD_TUPLE_UNPACK_WITH_CALL = build_tuple

    def build_list(self, inst: Instruction):
        args = [self.state.stack.pop() for _ in range(inst.argval)]
        args = args[::-1]
        if "UNPACK" in inst.opname:
            args = [f"*{arg}" for arg in args]
        self.state.stack.append(f"[{', '.join(args)}]")
    
    BUILD_LIST = BUILD_LIST_UNPACK = build_list

    def build_set(self, inst: Instruction):
        if inst.argval == 0:
            self.state.stack.append("set()")
        else:
            args = [self.state.stack.pop() for _ in range(inst.argval)]
            args = args[::-1]
            if "UNPACK" in inst.opname:
                args = [f"*{arg}" for arg in args]
            self.state.stack.append(f"{{{', '.join(args)}}}")
    
    BUILD_SET = BUILD_SET_UNPACK = build_set

    def build_map_unpack(self, inst: Instruction):
        if inst.argval == 0:
            self.state.stack.append("dict()")
        else:
            args = [self.state.stack.pop() for _ in range(inst.argval)]
            args = args[::-1]
            args = [f"**{arg}" for arg in args]
            self.state.stack.append(f"{{{', '.join(args)}}}")

    BUILD_MAP_UNPACK = BUILD_MAP_UNPACK_WITH_CALL = build_map_unpack

    def BUILD_MAP(self, inst: Instruction):
        args = [self.state.stack.pop() for _ in range(inst.argval * 2)]
        args = args[::-1]
        keys = args[::2]
        values = args[1::2]
        self.state.stack.append(f"{{{', '.join([f'{k}: {v}' for k, v in zip(keys, values)])}}}")

    def BUILD_CONST_KEY_MAP(self, inst: Instruction):
        keys = eval(self.state.stack.pop())
        args = [self.state.stack.pop() for _ in range(inst.argval)]
        values = args[::-1]
        self.state.stack.append(f"{{{', '.join([f'{k}: {v}' for k, v in zip(keys, values)])}}}")

    def BUILD_STRING(self, inst: Instruction):
        args = [self.state.stack.pop() for _ in range(inst.argval)]
        args = args[::-1]
        values = " + ".join(args)
        self.state.stack.append(values)

    def LIST_TO_TUPLE(self, inst: Instruction):
        item = self.state.stack.pop()
        self.state.stack.append(f"tuple({item})")

    def LIST_EXTEND(self, inst: Instruction):
        assert inst.argval == 1, "Only tested for argval==1"
        values = self.state.stack.pop()
        temp = self.get_temp_name()
        x = self.state.stack.pop()
        self.state.source_code += f"{temp} = {x}\n"
        self.state.source_code += f"{temp}.extend({values})\n"
        self.state.stack.append(temp)

    def generic_update(self, inst: Instruction):
        assert inst.argval == 1, "Only tested for argval==1"
        values = self.state.stack.pop()
        temp = self.get_temp_name()
        x = self.state.stack.pop()
        self.state.source_code += f"{temp} = {x}\n"
        self.state.source_code += f"{temp}.update({values})\n"
        self.state.stack.append(temp)
    
    SET_UPDATE = DICT_UPDATE = DICT_MERGE = generic_update

# ==================== Misc Instructions =============================
    def RAISE_VARARGS(self, inst: Instruction):
        if inst.argval == 0:
            self.state.source_code += "raise\n"
        elif inst.argval == 1:
            self.state.source_code += f"raise {self.state.stack.pop()}\n"
        elif inst.argval == 2:
            tos = self.state.stack.pop()
            tos1 = self.state.stack.pop()
            self.state.source_code += f"raise {tos1} from {tos}\n"
    
    def FORMAT_VALUE(self, inst: Instruction):
        func, spec = inst.argval
        if spec:
            form_spec = self.state.stack.pop()
            value = self.state.stack.pop()
            self.state.stack.append(f"format({value}, {form_spec})")
        else:
            value = self.state.stack.pop()
            func = str if func is None else func
            self.state.stack.append(f"{func.__name__}({value})")


# ==================== NOP Instructions =============================
    def generic_nop(self, inst: Instruction):
        pass

    # "EXTENDED_ARG" is treated as NOP here, because it has been handled by `dis.get_instructions`.
    # The extended args are already merged into the following instruction's `inst.argval`.
    EXTENDED_ARG = generic_nop

    NOP = RESUME = SETUP_LOOP = POP_BLOCK = PRECALL = generic_nop

# ==================== Unsupported Instructions =============================
    def unimplemented_instruction(self, inst: Instruction):
        raise NotImplementedError(f"Unsupported instruction: {inst.opname}")

    GET_YIELD_FROM_ITER = GET_ITER = FOR_ITER = unimplemented_instruction

    # we don't support try-except/try-finally
    POP_EXCEPT = RERAISE = WITH_EXCEPT_START = JUMP_IF_NOT_EXC_MATCH = SETUP_FINALLY = CHECK_EG_MATCH = PUSH_EXC_INFO = PREP_RERAISE_STAR = BEGIN_FINALLY = END_FINALLY = WITH_CLEANUP_FINISH = CALL_FINALLY = POP_FINALLY = WITH_CLEANUP_START = SETUP_EXCEPT = CHECK_EXC_MATCH = unimplemented_instruction

    # we don't support async/await
    GET_AWAITABLE = GET_AITER = GET_ANEXT = END_ASYNC_FOR = BEFORE_ASYNC_WITH = SETUP_ASYNC_WITH = SEND = ASYNC_GEN_WRAP = unimplemented_instruction

    CACHE = unimplemented_instruction
    
    MAKE_CELL = unimplemented_instruction
    
    # we don't know these instructions
    PRINT_EXPR = COPY_DICT_WITHOUT_KEYS = unimplemented_instruction

    # we only support bytecode for functions
    IMPORT_STAR = unimplemented_instruction
    
    YIELD_FROM = SETUP_ANNOTATIONS = LOAD_BUILD_CLASS = SETUP_WITH = BEFORE_WITH = MATCH_MAPPING = MATCH_SEQUENCE = MATCH_KEYS = MATCH_CLASS = unimplemented_instruction

    # we cannot use list/set/map comprehension
    SET_ADD = MAP_ADD = LIST_APPEND = unimplemented_instruction

    def decompile_range(self, start: int, end: int):
        running_index = start
        while running_index < end:
            inst = self.instructions[running_index]
            method = getattr(Decompiler, inst.opname, self.unimplemented_instruction)
            output = method(self, inst)
            if output:
                running_index = output
            else:
                running_index += 1

    def index_of(self, offset: int):
        for idx, inst in enumerate(self.instructions):
            if inst.offset == offset:
                return idx
        raise ValueError(f"Cannot find instruction with offset {offset}")

    @staticmethod
    def cleanup_instructions(instructions: List[Instruction]):
        propagate_line_nums(instructions)
        simplify_with_statement(instructions)
        nop_unreachable_bytecode(instructions)

    def __init__(self, code: Union[CodeType, Callable]):
        if callable(code):
            code = code.__code__
        self.code = code
        instructions = list(convert_instruction(_) for _ in dis.get_instructions(code))
        self.cleanup_instructions(instructions)
        self.instructions = instructions
        # supported_opnames = self.supported_opnames()
        # for inst in self.instructions:
        #     if inst.opname not in supported_opnames:
        #         raise NotImplementedError(f"Unsupported instruction: {inst.opname}")
        self.blocks = BasicBlock.decompose_basic_blocks(self.instructions)
        self.blocks_index_map = {str(block): idx for idx, block in enumerate(self.blocks)}
        self.state = DecompilerState(source_code="", stack=[])

    def visualize_cfg(self, filepath: str="cfg.png"):
        self.decompile()
        for block in self.blocks:
            lines = [x for x in block.decompiled_code.splitlines()]
            if ("if" in lines[-1] or "for" in lines[-1]) and lines[-1].endswith(":"):
                valid_lines = lines[:-1]
                valid_code = "".join([x + "\n" for x in valid_lines])
                simplified_code = remove_some_temp(valid_code, self.temp_prefix)
                block.decompiled_code = simplified_code + lines[-1] + "\n"
            else:
                block.decompiled_code = remove_some_temp(block.decompiled_code, self.temp_prefix)

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
                if left_block.end_misc or left_block.end_with_return:
                    continue
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

    def decompose_indentation_blocks(self, blocks: List[BasicBlock]) -> List[IndentationBlock]:
        start_index = self.blocks_index_map[str(blocks[0])]
        end_index = self.blocks_index_map[str(blocks[-1])] + 1
        indentation_blocks = []
        while start_index < end_index:
            indentation_block = self.get_indentation_block(self.blocks[start_index])
            indentation_blocks.append(indentation_block)
            start_index = self.blocks_index_map[str(indentation_block.blocks[-1])] + 1
        return indentation_blocks

    def get_temp_name(self):
        self.temp_count += 1
        return f"{self.temp_prefix}{self.temp_count}"

    @staticmethod
    def supported_opnames():
        return get_supported_opnames(Decompiler.decompile_block.__code__)

    @functools.lru_cache(maxsize=None)
    def decompile(self, indentation=4, temp_prefix: str="__temp_", overwite_fn_name: Optional[str]=None) -> str:
        self.indentation = indentation
        self.temp_prefix = temp_prefix
        self.decompile_range(0, len(self.instructions))
        source_code = self.state.source_code
        # the header might have invalid function name in torchdynamo. only optimize the function body.
        source_code = remove_some_temp(source_code, self.temp_prefix, indentation)
        header = get_function_signature(self.code, overwite_fn_name)
        source_code = header + add_indentation(source_code, indentation)
        return source_code

    def decompile_blocks(
            self,
            blocks: List[BasicBlock],
            stack: List[str],
            indentation: int=4,
        ) -> Tuple[str, List[str]]:
        indentation_blocks = self.decompose_indentation_blocks(blocks)
        # indentation blocks are decompiled in order
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
        This function is responsible to handle if-else, while, for, etc.
        ============ Deal with While Loop ============
while cond:
    body1
else:
    body2

is transformed into

while True:
    if cond:
        body1
    else:
        body2
        break

We identify the while loop by checking if the first basic block is jumped to from internal blocks.
        ============ Deal with For Loop ============

        ============ Deal with If Structure ============
if cond1:
    body1
else:
    body2

We identify the start of body2 by checking if the first basic block is jumping to internal blocks.
        """
        i = 0
        source_code = ""
        while True:
            blockresult = self.decompile_block(indentation_block.blocks[i], stack.copy(), indentation, indentation_block)
            indentation_block.blocks[i].decompiled_code = blockresult.source_code
            source_code += blockresult.source_code
            fallthrough_stack = blockresult.fallthrough_stack
            jump_stack = blockresult.jump_stack
            if i < len(indentation_block.blocks) - 1 and indentation_block.blocks[i].end_misc:
                i += 1
                stack = fallthrough_stack
                continue
            else:
                break
        
        is_for_loop = indentation_block.blocks[i].instructions[-1].opname == "FOR_ITER"
        if is_for_loop:
            to_block = indentation_block.blocks[i].jump_to_block
            assert to_block > indentation_block.blocks[i]
            for_blocks = [block for block in indentation_block.blocks[i + 1:] if block < to_block]
            block_code, fallthrough_stack = self.decompile_blocks(for_blocks, fallthrough_stack.copy(), indentation)
            source_code += add_indentation(block_code, indentation)
            rest_blocks = [block for block in indentation_block.blocks[i + 1:] if block >= to_block]
            block_code, fallthrough_stack = self.decompile_blocks(rest_blocks, fallthrough_stack.copy(), indentation)
            source_code += block_code
            return source_code, fallthrough_stack

        has_loop = any(block in indentation_block.blocks[i:] for block in indentation_block.blocks[i].from_blocks)

        # split blocks into if-else
        has_if = indentation_block.blocks[i].end_with_if_jmp
        if has_if:
            to_block = indentation_block.blocks[i].jump_to_block
            if to_block > indentation_block.blocks[i]:
                if_blocks = [block for block in indentation_block.blocks[i + 1:] if block < to_block]
                block_code, fallthrough_stack = self.decompile_blocks(if_blocks, fallthrough_stack.copy(), indentation)
                source_code += add_indentation(block_code, indentation)
            else:
                source_code += add_indentation("continue\n", indentation)
            else_blocks = [block for block in indentation_block.blocks[i + 1:] if block >= to_block]
            if else_blocks:
                block_code, jump_stack = self.decompile_blocks(else_blocks, jump_stack.copy(), indentation)
                if has_loop:
                    block_code = block_code + "break\n"
                source_code += f"else:\n" + add_indentation(block_code, indentation)
            else:
                if has_loop:
                    source_code += f"else:\n" + add_indentation("break\n", indentation)
        if has_loop:
            source_code = "while True:\n" + add_indentation(source_code, indentation)

        # We actually have two stacks here, both should be valid to go to the next block
        return source_code, fallthrough_stack

    def __hash__(self):
        return hash(self.code)

    def decompile_block(
            self,
            block: BasicBlock,
            stack: List[str],
            indentation: int=4,
            indentation_block: Optional[IndentationBlock] = None,
        ) -> BlockResult:
        """Decompile a basic block into source code.
        The `stack` holds expressions in string, like "3 + 4".
        """
        source_code = ""
        # source_code += "=" * 40 + "Basic Block Start" + "=" * 40 + "\n"
        # for inst in block.instructions:
        #     source_code += f"{inst.offset} {inst.opname} {inst.argval} ({inst.argrepr})\n"
        # source_code += "=" * 40 + "Basic Block End" + "=" * 40 + "\n"
        # return source_code, stack

