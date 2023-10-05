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
    simplify_finally_statement,
    Instruction,
)
from .utils import (
    get_function_signature,
)


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
            if isinstance(inst.argval, type):
                # Don't know why a class type get here, support this corner case anyway.
                module = inst.argval.__module__
                name = inst.argval.__name__
                self.state.source_code += "import importlib\n"
                temp_name = self.get_temp_name()
                self.state.source_code += f'{temp_name} = importlib.import_module("{module}").{name}\n'
                self.state.stack.append(temp_name)
            elif inst.argrepr.startswith("torch."):
                # Don't know why torch.xxx get here, support this corner case anyway. This deals with something like `torch.float`.
                self.state.source_code += "import torch\n"
                temp_name = self.get_temp_name()
                self.state.source_code += f'{temp_name} = {inst.argval}\n'
                self.state.stack.append(temp_name)
            else:
                self.state.stack.append(inst.argval)

    def generic_load(self, inst: Instruction):
        """`inst.argval` is the variable name, in string"""
        if "NULL + " in inst.argrepr:
            # Python 3.11 support
            self.state.stack.append(None)
        if inst.argrepr.startswith("."):
            # list/set/tuple comprehension.
            self.state.stack.append(inst.argval.replace(".", "comp_arg_"))
        else:
            self.state.stack.append(inst.argval)

    LOAD_FAST = LOAD_GLOBAL = LOAD_DEREF = LOAD_NAME = LOAD_CLASSDEREF = LOAD_CLOSURE = generic_load

    def MAKE_FUNCTION(self, inst: Instruction):
        if sys.version_info < (3, 11):
            qual_name = self.state.stack.pop()
            try:
                qual_name = eval(qual_name)
            except Exception:
                pass
            func_name = qual_name
            if "<" in func_name:
                self.state.source_code += f'"original name {qual_name} is illegal, use a temp name."\n'
                func_name = self.get_temp_name()
        else:
            # Python 3.11 support, see https://docs.python.org/3.11/library/dis.html#opcode-MAKE_FUNCTION
            func_name = self.get_temp_name()
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
        immediately_used = False
        if self.instructions[this_index + 1].opname == "STORE_FAST":
            # the function is immediately stored in a variable, use that variable name
            func_name = self.instructions[this_index + 1].argval
            immediately_used = True
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

    def GET_ITER(self, inst: Instruction):
        tos = self.state.stack.pop()
        self.state.stack.append(f"iter({tos})")

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
        """Only support consecutive and/or, do not support mixed and/or."""
        jump_offset = inst.get_jump_target()
        jump_index = self.index_of(jump_offset)
        this_index = self.index_of(inst.offset)
        cond = self.state.stack[-1]

        if "ASSERT" in self.instructions[this_index + 1].opname:
            with self.new_state(self.state.stack):
                self.decompile_range(this_index + 1, jump_index)
                source_code = self.state.source_code
            source_code = add_indentation(source_code, self.indentation)
            self.state.source_code += f"if not {cond}:\n{source_code}"
            return jump_index

        fallthrough_stack = self.state.stack

        if_body_start_offset = None
        if_body_end_offset = None
        last_index = this_index
        conditions = [cond]
        for _index in range(this_index, jump_index):
            _inst = self.instructions[_index]
            if "IF_FALSE" in _inst.opname:
                # JUMP_IF_FALSE, followed by "and", short-circuit evaluation means we jump to the end of if-block if the condition is false
                if if_body_end_offset is None:
                    if_body_end_offset = _inst.get_jump_target()
                if _inst.get_jump_target() == if_body_end_offset:
                    if _index != this_index:
                        with self.new_state(fallthrough_stack):
                            self.decompile_range(last_index + 1, _index)
                            source_code = self.state.source_code
                        self.state.source_code += source_code
                        conditions.append(self.state.stack[-1])
                        last_index = _index
                    conditions.append("and")

                    jump_stack = fallthrough_stack.copy()
                    fallthrough_stack.pop()
                    # POP_AND_JUMP / JUMP_OR_POP
                    if "POP_JUMP" in _inst.opname:
                        jump_stack.pop()
                    elif "OR_POP" in _inst.opname:
                        pass

            elif "IF_TRUE" in _inst.opname:
                # JUMP_IF_TRUE, followed by "or", short-circuit evaluation means we jump to the start of if-block if the condition is true
                if if_body_start_offset is None:
                    if_body_start_offset = _inst.get_jump_target()
                if _inst.get_jump_target() == if_body_start_offset:
                    if _index != this_index:
                        with self.new_state(fallthrough_stack):
                            self.decompile_range(last_index + 1, _index)
                            source_code = self.state.source_code
                        self.state.source_code += source_code
                        conditions.append(self.state.stack[-1])
                        last_index = _index
                    conditions.append("or")

                    jump_stack = fallthrough_stack.copy()
                    fallthrough_stack.pop()
                    # POP_AND_JUMP / JUMP_OR_POP
                    if "POP_JUMP" in _inst.opname:
                        jump_stack.pop()
                    elif "OR_POP" in _inst.opname:
                        pass

            elif "IF_NOT_NONE" in _inst.opname:
                # TODO to check, in 3.11
                self.state.source_code += f"if {cond} is None:\n"
            elif "IF_NONE" in _inst.opname:
                # TODO to check, in 3.11
                self.state.source_code += f"if {cond} is not None:\n"

        conditions.pop()

        if if_body_start_offset is None:
            if_body_start_offset = self.instructions[last_index + 1].offset
            
        if_body_start = self.index_of(if_body_start_offset)

        if if_body_end_offset is None:
            # Don't know where the if body ends, so we have to find the next jump instruction
            if_body_end = if_body_start + 1
            while if_body_end < len(self.instructions) and not self.instructions[if_body_end].is_jump():
                if_body_end += 1
            if if_body_end == len(self.instructions):
                if_body_end -= 1
            if_body_end_offset = self.instructions[if_body_end].offset
        else:
            if_body_end = self.index_of(if_body_end_offset)
        if jump_index < if_body_start:
            self.state.source_code += add_indentation("continue\n", self.indentation)
            return

        jump_targets = [i.get_jump_target() for i in self.instructions[if_body_start + 1: if_body_end] if i.is_jump() and i.get_jump_target() > if_body_end_offset]
        else_code = ""
        if jump_targets:
            # has "else" branch
            else_start_offset = if_body_end_offset
            else_start_index = self.index_of(else_start_offset)
            max_jump = max(jump_targets)
            max_jump_index = self.index_of(max_jump)
            else_code += "else:\n"
            with self.new_state(jump_stack):
                self.decompile_range(else_start_index, max_jump_index)
                source_code = self.state.source_code
            else_code += add_indentation(source_code, self.indentation)

        with self.new_state(fallthrough_stack):
            if else_code and self.instructions[if_body_end - 1].is_jump():
                # the last instruction is a jump, so it is not part of the if body, but the jump out of the if-else block
                if_body_end -= 1
            self.decompile_range(if_body_start, if_body_end)
            if_code = "if " + " ".join(conditions) + ":\n"
            if_code = if_code + add_indentation(self.state.source_code, self.indentation)
        
        self.state.source_code += if_code + else_code
        self.state.stack = fallthrough_stack

        return max_jump_index if else_code else jump_index

    POP_JUMP_IF_TRUE = POP_JUMP_IF_FALSE = generic_jump_if
    POP_JUMP_FORWARD_IF_TRUE = POP_JUMP_FORWARD_IF_FALSE = generic_jump_if
    POP_JUMP_BACKWARD_IF_TRUE = POP_JUMP_BACKWARD_IF_FALSE = generic_jump_if
    POP_JUMP_FORWARD_IF_NONE = POP_JUMP_FORWARD_IF_NOT_NONE = generic_jump_if
    POP_JUMP_BACKWARD_IF_NONE = POP_JUMP_BACKWARD_IF_NOT_NONE = generic_jump_if
    JUMP_IF_TRUE_OR_POP = JUMP_IF_FALSE_OR_POP = generic_jump_if

    def SETUP_FINALLY(self, inst: Instruction):
        start_index = self.index_of(inst.offset)
        end_index = self.index_of(inst.get_jump_target())
        pop_block_index = [i for i, x in enumerate(self.instructions) if x.opname == "POP_BLOCK" and start_index <= i < end_index][-1]

        try_code = ""
        with self.new_state(self.state.stack):
            self.decompile_range(start_index + 1, pop_block_index)
            try_code = self.state.source_code
            try_code = add_indentation(try_code, self.indentation)
            try_code = "try:\n" + try_code
        
        finally_code = ""
        with self.new_state(self.state.stack):
            end_finally_index = [i for i, x in enumerate(self.instructions) if x.opname == "END_FINALLY" and start_index <= i]
            if end_finally_index:
                end_index = end_finally_index[0]
            finally_end_index = end_index
            if self.instructions[finally_end_index - 1].is_jump():
                finally_end_index -= 1
            self.decompile_range(pop_block_index + 1, finally_end_index)
            finally_code = self.state.source_code
            finally_code = add_indentation(finally_code, self.indentation)
            finally_code = "finally:\n" + finally_code
        
        self.state.source_code += try_code + finally_code
        return end_index

    def FOR_ITER(self, inst: Instruction):
        start_index = self.index_of(inst.offset)
        end_index = self.index_of(inst.get_jump_target())

        temp_name = self.get_temp_name()
        for_code = f"for {temp_name} in {self.state.stack.pop()}:\n"
        self.state.stack.append(temp_name)
        with self.new_state(self.state.stack):
            self.decompile_range(start_index + 1, end_index)
            code = self.state.source_code
            for_code = for_code + add_indentation(code, self.indentation)

        self.state.source_code += for_code
        return end_index

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
        last_inst = [x for x in self.instructions if x.offset < inst.offset]
        has_kw_names = False
        if last_inst:
            if last_inst[-1].opname == "KW_NAMES" or (len(last_inst) > 1 and last_inst[-2].opname == "KW_NAMES" and last_inst[-1].opname == "PRECALL"):
                has_kw_names = True
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
        self.state.stack.append(f"{func}({', '.join(pos_args + kwcalls)})")
        self.replace_mutable_tos_with_temp()

    def generic_call(self, inst: Instruction):
        args = [(self.state.stack.pop()) for _ in range(inst.argval)]
        args = args[::-1]
        func = self.state.stack.pop()
        self.state.stack.append(f"{func}({', '.join(args)})")
        self.replace_mutable_tos_with_temp()
    
    CALL_FUNCTION = CALL_METHOD = generic_call

    def CALL_FUNCTION_KW(self, inst: Instruction):
        kw_args = eval(self.state.stack.pop())
        kw_vals = [(self.state.stack.pop()) for _ in range(len(kw_args))]
        kw_vals.reverse()
        kwcalls = []
        for name, val in zip(kw_args, kw_vals):
            kwcalls.append(f"{name}={val}")
        pos_args = [(self.state.stack.pop()) for _ in range(inst.argval - len(kw_args))]
        pos_args = pos_args[::-1]
        func = self.state.stack.pop()
        self.state.stack.append(f"{func}({', '.join(pos_args + kwcalls)})")
        self.replace_mutable_tos_with_temp()

    def CALL_FUNCTION_EX(self, inst: Instruction):
        if inst.argval == 0:
            args = self.state.stack.pop()
            func = self.state.stack.pop()
            self.state.stack.append(f"{func}(*{args})")
        elif inst.argval == 1:
            kw_args = self.state.stack.pop()
            args = self.state.stack.pop()
            func = self.state.stack.pop()
            self.state.stack.append(f"{func}(*{args}, **{kw_args})")
        self.replace_mutable_tos_with_temp()

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
        self.replace_mutable_tos_with_temp()
    
    BUILD_LIST = BUILD_LIST_UNPACK = build_list

    def build_set(self, inst: Instruction):
        ans = ""
        if inst.argval == 0:
            ans = "set()"
        else:
            args = [self.state.stack.pop() for _ in range(inst.argval)]
            args = args[::-1]
            if "UNPACK" in inst.opname:
                args = [f"*{arg}" for arg in args]
            ans = f"{{{', '.join(args)}}}"
        self.state.stack.append(ans)
        self.replace_mutable_tos_with_temp()
    
    BUILD_SET = BUILD_SET_UNPACK = build_set

    def build_map_unpack(self, inst: Instruction):
        if inst.argval == 0:
            self.state.stack.append("dict()")
        else:
            args = [self.state.stack.pop() for _ in range(inst.argval)]
            args = args[::-1]
            args = [f"**{arg}" for arg in args]
            self.state.stack.append(f"{{{', '.join(args)}}}")
        self.replace_mutable_tos_with_temp()

    BUILD_MAP_UNPACK = BUILD_MAP_UNPACK_WITH_CALL = build_map_unpack

    def BUILD_MAP(self, inst: Instruction):
        args = [self.state.stack.pop() for _ in range(inst.argval * 2)]
        args = args[::-1]
        keys = args[::2]
        values = args[1::2]
        self.state.stack.append(f"{{{', '.join([f'{k}: {v}' for k, v in zip(keys, values)])}}}")
        self.replace_mutable_tos_with_temp()

    def BUILD_CONST_KEY_MAP(self, inst: Instruction):
        keys = eval(self.state.stack.pop())
        args = [self.state.stack.pop() for _ in range(inst.argval)]
        values = args[::-1]
        self.state.stack.append(f"{{{', '.join([f'{k}: {v}' for k, v in zip(keys, values)])}}}")
        self.replace_mutable_tos_with_temp()

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
        temp = self.replace_mutable_tos_with_temp()
        self.state.source_code += f"{temp}.extend({values})\n"

    def LIST_APPEND(self, inst: Instruction):
        if inst.argval == 1:
            # it should be a bug, the tos should be the value. fix it anyway.
            inst.argval += 1
        container = self.state.stack[-inst.argval]
        value = self.state.stack.pop()
        self.state.source_code += f"{container}.append({value})\n"

    def generic_update(self, inst: Instruction):
        assert inst.argval == 1, "Only tested for argval==1"
        values = self.state.stack.pop()
        temp = self.replace_mutable_tos_with_temp()
        self.state.source_code += f"{temp}.update({values})\n"
    
    SET_UPDATE = DICT_UPDATE = DICT_MERGE = generic_update

    def SET_ADD(self, inst: Instruction):
        if inst.argval == 1:
            # it should be a bug, the tos should be the value. fix it anyway.
            inst.argval += 1
        container = self.state.stack[-inst.argval]
        value = self.state.stack.pop()
        self.state.source_code += f"{container}.add({value})\n"

    def MAP_ADD(self, inst: Instruction):
        container = self.state.stack[-inst.argval - 1]
        # see https://docs.python.org/3.10/library/dis.html#opcode-MAP_ADD
        if sys.version_info >= (3, 8):
            value = self.state.stack.pop()
            key = self.state.stack.pop()
        else:
            key = self.state.stack.pop()
            value = self.state.stack.pop()
        self.state.source_code += f"{container}.__setitem__({key}, {value})\n"

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

    NOP = RESUME = SETUP_LOOP = POP_BLOCK = PRECALL = BEGIN_FINALLY = END_FINALLY = generic_nop

    MAKE_CELL = generic_nop

# ==================== Unsupported Instructions =============================
    def unimplemented_instruction(self, inst: Instruction):
        raise NotImplementedError(f"Unsupported instruction: {inst.opname}")

    GET_YIELD_FROM_ITER = unimplemented_instruction

    # we don't support try-except/try-finally
    POP_EXCEPT = RERAISE = WITH_EXCEPT_START = JUMP_IF_NOT_EXC_MATCH = CHECK_EG_MATCH = PUSH_EXC_INFO = PREP_RERAISE_STAR = WITH_CLEANUP_FINISH = CALL_FINALLY = POP_FINALLY = WITH_CLEANUP_START = SETUP_EXCEPT = CHECK_EXC_MATCH = unimplemented_instruction

    # we don't support async/await
    GET_AWAITABLE = GET_AITER = GET_ANEXT = END_ASYNC_FOR = BEFORE_ASYNC_WITH = SETUP_ASYNC_WITH = SEND = ASYNC_GEN_WRAP = unimplemented_instruction

    CACHE = unimplemented_instruction
        
    # we don't know these instructions
    PRINT_EXPR = COPY_DICT_WITHOUT_KEYS = unimplemented_instruction

    # we only support bytecode for functions
    IMPORT_STAR = unimplemented_instruction
    
    YIELD_FROM = SETUP_ANNOTATIONS = LOAD_BUILD_CLASS = SETUP_WITH = BEFORE_WITH = MATCH_MAPPING = MATCH_SEQUENCE = MATCH_KEYS = MATCH_CLASS = unimplemented_instruction

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
        simplify_finally_statement(instructions)
        nop_unreachable_bytecode(instructions)

    def __init__(self, code: Union[CodeType, Callable]):
        if callable(code):
            code = code.__code__
        self.code = code
        instructions = list(convert_instruction(_) for _ in dis.get_instructions(code))
        Decompiler.cleanup_instructions(instructions)
        self.instructions = instructions
        self.state = DecompilerState(source_code="", stack=[])

    def get_temp_name(self):
        Decompiler.temp_count += 1
        return f"{self.temp_prefix}{Decompiler.temp_count}"

    def replace_mutable_tos_with_temp(self):
        ans = self.state.stack.pop()
        temp_name = self.get_temp_name()
        self.state.source_code += f"{temp_name} = {ans}\n"
        self.state.stack.append(temp_name)
        return temp_name

    @staticmethod
    def supported_opnames():
        opnames = []
        for x in dis.opname:
            if getattr(Decompiler, x, Decompiler.unimplemented_instruction) is not Decompiler.unimplemented_instruction:
                opnames.append(x)
        return opnames

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

    def __hash__(self):
        return hash(self.code)
