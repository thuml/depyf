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

from .code_transform import (
    nop_unreachable_bytecode,
    nop_instruction,
    add_indentation,
    remove_indentation,
    remove_some_temp,
    propagate_line_nums,
    convert_instruction,
    simplify_finally_statement,
    Instruction,
)
from .utils import (
    get_function_signature,
)


class DecompilationError(Exception):
    """Custom exception class for decompilation."""

    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'DecompilationError: {self.message}'


@dataclasses.dataclass
class DecompilerState:
    """State of decompiler, keep track of the evaluation stack, as well as the decompiled source code."""
    source_code: str
    stack: list
    inside_loop: bool = False
    loop_start_index: int = -1 # inclusive
    loop_end_index: int = -1 # exclusive


@dataclasses.dataclass
class Decompiler:
    """A decompiler for a code object."""
    code: CodeType
    temp_count: int = 0
    temp_prefix: str = "__temp_"
    state: DecompilerState = dataclasses.field(
        default_factory=lambda: DecompilerState(
            source_code="", stack=[]))
    indentation: int = 4

    @contextlib.contextmanager
    def new_state(self, stack, inside_loop=False, loop_start_index=-1, loop_end_index=-1):
        """Create a new state for decompiler."""
        state = DecompilerState(source_code="", stack=stack, inside_loop=inside_loop, loop_start_index=loop_start_index, loop_end_index=loop_end_index)
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
        except BaseException:
            pass
        if can_repr:
            self.state.stack.append(repr(inst.argval))
        else:
            if isinstance(inst.argval, type):
                # Don't know why a class type get here, support this corner
                # case anyway.
                module = inst.argval.__module__
                name = inst.argval.__name__
                self.state.source_code += "import importlib\n"
                temp_name = self.get_temp_name()
                self.state.source_code += f'{temp_name} = importlib.import_module("{module}").{name}\n'
                self.state.stack.append(temp_name)
            elif inst.argrepr.startswith("torch."):
                # Don't know why torch.xxx get here, support this corner case
                # anyway. This deals with something like `torch.float`.
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

    LOAD_FAST = LOAD_FAST_AND_CLEAR = LOAD_FAST_CHECK = LOAD_GLOBAL = LOAD_DEREF = LOAD_NAME = LOAD_CLASSDEREF = LOAD_CLOSURE = generic_load

    def LOAD_LOCALS(self, inst: Instruction):
        self.state.stack.append("locals()")
        self.replace_mutable_tos_with_temp()

    def LOAD_FROM_DICT_OR_GLOBALS(self, inst: Instruction):
        tos = self.state.stack.pop()
        self.state.stack.append(
            f"{tos}[{inst.argval}] if '{inst.argval}' in {tos} else {inst.argval}")
        self.replace_mutable_tos_with_temp()

    LOAD_FROM_DICT_OR_DEREF = LOAD_FROM_DICT_OR_GLOBALS

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
            # Python 3.11 support, see
            # https://docs.python.org/3.11/library/dis.html#opcode-MAKE_FUNCTION
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
            print(
                "Function with default arguments is not supported, ignore the default arguments")
        this_index = self.index_of(inst.offset)
        immediately_used = False
        if self.instructions[this_index + 1].opname == "STORE_FAST":
            # the function is immediately stored in a variable, use that
            # variable name
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
        lhs = str(self.state.stack.pop())
        rhs = inst.argval
        if rhs.isidentifier():
            self.state.stack.append(f"{lhs}.{rhs}")
        else:
            self.state.stack.append(f"getattr({lhs}, {repr(rhs)})")

    def LOAD_SUPER_ATTR(self, inst: Instruction):
        # not tested
        self_obj = self.state.stack.pop()
        cls_obj = self.state.stack.pop()
        super_obj = self.state.stack.pop()
        self.state.stack.append(
            f"{super_obj}({cls_obj}, {self_obj}).{inst.argval}")
        self.replace_mutable_tos_with_temp()

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

    def STORE_SLICE(self, inst: Instruction):
        # not tested, code according to
        # https://docs.python.org/3.12/library/dis.html#opcode-STORE_SLICE
        end = self.state.stack.pop()
        start = self.state.stack.pop()
        container = self.state.stack.pop()
        value = self.state.stack.pop()
        self.state.source_code += f"{container}[{start}:{end}] = {value}\n"

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

    def BINARY_SLICE(self, inst: Instruction):
        end = self.state.stack.pop()
        start = self.state.stack.pop()
        container = self.state.stack.pop()
        self.state.stack.append(f"{container}[{start}:{end}]")

# ==================== Binary Inplace Instructions =======================
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

# ==================== Conditional Test Instructions =====================
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
        jump_index = self.index_of(jump_offset)
        if self.state.inside_loop:
            if jump_index >= self.state.loop_end_index:
                self.state.source_code += "break\n"
            elif jump_index <= self.state.loop_start_index:
                self.state.source_code += "continue\n"
        else:
            return jump_index

    JUMP_ABSOLUTE = JUMP_FORWARD = JUMP_BACKWARD = JUMP_BACKWARD_NO_INTERRUPT = generic_abs_jump

    def RETURN_VALUE(self, inst: Instruction):
        self.state.source_code += f"return {self.state.stack[-1]}\n"
        self.state.stack.pop()

    def RETURN_CONST(self, inst: Instruction):
        self.state.source_code += f"return {inst.argval}\n"

    def YIELD_VALUE(self, inst: Instruction):
        if sys.version_info >= (3, 12):
            raise NotImplementedError(
                "YIELD_VALUE is not supported in Python 3.12")
        self.state.source_code += f"yield {self.state.stack[-1]}\n"

    def RETURN_GENERATOR(self, inst: Instruction):
        # we don't handle generator/coroutine, add this to support simple yield
        self.state.stack.append(None)

    def GEN_START(self, inst: Instruction):
        # self.state.stack.pop()
        assert inst.argval == 0, "Only generator expression is supported"

    def generic_jump_if(self, inst: Instruction):
        """How we support if-else:

        Failed idea: try to find the block of instructions for if and else.
        This is not possible, as the if-else block might have non-continuous instructions.
        Take this function as an example:

        def f(a):
            b = 1 if a else 2
            print(b)
        
        The bytecode is:
  2           0 LOAD_FAST                0 (a)
              2 POP_JUMP_IF_FALSE        4 (to 8)
              4 LOAD_CONST               1 (1)
              6 JUMP_FORWARD             1 (to 10)
        >>    8 LOAD_CONST               2 (2)
        >>   10 STORE_FAST               1 (b)

  3          12 LOAD_GLOBAL              0 (print)
             14 LOAD_FAST                1 (b)
             16 CALL_FUNCTION            1
             18 POP_TOP
             20 LOAD_CONST               0 (None)
             22 RETURN_VALUE
        
        The instructions for if branch: 2, 4, 6, 10
        The instructions for else branch: 8, 10


        Current idea:

        We take advantage of the following fact:

        This code snippet:

        if cond:
            if-body
        else:
            else-body
        rest-body

        is equivalent to:

        if cond：
            if-body
            rest-body
        else:
            else-body
            rest-body
        
        By duplicating the rest-body, we can decompile the if-else block separately.
        """
        jump_offset = inst.get_jump_target()
        jump_index = self.index_of(jump_offset)
        this_index = self.index_of(inst.offset)
        cond = self.state.stack[-1]
        fallthrough_stack = self.state.stack.copy()
        jump_stack = self.state.stack.copy()

        if "IF_NOT_NONE" in inst.opname:
            cond = f"{cond} is None"
        elif "IF_NONE" in inst.opname:
            cond = f"{cond} is not None"
        elif "IF_TRUE" in inst.opname:
            cond = f"not {cond}"
        elif "IF_FALSE" in inst.opname:
            cond = f"{cond}"

        # POP_AND_JUMP / JUMP_OR_POP
        if "POP_JUMP" in inst.opname:
            jump_stack.pop()
            fallthrough_stack.pop()
        elif "OR_POP" in inst.opname:
            fallthrough_stack.pop()

        end_index_candidates = [len(self.instructions)]
        if self.state.inside_loop:
            end_index_candidates.append(self.state.loop_end_index)

        def qualified_jump(i: Instruction):
            return i.is_jump() and i.get_jump_target() >= jump_offset

        jump_targets = [i.get_jump_target() for i in self.instructions[this_index: jump_index] if qualified_jump(i)]
        max_jump = max(jump_targets)
        max_jump_index = self.index_of(max_jump)
        # else branch might have jumps, we need to find the end of the else
        all_jump_targets = [i.get_jump_target() for i in self.instructions[this_index: max_jump_index] if qualified_jump(i)]
        max_jump_index = self.index_of(max(all_jump_targets))
        # extend one more instruction, because sometimes if-body and else-body share the same instruction
        # TODO how to determine if we need to extend more instructions?
        max_jump_index += 1
        end_index_candidates.append(max_jump_index)

        end_index = min(end_index_candidates)

        with self.new_state(fallthrough_stack):
            self.decompile_range(this_index + 1, end_index)
            if_body = self.state.source_code
            if_body = add_indentation(if_body, self.indentation)
            if_end_stack = self.state.stack.copy()
        if_code = f"if {cond}:\n{if_body}"
        self.state.source_code += if_code

        with self.new_state(jump_stack):
            self.decompile_range(jump_index, end_index)
            else_body = self.state.source_code
        if else_body:
            else_body = add_indentation(else_body, self.indentation)
            else_code = f"else:\n{else_body}"
            self.state.source_code += else_code

        self.state.stack = if_end_stack
        return end_index

        # if "ASSERT" in self.instructions[this_index + 1].opname:
        #     with self.new_state(self.state.stack):
        #         self.decompile_range(this_index + 1, jump_index)
        #         source_code = self.state.source_code
        #     source_code = add_indentation(source_code, self.indentation)
        #     self.state.source_code += f"if not {cond}:\n{source_code}"
        #     return jump_index

        

        # if_body_start_offset = None
        # if_body_end_offset = None
        # last_index = this_index
        # conditions = [cond]
        # for _index in range(this_index, jump_index):
        #     _inst = self.instructions[_index]
        #     if "IF_FALSE" in _inst.opname or "IF_NOT_NONE" in _inst.opname or "IF_NONE" in _inst.opname:
        #         # JUMP_IF_FALSE, followed by "and", short-circuit evaluation
        #         # means we jump to the end of if-block if the condition is
        #         # false
        #         if if_body_end_offset is None:
        #             if_body_end_offset = _inst.get_jump_target()
        #         if _inst.get_jump_target() == if_body_end_offset:
        #             if _index != this_index:
        #                 with self.new_state(fallthrough_stack):
        #                     self.decompile_range(last_index + 1, _index)
        #                     source_code = self.state.source_code
        #                 self.state.source_code += source_code
        #                 conditions.append(self.state.stack[-1])
        #                 last_index = _index

        #             conditions.append("and")

        #             jump_stack = fallthrough_stack.copy()
        #             fallthrough_stack.pop()


        #     elif "IF_TRUE" in _inst.opname:
        #         # JUMP_IF_TRUE, followed by "or", short-circuit evaluation
        #         # means we jump to the start of if-block if the condition is
        #         # true
        #         if if_body_start_offset is None:
        #             if_body_start_offset = _inst.get_jump_target()
        #         if _inst.get_jump_target() == if_body_start_offset:
        #             if _index != this_index:
        #                 with self.new_state(fallthrough_stack):
        #                     self.decompile_range(last_index + 1, _index)
        #                     source_code = self.state.source_code
        #                 self.state.source_code += source_code
        #                 conditions.append(self.state.stack[-1])
        #                 last_index = _index
        #             conditions.append("or")

        #             jump_stack = fallthrough_stack.copy()
        #             fallthrough_stack.pop()
        #             # POP_AND_JUMP / JUMP_OR_POP
        #             if "POP_JUMP" in _inst.opname:
        #                 jump_stack.pop()
        #             elif "OR_POP" in _inst.opname:
        #                 pass

        # conditions.pop()

        # if if_body_start_offset is None:
        #     if_body_start_offset = self.instructions[last_index + 1].offset

        # if_body_start = self.index_of(if_body_start_offset)

        # if if_body_end_offset is None:
        #     # Don't know where the if body ends, so we have to find the next
        #     # jump instruction
        #     if_body_end = if_body_start + 1
        #     while if_body_end < len(
        #             self.instructions) and not self.instructions[if_body_end].is_jump():
        #         if_body_end += 1
        #     if if_body_end == len(self.instructions):
        #         if_body_end -= 1
        #     if_body_end_offset = self.instructions[if_body_end].offset
        # else:
        #     if_body_end = self.index_of(if_body_end_offset)
        # if jump_index < if_body_start:
        #     self.state.source_code += add_indentation(
        #         "continue\n", self.indentation)
        #     return

        # with self.new_state(fallthrough_stack):
        #     if else_code and self.instructions[if_body_end - 1].is_jump():
        #         # the last instruction is a jump, so it is not part of the if
        #         # body, but the jump out of the if-else block
        #         if_body_end -= 1
        #     self.decompile_range(if_body_start, if_body_end)
        #     if_code = "if " + " ".join(conditions) + ":\n"
        #     if_code = if_code + \
        #         add_indentation(self.state.source_code, self.indentation)

        # self.state.source_code += if_code + else_code
        # self.state.stack = fallthrough_stack

        # return max_jump_index if else_code else jump_index

    POP_JUMP_IF_TRUE = POP_JUMP_IF_FALSE = generic_jump_if
    POP_JUMP_FORWARD_IF_TRUE = POP_JUMP_FORWARD_IF_FALSE = generic_jump_if
    POP_JUMP_BACKWARD_IF_TRUE = POP_JUMP_BACKWARD_IF_FALSE = generic_jump_if
    POP_JUMP_FORWARD_IF_NONE = POP_JUMP_FORWARD_IF_NOT_NONE = generic_jump_if
    POP_JUMP_BACKWARD_IF_NONE = POP_JUMP_BACKWARD_IF_NOT_NONE = generic_jump_if
    JUMP_IF_TRUE_OR_POP = JUMP_IF_FALSE_OR_POP = generic_jump_if
    POP_JUMP_IF_NOT_NONE = POP_JUMP_BACKWARD_IF_NOT_NONE
    POP_JUMP_IF_NONE = POP_JUMP_BACKWARD_IF_NONE

    def SETUP_FINALLY(self, inst: Instruction):
        start_index = self.index_of(inst.offset)
        end_index = self.index_of(inst.get_jump_target())
        pop_block_index = [i for i, x in enumerate(
            self.instructions) if x.opname == "POP_BLOCK" and start_index <= i < end_index][-1]

        try_code = ""
        with self.new_state(self.state.stack):
            self.decompile_range(start_index + 1, pop_block_index)
            try_code = self.state.source_code
            try_code = add_indentation(try_code, self.indentation)
            try_code = "try:\n" + try_code

        finally_code = ""
        with self.new_state(self.state.stack):
            end_finally_index = [
                i for i, x in enumerate(
                    self.instructions) if x.opname == "END_FINALLY" and start_index <= i]
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

    def SETUP_WITH(self, inst: Instruction):
        """
        with expression as var:
            body

        is equivalent to:

        var = expression
        var.__enter__()
        try:
            body
        finally:
            var.__exit__()

        We find the start of `finally` by `WITH_EXCEPT_START`, and the end of `finally` by `POP_EXCEPT`.
        In early python version, the start is `WITH_CLEANUP_START` and the end is `WITH_CLEANUP_FINISH`.
        """
        start_index = self.index_of(inst.offset)
        with_except_index = [i for i, x in enumerate(
            self.instructions) if x.opname in ["WITH_EXCEPT_START", "WITH_CLEANUP_START"] and i > start_index][-1]
        end_index = with_except_index
        nop_instruction(self.instructions[end_index])

        # NOP PUSH_EXC_INFO and JUMP_FORWARD
        i = end_index - 1
        while end_index - i <= 2:
            _inst = self.instructions[i]
            if _inst.opname.startswith("JUMP") or _inst.opname == "PUSH_EXC_INFO":
                nop_instruction(_inst)
            i -= 1

        pop_except_indices = [i for i, x in enumerate(
            self.instructions) if x.opname in ["POP_EXCEPT", "WITH_CLEANUP_FINISH"] and i > end_index]
        if sys.version_info >= (3, 11):
            # Python 3.11 seems to have two `POP_EXCEPT` instructions, not sure why.
            pop_except_index = pop_except_indices[1]
        else:
            pop_except_index = pop_except_indices[0]
        for i in range(end_index, pop_except_index + 1):
            nop_instruction(self.instructions[i])
        tos = self.state.stack[-1]
        temp = self.get_temp_name()
        self.state.stack.append(f"{temp}.__exit__")
        self.state.stack.append(temp)
        with_clause = f"with {tos} as {temp}:\n"
        with_body = ""
        with self.new_state(self.state.stack):
            self.decompile_range(start_index + 1, end_index)
            with_body = self.state.source_code
            with_body = add_indentation(with_body, self.indentation)
            lines = with_body.splitlines()
            ans = []
            for line in lines:
                if f"{temp}.__exit__" in line or "None(None, None)" in line.strip():
                    # this is the line that calls __exit__, we need to remove it, as it is managed by `with` statement.
                    # `None(None, None)` is used for Python 3.11. Who knows why it loads three Nones but call with 2 args for the following simple code:
                    # def f():
                    #     with a:
                    #         print(2)
                    continue
                ans.append(line)
            with_body = "".join([x + "\n" for x in ans])

        self.state.source_code += with_clause + with_body
        return pop_except_index + 1

    BEFORE_WITH = SETUP_WITH

    def FOR_ITER(self, inst: Instruction):
        start_index = self.index_of(inst.offset)
        end_index = self.index_of(inst.get_jump_target())

        temp_name = self.get_temp_name()
        for_code = f"for {temp_name} in {self.state.stack.pop()}:\n"
        self.state.stack.append(temp_name)
        with self.new_state(self.state.stack, inside_loop=True, loop_start_index=start_index, loop_end_index=end_index):
            self.decompile_range(start_index + 1, end_index)
            code = self.state.source_code
            for_code = for_code + add_indentation(code, self.indentation)

        self.state.source_code += for_code
        return end_index

# ==================== Stack Manipulation Instructions ===================
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
        n = inst.argval
        tos = self.state.stack[-1]
        value = self.state.stack[- n]
        tos, value = value, tos
        self.state.stack[-1] = tos
        self.state.stack[- n] = value

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
            if last_inst[-1].opname == "KW_NAMES" or (len(
                    last_inst) > 1 and last_inst[-2].opname == "KW_NAMES" and last_inst[-1].opname == "PRECALL"):
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
        if "iter(" in func:
            # Why do we need this? Don't know. But sometimes CPython generates
            # CALL with argval=0, but the function actually needs an arg (for
            # list/set/map comprehension).
            pos_args = [func]
            func = self.state.stack.pop()
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
        pos_args = [(self.state.stack.pop())
                    for _ in range(inst.argval - len(kw_args))]
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

    def CALL_INTRINSIC_1(self, inst: Instruction):
        if inst.argrepr in [
            "INTRINSIC_1_INVALID",
            "INTRINSIC_IMPORT_STAR",
            "INTRINSIC_STOPITERATION_ERROR",
                "INTRINSIC_ASYNC_GEN_WRAP"]:
            # invalid intrinsic, skip
            pass
        elif inst.argrepr in ["INTRINSIC_TYPEVAR", "INTRINSIC_PARAMSPEC", "INTRINSIC_TYPEVARTUPLE", "INTRINSIC_SUBSCRIPT_GENERIC", "INTRINSIC_TYPEALIAS"]:
            # not tested, skip
            pass
        elif inst.argrepr == "INTRINSIC_PRINT":
            self.state.source_code += f"print({self.state.stack.pop()})\n"
            self.state.stack.append("None")
        elif inst.argrepr == "INTRINSIC_UNARY_POSITIVE":
            self.state.stack[-1] = f"+{self.state.stack[-1]}"
        elif inst.argrepr == "INTRINSIC_LIST_TO_TUPLE":
            return self.LIST_TO_TUPLE(inst)


# ==================== Container Related Instructions (tuple, list, set, d

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
        self.state.stack.append(
            f"{{{', '.join([f'{k}: {v}' for k, v in zip(keys, values)])}}}")
        self.replace_mutable_tos_with_temp()

    def BUILD_CONST_KEY_MAP(self, inst: Instruction):
        keys = eval(self.state.stack.pop())
        args = [self.state.stack.pop() for _ in range(inst.argval)]
        values = args[::-1]
        self.state.stack.append(
            f"{{{', '.join([f'{k}: {v}' for k, v in zip(keys, values)])}}}")
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
    # The extended args are already merged into the following instruction's
    # `inst.argval`.
    EXTENDED_ARG = generic_nop

    NOP = RESUME = SETUP_LOOP = POP_BLOCK = PRECALL = BEGIN_FINALLY = END_FINALLY = generic_nop

    MAKE_CELL = generic_nop

    RERAISE = generic_nop

    # our FOR_ITER is different from CPython's FOR_ITER (as it does not need
    # to explicitly consider the case of exhausted iterator), so we don't need
    # to do anything here
    END_FOR = generic_nop

# ==================== Unsupported Instructions =============================
    def unimplemented_instruction(self, inst: Instruction):
        raise NotImplementedError(f"Unsupported instruction: {inst.opname}")

    GET_YIELD_FROM_ITER = unimplemented_instruction

    # we don't support try-except/try-finally
    POP_EXCEPT = WITH_EXCEPT_START = JUMP_IF_NOT_EXC_MATCH = CHECK_EG_MATCH = PUSH_EXC_INFO = PREP_RERAISE_STAR = WITH_CLEANUP_FINISH = CALL_FINALLY = POP_FINALLY = WITH_CLEANUP_START = SETUP_EXCEPT = CHECK_EXC_MATCH = CLEANUP_THROW = unimplemented_instruction

    # we don't support async/await
    GET_AWAITABLE = GET_AITER = GET_ANEXT = END_ASYNC_FOR = BEFORE_ASYNC_WITH = SETUP_ASYNC_WITH = SEND = ASYNC_GEN_WRAP = unimplemented_instruction

    CACHE = unimplemented_instruction

    # we don't know these instructions
    PRINT_EXPR = COPY_DICT_WITHOUT_KEYS = unimplemented_instruction

    # we only support bytecode for functions
    IMPORT_STAR = unimplemented_instruction

    YIELD_FROM = SETUP_ANNOTATIONS = LOAD_BUILD_CLASS = MATCH_MAPPING = MATCH_SEQUENCE = MATCH_KEYS = MATCH_CLASS = unimplemented_instruction

    # don't find any interesting use case for these instructions
    CALL_INTRINSIC_2 = unimplemented_instruction

    def decompile_range(self, start: int, end: int):
        running_index = start
        while running_index < end:
            inst = self.instructions[running_index]
            method = getattr(
                Decompiler,
                inst.opname,
                Decompiler.unimplemented_instruction)
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
    def cleanup_instructions(code, instructions: List[Instruction]):
        propagate_line_nums(instructions)
        simplify_finally_statement(instructions)
        nop_unreachable_bytecode(code, instructions)

    def __init__(self, code: Union[CodeType, Callable]):
        if callable(code):
            code = code.__code__
        self.code = code
        instructions = list(convert_instruction(_)
                            for _ in dis.get_instructions(code))
        Decompiler.cleanup_instructions(code, instructions)
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
            if getattr(
                    Decompiler,
                    x,
                    Decompiler.unimplemented_instruction) is not Decompiler.unimplemented_instruction:
                opnames.append(x)
        return opnames

    @functools.lru_cache(maxsize=None)
    def decompile(
            self,
            indentation=4,
            temp_prefix: str = "__temp_",
            overwite_fn_name: Optional[str] = None) -> str:
        try:
            self.indentation = indentation
            self.temp_prefix = temp_prefix
            self.decompile_range(0, len(self.instructions))
            source_code = self.state.source_code
            # the header might have invalid function name in torchdynamo. only
            # optimize the function body.
            source_code = remove_some_temp(
                source_code, self.temp_prefix, indentation)
            header = get_function_signature(self.code, overwite_fn_name)
            # we cannot rely on `co_names`. For example, `from math import sqrt` will make `math` and `sqrt` in `co_names`.
            global_names = set(inst.argval for inst in dis.get_instructions(self.code) if inst.opname == "STORE_GLOBAL")
            global_statements = "global " + ", ".join(
                global_names) + "\n" if global_names else ""
            nonlocal_statement = "nonlocal " + ", ".join(
                self.code.co_freevars) + "\n" if self.code.co_freevars else ""
            source_code = global_statements + nonlocal_statement + source_code
            source_code = header + add_indentation(source_code, indentation)
            return source_code
        except Exception as e:
            raise DecompilationError(
                f"Failed to decompile {self.code.co_name}") from e

    def __hash__(self):
        return hash(self.code)


def decompile(code: Union[CodeType, Callable]):
    """Decompile a code object or a function."""
    return Decompiler(code).decompile()

