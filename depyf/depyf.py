"""A simple program to transform bytecode into more readable source code."""

import sys
import os
import dis
from types import CodeType
from typing import List, Tuple, Dict, Union
import dataclasses
import inspect

def get_jump_target(self: dis.Instruction):
    if self.opcode in dis.hasjabs:
        return self.argval
    elif self.opcode in dis.hasjrel:
        return self.offset + self.argval
    else:
        raise ValueError(f"Instruction {self.opname} does not have jump target")

dis.Instruction.get_jump_target = get_jump_target

@dataclasses.dataclass
class BasicBlock:
    """A basic block without control flow"""
    instructions: List[dis.Instruction]
    to_blocks: List['BasicBlock']
    from_blocks: List['BasicBlock']

    def code_range(self):
        return (self.instructions[0].offset, self.instructions[-1].offset + 2)

    def jump_to_block(self, offset: int):
        return [b for b in self.to_blocks if b.instructions[0].offset == offset][0]


def decompose_basic_blocks(code: CodeType) -> List[BasicBlock]:
    """Decompose a code object into basic blocks."""
    insts = list(dis.get_instructions(code))
    block_starts = set()
    block_starts.add(0)
    jumps = set(dis.hasjabs) | set(dis.hasjrel)
    for i, inst in enumerate(insts):
        # the instruction below jump starts a new block
        if inst.opcode in jumps or inst.opname == "RETURN_VALUE":
            block_starts.add(inst.offset + 2)
        # the instruction is the target of a jump
        if inst.is_jump_target:
            block_starts.add(inst.offset)
        if inst.opcode in dis.hasjabs:
            block_starts.add(inst.argval)
        if inst.opcode in dis.hasjrel:
            block_starts.add(inst.offset + inst.argval)
    block_starts.add(insts[-1].offset + 2)
    block_starts = sorted(block_starts)
    # split into basic blocks
    blocks = []
    for start, end in zip(block_starts[:-1], block_starts[1:]):
        block_insts = [inst for inst in insts if start <= inst.offset < end]
        blocks.append(BasicBlock(block_insts, [], []))
    # connect basic blocks
    for block in blocks:
        last_inst = block.instructions[-1]
        if last_inst.opcode in jumps:
            jump_offset = last_inst.get_jump_target()
            fallthrough_offset = last_inst.offset + 2
            to_block = [b for b in blocks if b.instructions[0].offset == jump_offset][0]
            fallthrough_block = [b for b in blocks if b.instructions[0].offset == fallthrough_offset][0]
            block.to_blocks.append(to_block)
            block.to_blocks.append(fallthrough_block)
            to_block.from_blocks.append(block)
            fallthrough_block.from_blocks.append(block)
    return blocks


temp_count = 0


def get_temp_name():
    global temp_count
    temp_count += 1
    return f"__temp_{temp_count}"


def decompile_block(block: BasicBlock, stack: List[str], indentation=4) -> str:
    """Decompile a basic block into source code.
    The `stack` holds expressions in string, like "3 + 4".
    """
    source_code = ""
    for inst in block.instructions:
        if inst.opname == "LOAD_CONST":
            # `inst.argval` is the constant value, we have to use `repr` to get the source code
            stack.append(repr(inst.argval))
        elif inst.opname == "LOAD_FAST" or inst.opname == "LOAD_GLOBAL" or inst.opname == "LOAD_DEREF" or inst.opname == "LOAD_CLOSURE" or inst.opname == "LOAD_NAME" or inst.opname == "LOAD_CLASSDEREF":
            # `inst.argval` is the variable name, in string
            stack.append(inst.argval)
        elif inst.opname == "LOAD_ATTR":
            stack.append(f"getattr({stack.pop()}, {repr(inst.argval)})")
        elif inst.opname == "LOAD_METHOD":
            stack.append(f"{stack.pop()}.{inst.argval}")
        elif inst.opname == "LOAD_ASSERTION_ERROR":
            stack.append("AssertionError")
        elif inst.opname == "RAISE_VARARGS":
            if inst.argval == 0:
                source_code += "raise\n"
            elif inst.argval == 1:
                source_code += f"raise {stack.pop()}\n"
            elif inst.argval == 2:
                tos = stack.pop()
                tos1 = stack.pop()
                source_code += f"raise {tos1} from {tos}\n"
        elif inst.opname == "STORE_FAST" or inst.opname == "STORE_GLOBAL" or inst.opname == "STORE_DEREF" or inst.opname == "STORE_NAME":
            source_code += f"{inst.argval} = {stack.pop()}\n"
            # stack.append(inst.argval)
        elif inst.opname == "STORE_SUBSCR":
            index = stack.pop()
            x = stack.pop()
            value = stack.pop()
            source_code += f"{x}[{index}] = {value}\n"
        elif inst.opname == "STORE_ATTR":
            x = stack.pop()
            value = stack.pop()
            source_code += f"{x}.{inst.argval} = {value}\n"
        elif inst.opname == "DELETE_SUBSCR":
            index = stack.pop()
            x = stack.pop()
            source_code += f"del {x}[{index}]\n"
        elif inst.opname == "DELETE_NAME" or inst.opname == "DELETE_FAST" or inst.opname == "DELETE_GLOBAL" or inst.opname == "DELETE_DEREF":
            x = inst.argval
            source_code += f"del {x}\n"
        elif inst.opname == "DELETE_ATTR":
            x = stack.pop()
            source_code += f"del {x}.{inst.argval}\n"
        elif inst.opname == "IMPORT_NAME":
            name = inst.argval
            fromlist = stack.pop()
            level = stack.pop()
            source_code += f"{inst.argval} = __import__({repr(name)}, fromlist={fromlist}, level={level})\n"
            stack.append(name)
        elif inst.opname == "IMPORT_FROM":
            name = inst.argval
            module = stack[-1]
            source_code += f"{name} = {module}.{name}\n"
            stack.append(name)
        elif inst.opname == "UNARY_NEGATIVE" or inst.opname == "UNARY_POSITIVE" or inst.opname == "UNARY_INVERT" or inst.opname == "UNARY_NOT":
            op = {
                "UNARY_NEGATIVE": "-",
                "UNARY_POSITIVE": "+",
                "UNARY_INVERT": "~",
                "UNARY_NOT": "not",
            }[inst.opname]
            stack.append(f"({op} {stack.pop()})")
        elif inst.opname == "GET_ITER":
            raise NotImplementedError(f"Unsupported instruction: {inst.opname}")
            # "GET_YIELD_FROM_ITER" is not supported
            stack.append(f"iter({stack.pop()})")
        elif inst.opname == "BINARY_MULTIPLY" or inst.opname == "BINARY_ADD" or inst.opname == "BINARY_SUBTRACT" or inst.opname == "BINARY_TRUE_DIVIDE" or inst.opname == "BINARY_FLOOR_DIVIDE" or inst.opname == "BINARY_MODULO" or inst.opname == "BINARY_POWER" or inst.opname == "BINARY_AND" or inst.opname == "BINARY_OR" or inst.opname == "BINARY_XOR" or inst.opname == "BINARY_LSHIFT" or inst.opname == "BINARY_RSHIFT" or inst.opname == "BINARY_MATRIX_MULTIPLY":
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
        elif inst.opname == "INPLACE_MULTIPLY" or inst.opname == "INPLACE_ADD" or inst.opname == "INPLACE_SUBTRACT" or inst.opname == "INPLACE_TRUE_DIVIDE" or inst.opname == "INPLACE_FLOOR_DIVIDE" or inst.opname == "INPLACE_MODULO" or inst.opname == "INPLACE_POWER" or inst.opname == "INPLACE_AND" or inst.opname == "INPLACE_OR" or inst.opname == "INPLACE_XOR" or inst.opname == "INPLACE_LSHIFT" or inst.opname == "INPLACE_RSHIFT" or inst.opname == "INPLACE_MATRIX_MULTIPLY":
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
        elif inst.opname == "BINARY_SUBSCR":
            rhs = stack.pop()
            lhs = stack.pop()
            stack.append(f"{lhs}[{rhs}]")
        elif inst.opname == "COMPARE_OP":
            rhs = stack.pop()
            lhs = stack.pop()
            stack.append(f"({lhs} {inst.argval} {rhs})")
        elif inst.opname == "IS_OP":
            rhs = stack.pop()
            lhs = stack.pop()
            op = "is" if inst.argval == 0 else "is not"
            stack.append(f"({lhs} {op} {rhs})")
        elif inst.opname == "CONTAINS_OP":
            rhs = stack.pop()
            lhs = stack.pop()
            op = "in" if inst.argval == 0 else "not in"
            stack.append(f"({lhs} {op} {rhs})")
        elif inst.opname == "POP_JUMP_IF_FALSE" or inst.opname == "POP_JUMP_IF_TRUE":
            cond = stack.pop()
            jump_offset = inst.get_jump_target()
            fallthrough_offset = inst.offset + 2
            if inst.opname == "POP_JUMP_IF_FALSE":
                source_code += f"if {cond}:\n"
            elif inst.opname == "POP_JUMP_IF_TRUE":
                source_code += f"if not {cond}:\n"
            source_code += decompile_block(block.jump_to_block(fallthrough_offset), stack.copy())
            source_code += "else:\n"
            source_code += decompile_block(block.jump_to_block(jump_offset), stack.copy())
        elif inst.opname == "JUMP_IF_TRUE_OR_POP" or inst.opname == "JUMP_IF_FALSE_OR_POP":
            # not tested, don't know how to force the compiler to generate this
            cond = stack[-1]
            jump_offset = inst.get_jump_target()
            fallthrough_offset = inst.offset + 2
            if inst.opname == "JUMP_IF_TRUE_OR_POP":
                source_code += f"if not {cond}:\n"
            elif inst.opname == "JUMP_IF_FALSE_OR_POP":
                source_code += f"if {cond}:\n"
            # The fallthrough block should pop one value from the stack
            source_code += decompile_block(block.jump_to_block(fallthrough_offset), stack.copy()[:-1])
            source_code += "else:\n"
            source_code += decompile_block(block.jump_to_block(jump_offset), stack.copy())
        elif inst.opname == "JUMP_FORWARD" or inst.opname == "JUMP_ABSOLUTE":
            jump_offset = inst.get_jump_target()
            if jump_offset > block.instructions[0].offset:
                source_code += decompile_block(block.jump_to_block(jump_offset), stack.copy())
            else:
                raise NotImplementedError(f"Unsupported jump backward")
        elif inst.opname == "RETURN_VALUE":
            source_code += f"return {stack[-1]}\n"
        elif inst.opname == "YIELD_VALUE":
            source_code += f"yield {stack[-1]}\n"
        elif inst.opname == "GEN_START":
            # stack.pop()
            assert inst.argval == 0, "Only generator expression is supported"
        elif inst.opname == "GET_LEN":
            stack.append(f"len({stack[-1]})")
        elif inst.opname == "CALL_FUNCTION" or inst.opname == "CALL_METHOD":
            args = [(stack.pop()) for _ in range(inst.argval)]
            args = args[::-1]
            func = stack.pop()
            temp = get_temp_name()
            source_code += f"{temp} = {func}({', '.join(args)})\n"
            stack.append(temp)
        elif inst.opname == "CALL_FUNCTION_KW":
            kw_args = eval(stack.pop())
            kwcalls = []
            for name in kw_args:
                kwcalls.append(f"{name}={stack.pop()}")
            pos_args = [(stack.pop()) for _ in range(inst.argval - len(kw_args))]
            pos_args = pos_args[::-1]
            func = stack.pop()
            temp = get_temp_name()
            source_code += f"{temp} = {func}({', '.join(pos_args + kwcalls)})\n"
            stack.append(temp)
        elif inst.opname == "CALL_FUNCTION_EX":
            if inst.argval == 0:
                args = stack.pop()
                func = stack.pop()
                temp = get_temp_name()
                source_code += f"{temp} = {func}(*{args})\n"
                stack.append(temp)
            elif inst.argval == 1:
                kw_args = stack.pop()
                args = stack.pop()
                func = stack.pop()
                temp = get_temp_name()
                source_code += f"{temp} = {func}(*{args}, **{kw_args})\n"
                stack.append(temp)
        elif inst.opname == "POP_TOP":
            stack.pop()
        elif inst.opname == "UNPACK_SEQUENCE":
            varname = stack.pop()
            for i in range(inst.argval):
                stack.append(f"{varname}[{inst.argval - 1 - i}]")
        elif inst.opname == "UNPACK_EX":
            varname = stack.pop()
            stack.append(f"list({varname}[{inst.argval}:])")
            for i in range(inst.argval):
                stack.append(f"{varname}[{inst.argval - 1 - i}]")
        elif inst.opname == "BUILD_SLICE":
            tos = stack.pop()
            tos1 = stack.pop()
            if inst.argval == 2:
                stack.append(f"slice({tos1}, {tos})")
            elif inst.argval == 3:
                tos2 = stack.pop()
                stack.append(f"slice({tos2}, {tos1}, {tos})")
        elif inst.opname == "BUILD_TUPLE":
            args = [stack.pop() for _ in range(inst.argval)]
            args = args[::-1]
            if inst.argval == 1:
                stack.append(f"({args[0]},)")
            else:
                stack.append(f"({', '.join(args)})")
        elif inst.opname == "BUILD_LIST":
            args = [stack.pop() for _ in range(inst.argval)]
            args = args[::-1]
            stack.append(f"[{', '.join(args)}]")
        elif inst.opname == "BUILD_SET":
            if inst.argval == 0:
                stack.append("set()")
            else:
                args = [stack.pop() for _ in range(inst.argval)]
                args = args[::-1]
                stack.append(f"{{{', '.join(args)}}}")
        elif inst.opname == "BUILD_MAP":
            args = [stack.pop() for _ in range(inst.argval * 2)]
            args = args[::-1]
            keys = args[::2]
            values = args[1::2]
            stack.append(f"{{{', '.join([f'{k}: {v}' for k, v in zip(keys, values)])}}}")
        elif inst.opname == "BUILD_CONST_KEY_MAP":
            keys = eval(stack.pop())
            args = [stack.pop() for _ in range(inst.argval)]
            values = args[::-1]
            stack.append(f"{{{', '.join([f'{k}: {v}' for k, v in zip(keys, values)])}}}")
        elif inst.opname == "BUILD_STRING":
            args = [stack.pop() for _ in range(inst.argval)]
            args = args[::-1]
            values = " + ".join(args)
            stack.append(values)
        elif inst.opname == "LIST_TO_TUPLE":
            item = stack.pop()
            stack.append(f"tuple({item})")
        elif inst.opname == "LIST_EXTEND":
            assert inst.argval == 1, "Only tested for argval==1"
            values = stack.pop()
            temp = get_temp_name()
            x = stack.pop()
            source_code += f"{temp} = {x}\n"
            source_code += f"{temp}.extend({values})\n"
            stack.append(temp)
        elif inst.opname == "SET_UPDATE" or inst.opname == "DICT_UPDATE" or inst.opname == "DICT_MERGE":
            assert inst.argval == 1, "Only tested for argval==1"
            values = stack.pop()
            temp = get_temp_name()
            x = stack.pop()
            source_code += f"{temp} = {x}\n"
            source_code += f"{temp}.update({values})\n"
            stack.append(temp)
        elif inst.opname == "FORMAT_VALUE":
            func, spec = inst.argval
            if spec:
                form_spec = stack.pop()
                value = stack.pop()
                stack.append(f"format({value}, {form_spec})")
            else:
                value = stack.pop()
                func = str if func is None else func
                stack.append(f"{func.__name__}({value})")
        elif inst.opname == "ROT_N" or inst.opname == "ROT_TWO" or inst.opname == "ROT_THREE" or inst.opname == "ROT_FOUR":
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
        elif inst.opname == "NOP":
            continue
        elif inst.opname == "DUP_TOP":
            # not tested
            stack.append(stack[-1])
        elif inst.opname == "DUP_TOP_TWO":
            # not tested
            tos = stack[-1]
            tos1 = stack[-2]
            stack.append(tos1)
            stack.append(tos)
        else:
            # "EXTENDED_ARG" is unsupported
            # "MAKE_FUNCTION" is unsupported
            # Coroutine opcodes like "GET_AWAITABLE"/"GET_AITER"/"GET_ANEXT"/"END_ASYNC_FOR"/"BEFORE_ASYNC_WITH"/"SETUP_ASYNC_WITH" are unsupported
            # "PRINT_EXPR"/"COPY_DICT_WITHOUT_KEYS" is I don't know
            # "SET_ADD"/"MAP_ADD"/"LIST_APPEND" are unsupported, no list/set/map comprehension
            # "YIELD_FROM"/"SETUP_ANNOTATIONS" is unsupported
            # "IMPORT_STAR" is unsupported, because we only support bytecode for functions
            # "POP_BLOCK"/"POP_EXCEPT"/"RERAISE"/"WITH_EXCEPT_START"/"JUMP_IF_NOT_EXC_MATCH"/"SETUP_FINALLY" is unsupported, this means we don't support try-except/try-finally
            # "LOAD_BUILD_CLASS"/"SETUP_WITH" is unsupported
            # "MATCH_MAPPING"/"MATCH_SEQUENCE"/"MATCH_KEYS"/"MATCH_CLASS" is unsupported
            # "FOR_ITER"/"GET_ITER" is unsupported
            raise NotImplementedError(f"Unsupported instruction: {inst.opname}")

    source_code = "".join([" " * indentation + line + "\n" for line in source_code.splitlines()])
    return source_code


def get_function_signature_from_codeobject(code_obj):
    # Extract all required details from the code object
    arg_names = code_obj.co_varnames[:code_obj.co_argcount]
    return ', '.join(arg_names)

def decompile(code: CodeType):
    global temp_count
    temp_count = 0
    blocks = decompose_basic_blocks(code)
    header = f"def {code.co_name}({get_function_signature_from_codeobject(code)}):\n"
    source_code = decompile_block(blocks[0], [])
    source_code = header + source_code
    return source_code
