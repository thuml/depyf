from typing import List, Tuple, Dict, Union, Callable, Optional
import dataclasses
import dis

from .utils import (
    generate_dot_table,
)

all_jump_opcode_set = set(dis.hasjabs) | set(dis.hasjrel)


@dataclasses.dataclass(order=True, eq=True)
class BasicBlock:
    """A basic block without internal control flow. The bytecode in this block is executed sequentially.
    The block ends with a jump instruction or a return instruction."""
    code_start: int = dataclasses.field(init=False)
    code_end: int = dataclasses.field(init=False)
    code_range: Tuple[int, int] = dataclasses.field(init=False)
    simple_repr: str = dataclasses.field(init=False)
    full_repr: str = dataclasses.field(init=False)
    end_with_return: bool = dataclasses.field(init=False, default=False)
    end_with_direct_jmp: bool = dataclasses.field(init=False, default=False)
    end_with_if_jmp: bool = dataclasses.field(init=False, default=False)

    to_blocks: List['BasicBlock'] = dataclasses.field(default_factory=list)
    from_blocks: List['BasicBlock'] = dataclasses.field(default_factory=list)

    instructions: List[dis.Instruction] = dataclasses.field(default_factory=list)

    def __init__(self, instructions: List[dis.Instruction]):
        self.instructions = instructions
        self.to_blocks = []
        self.from_blocks = []
        self.code_start = self.instructions[0].offset
        self.code_end = self.instructions[-1].offset + 2
        self.code_range = (self.code_start, self.code_end)
        self.simple_repr = f"{self.code_range}"
        lines = []
        for inst in self.instructions:
            line = [">>" if inst.is_jump_target else "  ", str(inst.offset), inst.opname, str(inst.argval), f"({inst.argrepr})"]
            lines.append(line)
        self.full_repr = generate_dot_table(self.simple_repr, lines)
        self.end_with_return = self.instructions[-1].opname == "RETURN_VALUE"
        end_with_jmp = self.instructions[-1].opcode in all_jump_opcode_set
        self.end_with_direct_jmp = end_with_jmp and "IF" not in self.instructions[-1].opname
        self.end_with_if_jmp = end_with_jmp and "IF" in self.instructions[-1].opname

    def __str__(self):
        return self.simple_repr

    def __repr__(self):
        return self.simple_repr

    def __eq__(self, other):
        return self.code_start == other.code_start

    def jump_to_block(self, offset: int) -> 'BasicBlock':
        blocks = [b for b in self.to_blocks if b.code_start >= offset]
        if not blocks:
            raise ValueError(f"Cannot find block that starts at {offset}")
        return blocks[0]

    @staticmethod
    def find_the_first_block(blocks: List['BasicBlock'], offset: int) -> Optional['BasicBlock']:
        candidates = [b for b in blocks if b.code_start >= offset]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x.code_start)

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
            block.to_blocks.sort(key=lambda x: x.code_start)
            block.from_blocks.sort(key=lambda x: x.code_start)
        return blocks
