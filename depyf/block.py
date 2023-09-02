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

    @property
    def fallthrough_block(self) -> Optional['BasicBlock']:
        offset = self.code_end
        blocks = [b for b in self.to_blocks if b.code_start >= offset]
        if not blocks:
            raise ValueError(f"Cannot find block that starts at {offset}")
        return blocks[0]

    @property
    def jump_to_block(self) -> 'BasicBlock':
        offset = self.instructions[-1].get_jump_target()
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
        for i, inst in enumerate(insts):
            if inst.opcode in all_jump_opcode_set:
                # both jump target and the instruction after the jump are block starts
                # even if this is a direct jump, the instruction after the jump is a block start
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
            if block.end_with_return:
                continue
            else:
                assert block.end_with_direct_jmp or block.end_with_if_jmp, f"Block {block} does not end with a jump or return"
                # here we cannot use the `jump_to_block/fallthrough_block` property, because the blocks are not connected yet
                to_block = BasicBlock.find_the_first_block(blocks, block.instructions[-1].get_jump_target())
                block.to_blocks.append(to_block)
                to_block.from_blocks.append(block)

                if block.end_with_if_jmp:
                    fallthrough_block = BasicBlock.find_the_first_block(blocks, block.code_end)
                    block.to_blocks.append(fallthrough_block)
                    fallthrough_block.from_blocks.append(block)

        for block in blocks:
            block.to_blocks.sort()
            block.from_blocks.sort()
        return blocks


@dataclasses.dataclass()
class IndentationBlock:
    """An indentation block consists several basic blocks. It represents any block that is indented,
     e.g. if-else, while, for, etc."""
    blocks: List[BasicBlock]
    start: int = dataclasses.field(init=False)
    end: int = dataclasses.field(init=False)

    def __init__(self, blocks: List[BasicBlock]):
        self.blocks = blocks
        self.start = self.blocks[0].code_start
        self.end = self.blocks[-1].code_end

    def __bool__(self):
        return bool(self.blocks)
