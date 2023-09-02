"""Add a method to `dis.Instruction` to get jump target."""

import dis
import sys

py311 = sys.version_info >= (3, 11)
all_jump_opcode_set = set(dis.hasjabs) | set(dis.hasjrel)

def get_jump_target(self: dis.Instruction):
    if self.opcode in all_jump_opcode_set:
        return int(self.argrepr.replace("to ", "").strip())
    # seems like a bug, "FOR_ITER" is in `dis.hasjrel`, but its `argval` is an absolute offset
    # if self.opcode in dis.hasjabs:
    #     return self.argval
    # elif self.opcode in dis.hasjrel:
    #     return self.offset + self.argval if not py311 else self.argval
    else:
        raise ValueError(f"Instruction {self.opname} does not have jump target")

dis.Instruction.get_jump_target = get_jump_target

del get_jump_target
