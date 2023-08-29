"""Add a method to `dis.Instruction` to get jump target."""

import dis
import sys

py311 = sys.version_info >= (3, 11)

def get_jump_target(self: dis.Instruction):
    if self.opcode in dis.hasjabs:
        return self.argval
    elif self.opcode in dis.hasjrel:
        return self.offset + self.argval if not py311 else self.argval
    else:
        raise ValueError(f"Instruction {self.opname} does not have jump target")

dis.Instruction.get_jump_target = get_jump_target

del get_jump_target
