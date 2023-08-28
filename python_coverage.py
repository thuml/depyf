import dis
import sys
import inspect

from depyf import decompile, Decompiler


filepath = inspect.getfile(Decompiler)

codecontent = open(filepath, "r").read()

unsupported_opnames = []

for k in dis.opmap:
    if k not in codecontent:
        unsupported_opnames.append(k)

print("Python version:", sys.version)
if len(unsupported_opnames) == 0:
    print("All opnames are supported!")
else:
    print(f"Total {len(unsupported_opnames)} unsupported opnames:")
    for k in unsupported_opnames:
        print(k)
