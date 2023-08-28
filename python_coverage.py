import dis
import sys
import inspect

from depyf import decompile, Decompiler


filepath = inspect.getfile(Decompiler)

codecontent = open(filepath, "r").read()

considered_opnames = []

for k in dis.opmap:
    if k in codecontent:
        considered_opnames.append(k)

all_opnames = set(dis.opmap)

considered_opnames = set(considered_opnames)

supported_opnames = set(Decompiler.supported_opnames())

print("Python version:", sys.version)
unconsidered = all_opnames - considered_opnames
unsupported = considered_opnames - supported_opnames
if len(unconsidered) == 0:
    print("All opnames are considered!")
else:
    print(f"Total {len(unconsidered)} unconsidered opnames:")
    for k in unconsidered:
        print(k)

if len(unsupported) == 0:
    print("All considered opnames are supported!")
else:
    print(f"Total {len(unsupported)} unsupported opnames:")
    for k in unsupported:
        print(k)
