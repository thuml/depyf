from depyf import decompile
import dis
import sys

filepath = decompile.__code__.co_filename

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
