#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
torch.compile internals - API walkthrough

Order:
  1. bytecode / CodeType / function / globals / frame
  2. register_bytecode_hook -> see Dynamo output (raw bytecode)
  3. mini_decompile -> make bytecode readable
  4. __compiled_fn in eager / inductor backends
  5. lazy_format_graph_code -> FX Graph transformation stages
  6. graph break -> resume recursive structure
  7. _debug_get_cache_entry_list + recompilation + dynamic shapes experiments
  8. guard tree traversal
  9. simple dispatch wrapper (hand-written, no depyf dependency)
  10. summary
"""
import dis
import sys
import os
import types
import textwrap
import contextlib

import torch
import torch._dynamo
import torch._dynamo.convert_frame as _cf
from torch._dynamo.eval_frame import (
    _debug_get_cache_entry_list,
    innermost_fn,
)

assert torch.cuda.is_available(), "This script requires CUDA"
torch.set_default_device("cuda")

def section(title):
    print("\n" + "=" * 70)
    print("  " + title)
    print("=" * 70 + "\n")

def subsection(title):
    print("\n--- " + title + " ---\n")


# =========================================================================
# Part 1: bytecode / CodeType / function / globals / frame
# =========================================================================

section("Part 1: bytecode / CodeType / function / globals / frame")

def greet(name):
    return "hello " + name

code = greet.__code__
print("A Python function = CodeType + __globals__ + __closure__")
print()
print("greet.__code__ =", code)
print("  co_code      =", code.co_code[:16].hex(), " (raw bytes)")
print("  co_name      =", repr(code.co_name))
print("  co_varnames  =", code.co_varnames, "  (local vars, including params)")
print("  co_consts    =", code.co_consts, "  (literal constants)")
print("  co_names     =", code.co_names, "  (global name references)")
print("  co_argcount  =", code.co_argcount)

subsection("co_names + __globals__: how LOAD_GLOBAL works")

def fn_using_global(x):
    return torch.relu(x)

code2 = fn_using_global.__code__
print("fn_using_global.co_names:", code2.co_names)
print()
print("LOAD_GLOBAL looks up the name in co_names, then fetches from __globals__:")
for inst in dis.get_instructions(code2):
    if inst.opname == "LOAD_GLOBAL":
        name = inst.argval
        val = fn_using_global.__globals__.get(name, "NOT FOUND")
        print("  LOAD_GLOBAL %s -> __globals__['%s'] = %s" % (name, name, type(val)))
print()
print("Dynamo exploits this: injects __compiled_fn_xxx into fn.__globals__,")
print("then transformed_code references it via co_names + LOAD_GLOBAL.")

subsection("CodeType -> function / compile() + exec()")

new_fn = types.FunctionType(code, greet.__globals__, "cloned_greet")
print("types.FunctionType(code, globals, name) constructs a function from code:")
print("  new_fn('world') =", new_fn("world"))
print()

src_text = "def dynamic_add(a, b):\n    return a + b\n"
print("compile() turns source text into a CodeType:")
print("  source text =", repr(src_text))
code_from_text = compile(src_text, "<dynamic>", "exec")
print("  compile(src, '<dynamic>', 'exec') =", code_from_text)
print("  type:", type(code_from_text))
print()
print("  The compiled code contains a nested code object for the function:")
for const in code_from_text.co_consts:
    if isinstance(const, types.CodeType):
        print("    found nested CodeType:", const)
        print("    co_name =", const.co_name, ", co_argcount =", const.co_argcount)
print()
ns = {}
exec(code_from_text, ns)
dynamic_add = ns["dynamic_add"]
print("  After exec(), the function is alive in the namespace:")
print("    dynamic_add =", dynamic_add)
print("    dynamic_add(3, 4) =", dynamic_add(3, 4))
print("    dynamic_add.__code__ =", dynamic_add.__code__)
print("    dynamic_add.__code__.co_consts =", dynamic_add.__code__.co_consts)
print()
print("  This is exactly how Inductor loads generated kernels:")
print("    PyCodeCache.load_by_key_path reads a .py file -> exec() -> returns callable")

subsection("Frame: the runtime context of a function call")

def show_frame(x):
    f = sys._getframe()
    print("  current frame:", f)
    print("  caller  frame:", f.f_back)
    print()
    print("  f.f_code     =", f.f_code)
    print("  f.f_locals   =", {k: v for k, v in f.f_locals.items() if k != "f"})
    print("  f.f_lineno   =", f.f_lineno)
    return x + 1

print("CodeType is static and immutable; Frame is created per call at runtime:")
print("  Frame = code + locals dict + globals dict + call stack position")
print()
print("Calling show_frame(42):")
show_frame(42)
print()
print("Dynamo's C-level eval_frame hook fires AFTER frame creation, BEFORE execution.")
print("It looks up frame.f_code -> finds CacheEntry chain -> swaps in transformed_code.")

subsection("How Dynamo associates transformed_code with a function")

print(
    "Dynamo does NOT do fn.__code__ = new_code (that's what dispatch_to_code does).\n"
    "Instead, it uses CPython's code extra mechanism:\n"
    "  code objects have a hidden co_extra pointer (C level)\n"
    "  Dynamo stores a CacheEntry linked list there\n"
    "  each CacheEntry = (guard_manager, transformed_code)\n"
    "\n"
    "eval_frame hook:\n"
    "  1. frame.f_code -> retrieve CacheEntry chain\n"
    "  2. check guards one by one\n"
    "  3. first match -> swap in transformed_code for this frame\n"
    "  4. hand back to CPython to execute\n"
)


# =========================================================================
# Part 2: register_bytecode_hook
# =========================================================================

section("Part 2: register_bytecode_hook -- see Dynamo's compilation output")

hook_results = []

def my_hook(old_code, new_code):
    hook_results.append((old_code, new_code))
    return None

handle = _cf.register_bytecode_hook(my_hook)

def add_fn(x, y):
    return x + y

torch._dynamo.reset()
compiled_add = torch.compile(add_fn, backend="eager")
out = compiled_add(torch.randn(3), torch.randn(3))
handle.remove()

print("Hook captured %d compilation event(s)" % len(hook_results))
if hook_results:
    old_c, new_c = hook_results[0]
    print("\nOriginal bytecode (old_code = add_fn.__code__):")
    dis.dis(old_c)
    print("\ntransformed_code (new_code, Dynamo's replacement bytecode):")
    dis.dis(new_c)
    print("\nWe can see LOAD_GLOBAL __compiled_fn_xxx in the transformed_code,")
    print("but raw bytecode is hard to read -> we need a decompiler.")


# =========================================================================
# Part 3: mini_decompile
# =========================================================================

section("Part 3: mini_decompile -- turn bytecode into readable pseudo-code")

print("Decompilation = simulate CPython's stack machine with strings instead of real values.\n")

def mini_decompile(code_obj):
    """Minimal decompiler: handles only the simplest opcodes.
    For complex bytecode (graph break etc.), use depyf.decompile instead."""
    stack = []
    lines = []
    args = list(code_obj.co_varnames[:code_obj.co_argcount])
    lines.append("def %s(%s):" % (code_obj.co_name, ", ".join(args)))

    for inst in dis.get_instructions(code_obj):
        op = inst.opname
        if op == "RESUME":
            continue
        elif op in ("LOAD_FAST", "LOAD_DEREF"):
            stack.append(inst.argval)
        elif op == "LOAD_CONST":
            stack.append(repr(inst.argval))
        elif op == "LOAD_GLOBAL":
            if inst.argval is not None:
                stack.append(str(inst.argval))
        elif op == "PUSH_NULL":
            stack.append(None)
        elif op == "STORE_FAST":
            val = stack.pop() if stack else "???"
            lines.append("    %s = %s" % (inst.argval, val))
        elif op == "DELETE_FAST":
            pass
        elif op.startswith("BINARY_OP"):
            right = stack.pop() if stack else "???"
            left = stack.pop() if stack else "???"
            stack.append("(%s %s %s)" % (left, inst.argrepr, right))
        elif op == "COMPARE_OP":
            right = stack.pop() if stack else "???"
            left = stack.pop() if stack else "???"
            stack.append("(%s %s %s)" % (left, inst.argrepr, right))
        elif op == "CALL":
            nargs = inst.argval
            call_args = []
            for _ in range(nargs):
                if stack:
                    call_args.insert(0, stack.pop())
            func = stack.pop() if stack else "???"
            if stack and stack[-1] is None:
                stack.pop()
            stack.append("%s(%s)" % (func, ", ".join(str(a) for a in call_args)))
        elif op == "POP_TOP":
            if stack:
                val = stack.pop()
                if val is not None:
                    lines.append("    " + str(val))
        elif op in ("RETURN_VALUE", "RETURN_CONST"):
            if op == "RETURN_CONST":
                lines.append("    return %s" % repr(inst.argval))
            elif stack:
                lines.append("    return %s" % stack.pop())
        elif op == "LOAD_ATTR":
            if stack:
                obj = stack.pop()
                stack.append("%s.%s" % (obj, inst.argval))
        elif op == "BINARY_SUBSCR":
            if len(stack) >= 2:
                idx = stack.pop()
                obj = stack.pop()
                stack.append("%s[%s]" % (obj, idx))
        elif op == "COPY":
            if len(stack) >= inst.argval:
                stack.append(stack[-inst.argval])
        elif op == "SWAP":
            if len(stack) >= inst.argval:
                stack[-1], stack[-inst.argval] = stack[-inst.argval], stack[-1]
        elif op == "BUILD_TUPLE":
            items = []
            for _ in range(inst.argval):
                if stack:
                    items.insert(0, str(stack.pop()))
            stack.append("(%s,)" % ", ".join(items) if len(items) == 1
                         else "(%s)" % ", ".join(items))
        # skip all other opcodes silently
    return "\n".join(lines)

def test_fn(a, b):
    c = a + b
    d = c * a
    return d

print("Verify with a simple function:")
print("  original: def test_fn(a, b): c = a+b; d = c*a; return d")
print("  bytecode:")
dis.dis(test_fn)
print("\n  mini_decompile:")
print(textwrap.indent(mini_decompile(test_fn.__code__), "  "))

subsection("Decompile Dynamo's transformed_code")

if hook_results:
    _, new_c = hook_results[0]
    print("mini_decompile(transformed_code):")
    print(textwrap.indent(mini_decompile(new_c), "  "))

    from depyf import decompile
    print("\ndepyf.decompile(transformed_code):")
    print(textwrap.indent(decompile(new_c), "  "))

    print("\nKey observation:")
    compiled_names = [n for n in new_c.co_names if n.startswith("__compiled")]
    print("  transformed_code calls: %s" % compiled_names)
    print("  This __compiled_fn_xxx is the callable returned by the backend.")
    print("  -> It points to different things depending on the backend. Let's look.")


# =========================================================================
# Part 4: __compiled_fn in eager / inductor backends
# =========================================================================

section("Part 4: __compiled_fn under different backends")

subsection("eager backend: __compiled_fn = gm.forward")

torch._dynamo.reset()
eager_gms = []

def eager_backend(gm, example_inputs):
    eager_gms.append(gm)
    return gm.forward

def mul_fn(x, y):
    return x * y + x

compiled_eager = torch.compile(mul_fn, backend=eager_backend)
compiled_eager(torch.randn(3), torch.randn(3))

entries_eager = _debug_get_cache_entry_list(mul_fn)
if entries_eager and eager_gms:
    gm = eager_gms[0]
    code_e = entries_eager[0].code
    for cn in code_e.co_names:
        if not cn.startswith("__compiled"):
            continue
        actual = mul_fn.__globals__.get(cn)
        inner = innermost_fn(actual)
        print("globals['%s']:" % cn)
        print("  type(actual)     =", type(actual))
        print("  innermost_fn     =", inner)
        print()

        if hasattr(inner, "__func__"):
            print("  inner.__func__ is gm.forward.__func__:",
                  inner.__func__ is gm.forward.__func__)
        print()

        print("  gm.code (Python code generated from FX Graph):")
        print(textwrap.indent(gm.code, "    "))

        print("  gm.graph.print_tabular() (FX Graph in table form):")
        gm.graph.print_tabular()

        print()
        print("  gm.print_readable():")
        try:
            readable = gm.print_readable(print_output=False)
            for line in readable.split("\n")[:20]:
                print("    " + line)
        except Exception:
            pass

subsection("inductor backend: __compiled_fn -> CompiledFxGraph -> kernel file")

torch._dynamo.reset()

def relu_fn(x):
    return torch.relu(x) + 1.0

compiled_ind = torch.compile(relu_fn, backend="inductor")
compiled_ind(torch.randn(8))

entries_ind = _debug_get_cache_entry_list(relu_fn)
if entries_ind:
    code_i = entries_ind[0].code
    for cn in code_i.co_names:
        if not cn.startswith("__compiled"):
            continue
        actual = relu_fn.__globals__.get(cn)
        inner = innermost_fn(actual)
        print("globals['%s']:" % cn)
        print("  type(actual) =", type(actual))
        print("  type(inner)  =", type(inner))
        print()
        print("  Inductor call chain:")
        print("    __compiled_fn_xxx")
        print("    -> @disable wrapper (prevents recursive compilation)")
        print("    -> aot_module_simplified")
        print("    -> CompiledFxGraph.__call__")
        print("    -> current_callable (codegen'd call function)")

subsection("Intercept inductor output via patch load_by_key_path")

from torch._inductor.codecache import PyCodeCache
from unittest.mock import patch

inductor_codes = []
_orig_load = PyCodeCache.load_by_key_path.__func__

@classmethod
def patched_load(cls, key, path, linemap=None, attrs=None):
    src = open(path).read()
    inductor_codes.append({"key": key, "path": path, "src": src})
    return _orig_load(cls, key, path, linemap, attrs)

torch._dynamo.reset()

def relu_fn2(x):
    return torch.relu(x) * 2

with patch.object(PyCodeCache, "load_by_key_path", patched_load):
    compiled_relu2 = torch.compile(relu_fn2, backend="inductor")
    compiled_relu2(torch.randn(16))

print("\nload_by_key_path was called %d time(s):" % len(inductor_codes))
for i, item in enumerate(inductor_codes):
    print("\n  [%d] key = %s..." % (i, item["key"][:30]))
    print("      path = %s" % item["path"])
    src_lines = item["src"].split("\n")
    print("      total lines: %d" % len(src_lines))
    # show the actual kernel code (inside the pybinding/triton string), not boilerplate
    in_kernel = False
    kernel_lines = []
    for line in src_lines:
        if "def call(" in line or "def triton_" in line or "extern \"C\"" in line:
            in_kernel = True
        if in_kernel:
            kernel_lines.append(line)
        if in_kernel and line.strip() == "":
            break
    if kernel_lines:
        print("      kernel excerpt:")
        for kl in kernel_lines[:20]:
            print("        " + kl)
        if len(kernel_lines) > 20:
            print("        ...")

print(
    "\nKey difference between backends:\n"
    "  eager:    __compiled_fn = gm.forward (runs FX Graph Python code directly)\n"
    "  inductor: __compiled_fn = CompiledFxGraph (runs codegen'd Triton/C++ kernel)\n"
    "  Both are wrapped with @disable to prevent Dynamo from recursively compiling.\n"
)


# =========================================================================
# Part 5: lazy_format_graph_code -- FX Graph stages
# =========================================================================

section("Part 5: lazy_format_graph_code -- FX Graph transformation stages")

print(
    "During compilation, the FX Graph goes through multiple stages:\n"
    "  1. Captured Graph   -- raw graph from Dynamo's symbolic execution\n"
    "  2. BEFORE PRE GRAD  -- before pre-grad optimizations\n"
    "  3. Joint graph      -- AOT Autograd's joint forward+backward graph\n"
    "  4. Forward graph    -- split forward graph\n"
    "  5. Backward graph   -- split backward graph\n"
    "\n"
    "These are passed to the logger via lazy_format_graph_code().\n"
    "depyf intercepts by replacing fn.__code__ (not mock.patch,\n"
    "because the function is imported into many modules).\n"
)

subsection("Custom backend to view Captured Graph directly")

torch._dynamo.reset()
captured_gms = []

def capture_backend(gm, example_inputs):
    captured_gms.append(gm)
    return gm.forward

def model_fn(x):
    y = torch.sin(x)
    z = y + x
    return z.sum()

compiled_model = torch.compile(model_fn, backend=capture_backend)
compiled_model(torch.randn(4, requires_grad=True))

if captured_gms:
    print("Captured Graph (what the backend receives):")
    print("  gm.code:")
    print(textwrap.indent(captured_gms[0].code, "    "))
    print("  gm.graph.print_tabular():")
    captured_gms[0].graph.print_tabular()

subsection("Patch __code__ to intercept all stages")

from torch.fx._utils import lazy_format_graph_code as _original_lazy_format

_target_globals = _original_lazy_format.__globals__
_original_code = _original_lazy_format.__code__

_target_globals["_captured_graph_stages"] = []
_target_globals["_saved_original_code"] = _original_code

def patched_lazy_format_graph_code(name, gm, maybe_id=None, **kwargs):
    try:
        readable = gm.print_readable(print_output=False)
    except Exception:
        readable = str(gm.graph)
    _captured_graph_stages.append({"name": name, "maybe_id": maybe_id,
                                   "code": readable[:600]})
    import types as _types
    orig_fn = _types.FunctionType(_saved_original_code,
                                  lazy_format_graph_code.__globals__)
    return orig_fn(name, gm, maybe_id, **kwargs)

torch._dynamo.reset()

def grad_model(x):
    y = torch.sin(x)
    z = y + x
    return z.sum()

_original_lazy_format.__code__ = patched_lazy_format_graph_code.__code__
try:
    compiled_grad = torch.compile(grad_model, backend="inductor")
    result = compiled_grad(torch.randn(4, requires_grad=True))
    result.backward()
finally:
    _original_lazy_format.__code__ = _original_code

graph_stages = _target_globals.pop("_captured_graph_stages")
_target_globals.pop("_saved_original_code", None)

print("lazy_format_graph_code was called %d time(s):" % len(graph_stages))
for i, stage in enumerate(graph_stages):
    name_str = stage["name"]
    if stage["maybe_id"] is not None:
        name_str += " (id=%s)" % stage["maybe_id"]
    print("\n  [%d] %s" % (i, name_str))
    code_lines = stage["code"].split("\n")
    for line in code_lines[:10]:
        print("      " + line)
    if len(code_lines) > 10:
        print("      ... (%d lines total)" % len(code_lines))

print(
    "\nThese correspond to the files depyf saves:\n"
    "  __compiled_fn_0.Captured_Graph.0.py  <-- lazy_format_graph_code\n"
    "  __compiled_fn_0.BEFORE_PRE_GRAD.0.py <-- lazy_format_graph_code\n"
    "  __compiled_fn_0.Forward_graph.0.py   <-- lazy_format_graph_code\n"
    "  __compiled_fn_0.kernel_0.py          <-- PyCodeCache.load_by_key_path\n"
)


# =========================================================================
# Part 6: Graph Break -> Resume recursive structure
# =========================================================================

section("Part 6: Graph Break -> __resume_at recursive structure")

torch._dynamo.reset()
break_hooks = []

def bh(old_code, new_code):
    break_hooks.append((old_code, new_code))
    return None

h = _cf.register_bytecode_hook(bh)

def fn_with_break(x, y):
    z = x + y
    print("  [inside] z.shape =", z.shape)
    return z * 2

compiled_break = torch.compile(fn_with_break, backend="eager")
out = compiled_break(torch.randn(3), torch.randn(3))
h.remove()

print("Bytecode hook captured %d events (graph break -> multiple compilations):" % len(break_hooks))
for i, (oc, nc) in enumerate(break_hooks):
    print("  [%d] %s -> %s" % (i, oc.co_name, nc.co_name))

from depyf import decompile

subsection("Main function's transformed_code")

entries_break = _debug_get_cache_entry_list(fn_with_break)
resume_names = []
if entries_break:
    tc = entries_break[0].code
    print("depyf.decompile(fn_with_break's transformed_code):")
    print(textwrap.indent(decompile(tc), "  "))

    compiled_names = [n for n in tc.co_names if n.startswith("__compiled")]
    resume_names = [n for n in tc.co_names if n.startswith("__resume")]
    print("\n  __compiled_fn in co_names:", compiled_names)
    print("  __resume_at  in co_names:", resume_names)

subsection("Resume functions have their own cache entries + transformed_code")

for rname in resume_names:
    resume_fn = fn_with_break.__globals__.get(rname)
    if resume_fn is None:
        continue
    print("Resume function: %s" % rname)
    print("  type:", type(resume_fn))
    print("  co_name:", resume_fn.__code__.co_name)

    re_entries = _debug_get_cache_entry_list(resume_fn.__code__)
    print("  cache entries: %d" % len(re_entries))

    if re_entries:
        re_tc = re_entries[0].code
        print("\n  depyf.decompile(resume's transformed_code):")
        print(textwrap.indent(decompile(re_tc), "    "))

        re_compiled = [n for n in re_tc.co_names if n.startswith("__compiled")]
        re_resume = [n for n in re_tc.co_names if n.startswith("__resume")]
        print("    __compiled_fn:", re_compiled)
        print("    __resume_at:", re_resume)

print(
    "\nRecursive structure:\n"
    "  fn_with_break\n"
    "    -> transformed_code:\n"
    "       __compiled_fn_A(x, y)  # sub-graph before break\n"
    "       print(...)             # graph break (non-compilable)\n"
    "       __resume_at_XX(z)      # continues after break\n"
    "          -> resume has its own transformed_code:\n"
    "             __compiled_fn_B(z)  # resume sub-graph\n"
    "\n"
    "  Each level: independent CacheEntry + guard + transformed_code\n"
)

subsection("Graph break example: if/else with data-dependent condition")

torch._dynamo.reset()
ifelse_hooks = []

def ifh(old_code, new_code):
    ifelse_hooks.append((old_code, new_code))
    return None

h2 = _cf.register_bytecode_hook(ifh)

def fn_with_if(x):
    y = x * 2
    if y.sum().item() > 0:  # .item() -> data-dependent -> graph break!
        z = y + 10
    else:
        z = y - 10
    return z.sum()

compiled_if = torch.compile(fn_with_if, backend="eager")
out_if = compiled_if(torch.ones(4))
h2.remove()

print(".item() forces a graph break because the value is data-dependent.")
print("Bytecode hook captured %d events:" % len(ifelse_hooks))
for i, (oc, nc) in enumerate(ifelse_hooks):
    print("  [%d] %s -> %s" % (i, oc.co_name, nc.co_name))

entries_if = _debug_get_cache_entry_list(fn_with_if)
if entries_if:
    tc_if = entries_if[0].code
    print("\ndepyf.decompile(fn_with_if's transformed_code):")
    print(textwrap.indent(decompile(tc_if), "  "))

    resume_if_names = [n for n in tc_if.co_names if n.startswith("__resume")]
    print("\n  __resume_at functions:", resume_if_names)
    for rn in resume_if_names:
        rfn = fn_with_if.__globals__.get(rn)
        if rfn is None:
            continue
        re = _debug_get_cache_entry_list(rfn.__code__)
        if re:
            print("\n  depyf.decompile(%s):" % rn)
            print(textwrap.indent(decompile(re[0].code), "    "))

print(
    "\nWith data-dependent if/else:\n"
    "  __compiled_fn_A computes y = x*2 and y.sum()\n"
    "  .item() triggers graph break (needs actual value)\n"
    "  Python evaluates the if/else using the real item() value\n"
    "  __resume_at_XX continues with the chosen branch\n"
    "  __compiled_fn_B inside resume computes the rest (z = y +/- 10, z.sum())\n"
)


# =========================================================================
# Part 7: _debug_get_cache_entry_list + recompilation + dynamic shapes
# =========================================================================

section("Part 7: _debug_get_cache_entry_list + recompilation experiments")

print("Each compilation produces a CacheEntry, stored in a linked list in code extra.\n")

# ---- Experiment A: default behavior (automatic dynamic) ----

subsection("Experiment A: Default behavior (automatic dynamic shapes)")

torch._dynamo.reset()

def shape_fn_a(x):
    return x.sum()

compiled_a = torch.compile(shape_fn_a, backend="eager")

print("Call 1 - shape=(3,):")
compiled_a(torch.randn(3))
entries_a = _debug_get_cache_entry_list(shape_fn_a)
print("  cache entries: %d" % len(entries_a))

print("Call 2 - shape=(3,) (guard passes, cache hit):")
compiled_a(torch.randn(3))
entries_a = _debug_get_cache_entry_list(shape_fn_a)
print("  cache entries: %d  (unchanged)" % len(entries_a))

print("Call 3 - shape=(5,) (guard fails, recompile):")
compiled_a(torch.randn(5))
entries_a = _debug_get_cache_entry_list(shape_fn_a)
print("  cache entries: %d  (new entry)" % len(entries_a))

print("Call 4 - shape=(7,):")
compiled_a(torch.randn(7))
entries_a = _debug_get_cache_entry_list(shape_fn_a)
print("  cache entries: %d" % len(entries_a))

print("Call 5 - shape=(11,):")
compiled_a(torch.randn(11))
entries_a = _debug_get_cache_entry_list(shape_fn_a)
print("  cache entries: %d" % len(entries_a))

print()
print("Result: only %d entries despite 4 different shapes!" % len(entries_a))
print("Reason: after seeing shapes 3 and 5, Dynamo marks the dimension as DYNAMIC.")
print("  entry[0]: static guard (shape==3), used only for shape=3")
print("  entry[1]: dynamic guard (shape>=2), accepts any valid shape")
print("  Shapes 7 and 11 both pass the dynamic guard -> reuse entry[1].")

print("\nGuards for each entry:")
for i, entry in enumerate(entries_a):
    all_guards = entry.guard_manager.root.get_leaf_guards()
    # find shape-related guards from child managers
    children = entry.guard_manager.root.get_child_managers()
    print("  entry[%d]: %d leaf guards, %d child managers" % (
        i, len(all_guards), len(children)))
    for child in children:
        for sub_child in child.get_child_managers():
            for lg in sub_child.get_leaf_guards():
                for part in lg.verbose_code_parts():
                    if "size" in part or "shape" in part:
                        print("    shape guard: %s" % part.strip()[:80])


# ---- Experiment B: dynamic=True (force dynamic from the start) ----

subsection("Experiment B: dynamic=True (force dynamic from the start)")

torch._dynamo.reset()

def shape_fn_b(x):
    return x.sum()

compiled_b = torch.compile(shape_fn_b, backend="eager", dynamic=True)

print("Call 1 - shape=(3,):")
compiled_b(torch.randn(3))
entries_b = _debug_get_cache_entry_list(shape_fn_b)
print("  cache entries: %d" % len(entries_b))

print("Call 2 - shape=(5,):")
compiled_b(torch.randn(5))
entries_b = _debug_get_cache_entry_list(shape_fn_b)
print("  cache entries: %d  (still just 1!)" % len(entries_b))

print("Call 3 - shape=(100,):")
compiled_b(torch.randn(100))
entries_b = _debug_get_cache_entry_list(shape_fn_b)
print("  cache entries: %d" % len(entries_b))

print()
print("Result: always %d entry -- dynamic=True means every dim is symbolic from call 1." % len(entries_b))
print("No recompilation needed because the guard only checks range constraints.")

print("\nGuards:")
if entries_b:
    children_b = entries_b[0].guard_manager.root.get_child_managers()
    for child in children_b:
        for sub_child in child.get_child_managers():
            for lg in sub_child.get_leaf_guards():
                for part in lg.verbose_code_parts():
                    if "size" in part or "shape" in part:
                        print("  shape guard: %s" % part.strip()[:80])


# ---- Experiment C: dynamic=False (force static, recompile every new shape) ----

subsection("Experiment C: dynamic=False (force static, recompile for every new shape)")

torch._dynamo.reset()

def shape_fn_c(x):
    return x.sum()

compiled_c = torch.compile(shape_fn_c, backend="eager", dynamic=False)

print("Call 1 - shape=(3,):")
compiled_c(torch.randn(3))
entries_c = _debug_get_cache_entry_list(shape_fn_c)
print("  cache entries: %d" % len(entries_c))

print("Call 2 - shape=(5,):")
compiled_c(torch.randn(5))
entries_c = _debug_get_cache_entry_list(shape_fn_c)
print("  cache entries: %d" % len(entries_c))

print("Call 3 - shape=(7,):")
compiled_c(torch.randn(7))
entries_c = _debug_get_cache_entry_list(shape_fn_c)
print("  cache entries: %d" % len(entries_c))

print()
print("Result: %d entries -- dynamic=False disables automatic dynamic marking." % len(entries_c))
print("Every new shape triggers a recompilation with a static guard.")

print("\nGuards per entry:")
for i, entry in enumerate(entries_c):
    children = entry.guard_manager.root.get_child_managers()
    for child in children:
        for sub_child in child.get_child_managers():
            for lg in sub_child.get_leaf_guards():
                for part in lg.verbose_code_parts():
                    if "size" in part or "shape" in part:
                        print("  entry[%d] shape guard: %s" % (i, part.strip()[:80]))


# ---- Experiment D: cache_size_limit ----

subsection("Experiment D: cache_size_limit (max entries before giving up)")

import torch._dynamo.config as dynamo_config

original_limit = dynamo_config.cache_size_limit
print("Default cache_size_limit: %d" % original_limit)
print()

dynamo_config.cache_size_limit = 2
torch._dynamo.reset()

def shape_fn_d(x):
    return x.sum()

compiled_d = torch.compile(shape_fn_d, backend="eager", dynamic=False)

print("Setting cache_size_limit = 2, dynamic=False:")
for size in [3, 5, 7, 9]:
    compiled_d(torch.randn(size))
    entries_d = _debug_get_cache_entry_list(shape_fn_d)
    print("  shape=(%d,) -> cache entries: %d" % (size, len(entries_d)))

print()
print("Result: after hitting the limit, Dynamo falls back to eager (no compilation).")
print("The cache does not grow beyond the limit.")

dynamo_config.cache_size_limit = original_limit


# ---- Experiment E: inductor + dynamic=True, verify generated kernel ----

subsection("Experiment E: inductor + dynamic=True -> verify kernel handles dynamic shapes")

torch._dynamo.reset()
inductor_dyn_codes = []

@classmethod
def patched_load_dyn(cls, key, path, linemap=None, attrs=None):
    src = open(path).read()
    inductor_dyn_codes.append({"key": key, "path": path, "src": src})
    return _orig_load(cls, key, path, linemap, attrs)

def dyn_relu(x):
    return torch.relu(x)

with patch.object(PyCodeCache, "load_by_key_path", patched_load_dyn):
    compiled_dyn_relu = torch.compile(dyn_relu, backend="inductor", dynamic=True)
    compiled_dyn_relu(torch.randn(8))

print("Inductor kernel with dynamic=True:")
if inductor_dyn_codes:
    item = inductor_dyn_codes[0]
    print("  kernel file: %s" % item["path"])
    src = item["src"]
    has_symbolic = any(kw in src for kw in ["s0", "ks0", "xnumel"])
    has_hardcoded = "static_cast<int64_t>(8L)" in src
    print("  contains symbolic size variable (s0/ks0): %s" % has_symbolic)
    print("  contains hard-coded size (8L): %s" % has_hardcoded)
    print()
    print("  With dynamic=True, the kernel uses symbolic size variables (e.g. ks0)")
    print("  instead of hard-coded constants, so one kernel works for all input sizes.")
    print("  You can open the file above to see the full generated C++/Triton code.")

print("\nRe-use with different shape (no recompilation):")
compiled_dyn_relu(torch.randn(32))
entries_e = _debug_get_cache_entry_list(dyn_relu)
print("  cache entries after shape=32: %d  (still 1, no recompile)" % len(entries_e))


# =========================================================================
# Part 8: Guard tree traversal
# =========================================================================

section("Part 8: Guard tree traversal")

print("Each CacheEntry's guard_manager.root is a tree of GuardManager nodes.\n"
      "API: root.get_leaf_guards() -> leaf conditions at this node\n"
      "     root.get_child_managers() -> sub-trees for nested attributes\n")

torch._dynamo.reset()

def guard_add(x):
    return x + 1

compiled_guard_add = torch.compile(guard_add, backend="eager")
compiled_guard_add(torch.randn(4))
guard_entries = _debug_get_cache_entry_list(guard_add)

if guard_entries:
    entry0 = guard_entries[0]
    root = entry0.guard_manager.root

    def print_guard_tree(node, depth=0, max_depth=2):
        prefix = "  " * depth
        leaf_guards = node.get_leaf_guards()
        children = node.get_child_managers()
        print("%s[%s] (%d leaf guards, %d children)" % (
            prefix, type(node).__name__, len(leaf_guards), len(children)))
        for lg in leaf_guards[:4]:
            for part in lg.verbose_code_parts():
                print("%s  LEAF: %s" % (prefix, part.strip()[:90]))
        if len(leaf_guards) > 4:
            print("%s  ... (%d more)" % (prefix, len(leaf_guards) - 4))
        if depth < max_depth:
            for i, child in enumerate(children):
                print("%s  child[%d]:" % (prefix, i))
                print_guard_tree(child, depth + 2, max_depth)
        elif children:
            print("%s  ... (%d children omitted, depth limit)" % (prefix, len(children)))

    print("guard tree for `guard_add` (x + 1):")
    print_guard_tree(root)

    gm0 = entry0.guard_manager
    if hasattr(gm0, "closure_vars") and gm0.closure_vars:
        print("\nclosure_vars (variables referenced by guards):")
        for k, v in list(gm0.closure_vars.items())[:5]:
            print("  %s = %s" % (k, repr(v)[:80]))

    print(
        "\nKey takeaway:\n"
        "  RootGuardManager checks global state (device, hooks, etc.)\n"
        "  Child GuardManagers check input tensor properties (dtype, shape, dispatch keys)\n"
        "  Leaf guards are the actual boolean conditions (check_tensor, obj_id, shape bounds)\n"
    )


# =========================================================================
# Part 9: Simple dispatch wrapper (hand-written)
# =========================================================================

section("Part 9: Simple dispatch wrapper (hand-written, no depyf)")

print(
    "Idea: compile once, then skip eval_frame hook and execute transformed_code directly.\n"
    "Principle: Dynamo guarantees transformed_code has the same signature as original code,\n"
    "so we can do fn.__code__ = transformed_code and call fn normally.\n"
)

class SimpleDispatchWrapper:
    """Hand-written minimal dispatch wrapper demonstrating the core principle"""

    def __init__(self):
        self.compiled_codes = []
        self._orig_code = self.__class__.forward.__code__
        self._hook_handle = _cf.register_bytecode_hook(self._bytecode_hook)
        self._compiled_callable = torch.compile(self.forward, backend="eager")

    def _bytecode_hook(self, old_code, new_code):
        if old_code is self._orig_code:
            self.compiled_codes.append(new_code)

    @contextlib.contextmanager
    def dispatch_to_code(self, index):
        """Temporarily swap forward's code object, bypassing eval_frame hook"""
        self.__class__.forward.__code__ = self.compiled_codes[index]
        try:
            yield
        finally:
            self.__class__.forward.__code__ = self._orig_code

    def forward(self, x):
        return x * 2 + 1

    def __call__(self, x):
        if self.compiled_codes:
            with self.dispatch_to_code(0):
                return self.forward(x)
        else:
            return self._compiled_callable(x)

torch._dynamo.reset()
wrapper = SimpleDispatchWrapper()

print("Call 1 - shape=(4,) -- triggers compilation:")
out1 = wrapper(torch.randn(4))
print("  result:", out1)
print("  compiled_codes count:", len(wrapper.compiled_codes))

print("\nCall 2 - shape=(8,) -- direct dispatch, skips guard:")
out2 = wrapper(torch.randn(8))
print("  result:", out2)
print("  compiled_codes count:", len(wrapper.compiled_codes), "(unchanged)")

print(
    "\nHow it works:\n"
    "  1. bytecode_hook captures transformed_code on first compilation\n"
    "  2. dispatch_to_code(0):\n"
    "     cls.forward.__code__ = compiled_codes[0]\n"
    "     -> self.forward(x) executes transformed_code directly\n"
    "     -> no eval_frame hook, no guard checking\n"
    "     -> restore original code after execution\n"
    "  3. Advantage: eliminates guard overhead; useful when you know\n"
    "     recompilation is unnecessary\n"
)

wrapper._hook_handle.remove()


# =========================================================================
# Part 10: Summary
# =========================================================================

section("Part 10: Summary -- how all APIs connect")

print(
    "  torch.compile(fn, backend=xxx)(inputs)\n"
    "  |\n"
    "  | [C level] eval_frame hook intercepts fn's frame\n"
    "  | [Dynamo]  symbolic execution -> FX Graph (Captured Graph)\n"
    "  | [AOTAutograd] Joint graph -> Forward/Backward graph\n"
    "  | [Backend] eager: gm.forward / inductor: CompiledFxGraph\n"
    "  |\n"
    "  v Output:\n"
    "  +-- CacheEntry (linked list in code extra)\n"
    "  |     .code = transformed_code\n"
    "  |     .guard_manager = guard tree\n"
    "  |       root -> get_leaf_guards()    leaf guards\n"
    "  |            -> get_child_managers() subtrees\n"
    "  |\n"
    "  +-- transformed_code's co_names contains:\n"
    "  |     __compiled_fn_xxx -- injected into fn.__globals__\n"
    "  |       eager: gm.forward (FX Graph viewable via gm.graph)\n"
    "  |       inductor: CompiledFxGraph (kernel loaded via load_by_key_path)\n"
    "  |     __resume_at_xxx  -- generated on graph break\n"
    "  |       has its own CacheEntry chain (recursive)\n"
    "  |\n"
    "  +-- depyf interception points:\n"
    "        register_bytecode_hook      -> old/new code\n"
    "        _debug_get_cache_entry_list  -> CacheEntry chain\n"
    "        lazy_format_graph_code       -> Graph at each stage\n"
    "        PyCodeCache.load_by_key_path -> Inductor kernel code\n"
    "\n"
    "  API quick reference:\n"
    "    register_bytecode_hook(hook)      intercept Dynamo output\n"
    "    _debug_get_cache_entry_list(fn)   read cache linked list\n"
    "    guard_manager.root                guard tree root\n"
    "    innermost_fn(fn)                  unwrap @disable wrapper\n"
    "    PyCodeCache.load_by_key_path      inductor code loading\n"
    "    dispatch_to_code                  bypass guards, execute directly\n"
    "    mini_decompile / decompile        bytecode decompilation\n"
    "\n"
    "  Compilation control:\n"
    "    dynamic=True           force all dims dynamic from the start\n"
    "    dynamic=False          force static, recompile on every new shape\n"
    "    cache_size_limit=N     max CacheEntry count before fallback\n"
)
