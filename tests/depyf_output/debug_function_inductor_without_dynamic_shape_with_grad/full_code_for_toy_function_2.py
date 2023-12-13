
def __guard_0_for_resume_in_toy_function_at_5(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['b'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['x'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(5078369616))) \
        and (___check_tensors(L['b'], L['x'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_3*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_3(*args, **kwargs):
    pass

def __transformed_code_0_for_resume_in_toy_function_at_5(b, x):
    a = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __compiled_fn_3(x, b)[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_38_2(b, x):
    return x * b

def transformed___resume_at_38_2(b, x):
    L = {"b": b, "x": x}
    if __guard_0_for_resume_in_toy_function_at_5(L):
        return __transformed_code_0_for_resume_in_toy_function_at_5(b, x)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_38_2(b, x)

#============ end of __resume_at_38_2 ============#

# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_30_1(b, x):
    b = b * -1
    return x * b

def transformed___resume_at_30_1(b, x):
    L = {"b": b, "x": x}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_30_1(b, x)

#============ end of __resume_at_30_1 ============#

def __guard_0_for_toy_function(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['a'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['b'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(5078369616))) \
        and (___check_tensors(L['a'], L['b'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_0*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_0(*args, **kwargs):
    pass

def __transformed_code_0_for_toy_function(a, b):
    __temp_1 = __compiled_fn_0(a, b)
    x = __temp_1[0]
    if __temp_1[1]:
        return __resume_at_30_1(b, x)
    return __resume_at_38_2(b, x)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def toy_function(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

def transformed_toy_function(a, b):
    L = {"a": a, "b": b}
    if __guard_0_for_toy_function(L):
        return __transformed_code_0_for_toy_function(a, b)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return toy_function(a, b)

#============ end of toy_function ============#
