
def __guard_0_for_torch_dynamo_resume_in_toy_function_at_5(L):
    return 

# Note: please refer to the graph code in __compiled_fn_5*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_5(*args, **kwargs):
    pass

def __transformed_code_0_for_torch_dynamo_resume_in_toy_function_at_5(b, x):
    a = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __compiled_fn_5(x, b)[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_38_3(b, x):
    return x * b

def transformed___resume_at_38_3(b, x):
    L = {"b": b, "x": x}
    if __guard_0_for_torch_dynamo_resume_in_toy_function_at_5(L):
        return __transformed_code_0_for_torch_dynamo_resume_in_toy_function_at_5(b, x)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_38_3(b, x)

#============ end of __resume_at_38_3 ============#

def __guard_1_for_torch_dynamo_resume_in_toy_function_at_5(L):
    return 

# Note: please refer to the graph code in __compiled_fn_11*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_11(*args, **kwargs):
    pass

def __transformed_code_1_for_torch_dynamo_resume_in_toy_function_at_5(b, x):
    a = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __compiled_fn_11(b, x)[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_30_2(b, x):
    b = b * -1
    return x * b

def transformed___resume_at_30_2(b, x):
    L = {"b": b, "x": x}
    if __guard_1_for_torch_dynamo_resume_in_toy_function_at_5(L):
        return __transformed_code_1_for_torch_dynamo_resume_in_toy_function_at_5(b, x)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_30_2(b, x)

#============ end of __resume_at_30_2 ============#

def __guard_0_for_torch_dynamo_resume_in_toy_function_at_5(L):
    return 

# Note: please refer to the graph code in __compiled_fn_5*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_5(*args, **kwargs):
    pass

def __transformed_code_0_for_torch_dynamo_resume_in_toy_function_at_5(b, x):
    a = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __compiled_fn_5(x, b)[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_38_9(b, x):
    return x * b

def transformed___resume_at_38_9(b, x):
    L = {"b": b, "x": x}
    if __guard_0_for_torch_dynamo_resume_in_toy_function_at_5(L):
        return __transformed_code_0_for_torch_dynamo_resume_in_toy_function_at_5(b, x)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_38_9(b, x)

#============ end of __resume_at_38_9 ============#

def __guard_1_for_torch_dynamo_resume_in_toy_function_at_5(L):
    return 

# Note: please refer to the graph code in __compiled_fn_11*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_11(*args, **kwargs):
    pass

def __transformed_code_1_for_torch_dynamo_resume_in_toy_function_at_5(b, x):
    a = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __compiled_fn_11(b, x)[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_30_8(b, x):
    b = b * -1
    return x * b

def transformed___resume_at_30_8(b, x):
    L = {"b": b, "x": x}
    if __guard_1_for_torch_dynamo_resume_in_toy_function_at_5(L):
        return __transformed_code_1_for_torch_dynamo_resume_in_toy_function_at_5(b, x)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_30_8(b, x)

#============ end of __resume_at_30_8 ============#

def __guard_1_for_toy_function(L):
    return 

# Note: please refer to the graph code in __compiled_fn_7*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_7(*args, **kwargs):
    pass

def __transformed_code_1_for_toy_function(a, b):
    __temp_7 = __compiled_fn_7(a.size(0), a, b.size(0), b)
    x = __temp_7[0]
    if __temp_7[1]:
        return __resume_at_30_8(b, x)
    return __resume_at_38_9(b, x)


def __guard_0_for_toy_function(L):
    return 

# Note: please refer to the graph code in __compiled_fn_1*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_1(*args, **kwargs):
    pass

def __transformed_code_0_for_toy_function(a, b):
    __temp_1 = __compiled_fn_1(a, b)
    x = __temp_1[0]
    if __temp_1[1]:
        return __resume_at_30_2(b, x)
    return __resume_at_38_3(b, x)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def toy_function(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

def transformed_toy_function(a, b):
    L = {"a": a, "b": b}
    if __guard_1_for_toy_function(L):
        return __transformed_code_1_for_toy_function(a, b)
    if __guard_0_for_toy_function(L):
        return __transformed_code_0_for_toy_function(a, b)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return toy_function(a, b)

#============ end of toy_function ============#
