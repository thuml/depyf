
def __guard_0_for_resume_in_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['b'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['x'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(5944251952))) \
        and (___compile_config_hash() == '4865d56f7233bad50ab45f1dfec713b9') \
        and (___check_tensors(L['b'], L['x'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_3*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_3(*args, **kwargs):
    pass

def __transformed_code_0_for_resume_in_forward(b, x):
    return __compiled_fn_3(x, b)[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_38_2(b, x):
    return x * b

def transformed___resume_at_38_2(b, x):
    L = {"b": b, "x": x}
    if __guard_0_for_resume_in_forward(L):
        return __transformed_code_0_for_resume_in_forward(b, x)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_38_2(b, x)

#============ end of __resume_at_38_2 ============#

def __guard_1_for_resume_in_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['b'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['x'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(5944251952))) \
        and (___compile_config_hash() == '4865d56f7233bad50ab45f1dfec713b9') \
        and (___check_tensors(L['b'], L['x'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_7*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_7(*args, **kwargs):
    pass

def __transformed_code_1_for_resume_in_forward(b, x):
    return __compiled_fn_7(b, x)[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_30_1(b, x):
    b = b * -1
    return x * b

def transformed___resume_at_30_1(b, x):
    L = {"b": b, "x": x}
    if __guard_1_for_resume_in_forward(L):
        return __transformed_code_1_for_resume_in_forward(b, x)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_30_1(b, x)

#============ end of __resume_at_30_1 ============#

def __guard_0_for_resume_in_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['b'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['x'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(5944251952))) \
        and (___compile_config_hash() == '4865d56f7233bad50ab45f1dfec713b9') \
        and (___check_tensors(L['b'], L['x'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_3*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_3(*args, **kwargs):
    pass

def __transformed_code_0_for_resume_in_forward(b, x):
    return __compiled_fn_3(x, b)[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_38_6(b, x):
    return x * b

def transformed___resume_at_38_6(b, x):
    L = {"b": b, "x": x}
    if __guard_0_for_resume_in_forward(L):
        return __transformed_code_0_for_resume_in_forward(b, x)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_38_6(b, x)

#============ end of __resume_at_38_6 ============#

def __guard_1_for_resume_in_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['b'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['x'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(5944251952))) \
        and (___compile_config_hash() == '4865d56f7233bad50ab45f1dfec713b9') \
        and (___check_tensors(L['b'], L['x'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_7*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_7(*args, **kwargs):
    pass

def __transformed_code_1_for_resume_in_forward(b, x):
    return __compiled_fn_7(b, x)[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_30_5(b, x):
    b = b * -1
    return x * b

def transformed___resume_at_30_5(b, x):
    L = {"b": b, "x": x}
    if __guard_1_for_resume_in_forward(L):
        return __transformed_code_1_for_resume_in_forward(b, x)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_30_5(b, x)

#============ end of __resume_at_30_5 ============#

def __guard_1_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['a'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['b'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(5944251952))) \
        and (___compile_config_hash() == '4865d56f7233bad50ab45f1dfec713b9') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['a'], L['b'], tensor_check_names=tensor_check_names)) \
        and (2 <= L['a'].size()[0]) \
        and (2 <= L['b'].size()[0]) \
        and (2 <= L['b'].size()[0])

# Note: please refer to the graph code in __compiled_fn_4*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_4(*args, **kwargs):
    pass

def __transformed_code_1_for_forward(self, a, b):
    __temp_7 = __compiled_fn_4(a.size(0), a, b.size(0), b)
    x = __temp_7[0]
    if __temp_7[1]:
        return __resume_at_30_5(b, x)
    return __resume_at_38_6(b, x)


def __guard_0_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['a'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['b'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(5944251952))) \
        and (___compile_config_hash() == '4865d56f7233bad50ab45f1dfec713b9') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['a'], L['b'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_0*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_0(*args, **kwargs):
    pass

def __transformed_code_0_for_forward(self, a, b):
    __temp_1 = __compiled_fn_0(a, b)
    x = __temp_1[0]
    if __temp_1[1]:
        return __resume_at_30_1(b, x)
    return __resume_at_38_2(b, x)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

def transformed_forward(self, a, b):
    L = {"self": self, "a": a, "b": b}
    if __guard_1_for_forward(L):
        return __transformed_code_1_for_forward(self, a, b)
    if __guard_0_for_forward(L):
        return __transformed_code_0_for_forward(self, a, b)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, a, b)

#============ end of forward ============#
