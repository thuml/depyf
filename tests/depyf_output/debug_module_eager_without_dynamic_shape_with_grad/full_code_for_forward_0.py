
# Note: the following variables are used inside the guard function.
___check_tensors = '''None'''
___check_tensors_verbose = '''None'''
___check_global_state = '''<built-in method check of torch._C._dynamo.guards.GlobalStateGuard object at 0x1437bc4b0>'''
___check_torch_function_mode_stack = '''<function make_torch_function_mode_stack_guard.<locals>.check_torch_function_mode_stack at 0x143502680>'''
IsNonOverlappingAndDenseIndicator = '''<function eval_is_non_overlapping_and_dense at 0x137c87a30>'''
cast_symbool_to_symint_guardless = '''<function cast_symbool_to_symint_guardless at 0x137c87be0>'''
math = '''<module 'math' from '/opt/homebrew/Cellar/python@3.10/3.10.16/Frameworks/Python.framework/Versions/3.10/lib/python3.10/lib-dynload/math.cpython-310-darwin.so'>'''
torch = '''<module 'torch' from '/Users/youkaichao/uv_envs/py310/lib/python3.10/site-packages/torch/__init__.py'>'''
___check_type_id = '''<built-in function check_type_id>'''
___check_obj_id = '''<built-in function check_obj_id>'''
___odict_getitem = '''<method '__getitem__' of 'dict' objects>'''
___key_to_id = '''<function key_to_id at 0x137d8d870>'''
___dict_version = '''<built-in function dict_version>'''
___dict_contains = '''<function _get_closure_vars.<locals>.<lambda> at 0x143797c70>'''
___tuple_iterator_len = '''<method '__length_hint__' of 'tuple_iterator' objects>'''
___normalize_range_iter = '''<function normalize_range_iter at 0x137d8d3f0>'''
___tuple_iterator_getitem = '''<function tuple_iterator_getitem at 0x137d8d360>'''
___get_torch_function_mode_stack_at = '''<function get_torch_function_mode_stack_at at 0x137d84b80>'''
__math_isnan = '''<built-in function isnan>'''
__numpy_isnan = '''<ufunc 'isnan'>'''
inf = '''inf'''
__load_module = '''<function import_module at 0x1044d7490>'''
utils_device = '''<module 'torch.utils._device' from '/Users/youkaichao/uv_envs/py310/lib/python3.10/site-packages/torch/utils/_device.py'>'''
device = '''<class 'torch.device'>'''
___from_numpy = '''<function from_numpy at 0x137e07370>'''
___as_tensor = '''<function _as_tensor_fullprec at 0x132c06560>'''
inspect = '''<module 'inspect' from '/opt/homebrew/Cellar/python@3.10/3.10.16/Frameworks/Python.framework/Versions/3.10/lib/python3.10/inspect.py'>'''
def __guard_0_for_torch_dynamo_resume_in_forward_at_15(L, G, **___kwargs_ignored):
    __guard_hit = True
    __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:551 in init_ambient_guards
    __guard_hit = __guard_hit and ___check_global_state()
    __guard_hit = __guard_hit and ___check_torch_function_mode_stack()
    __guard_hit = __guard_hit and check_tensor(L['b'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[10], stride=[1])
    __guard_hit = __guard_hit and hasattr(L['b'], '_dynamo_dynamic_indices') == False
    __guard_hit = __guard_hit and check_no_aliasing(L['b'], L['x'])
    __guard_hit = __guard_hit and check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[10], stride=[1])
    __guard_hit = __guard_hit and hasattr(L['x'], '_dynamo_dynamic_indices') == False
    return __guard_hit

# Note: please refer to the graph code in __compiled_fn_5*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_5(*args, **kwargs):
    pass

def __transformed_code_0_for_torch_dynamo_resume_in_forward_at_15(b, x):
    a = None; self = None # this line helps Python to generate bytecode with at least the same number of local variables as the original function
    tmp_1 = x
    tmp_2 = b
    graph_out_0 = __compiled_fn_5(x, b)
    return graph_out_0[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_38_3(b, x):
    return x * b

def transformed___resume_at_38_3(b, x):
    __local_dict = {"b": b, "x": x}
    __global_dict = globals()
    if __guard_0_for_torch_dynamo_resume_in_forward_at_15(__local_dict, __global_dict):
        return __transformed_code_0_for_torch_dynamo_resume_in_forward_at_15(b, x)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_38_3(b, x)

#============ end of __resume_at_38_3 ============#

# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_30_2(b, x):
    b = b * -1
    return x * b

def transformed___resume_at_30_2(b, x):
    __local_dict = {"b": b, "x": x}
    __global_dict = globals()
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_30_2(b, x)

#============ end of __resume_at_30_2 ============#

# Note: the following variables are used inside the guard function.
___check_tensors = '''None'''
___check_tensors_verbose = '''None'''
___check_global_state = '''<built-in method check of torch._C._dynamo.guards.GlobalStateGuard object at 0x14350a030>'''
___check_torch_function_mode_stack = '''<function make_torch_function_mode_stack_guard.<locals>.check_torch_function_mode_stack at 0x143797e20>'''
IsNonOverlappingAndDenseIndicator = '''<function eval_is_non_overlapping_and_dense at 0x137c87a30>'''
cast_symbool_to_symint_guardless = '''<function cast_symbool_to_symint_guardless at 0x137c87be0>'''
math = '''<module 'math' from '/opt/homebrew/Cellar/python@3.10/3.10.16/Frameworks/Python.framework/Versions/3.10/lib/python3.10/lib-dynload/math.cpython-310-darwin.so'>'''
torch = '''<module 'torch' from '/Users/youkaichao/uv_envs/py310/lib/python3.10/site-packages/torch/__init__.py'>'''
___check_type_id = '''<built-in function check_type_id>'''
___check_obj_id = '''<built-in function check_obj_id>'''
___odict_getitem = '''<method '__getitem__' of 'dict' objects>'''
___key_to_id = '''<function key_to_id at 0x137d8d870>'''
___dict_version = '''<built-in function dict_version>'''
___dict_contains = '''<function _get_closure_vars.<locals>.<lambda> at 0x143797c70>'''
___tuple_iterator_len = '''<method '__length_hint__' of 'tuple_iterator' objects>'''
___normalize_range_iter = '''<function normalize_range_iter at 0x137d8d3f0>'''
___tuple_iterator_getitem = '''<function tuple_iterator_getitem at 0x137d8d360>'''
___get_torch_function_mode_stack_at = '''<function get_torch_function_mode_stack_at at 0x137d84b80>'''
__math_isnan = '''<built-in function isnan>'''
__numpy_isnan = '''<ufunc 'isnan'>'''
inf = '''inf'''
__load_module = '''<function import_module at 0x1044d7490>'''
utils_device = '''<module 'torch.utils._device' from '/Users/youkaichao/uv_envs/py310/lib/python3.10/site-packages/torch/utils/_device.py'>'''
device = '''<class 'torch.device'>'''
___from_numpy = '''<function from_numpy at 0x137e07370>'''
___as_tensor = '''<function _as_tensor_fullprec at 0x132c06560>'''
inspect = '''<module 'inspect' from '/opt/homebrew/Cellar/python@3.10/3.10.16/Frameworks/Python.framework/Versions/3.10/lib/python3.10/inspect.py'>'''
def __guard_0_for_forward(L, G, **___kwargs_ignored):
    __guard_hit = True
    __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:551 in init_ambient_guards
    __guard_hit = __guard_hit and ___check_global_state()
    __guard_hit = __guard_hit and ___check_torch_function_mode_stack()
    __guard_hit = __guard_hit and check_tensor(L['a'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[10], stride=[1])
    __guard_hit = __guard_hit and hasattr(L['a'], '_dynamo_dynamic_indices') == False
    __guard_hit = __guard_hit and check_no_aliasing(L['a'], L['b'])
    __guard_hit = __guard_hit and check_tensor(L['b'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[10], stride=[1])
    __guard_hit = __guard_hit and hasattr(L['b'], '_dynamo_dynamic_indices') == False
    __guard_hit = __guard_hit and ___check_obj_id(G['torch'], 4385618208)
    __guard_hit = __guard_hit and ___check_obj_id(G['torch'].abs, 4386913024)
    return __guard_hit

# Note: please refer to the graph code in __compiled_fn_1*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_1(*args, **kwargs):
    pass

def __transformed_code_0_for_forward(self, a, b):
    tmp_1 = a
    tmp_2 = b
    graph_out_0 = __compiled_fn_1(a, b)
    x = graph_out_0[1]
    if graph_out_0[0]:
        return __resume_at_30_2(b, x)
    return __resume_at_38_3(b, x)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

def transformed_forward(self, a, b):
    __local_dict = {"self": self, "a": a, "b": b}
    __global_dict = globals()
    if __guard_0_for_forward(__local_dict, __global_dict):
        return __transformed_code_0_for_forward(self, a, b)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, a, b)

#============ end of forward ============#
