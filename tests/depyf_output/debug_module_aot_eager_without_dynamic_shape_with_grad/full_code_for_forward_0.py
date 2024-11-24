
# Note: the following variables are used inside the guard function.
___check_tensors = '''None'''
___check_tensors_verbose = '''None'''
___check_global_state = '''<built-in method check of torch._C._dynamo.guards.GlobalStateGuard object at 0x118000dd0>'''
___check_torch_function_mode_stack = '''<function make_torch_function_mode_stack_guard.<locals>.check_torch_function_mode_stack at 0x11f2a2b90>'''
Abs = '''<built-in function abs>'''
Eq = '''<built-in function eq>'''
Ne = '''<built-in function ne>'''
Gt = '''<built-in function gt>'''
Lt = '''<built-in function lt>'''
Le = '''<built-in function le>'''
Ge = '''<built-in function ge>'''
Min = '''<built-in function min>'''
Max = '''<built-in function max>'''
Mod = '''<built-in function mod>'''
PythonMod = '''<built-in function mod>'''
FloorDiv = '''<built-in function floordiv>'''
TrueDiv = '''<built-in function truediv>'''
PowByNatural = '''<built-in function pow>'''
IsNonOverlappingAndDenseIndicator = '''<function eval_is_non_overlapping_and_dense at 0x10ba5d240>'''
floor = '''<built-in function floor>'''
ceiling = '''<built-in function ceil>'''
FloorToInt = '''<built-in function floor>'''
FloatPow = '''<built-in function pow>'''
CeilToInt = '''<built-in function ceil>'''
cast_symbool_to_symint_guardless = '''<function cast_symbool_to_symint_guardless at 0x10ba5d3f0>'''
RoundToInt = '''<built-in function round>'''
RoundDecimal = '''<built-in function round>'''
TruncToInt = '''<built-in function trunc>'''
IntTrueDiv = '''<built-in function truediv>'''
FloatTrueDiv = '''<built-in function truediv>'''
ToFloat = '''<class 'float'>'''
OpaqueUnaryFn_cos = '''<built-in function cos>'''
OpaqueUnaryFn_cosh = '''<built-in function cosh>'''
OpaqueUnaryFn_acos = '''<built-in function acos>'''
OpaqueUnaryFn_sin = '''<built-in function sin>'''
OpaqueUnaryFn_sinh = '''<built-in function sinh>'''
OpaqueUnaryFn_asin = '''<built-in function asin>'''
OpaqueUnaryFn_tan = '''<built-in function tan>'''
OpaqueUnaryFn_tanh = '''<built-in function tanh>'''
OpaqueUnaryFn_atan = '''<built-in function atan>'''
OpaqueUnaryFn_sqrt = '''<built-in function sqrt>'''
___check_type_id = '''<built-in function check_type_id>'''
___check_obj_id = '''<built-in function check_obj_id>'''
___odict_getitem = '''<method '__getitem__' of 'dict' objects>'''
___key_to_id = '''<function key_to_id at 0x10bbabac0>'''
___dict_version = '''<built-in function dict_version>'''
___dict_contains = '''<function _get_closure_vars.<locals>.<lambda> at 0x11f1db130>'''
___tuple_iterator_len = '''<method '__length_hint__' of 'tuple_iterator' objects>'''
___tuple_iterator_getitem = '''<function tuple_iterator_getitem at 0x10bbab640>'''
___get_torch_function_mode_stack_at = '''<function get_torch_function_mode_stack_at at 0x10bbbac20>'''
__math_isnan = '''<built-in function isnan>'''
__numpy_isnan = '''<ufunc 'isnan'>'''
inf = '''inf'''
__load_module = '''<function import_module at 0x103263910>'''
utils_device = '''<module 'torch.utils._device' from '/Users/youkaichao/anaconda3/envs/py310/lib/python3.10/site-packages/torch/utils/_device.py'>'''
device = '''<class 'torch.device'>'''
___from_numpy = '''<function from_numpy at 0x10bfac3a0>'''
___as_tensor = '''<function _as_tensor_fullprec at 0x10a19ca60>'''
torch = '''<module 'torch' from '/Users/youkaichao/anaconda3/envs/py310/lib/python3.10/site-packages/torch/__init__.py'>'''
inspect = '''<module 'inspect' from '/Users/youkaichao/anaconda3/envs/py310/lib/python3.10/inspect.py'>'''
def __guard_0_for_torch_dynamo_resume_in_forward_at_15(L, G, **___kwargs_ignored):
    __guard_hit = True
    __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:483 in init_ambient_guards
    __guard_hit = __guard_hit and ___check_global_state()
    __guard_hit = __guard_hit and ___check_torch_function_mode_stack()
    __guard_hit = __guard_hit and check_tensor(L['b'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[10], stride=[1])
    __guard_hit = __guard_hit and hasattr(L['b'], '_dynamo_dynamic_indices') == False
    __guard_hit = __guard_hit and check_no_aliasing(L['b'], L['x'])
    __guard_hit = __guard_hit and check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[10], stride=[1])
    __guard_hit = __guard_hit and hasattr(L['x'], '_dynamo_dynamic_indices') == False
    __guard_hit = __guard_hit and check_no_aliasing(L['b'], L['x'])
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
    __temp_7, = __compiled_fn_5(x, b)
    return __temp_7


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
___check_global_state = '''<built-in method check of torch._C._dynamo.guards.GlobalStateGuard object at 0x10fe5dd50>'''
___check_torch_function_mode_stack = '''<function make_torch_function_mode_stack_guard.<locals>.check_torch_function_mode_stack at 0x11f2631c0>'''
Abs = '''<built-in function abs>'''
Eq = '''<built-in function eq>'''
Ne = '''<built-in function ne>'''
Gt = '''<built-in function gt>'''
Lt = '''<built-in function lt>'''
Le = '''<built-in function le>'''
Ge = '''<built-in function ge>'''
Min = '''<built-in function min>'''
Max = '''<built-in function max>'''
Mod = '''<built-in function mod>'''
PythonMod = '''<built-in function mod>'''
FloorDiv = '''<built-in function floordiv>'''
TrueDiv = '''<built-in function truediv>'''
PowByNatural = '''<built-in function pow>'''
IsNonOverlappingAndDenseIndicator = '''<function eval_is_non_overlapping_and_dense at 0x10ba5d240>'''
floor = '''<built-in function floor>'''
ceiling = '''<built-in function ceil>'''
FloorToInt = '''<built-in function floor>'''
FloatPow = '''<built-in function pow>'''
CeilToInt = '''<built-in function ceil>'''
cast_symbool_to_symint_guardless = '''<function cast_symbool_to_symint_guardless at 0x10ba5d3f0>'''
RoundToInt = '''<built-in function round>'''
RoundDecimal = '''<built-in function round>'''
TruncToInt = '''<built-in function trunc>'''
IntTrueDiv = '''<built-in function truediv>'''
FloatTrueDiv = '''<built-in function truediv>'''
ToFloat = '''<class 'float'>'''
OpaqueUnaryFn_cos = '''<built-in function cos>'''
OpaqueUnaryFn_cosh = '''<built-in function cosh>'''
OpaqueUnaryFn_acos = '''<built-in function acos>'''
OpaqueUnaryFn_sin = '''<built-in function sin>'''
OpaqueUnaryFn_sinh = '''<built-in function sinh>'''
OpaqueUnaryFn_asin = '''<built-in function asin>'''
OpaqueUnaryFn_tan = '''<built-in function tan>'''
OpaqueUnaryFn_tanh = '''<built-in function tanh>'''
OpaqueUnaryFn_atan = '''<built-in function atan>'''
OpaqueUnaryFn_sqrt = '''<built-in function sqrt>'''
___check_type_id = '''<built-in function check_type_id>'''
___check_obj_id = '''<built-in function check_obj_id>'''
___odict_getitem = '''<method '__getitem__' of 'dict' objects>'''
___key_to_id = '''<function key_to_id at 0x10bbabac0>'''
___dict_version = '''<built-in function dict_version>'''
___dict_contains = '''<function _get_closure_vars.<locals>.<lambda> at 0x11f1db130>'''
___tuple_iterator_len = '''<method '__length_hint__' of 'tuple_iterator' objects>'''
___tuple_iterator_getitem = '''<function tuple_iterator_getitem at 0x10bbab640>'''
___get_torch_function_mode_stack_at = '''<function get_torch_function_mode_stack_at at 0x10bbbac20>'''
__math_isnan = '''<built-in function isnan>'''
__numpy_isnan = '''<ufunc 'isnan'>'''
inf = '''inf'''
__load_module = '''<function import_module at 0x103263910>'''
utils_device = '''<module 'torch.utils._device' from '/Users/youkaichao/anaconda3/envs/py310/lib/python3.10/site-packages/torch/utils/_device.py'>'''
device = '''<class 'torch.device'>'''
___from_numpy = '''<function from_numpy at 0x10bfac3a0>'''
___as_tensor = '''<function _as_tensor_fullprec at 0x10a19ca60>'''
torch = '''<module 'torch' from '/Users/youkaichao/anaconda3/envs/py310/lib/python3.10/site-packages/torch/__init__.py'>'''
inspect = '''<module 'inspect' from '/Users/youkaichao/anaconda3/envs/py310/lib/python3.10/inspect.py'>'''
def __guard_0_for_forward(L, G, **___kwargs_ignored):
    __guard_hit = True
    __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:483 in init_ambient_guards
    __guard_hit = __guard_hit and ___check_global_state()
    __guard_hit = __guard_hit and ___check_torch_function_mode_stack()
    __guard_hit = __guard_hit and check_tensor(L['a'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[10], stride=[1])
    __guard_hit = __guard_hit and hasattr(L['a'], '_dynamo_dynamic_indices') == False
    __guard_hit = __guard_hit and check_no_aliasing(L['a'], L['b'])
    __guard_hit = __guard_hit and check_tensor(L['b'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[10], stride=[1])
    __guard_hit = __guard_hit and hasattr(L['b'], '_dynamo_dynamic_indices') == False
    __guard_hit = __guard_hit and check_no_aliasing(L['a'], L['b'])
    __guard_hit = __guard_hit and ___check_obj_id(G['torch'], 4364233408)
    __guard_hit = __guard_hit and ___check_obj_id(G['torch'].abs, 4369569792)
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
    __temp_2, __temp_3 = __compiled_fn_1(a, b)
    x = __temp_2
    if __temp_3:
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
