
# Note: the following variables are used inside the guard function.
___check_tensors = '''None'''
___check_tensors_verbose = '''None'''
___check_global_state = '''<built-in method check of torch._C._dynamo.guards.GlobalStateGuard object at 0x123026cf0>'''
___check_torch_function_mode_stack = '''<function make_torch_function_mode_stack_guard.<locals>.check_torch_function_mode_stack at 0x123441bd0>'''
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
IsNonOverlappingAndDenseIndicator = '''<function eval_is_non_overlapping_and_dense at 0x121e12950>'''
floor = '''<built-in function floor>'''
ceiling = '''<built-in function ceil>'''
FloorToInt = '''<built-in function floor>'''
FloatPow = '''<built-in function pow>'''
CeilToInt = '''<built-in function ceil>'''
cast_symbool_to_symint_guardless = '''<function cast_symbool_to_symint_guardless at 0x121e12b00>'''
RoundToInt = '''<built-in function round>'''
RoundDecimal = '''<built-in function round>'''
TruncToInt = '''<built-in function trunc>'''
IntTrueDiv = '''<built-in function truediv>'''
FloatTrueDiv = '''<built-in function truediv>'''
ToFloat = '''<class 'float'>'''
___check_type_id = '''<built-in function check_type_id>'''
___check_obj_id = '''<built-in function check_obj_id>'''
___odict_getitem = '''<method '__getitem__' of 'dict' objects>'''
___key_to_id = '''<function key_to_id at 0x121ef8310>'''
___dict_version = '''<built-in function dict_version>'''
___dict_contains = '''<function _get_closure_vars.<locals>.<lambda> at 0x1232f5240>'''
___tuple_iterator_len = '''<method '__length_hint__' of 'tuple_iterator' objects>'''
___tuple_iterator_getitem = '''<function tuple_iterator_getitem at 0x121ef7e20>'''
___get_torch_function_mode_stack_at = '''<function get_torch_function_mode_stack_at at 0x121efb2e0>'''
__math_isnan = '''<built-in function isnan>'''
__numpy_isnan = '''None'''
inf = '''inf'''
__load_module = '''<function import_module at 0x10465b9a0>'''
utils_device = '''<module 'torch.utils._device' from '/Users/youkaichao/anaconda3/envs/py310/lib/python3.10/site-packages/torch/utils/_device.py'>'''
device = '''<class 'torch.device'>'''
___from_numpy = '''<function from_numpy at 0x121ffb640>'''
___as_tensor = '''<function _as_tensor_fullprec at 0x1201ca560>'''
torch = '''<module 'torch' from '/Users/youkaichao/anaconda3/envs/py310/lib/python3.10/site-packages/torch/__init__.py'>'''
inspect = '''<module 'inspect' from '/Users/youkaichao/anaconda3/envs/py310/lib/python3.10/inspect.py'>'''
def __guard_0_for_f(L, G, **___kwargs_ignored):
    __guard_hit = True
    __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:479 in init_ambient_guards
    __guard_hit = __guard_hit and ___check_global_state()
    __guard_hit = __guard_hit and ___check_torch_function_mode_stack()
    __guard_hit = __guard_hit and check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[5], stride=[1])
    __guard_hit = __guard_hit and hasattr(L['x'], '_dynamo_dynamic_indices') == False
    return __guard_hit

# Note: please refer to the graph code in __compiled_fn_1*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_1(*args, **kwargs):
    pass

def __transformed_code_0_for_f(x):
    __temp_2, = __compiled_fn_1(x)
    return __temp_2


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def f(x):
    return x + 1

def transformed_f(x):
    __local_dict = {"x": x}
    __global_dict = globals()
    if __guard_0_for_f(__local_dict, __global_dict):
        return __transformed_code_0_for_f(x)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return f(x)

#============ end of f ============#
