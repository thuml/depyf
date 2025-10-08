
# Note: the following variables are used inside the guard function.
___check_tensors = '''None'''
___check_tensors_verbose = '''None'''
___check_global_state = '''<built-in method check of torch._C._dynamo.guards.GlobalStateGuard object at 0x11381e0b0>'''
___check_torch_function_mode_stack = '''<function make_torch_function_mode_stack_guard.<locals>.check_torch_function_mode_stack at 0x115ea7d80>'''
IsNonOverlappingAndDenseIndicator = '''<function eval_is_non_overlapping_and_dense at 0x110054a40>'''
cast_symbool_to_symint_guardless = '''<function cast_symbool_to_symint_guardless at 0x110054c20>'''
math = '''<module 'math' (built-in)>'''
torch = '''<module 'torch' from '/Users/youkaichao/uv_envs/py312/lib/python3.12/site-packages/torch/__init__.py'>'''
___check_type_id = '''<built-in function check_type_id>'''
___check_obj_id = '''<built-in function check_obj_id>'''
___odict_getitem = '''<method '__getitem__' of 'dict' objects>'''
___key_to_id = '''<function key_to_id at 0x1101a0540>'''
___dict_version = '''<built-in function dict_version>'''
___dict_contains = '''<function _get_closure_vars.<locals>.<lambda> at 0x113b90f40>'''
___tuple_iterator_len = '''<method '__length_hint__' of 'tuple_iterator' objects>'''
___normalize_range_iter = '''<function normalize_range_iter at 0x11018fe20>'''
___tuple_iterator_getitem = '''<function tuple_iterator_getitem at 0x11018fce0>'''
___dataclass_fields = '''<function dataclass_fields at 0x11018fd80>'''
___namedtuple_fields = '''<function _get_closure_vars.<locals>.<lambda> at 0x113b90fe0>'''
___get_torch_function_mode_stack_at = '''<function get_torch_function_mode_stack_at at 0x1101b80e0>'''
__math_isnan = '''<built-in function isnan>'''
__numpy_isnan = '''<ufunc 'isnan'>'''
inf = '''inf'''
__load_module = '''<function import_module at 0x102a3a840>'''
utils_device = '''<module 'torch.utils._device' from '/Users/youkaichao/uv_envs/py312/lib/python3.12/site-packages/torch/utils/_device.py'>'''
device = '''<class 'torch.device'>'''
___from_numpy = '''<function from_numpy at 0x1107c8ae0>'''
___as_tensor = '''<function _as_tensor_fullprec at 0x10e3d6e80>'''
inspect = '''<module 'inspect' from '/Users/youkaichao/.local/share/uv/python/cpython-3.12.8-macos-aarch64-none/lib/python3.12/inspect.py'>'''
def __guard_5_for_forward(L, G, **___kwargs_ignored):
    __guard_hit = True
    __guard_hit = __guard_hit and torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:800 in init_ambient_guards
    __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:788 in init_ambient_guards
    __guard_hit = __guard_hit and ___check_global_state()
    __guard_hit = __guard_hit and ___check_torch_function_mode_stack()
    __guard_hit = __guard_hit and check_tensor(L['a'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[None], stride=[1])  # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    __guard_hit = __guard_hit and hasattr(L['a'], '_dynamo_dynamic_indices') == False           # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    __guard_hit = __guard_hit and check_no_aliasing(L['a'], L['b'])
    __guard_hit = __guard_hit and L['b'].size()[0] == L['a'].size()[0]  # (unknown var s52, please file a bug)
    __guard_hit = __guard_hit and 2 <= L['a'].size()[0]  # b = b * -1 + b.sum()  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward (user code shown is first use of this value--the guard itself is not due user code but due to 0/1 specialization in the framework; to avoid specialization try torch._dynamo.decorators.mark_unbacked(tensor, dim))
    __guard_hit = __guard_hit and check_tensor(L['b'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[None], stride=[1])  # b = b * -1 + b.sum()  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward
    __guard_hit = __guard_hit and hasattr(L['b'], '_dynamo_dynamic_indices') == False           # b = b * -1 + b.sum()  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward
    __guard_hit = __guard_hit and L['b'].size()[0] == L['a'].size()[0]  # (unknown var s52, please file a bug)
    __guard_hit = __guard_hit and 2 <= L['a'].size()[0]  # b = b * -1 + b.sum()  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward (user code shown is first use of this value--the guard itself is not due user code but due to 0/1 specialization in the framework; to avoid specialization try torch._dynamo.decorators.mark_unbacked(tensor, dim))
    __guard_hit = __guard_hit and ___check_obj_id(G['torch'], 4346003840)                       # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    __guard_hit = __guard_hit and ___check_obj_id(G['torch'].abs, 4347732160)                   # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    return __guard_hit

# Note: please refer to the graph code in __compiled_fn_3_8603b153_36ae_4ac8_a89b_3187562ff62e*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_3_8603b153_36ae_4ac8_a89b_3187562ff62e(*args, **kwargs):
    pass

def __transformed_code_5_for_forward(self, a, b):
    x = None # this line helps Python to generate bytecode with at least the same number of local variables as the original function
    tmp_1 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_enter
    tmp_2 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_enter()
    tmp_3 = __import_torch_dot__dynamo_dot_utils.call_size
    tmp_4 = a
    tmp_5 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_exit
    __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_exit(tmp_2)
    graph_out_0 = __compiled_fn_3_8603b153_36ae_4ac8_a89b_3187562ff62e(
        __import_torch_dot__dynamo_dot_utils.call_size(a, 0), tmp_4, b)
    []
    __temp_16 = []
    __temp_16.extend([])
    [__temp_16]
    return graph_out_0[0]


# Note: the following variables are used inside the guard function.
___check_tensors = '''None'''
___check_tensors_verbose = '''None'''
___check_global_state = '''<built-in method check of torch._C._dynamo.guards.GlobalStateGuard object at 0x1111b2e30>'''
___check_torch_function_mode_stack = '''<function make_torch_function_mode_stack_guard.<locals>.check_torch_function_mode_stack at 0x113b66a20>'''
IsNonOverlappingAndDenseIndicator = '''<function eval_is_non_overlapping_and_dense at 0x110054a40>'''
cast_symbool_to_symint_guardless = '''<function cast_symbool_to_symint_guardless at 0x110054c20>'''
math = '''<module 'math' (built-in)>'''
torch = '''<module 'torch' from '/Users/youkaichao/uv_envs/py312/lib/python3.12/site-packages/torch/__init__.py'>'''
___check_type_id = '''<built-in function check_type_id>'''
___check_obj_id = '''<built-in function check_obj_id>'''
___odict_getitem = '''<method '__getitem__' of 'dict' objects>'''
___key_to_id = '''<function key_to_id at 0x1101a0540>'''
___dict_version = '''<built-in function dict_version>'''
___dict_contains = '''<function _get_closure_vars.<locals>.<lambda> at 0x113b90f40>'''
___tuple_iterator_len = '''<method '__length_hint__' of 'tuple_iterator' objects>'''
___normalize_range_iter = '''<function normalize_range_iter at 0x11018fe20>'''
___tuple_iterator_getitem = '''<function tuple_iterator_getitem at 0x11018fce0>'''
___dataclass_fields = '''<function dataclass_fields at 0x11018fd80>'''
___namedtuple_fields = '''<function _get_closure_vars.<locals>.<lambda> at 0x113b90fe0>'''
___get_torch_function_mode_stack_at = '''<function get_torch_function_mode_stack_at at 0x1101b80e0>'''
__math_isnan = '''<built-in function isnan>'''
__numpy_isnan = '''<ufunc 'isnan'>'''
inf = '''inf'''
__load_module = '''<function import_module at 0x102a3a840>'''
utils_device = '''<module 'torch.utils._device' from '/Users/youkaichao/uv_envs/py312/lib/python3.12/site-packages/torch/utils/_device.py'>'''
device = '''<class 'torch.device'>'''
___from_numpy = '''<function from_numpy at 0x1107c8ae0>'''
___as_tensor = '''<function _as_tensor_fullprec at 0x10e3d6e80>'''
inspect = '''<module 'inspect' from '/Users/youkaichao/.local/share/uv/python/cpython-3.12.8-macos-aarch64-none/lib/python3.12/inspect.py'>'''
def __guard_3_for_forward(L, G, **___kwargs_ignored):
    __guard_hit = True
    __guard_hit = __guard_hit and torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:800 in init_ambient_guards
    __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:788 in init_ambient_guards
    __guard_hit = __guard_hit and ___check_global_state()
    __guard_hit = __guard_hit and ___check_torch_function_mode_stack()
    __guard_hit = __guard_hit and check_tensor(L['a'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[10], stride=[1])  # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    __guard_hit = __guard_hit and hasattr(L['a'], '_dynamo_dynamic_indices') == False           # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    __guard_hit = __guard_hit and check_no_aliasing(L['a'], L['b'])
    __guard_hit = __guard_hit and check_tensor(L['b'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[10], stride=[1])  # b = b * -1 + b.sum()  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward
    __guard_hit = __guard_hit and hasattr(L['b'], '_dynamo_dynamic_indices') == False           # b = b * -1 + b.sum()  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward
    __guard_hit = __guard_hit and ___check_obj_id(G['torch'], 4346003840)                       # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    __guard_hit = __guard_hit and ___check_obj_id(G['torch'].abs, 4347732160)                   # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    return __guard_hit

# Note: please refer to the graph code in __compiled_fn_1_cbdc5c3f_e983_46fb_89e4_60bc145d60ec*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_1_cbdc5c3f_e983_46fb_89e4_60bc145d60ec(*args, **kwargs):
    pass

def __transformed_code_3_for_forward(self, a, b):
    x = None # this line helps Python to generate bytecode with at least the same number of local variables as the original function
    tmp_1 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_enter
    tmp_2 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_enter()
    tmp_3 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_exit
    __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_exit(tmp_2)
    graph_out_0 = __compiled_fn_1_cbdc5c3f_e983_46fb_89e4_60bc145d60ec(a, b)
    []
    __temp_7 = []
    __temp_7.extend([])
    [__temp_7]
    return graph_out_0[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, a, b):
    x = a / (torch.abs(a) + 1)
    b = b * -1 + b.sum()
    return x * b

def transformed_forward(self, a, b):
    __local_dict = {"self": self, "a": a, "b": b}
    __global_dict = globals()
    if __guard_5_for_forward(__local_dict, __global_dict):
        return __transformed_code_5_for_forward(self, a, b)
    if __guard_3_for_forward(__local_dict, __global_dict):
        return __transformed_code_3_for_forward(self, a, b)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, a, b)

#============ end of forward ============#
