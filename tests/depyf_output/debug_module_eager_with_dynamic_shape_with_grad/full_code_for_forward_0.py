
# Note: the following variables are used inside the guard function.
___check_tensors = '''None'''
___check_tensors_verbose = '''None'''
___check_global_state = '''<built-in method check of torch._C._dynamo.guards.GlobalStateGuard object at 0x123d1d8f0>'''
___check_torch_function_mode_stack = '''<function make_torch_function_mode_stack_guard.<locals>.check_torch_function_mode_stack at 0x124ba7ba0>'''
IsNonOverlappingAndDenseIndicator = '''<function eval_is_non_overlapping_and_dense at 0x121b54a40>'''
cast_symbool_to_symint_guardless = '''<function cast_symbool_to_symint_guardless at 0x121b54c20>'''
math = '''<module 'math' (built-in)>'''
torch = '''<module 'torch' from '/Users/youkaichao/uv_envs/py312/lib/python3.12/site-packages/torch/__init__.py'>'''
___check_type_id = '''<built-in function check_type_id>'''
___check_obj_id = '''<built-in function check_obj_id>'''
___odict_getitem = '''<method '__getitem__' of 'dict' objects>'''
___key_to_id = '''<function key_to_id at 0x121ca8540>'''
___dict_version = '''<built-in function dict_version>'''
___dict_contains = '''<function _get_closure_vars.<locals>.<lambda> at 0x124890ea0>'''
___tuple_iterator_len = '''<method '__length_hint__' of 'tuple_iterator' objects>'''
___normalize_range_iter = '''<function normalize_range_iter at 0x121c8be20>'''
___tuple_iterator_getitem = '''<function tuple_iterator_getitem at 0x121c8bce0>'''
___dataclass_fields = '''<function dataclass_fields at 0x121c8bd80>'''
___namedtuple_fields = '''<function _get_closure_vars.<locals>.<lambda> at 0x124890f40>'''
___get_torch_function_mode_stack_at = '''<function get_torch_function_mode_stack_at at 0x121c940e0>'''
__math_isnan = '''<built-in function isnan>'''
__numpy_isnan = '''<ufunc 'isnan'>'''
inf = '''inf'''
__load_module = '''<function import_module at 0x1025ae840>'''
utils_device = '''<module 'torch.utils._device' from '/Users/youkaichao/uv_envs/py312/lib/python3.12/site-packages/torch/utils/_device.py'>'''
device = '''<class 'torch.device'>'''
___from_numpy = '''<function from_numpy at 0x122224ae0>'''
___as_tensor = '''<function _as_tensor_fullprec at 0x1200d6e80>'''
inspect = '''<module 'inspect' from '/Users/youkaichao/.local/share/uv/python/cpython-3.12.8-macos-aarch64-none/lib/python3.12/inspect.py'>'''
def __guard_3_for_forward(L, G, **___kwargs_ignored):
    __guard_hit = True
    __guard_hit = __guard_hit and torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:800 in init_ambient_guards
    __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:788 in init_ambient_guards
    __guard_hit = __guard_hit and ___check_global_state()
    __guard_hit = __guard_hit and ___check_torch_function_mode_stack()
    __guard_hit = __guard_hit and check_tensor(L['a'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[None], stride=[1])  # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    __guard_hit = __guard_hit and hasattr(L['a'], '_dynamo_dynamic_indices') == False           # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    __guard_hit = __guard_hit and check_no_aliasing(L['a'], L['b'])
    __guard_hit = __guard_hit and L['b'].size()[0] == L['a'].size()[0]  # (unknown var s52, please file a bug)
    __guard_hit = __guard_hit and 2 <= L['a'].size()[0]  # b = b * -1 + b.sum()  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward (user code shown is first use of this value--the guard itself is not due user code but due to 0/1 specialization in the framework; to avoid specialization try torch._dynamo.decorators.mark_unbacked(tensor, dim))
    __guard_hit = __guard_hit and check_tensor(L['b'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[None], stride=[1])  # b = b * -1 + b.sum()  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward
    __guard_hit = __guard_hit and hasattr(L['b'], '_dynamo_dynamic_indices') == False           # b = b * -1 + b.sum()  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward
    __guard_hit = __guard_hit and L['b'].size()[0] == L['a'].size()[0]  # (unknown var s52, please file a bug)
    __guard_hit = __guard_hit and 2 <= L['a'].size()[0]  # b = b * -1 + b.sum()  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward (user code shown is first use of this value--the guard itself is not due user code but due to 0/1 specialization in the framework; to avoid specialization try torch._dynamo.decorators.mark_unbacked(tensor, dim))
    __guard_hit = __guard_hit and ___check_obj_id(G['torch'], 4341236096)                       # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    __guard_hit = __guard_hit and ___check_obj_id(G['torch'].abs, 4601209024)                   # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    return __guard_hit

# Note: please refer to the graph code in __compiled_fn_3_a7254709_067c_4c63_9994_672d84769ff2*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_3_a7254709_067c_4c63_9994_672d84769ff2(*args, **kwargs):
    pass

def __transformed_code_3_for_forward(self, a, b):
    x = None # this line helps Python to generate bytecode with at least the same number of local variables as the original function
    tmp_1 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_enter
    tmp_2 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_enter()
    tmp_3 = __import_torch_dot__dynamo_dot_utils.call_size
    tmp_4 = a
    tmp_5 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_exit
    __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_exit(tmp_2)
    graph_out_0 = __compiled_fn_3_a7254709_067c_4c63_9994_672d84769ff2(
        __import_torch_dot__dynamo_dot_utils.call_size(a, 0), tmp_4, b)
    []
    __temp_16 = []
    __temp_16.extend([])
    [__temp_16]
    return graph_out_0[0]


# Note: the following variables are used inside the guard function.
___check_tensors = '''None'''
___check_tensors_verbose = '''None'''
___check_global_state = '''<built-in method check of torch._C._dynamo.guards.GlobalStateGuard object at 0x123d35eb0>'''
___check_torch_function_mode_stack = '''<function make_torch_function_mode_stack_guard.<locals>.check_torch_function_mode_stack at 0x12486b6a0>'''
IsNonOverlappingAndDenseIndicator = '''<function eval_is_non_overlapping_and_dense at 0x121b54a40>'''
cast_symbool_to_symint_guardless = '''<function cast_symbool_to_symint_guardless at 0x121b54c20>'''
math = '''<module 'math' (built-in)>'''
torch = '''<module 'torch' from '/Users/youkaichao/uv_envs/py312/lib/python3.12/site-packages/torch/__init__.py'>'''
___check_type_id = '''<built-in function check_type_id>'''
___check_obj_id = '''<built-in function check_obj_id>'''
___odict_getitem = '''<method '__getitem__' of 'dict' objects>'''
___key_to_id = '''<function key_to_id at 0x121ca8540>'''
___dict_version = '''<built-in function dict_version>'''
___dict_contains = '''<function _get_closure_vars.<locals>.<lambda> at 0x124890ea0>'''
___tuple_iterator_len = '''<method '__length_hint__' of 'tuple_iterator' objects>'''
___normalize_range_iter = '''<function normalize_range_iter at 0x121c8be20>'''
___tuple_iterator_getitem = '''<function tuple_iterator_getitem at 0x121c8bce0>'''
___dataclass_fields = '''<function dataclass_fields at 0x121c8bd80>'''
___namedtuple_fields = '''<function _get_closure_vars.<locals>.<lambda> at 0x124890f40>'''
___get_torch_function_mode_stack_at = '''<function get_torch_function_mode_stack_at at 0x121c940e0>'''
__math_isnan = '''<built-in function isnan>'''
__numpy_isnan = '''<ufunc 'isnan'>'''
inf = '''inf'''
__load_module = '''<function import_module at 0x1025ae840>'''
utils_device = '''<module 'torch.utils._device' from '/Users/youkaichao/uv_envs/py312/lib/python3.12/site-packages/torch/utils/_device.py'>'''
device = '''<class 'torch.device'>'''
___from_numpy = '''<function from_numpy at 0x122224ae0>'''
___as_tensor = '''<function _as_tensor_fullprec at 0x1200d6e80>'''
inspect = '''<module 'inspect' from '/Users/youkaichao/.local/share/uv/python/cpython-3.12.8-macos-aarch64-none/lib/python3.12/inspect.py'>'''
def __guard_0_for_forward(L, G, **___kwargs_ignored):
    __guard_hit = True
    __guard_hit = __guard_hit and torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:800 in init_ambient_guards
    __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:788 in init_ambient_guards
    __guard_hit = __guard_hit and ___check_global_state()
    __guard_hit = __guard_hit and ___check_torch_function_mode_stack()
    __guard_hit = __guard_hit and check_tensor(L['a'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[10], stride=[1])  # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    __guard_hit = __guard_hit and hasattr(L['a'], '_dynamo_dynamic_indices') == False           # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    __guard_hit = __guard_hit and check_no_aliasing(L['a'], L['b'])
    __guard_hit = __guard_hit and check_tensor(L['b'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[10], stride=[1])  # b = b * -1 + b.sum()  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward
    __guard_hit = __guard_hit and hasattr(L['b'], '_dynamo_dynamic_indices') == False           # b = b * -1 + b.sum()  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:14 in forward
    __guard_hit = __guard_hit and ___check_obj_id(G['torch'], 4341236096)                       # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    __guard_hit = __guard_hit and ___check_obj_id(G['torch'].abs, 4601209024)                   # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:13 in forward
    return __guard_hit

# Note: please refer to the graph code in __compiled_fn_1_2463bd09_01cf_4d4e_a7e9_7751c2dcbe61*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_1_2463bd09_01cf_4d4e_a7e9_7751c2dcbe61(*args, **kwargs):
    pass

def __transformed_code_0_for_forward(self, a, b):
    x = None # this line helps Python to generate bytecode with at least the same number of local variables as the original function
    tmp_1 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_enter
    tmp_2 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_enter()
    tmp_3 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_exit
    __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_exit(tmp_2)
    graph_out_0 = __compiled_fn_1_2463bd09_01cf_4d4e_a7e9_7751c2dcbe61(a, b)
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
    if __guard_3_for_forward(__local_dict, __global_dict):
        return __transformed_code_3_for_forward(self, a, b)
    if __guard_0_for_forward(__local_dict, __global_dict):
        return __transformed_code_0_for_forward(self, a, b)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, a, b)

#============ end of forward ============#
