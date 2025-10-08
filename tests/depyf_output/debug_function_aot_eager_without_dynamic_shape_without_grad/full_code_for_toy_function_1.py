
# Note: the following variables are used inside the guard function.
___check_tensors = '''None'''
___check_tensors_verbose = '''None'''
___check_global_state = '''<built-in method check of torch._C._dynamo.guards.GlobalStateGuard object at 0x117790c30>'''
___check_torch_function_mode_stack = '''<function make_torch_function_mode_stack_guard.<locals>.check_torch_function_mode_stack at 0x12201fb00>'''
IsNonOverlappingAndDenseIndicator = '''<function eval_is_non_overlapping_and_dense at 0x116d4ca40>'''
cast_symbool_to_symint_guardless = '''<function cast_symbool_to_symint_guardless at 0x116d4cc20>'''
math = '''<module 'math' (built-in)>'''
torch = '''<module 'torch' from '/Users/youkaichao/uv_envs/py312/lib/python3.12/site-packages/torch/__init__.py'>'''
___check_type_id = '''<built-in function check_type_id>'''
___check_obj_id = '''<built-in function check_obj_id>'''
___odict_getitem = '''<method '__getitem__' of 'dict' objects>'''
___key_to_id = '''<function key_to_id at 0x116ea8540>'''
___dict_version = '''<built-in function dict_version>'''
___dict_contains = '''<function _get_closure_vars.<locals>.<lambda> at 0x12201f1a0>'''
___tuple_iterator_len = '''<method '__length_hint__' of 'tuple_iterator' objects>'''
___normalize_range_iter = '''<function normalize_range_iter at 0x116e87e20>'''
___tuple_iterator_getitem = '''<function tuple_iterator_getitem at 0x116e87ce0>'''
___dataclass_fields = '''<function dataclass_fields at 0x116e87d80>'''
___namedtuple_fields = '''<function _get_closure_vars.<locals>.<lambda> at 0x12201f240>'''
___get_torch_function_mode_stack_at = '''<function get_torch_function_mode_stack_at at 0x116e940e0>'''
__math_isnan = '''<built-in function isnan>'''
__numpy_isnan = '''<ufunc 'isnan'>'''
inf = '''inf'''
__load_module = '''<function import_module at 0x100922840>'''
utils_device = '''<module 'torch.utils._device' from '/Users/youkaichao/uv_envs/py312/lib/python3.12/site-packages/torch/utils/_device.py'>'''
device = '''<class 'torch.device'>'''
___from_numpy = '''<function from_numpy at 0x1175f4ae0>'''
___as_tensor = '''<function _as_tensor_fullprec at 0x1150d6e80>'''
inspect = '''<module 'inspect' from '/Users/youkaichao/.local/share/uv/python/cpython-3.12.8-macos-aarch64-none/lib/python3.12/inspect.py'>'''
def __guard_1_for_toy_function(L, G, **___kwargs_ignored):
    __guard_hit = True
    __guard_hit = __guard_hit and torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:800 in init_ambient_guards
    __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:788 in init_ambient_guards
    __guard_hit = __guard_hit and ___check_global_state()
    __guard_hit = __guard_hit and ___check_torch_function_mode_stack()
    __guard_hit = __guard_hit and check_tensor(L['a'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[10], stride=[1])  # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function
    __guard_hit = __guard_hit and hasattr(L['a'], '_dynamo_dynamic_indices') == False           # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function
    __guard_hit = __guard_hit and check_no_aliasing(L['a'], L['b'])
    __guard_hit = __guard_hit and check_tensor(L['b'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[10], stride=[1])  # b = b * -1 + b.sum()  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:5 in toy_function
    __guard_hit = __guard_hit and hasattr(L['b'], '_dynamo_dynamic_indices') == False           # b = b * -1 + b.sum()  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:5 in toy_function
    __guard_hit = __guard_hit and ___check_obj_id(G['torch'], 4311302528)                       # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function
    __guard_hit = __guard_hit and ___check_obj_id(G['torch'].abs, 4312867008)                   # x = a / (torch.abs(a) + 1)  # data/DeepLearning/depyf/tests/test_pytorch/test_pytorch.py:4 in toy_function
    return __guard_hit

# Note: please refer to the graph code in __compiled_fn_1_4b1a426f_b748_4f5d_af75_b0986a5cfa2d*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_1_4b1a426f_b748_4f5d_af75_b0986a5cfa2d(*args, **kwargs):
    pass

def __transformed_code_1_for_toy_function(a, b):
    x = None # this line helps Python to generate bytecode with at least the same number of local variables as the original function
    tmp_1 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_enter
    tmp_2 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_enter()
    tmp_3 = __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_exit
    __import_torch_dot__dynamo_dot_utils.record_pregraph_bytecode_exit(tmp_2)
    graph_out_0 = __compiled_fn_1_4b1a426f_b748_4f5d_af75_b0986a5cfa2d(a, b)
    []
    __temp_7 = []
    __temp_7.extend([])
    [__temp_7]
    return graph_out_0[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def toy_function(a, b):
    x = a / (torch.abs(a) + 1)
    b = b * -1 + b.sum()
    return x * b

def transformed_toy_function(a, b):
    __local_dict = {"a": a, "b": b}
    __global_dict = globals()
    if __guard_1_for_toy_function(__local_dict, __global_dict):
        return __transformed_code_1_for_toy_function(a, b)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return toy_function(a, b)

#============ end of toy_function ============#
