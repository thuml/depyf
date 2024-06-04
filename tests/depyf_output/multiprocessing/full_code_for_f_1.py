
# Note: the following variables are used inside the guard function.
___check_global_state = '''<built-in method check of torch._C._dynamo.guards.GlobalStateGuard object at 0x15df32a30>'''
___check_tensors = '''<built-in method check of torch._C._dynamo.guards.TensorGuards object at 0x15df33190>'''
tensor_check_names = '''["L['x']"]'''
utils_device = '''<module 'torch.utils._device' from '/Users/youkaichao/anaconda3/envs/py310/lib/python3.10/site-packages/torch/utils/_device.py'>'''
def __guard_0_for_f(L, G, **___kwargs_ignored):
    return (___check_global_state()) \
        and (hasattr(L['x'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and (___check_tensors(L['x'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_1*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_1(*args, **kwargs):
    pass

def __transformed_code_0_for_f(x):
    return __compiled_fn_1(x)


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
