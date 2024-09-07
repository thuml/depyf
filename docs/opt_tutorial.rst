Optimization Tutorial
===========================================

This tutorial will show the process of optimizing code with ``torch.compile``, using the ``depyf`` library.

The example code we want to optimize is as follows:

.. code-block:: python

    import torch

    class F(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            self.i = i

        def forward(self, x):
            return x + self.i

    class Mod(torch.nn.Module):
        def __init__(self, n: int):
            super().__init__()
            self.fs = torch.nn.ModuleList([F(i) for i in range(n)])

        @torch.compile
        def forward(self, x):
            for f in self.fs:
                x = f(x)
            return x

    total_time = 0
    import time

    mod = Mod(100)
    mod(torch.tensor([1])) # compile

    x = torch.tensor([2]) # create input tensor
    start = time.time()
    for i in range(10000):
        y = mod(x)
        # do something with y
    end = time.time()
    total_time += end - start
    print(total_time)

For illustration purposes, we make the computation in the function trivial to understand. In practice, the function can be a complex function that does some real computation. On an MacOS machine, running the compiled function 10000 times takes about 0.7 seconds. We want to optimize the code to make it run faster.

To optimize the code, we need to first get an understanding of what's going on in the code. We can use the ``depyf`` library to decompile the bytecode, with just two more lines:

.. code-block:: python

    ... # the code above `mod(torch.tensor([1]))`
    import depyf
    with depyf.prepare_debug("dump_src_dir/"):
        mod(torch.tensor([1])) # compile
    ... # the code below `mod(torch.tensor([1]))`

After running the code above, we will find a new directory ``dump_src_dir/`` in the current directory. The directory contains the decompiled source code. Inside the ``full_code_for_forward_0.py`` file, we can find:

.. code-block:: python

    def __guard_0_for_forward(L, G, **___kwargs_ignored):
        __guard_hit = True
        __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:460 in init_ambient_guards
        __guard_hit = __guard_hit and ___check_global_state()
        __guard_hit = __guard_hit and check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.int64, device=None, requires_grad=False, size=[1], stride=[1])
        __guard_hit = __guard_hit and hasattr(L['x'], '_dynamo_dynamic_indices') == False
        __guard_hit = __guard_hit and ___check_obj_id(L['self'], 4381402176)
        __guard_hit = __guard_hit and ___check_obj_id(L['self'].training, 4378113208)
        __guard_hit = __guard_hit and ___check_obj_id(L['self'].fs, 4381406064)
        __guard_hit = __guard_hit and ___check_type_id(L['self'].fs, 4451582432)
        __guard_hit = __guard_hit and len(L['self'].fs) == 100
        __guard_hit = __guard_hit and ___check_obj_id(L['self'].fs.training, 4378113208)
        __guard_hit = __guard_hit and ___check_obj_id(L['self'].fs[0], 4381402320)
        __guard_hit = __guard_hit and not ___dict_contains('forward', L['self'].fs[0].__dict__)
        __guard_hit = __guard_hit and ___check_obj_id(L['self'].fs[0].training, 4378113208)
        __guard_hit = __guard_hit and L['self'].fs[0].i == 0
        __guard_hit = __guard_hit and ___check_obj_id(L['self'].fs[1], 4381407936)
        __guard_hit = __guard_hit and not ___dict_contains('forward', L['self'].fs[1].__dict__)
        __guard_hit = __guard_hit and ___check_obj_id(L['self'].fs[1].training, 4378113208)
        __guard_hit = __guard_hit and L['self'].fs[1].i == 1
        __guard_hit = __guard_hit and ___check_obj_id(L['self'].fs[2], 4381407888)
        __guard_hit = __guard_hit and not ___dict_contains('forward', L['self'].fs[2].__dict__)
        __guard_hit = __guard_hit and ___check_obj_id(L['self'].fs[2].training, 4378113208)
        __guard_hit = __guard_hit and L['self'].fs[2].i == 2
      ... # more checks for i in range(3, 100)
        __guard_hit = __guard_hit and ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_module'], 4453328560)
        __guard_hit = __guard_hit and ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_module'].torch, 4381310736)
        __guard_hit = __guard_hit and ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_module'].torch._C, 4386445312)
        __guard_hit = __guard_hit and ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_module'].torch._C._get_tracing_state, 4435369296)
        __guard_hit = __guard_hit and ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 4378072768)
        __guard_hit = __guard_hit and not G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks
        __guard_hit = __guard_hit and ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 4378072768)
        __guard_hit = __guard_hit and not G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks
        __guard_hit = __guard_hit and ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 4378072768)
        __guard_hit = __guard_hit and not G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks
        __guard_hit = __guard_hit and ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 4378072768)
        __guard_hit = __guard_hit and not G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks
        return __guard_hit

This is the code ``torch.compile`` generates to check the input to see if the compiled function can be called. However, we can see it is way too conservative. It is checking a lot of things that will be constants during the whole execution, e.g. ``L['self'].fs[0].i == 0`` wants to make sure ``self.fs[0].i`` is still ``0``. Technically, we can indeed change ``self.fs[0].i`` during the execution, but our code is not doing this. To optimize the code, we should be able to bypass these checks.

Remember from the :doc:`walk_through`, that ``torch.compile`` is a just-in-time compiler. It means all the above checks are executed every time we call the function, introducing significant overhead.

With the help of ``depyf``, one obvious way to optimize the code is to make ``__guard_0_for_forward`` function directly return ``True``. But this would require ``torch.compile`` to provide a way to skip the checks. Currently, ``torch.compile`` does not provide such a way. An alternative approach, is to use ``depyf`` to directly call the compiled function without the checks, through the bytecode hook mechanism:

.. code-block:: python

    import torch

    class F(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            self.i = i

        def forward(self, x):
            return x + self.i

    import depyf
    from depyf.optimization import TorchCompileWrapperWithCustomDispacther

    class Mod(TorchCompileWrapperWithCustomDispacther):
        def __init__(self, n: int):
            self.fs = torch.nn.ModuleList([F(i) for i in range(n)])
            compiled_callable = torch.compile(self.forward)
            super().__init__(compiled_callable)

        def forward(self, x):
            for f in self.fs:
                x = f(x)
            return x

        def __call__(self, x):
            if len(self.compiled_codes) == 1:
                with self.dispatch_to_code(0):
                    return self.forward(x)
            else:
                return self.compiled_callable(x)

    total_time = 0
    import time

    mod = Mod(100)
    mod(torch.tensor([1])) # compile

    x = torch.tensor([2]) # create input tensor
    start = time.time()
    for i in range(10000):
        y = mod(x)
        # do something with y
    end = time.time()
    total_time += end - start
    print(total_time)

It executes the same code as before, but with the ``TorchCompileWrapperWithCustomDispacther`` class. Running the code, we can see the execution time is reduced to about 0.05 seconds, which means we previously spend almost 0.65 seconds on the checks.

Under the hood, ``TorchCompileWrapperWithCustomDispacther`` will hijack the bytecode compiled by ``torch.compile`` and directly call the compiled function without the checks. As we can see in the ``__call__`` method, if there is already one compiled code, it will directly call the compiled code. Otherwise, it will call the ``torch.compile`` function to compile the code. This will remove the Dynamo overhead.

The code above is an extreme example of running a very tiny computation while Dynamo introduces significant overhead. In practice, the overhead of Dynamo is not as significant as 10x of the main computation. But it can still be significant enough to be optimized. For example, in `vLLM's TPU integration <https://github.com/vllm-project/vllm/pull/7898>_` , it uses this technique to remove the overhead of the Dynamo checks, because TPU is very fast and the overhead of the checks is significant. With this technique, it helps to improve the throughput of the TPU by 4%.

This is just one example of how to optimize code with ``torch.compile``. The general workflow is:
1. Use ``depyf`` to decompile the bytecode to understand what's going on.
2. Find the bottleneck in the code.
3. Either refactor the code to remove the bottleneck, or use ``depyf`` to directly call the compiled function without the bottleneck.
