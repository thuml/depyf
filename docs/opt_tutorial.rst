Optimization Tutorial
===========================================

This tutorial will guide you through the process of optimizing code with ``torch.compile``, using the ``depyf`` library.

The example code we want to optimize is as follows:

.. code-block:: python

    import torch

    class F(torch.nn.Module):
        def forward(self, x, i):
            return x + i

    @torch.compile
    def g(x):
        x = F()(x, 5)
        return x

    for i in range(1000):
        x = torch.tensor([i]) # create input tensor
        y = g(x)
        # do something with y

For illustration purposes, we make the computation in the function ``g`` trivial. In practice, the function ``g`` can be a complex function that does some real computation.

To optimize the code, we need to first get an understanding of what's going on in the code. We can use the ``depyf`` library to decompile the bytecode of the function ``g``, with just two more lines:

.. code-block:: python

    import torch

    class F(torch.nn.Module):
        def forward(self, x, i):
            return x + i

    @torch.compile
    def g(x):
        x = F()(x, 5)
        return x

    import depyf
    with depyf.prepare_debug("dump_src_dir/"):
        for i in range(1000):
            x = torch.tensor([i]) # create input tensor
            y = g(x)
            # do something with y

After running the code above, you will find a new directory ``dump_src_dir/`` in the current directory. The directory contains the decompiled source code of the function ``g``. Inside the ``full_code_for_g_0.py`` file, you can find:

.. code-block:: python

    def __guard_0_for_g(L, G, **___kwargs_ignored):
        __guard_hit = True
        __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:460 in init_ambient_guards
        __guard_hit = __guard_hit and ___check_global_state()
        __guard_hit = __guard_hit and check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[1], stride=[1])
        __guard_hit = __guard_hit and hasattr(L['x'], '_dynamo_dynamic_indices') == False
        __guard_hit = __guard_hit and ___check_obj_id(G['F'], 4576341520)
        __guard_hit = __guard_hit and ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_module'], 4413465488)
        __guard_hit = __guard_hit and ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_module'].torch, 4309172144)
        __guard_hit = __guard_hit and ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_module'].torch._C, 4314290416)
        __guard_hit = __guard_hit and ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_module'].torch._C._get_tracing_state, 4337294032)
        __guard_hit = __guard_hit and ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 4305934016)
        __guard_hit = __guard_hit and not G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks
        __guard_hit = __guard_hit and ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 4305934016)
        __guard_hit = __guard_hit and not G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks
        __guard_hit = __guard_hit and ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 4305934016)
        __guard_hit = __guard_hit and not G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks
        __guard_hit = __guard_hit and ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 4305934016)
        __guard_hit = __guard_hit and not G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks
        return __guard_hit

This is the code ``torch.compile`` generates to check the input to see if the compiled function can be called. However, we can see it is way too conservative. It is checking a lot of things that will be constants during the whole execution, e.g. ``___check_obj_id(G['F'], 4576341520)`` wants to make sure ``F`` is still a class object. Technically, we can indeed change the class object during the execution, but it is not a common practice. And these checks are executed every time we call the function ``g``, which counts as overhead.

If we just want to use ``torch.compile`` to compile the code, but skip the checks, we can use ``TorchCompileWrapperWithCustomDispacther`` from ``depyf``:

.. code-block:: python

    import torch

    class F(torch.nn.Module):
        def forward(self, x, i):
            return x + i

    def g(x):
        x = F()(x, 5)
        return x

    import depyf
    from depyf.optimization import TorchCompileWrapperWithCustomDispacther

    class MyMod(TorchCompileWrapperWithCustomDispacther):
        def __init__(self):
            compiled_callable = torch.compile(g)
            super().__init__(compiled_callable)
        
        def forward(self, x):
            return g(x)

        def __call__(self, x):
            if len(self.compiled_codes) == 1:
                with self.dispatch_to_code(0):
                    return self.forward(x)
            else:
                return self.compiled_callable(x)

    mod = MyMod()

    for i in range(1000):
        x = torch.tensor([i]) # create input tensor
        y = mod(x)
        # do something with y

Under the hood, it will hijack the bytecode compiled by ``torch.compile`` and directly call the compiled function without the checks. As we can see in the ``__call__`` method, if there is already one compiled code, it will directly call the compiled code. Otherwise, it will call the ``torch.compile`` function to compile the code. This will remove the Dynamo overhead.

This technique is used in `vLLM's TPU integration <https://github.com/vllm-project/vllm/pull/7898>_` to remove the overhead of the Dynamo checks, because TPU is very fast and the overhead of the checks is significant. With this technique, it helps to improve the throughput of the TPU by 4%.

This is just one example of how to optimize code with ``torch.compile``. You can also use the decompiled source code to understand the code better and optimize it in other ways.