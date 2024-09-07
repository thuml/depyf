Optimization Tutorial
===========================================

In this tutorial, we will demonstrate how to optimize code using ``torch.compile``, with the help of the ``depyf`` library.

Example Code
------------

Let's start with a simple example that we want to optimize:

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
    mod(torch.tensor([1]))  # Compile the function

    x = torch.tensor([2])  # Create input tensor
    start = time.time()
    for i in range(10000):
        y = mod(x)
        # do something with y
    end = time.time()
    total_time += end - start
    print(total_time)

This example is intentionally simplified to make the computation easy to follow. In a real-world scenario, the function would perform more complex operations. On a MacOS machine, running the compiled function 10,000 times takes around 0.7 seconds. Our goal is to optimize the code to reduce this execution time.

Understanding the Code with Depyf
---------------------------------

To optimize the code, we first need to understand what's happening during execution. The ``depyf`` library can decompile the bytecode and provide insights. We can add two lines to the previous code:

.. code-block:: python

    # Insert these lines before ``mod(torch.tensor([1]))``
    import depyf
    with depyf.prepare_debug("dump_src_dir/"):
        mod(torch.tensor([1]))  # Compile the function
    # Remaining code as above

After running the code, a new directory named ``dump_src_dir/`` will appear. This directory contains the decompiled source code. For example, in the file ``full_code_for_forward_0.py``, you will find:

.. code-block:: python

    def __guard_0_for_forward(L, G, **___kwargs_ignored):
        __guard_hit = True
        __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None
        __guard_hit = __guard_hit and ___check_global_state()
        __guard_hit = __guard_hit and check_tensor(L['x'], Tensor, DispatchKeySet(CPU, ...), ...)
        ...
        __guard_hit = __guard_hit and len(L['self'].fs) == 100
        __guard_hit = __guard_hit and L['self'].fs[0].i == 0
        __guard_hit = __guard_hit and L['self'].fs[1].i == 1
        __guard_hit = __guard_hit and L['self'].fs[2].i == 2
        ...
        return __guard_hit

This is the code that ``torch.compile`` generates to check whether the function's input remains valid for the compiled function. However, many of these checks are overly conservative. For example, ``L['self'].fs[0].i == 0`` checks that ``self.fs[0].i`` is still ``0``, even though our code doesn't modify this value.

Remember from the :doc:`walk_through`, that ``torch.compile`` is a just-in-time compiler. It means all the above checks are executed every time we call the function, introducing significant overhead.

Optimizing the Code
-------------------

Since ``torch.compile`` performs these checks every time the function is called, they introduce overhead. To optimize the code, we can bypass these checks. One approach is to modify the ``__guard_0_for_forward`` function to return ``True`` immediately, but ``torch.compile`` doesn't provide a direct mechanism for this.

Instead, we can use ``depyf`` to directly call the compiled function without the checks. The following code demonstrates this approach:

.. code-block:: python

    import torch
    import depyf
    from depyf.optimization import TorchCompileWrapperWithCustomDispatcher

    class F(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            self.i = i

        def forward(self, x):
            return x + self.i

    class Mod(TorchCompileWrapperWithCustomDispatcher):
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
    mod(torch.tensor([1]))  # Compile

    x = torch.tensor([2])  # Input tensor
    start = time.time()
    for i in range(10000):
        y = mod(x)
    end = time.time()
    total_time += end - start
    print(total_time)

In this code, the ``TorchCompileWrapperWithCustomDispatcher`` class is used to bypass the checks. By doing this, the execution time drops to about 0.05 seconds, compared to the original 0.7 seconds. This shows that the checks were responsible for most of the overhead.

How It Works
------------

``TorchCompileWrapperWithCustomDispatcher`` hijacks the bytecode generated by ``torch.compile`` and directly calls the compiled function without the guards. The ``__call__`` method checks whether a compiled version already exists, and if so, it dispatches directly to the compiled code.

Real-World Applications
-----------------------

This is an extreme example with a trivial computation, where the overhead introduced by Dynamo is disproportionately large. In practice, the overhead is typically not as severe. However, it can still be significant in high-performance environments, such as when running code on TPU. TPU code is often performance-sensitive, and removing unnecessary checks can lead to substantial speedups.

For example, in `vLLM's TPU integration <https://github.com/vllm-project/vllm/pull/7898>`_, this optimization technique is used to remove Dynamo's overhead, improving TPU throughput by 4%.

Conclusion
----------

Optimizing code with ``torch.compile`` involves:

1. Using ``depyf`` to decompile the bytecode and understand the performance bottlenecks.
2. Identifying and addressing unnecessary checks or other sources of overhead.
3. Using ``depyf`` to directly call the compiled function, bypassing unnecessary steps where appropriate.

By following these steps, we can significantly improve performance, especially in environments where execution time is critical.
