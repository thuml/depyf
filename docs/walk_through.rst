A Walk Through Example of ``torch.compile``
===========================================

In this tutorial, we will learn how does PyTorch compiler work for the following code:

.. code-block:: python

    import torch

    @torch.compile
    def modified_sigmoid(inputs):
        x = inputs["x"]
        y = inputs["y"]
        x = 1.0 / (torch.exp(-x) + 5)
        if x.mean() > 0.5:
            x = x / 1.1
        return x * y

    shape_10_inputs = {"x": torch.randn(10, requires_grad=True), "y": torch.randn(10, requires_grad=True)}
    shape_8_inputs = {"x": torch.randn(8, requires_grad=True), "y": torch.randn(8, requires_grad=True)}
    # warmup
    for i in range(100):
        output = modified_sigmoid(*shape_10_inputs)
        output = modified_sigmoid(*shape_8_inputs)
    
    # execution of compiled functions
    output = modified_sigmoid(*shape_10_inputs)

The code tries to implement a strange sigmoid function :math:`\frac{1}{e^{-x} + 5}`, and scales the output according to its activation value, then multiplies the output with another tensor ``y``.

The tutorial intends to cover the following aspects of PyTorch compiler:

- Dynamo (graph capture)
- AOTAutograd (forward graph and backward graph)
- Inductor (compile graph to kernel)

PyTorch compiler is a Just-In-Time compiler
--------------------------------------------

The first concept we have to know is that PyTorch compiler is a Just-In-Time compiler. So what does `Just-In-Time compiler` mean? Well, let's look at another example:

.. code-block:: python

    import torch

    class A(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.exp(2 * x)

    class B(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.exp(-x)

    def f(x, mod):
        y = mod(x)
        z = torch.log(y)
        return z

We write this funny function ``f``, that contains a module call, and a ``torch.log`` call. Anyone with elementary math knowledge cannot wait to optimize the code as follows:

.. code-block:: python

    def f(x, mod):
        if isinstance(mod, A):
            return 2 * x
        elif isinstance(mod, B):
            return -x

That's it! We can call it our fist Just-In-Time compiler, although it is `compiled` by our brain rather than an automated program.

The basic workflow of a Just-In-Time compiler is: right before the function is executed, it analyzes if the execution can be optimized, and what is the condition under which the function execution can be optimized. Hopefully, the condition is general enough for new inputs, so that the benfit outweights the cost of Just-In-Time compilation.

This leads to two basic concepts in Just-In-Time compilers: guards, and compiled code. Guards are conditions when the functions can be optimized, and compiled code is the optimized version of functions. In the above simple Just-In-Time compiler example, ``isinstance(mod, A)`` is a guard, and ``return 2 * x`` is the corresponding compiled code that is equivalent to the original code under the guarding condition, but is significantly faster.

And if we want to be rigorous, our Just-In-Time example should be updated as follows:

.. code-block:: python

    def f(x, mod):
        if isinstance(x, torch.Tensor) and isinstance(mod, A):
            return 2 * x
        elif isinstance(x, torch.Tensor) and isinstance(mod, B):
            return -x
        else:
            y = mod(x)
            z = torch.log(y)
            return z

We have to check each parameter so that our guards are sound, and also fallback to the original code if we fail to optimize the code.

Going more rigorous, the above example is actually an Ahead-of-time compiler: we inspect all the available source code, and before running any function, we write the optimized function in terms of guards and compiled code. A real Just-In-Time procedure should be:

.. code-block:: python

    def f(x, mod):
        for guard, compiled_code in f.compiled_entries:
            if guard(x, mod):
                return compiled_code(x, mod)
        try:
            guard, compiled_code = compile_and_optimize(x, mod)
            f.compiled_entries.append([guard, compiled_code])
            return compiled_code(x, mod)
        except FailToCompileError:
            y = mod(x)
            z = torch.log(y)
            return z

A Just-In-Time compiler just optimizes for what it has seen. Everytime it sees a new input that does not satisfy any guarding condition, it compiles a new guard and compiled code for the new input.

Let's explain it step-by-step:

.. code-block:: python

    import torch

    class A(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.exp(2 * x)

    class B(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.exp(-x)

    @just_in_time_compile
    def f(x, mod):
        y = mod(x)
        z = torch.log(y)
        return z

    a = A()
    b = B()
    x = torch.randn((5, 5, 5))
    # before executing f(x, a), f.compiled_entries == []
    # after executing f(x, a), f.compiled_entries == [Guard("isinstance(x, torch.Tensor) and isinstance(mod, A)"), CompiledCode("return 2 * x")]
    f(x, a)
    # the second call of f(x, a) hit a condition, so we can just execute the compiled code
    f(x, a)
    # f(x, b) will trigger compilation and add a new compiled entry
    # before executing f(x, b), f.compiled_entries == [Guard("isinstance(x, torch.Tensor) and isinstance(mod, A)"), CompiledCode("return 2 * x")]
    # after executing f(x, b), f.compiled_entries == [Guard("isinstance(x, torch.Tensor) and isinstance(mod, A)"), CompiledCode("return 2 * x"), Guard("isinstance(x, torch.Tensor) and isinstance(mod, B)"), CompiledCode("return -x")]
    f(x, b)
    # the second call of f(x, b) hit a condition, so we can just execute the compiled code
    f(x, b)

That's basically how ``torch.compile`` works as a Just-In-Time compiler. We can even extract those compiled entries from functions, see the `PyTorch documentation <https://pytorch.org/docs/main/torch.compiler_deepdive.html#how-to-inspect-artifacts-generated-by-torchdynamo>`_ for more details.