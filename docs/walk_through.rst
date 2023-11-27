A Walk Through Example of ``torch.compile``
===========================================

In this tutorial, we will learn how does PyTorch compiler work for the following code:

.. code-block:: python

    import torch

    @torch.compile
    def function(inputs):
        x = inputs["x"]
        y = inputs["y"]
        x = x.cos().cos()
        if x.mean() > 0.5:
            x = x / 1.1
        return x * y

    shape_10_inputs = {"x": torch.randn(10, requires_grad=True), "y": torch.randn(10, requires_grad=True)}
    shape_8_inputs = {"x": torch.randn(8, requires_grad=True), "y": torch.randn(8, requires_grad=True)}
    # warmup
    for i in range(100):
        output = function(shape_10_inputs)
        output = function(shape_8_inputs)
    
    # execution of compiled functions
    output = function(shape_10_inputs)

The code tries to implement an activation function :math:`\text{cos}(\text{cos}(x))`, and scales the output according to its activation value, then multiplies the output with another tensor ``y``.

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

This leads to two basic concepts in Just-In-Time compilers: guards, and transformed code. Guards are conditions when the functions can be optimized, and transformed code is the optimized version of functions. In the above simple Just-In-Time compiler example, ``isinstance(mod, A)`` is a guard, and ``return 2 * x`` is the corresponding transformed code that is equivalent to the original code under the guarding condition, but is significantly faster.

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

Going more rigorous, the above example is actually an Ahead-of-time compiler: we inspect all the available source code, and before running any function, we write the optimized function in terms of guards and transformed code. A real Just-In-Time procedure should be:

.. code-block:: python

    def f(x, mod):
        for guard, transformed_code in f.compiled_entries:
            if guard(x, mod):
                return transformed_code(x, mod)
        try:
            guard, transformed_code = compile_and_optimize(x, mod)
            f.compiled_entries.append([guard, transformed_code])
            return transformed_code(x, mod)
        except FailToCompileError:
            y = mod(x)
            z = torch.log(y)
            return z

A Just-In-Time compiler just optimizes for what it has seen. Everytime it sees a new input that does not satisfy any guarding condition, it compiles a new guard and transformed code for the new input.

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

    @just_in_time_compile # an imaginary compiler function
    def f(x, mod):
        y = mod(x)
        z = torch.log(y)
        return z

    a = A()
    b = B()
    x = torch.randn((5, 5, 5))
    # before executing f(x, a), f.compiled_entries == [] is empty.
    # after executing f(x, a), f.compiled_entries == [Guard("isinstance(x, torch.Tensor) and isinstance(mod, A)"), TransformedCode("return 2 * x")]
    f(x, a)
    # the second call of f(x, a) hit a condition, so we can just execute the transformed code
    f(x, a)
    # f(x, b) will trigger compilation and add a new compiled entry
    # before executing f(x, b), f.compiled_entries == [Guard("isinstance(x, torch.Tensor) and isinstance(mod, A)"), TransformedCode("return 2 * x")]
    # after executing f(x, b), f.compiled_entries == [Guard("isinstance(x, torch.Tensor) and isinstance(mod, A)"), TransformedCode("return 2 * x"), Guard("isinstance(x, torch.Tensor) and isinstance(mod, B)"), TransformedCode("return -x")]
    f(x, b)
    # the second call of f(x, b) hit a condition, so we can just execute the transformed code
    f(x, b)

That's basically how ``torch.compile`` works as a Just-In-Time compiler. We can even extract those compiled entries from functions, see the `PyTorch documentation <https://pytorch.org/docs/main/torch.compiler_deepdive.html#how-to-inspect-artifacts-generated-by-torchdynamo>`_ for more details.

How does Dynamo transform and modify the function?
---------------------------------------------------

As we understand the global picture of ``torch.compile`` as a Just-In-Time compiler, we can diver deeper in how it works. Unlike general purpose compilers like ``gcc`` or ``llvm``, ``torch.compile`` is a domain-specific compiler: it only focuses on PyTorch related computation graph. Therefore, we need a tool to separate users code into two parts: plain python code and computation graph code.

``Dynamo``, living inside the module ``torch._dynamo``, is the tool for doing this. Normally we don't interact with this module directly. It is called inside the ``torch.compile`` function.

Conceptually, ``Dynamo`` does the following things:

- Find the first operation that cannot be represented in computation graph but requires the value of computed value in the graph (e.g. ``print`` a tensor's value, use a tensor's value to decide ``if`` statements control flow in Python).
- Split the previous operations into two parts: a computation graph that is purely about tensor computation, and some Python code about manipulating Python objects.
- Leave the rest operations as one or two new functions (called ``resume functions``), and trigger the above analysis again.

To enable such a fine-grained manipulation of functions, ``Dynamo`` operates on the level of Python bytecode, a level that is lower than Python source code.

The following procedure describes what Dynamo does to our function ``function``.

.. image:: _static/images/dynamo-workflow.png
  :width: 1200
  :alt: Dynamo workflow

One important feature of ``Dynamo``, is that it can analyze all the functions called inside the ``function`` function. If a function can be represented entirely in a computation graph, that function call will be inlined and the function call is eliminated.

The mission of ``Dynamo``, is to extract computation graphs from Python code in a safe and sound way. Once we have the computation graphs, we can enter the world of computation graph optimization now.

Dynamic shape support from Dynamo
---------------------------------------------------
Deep learning compilers usually favor static shape inputs. That's why the guarding conditions above include shape guards. Our first function call uses input of shape ``[10]``, but the second function call uses input of shape ``[8]``. It will fail the shape guards, therefore trigger a new code transform.

By default, Dynamo supports dynamic shapes. When the shape guards fail, it will analyze and compare the shapes, and try to generalize the shape. In this case, after seeing input of shape ``[8]``, it will try to generalize to arbitary one-dimensional shape ``[s0]``, known as dynamic shape or symbolic shape.

AOTAutograd: generate backward computation graph from forward graph
------------------------------------------------------------------------

The above code only deals with forward computation graph. One important missing piece is how to get the backward computation graph to compute the gradient.

In plain PyTorch code, backward computation is triggered by the ``backward`` function call on some scalar loss value. Each PyTorch function stores what is required for backward during forward computation.

To explain what happens in eager mode during backward, we have the following implementation mimicing the builtin behavior of ``torch.cos`` function (some `background knowledge <https://pytorch.org/docs/main/notes/extending.html#extending-torch-autograd>`_ about how to write custom function with autograd support in PyTorch is required):

.. code-block:: python

    import torch
    class Cosine(torch.autograd.Function):
        @staticmethod
        def forward(x0):
            x1 = torch.cos(x0)
            return x1, x0

        @staticmethod
        def setup_context(ctx, inputs, output):
            x1, x0 = output
            print(f"saving tensor of size {x0.shape}")
            ctx.save_for_backward(x0)

        @staticmethod
        def backward(ctx, grad_output):
            x0, = ctx.saved_tensors
            result = (-torch.sin(x0)) * grad_output
            return result

    # Wrap Cosine in a function so that it is clearer what the output is
    def cosine(x):
        y, x= Cosine.apply(x)
        return y

    def naive_two_cosine(x0):
        x1 = cosine(x0)
        x2 = cosine(x1)
        return x2

Running the above function with an input that requires grad, we can see that two tensors are saved:

.. code-block:: python

    input = torch.randn((5, 5, 5), requires_grad=True)
    output = naive_two_cosine(input)

The output:

.. code-block:: text

    saving tensor of size torch.Size([5, 5, 5])
    saving tensor of size torch.Size([5, 5, 5])

If we have the computation graph ahead-of-time, we can optimize the computation as follows:

.. code-block:: python

    class OptimizedTwoCosine(torch.autograd.Function):
        @staticmethod
        def forward(x0):
            x1 = torch.cos(x0)
            x2 = torch.cos(x1)
            return x2, x0

        @staticmethod
        def setup_context(ctx, inputs, output):
            x2, x0 = output
            print(f"saving tensor of size {x0.shape}")
            ctx.save_for_backward(x0)

        @staticmethod
        def backward(ctx, grad_x2):
            x0, = ctx.saved_tensors
            # re-compute in backward
            x1 = torch.cos(x0)
            grad_x1 = (-torch.sin(x1)) * grad_x2
            grad_x0 = (-torch.sin(x0)) * grad_x1
            return grad_x0

    def optimized_two_cosine(x):
        x2, x0 = OptimizedTwoCosine.apply(x)
        return x2

Running the above function with an input that requires grad, we can see that only one tensor is saved:

.. code-block:: python

    input = torch.randn((5, 5, 5), requires_grad=True)
    output = optimized_two_cosine(input)

The output:

.. code-block:: text

    saving tensor of size torch.Size([5, 5, 5])

And we can check the correctness of two implementations against native PyTorch implementation:

.. code-block:: python

    input = torch.randn((5, 5, 5), requires_grad=True)
    grad_output = torch.randn((5, 5, 5), requires_grad=True)

    output1 = torch.cos(torch.cos(input))
    (output1 * grad_output).sum().backward()
    grad_input1 = input.grad; input.grad = None

    output2 = naive_two_cosine(input)
    (output2 * grad_output).sum().backward()
    grad_input2 = input.grad; input.grad = None

    output3 = optimized_two_cosine(input)
    (output3 * grad_output).sum().backward()
    grad_input3 = input.grad; input.grad = None

    assert torch.allclose(output1, output2)
    assert torch.allclose(output1, output3)
    assert torch.allclose(grad_input1, grad_input2)
    assert torch.allclose(grad_input1, grad_input3)

The following computation graph shows the details of a naive implementation:

.. image:: _static/images/eager-joint-graph.png
  :width: 1200
  :alt: Eager mode autograd

And the following computation graph shows the details of an optimized implementation:

.. image:: _static/images/aot-joint-graph.png
  :width: 1200
  :alt: AOT mode autograd

We can only save one value, and recompute the first ``cos`` function to get another value for backward.

AOTAutograd does the above optimization automatically. In essense, it dynamically generates a function like the following:

.. code-block:: python

    class OptimizedFunction(torch.autograd.Function):
        @staticmethod
        def forward(inputs):
            outputs, saved_tensors = forward_graph(inputs)
            return outputs, saved_tensors

        @staticmethod
        def setup_context(ctx, inputs, output):
            outputs, saved_tensors = output
            ctx.save_for_backward(saved_tensors)

        @staticmethod
        def backward(ctx, grad_outputs):
            saved_tensors = ctx.saved_tensors
            grad_inputs = backward_graph(grad_outputs, saved_tensors)
            return grad_inputs

    def optimized_function(inputs):
        outputs, saved_tensors = OptimizedFunction.apply(inputs)
        return outputs

This way, the saved tensors are made explicit, and the ``optimized_function`` accepts exactly the same inputs as the original function, while the producing exactly the same output as the original function and having exactly the same backward behavior as the original function.

By varying the amount of ``saved_tensors``, we can:

- Save more tensors for backward, so that backward computation is less heavy.
- Save less tensors for backward, so that the memory footprint of forward is less heavy.

Usually people goes the second way, i.e., saving memory by having more computation in the backward pass. And AOTAutograd will automatically select the optimal way to save memory.

That is basically how AOT Autograd works!

Backend: compile and optimize computation graph
--------------------------------------------------

Finally, after ``Dynamo`` separates PyTorch code from Python code, and after ``AOTAutograd`` generates the backward computation graph from the forward computation graph, we entered the world of pure computation graphs.

This is how the ``backend`` argument in ``torch.compile`` comes into play. It takes the above computation graphs as input, and generates optimized code that can execute the above computation graphs.

In general, a backend will try every optimize techniques it knows for the computation graphs. Each optimization technique is called one ``pass``. Some optimization passes from the PyTorch builtin backend, namely the ``Inductor`` backend, can be found `here <https://github.com/pytorch/pytorch/tree/main/torch/_inductor/fx_passes>`_.

In addition, no optimization is also a possible optimization. This is called ``eager`` backend in PyTorch.

In a strict sense, the ``backend`` option in ``torch.compile`` affects whether backward computation graph exists and how the computation graphs are optimized. In practice, custom backends usually work with ``AOTAutograd`` to obtain backward computation graphs, and they only need to deal with computation graph optimization, no matter it is forward graph or backward graph.

Summary
--------------------------------------------------

The following table shows the difference among several ``backend`` option in ``torch.compile``. If we want to adapt our code to ``torch.compile``, it is recommended to try ``backend="eager"`` first to see how our code is transformed into computation graph, and then to try ``backend="aot_eager"`` to see if we are satisfied with the backward graph, and finally try ``backend="inductor"`` to see if we can get any performance benefit.

.. list-table:: Summary of backends
   :header-rows: 1

   * - backend
     - forward computation graph
     - backward computation graph
     - computation graph optimization
   * - ``eager``
     - captured by ``Dynamo``
     - N/A
     - N/A
   * - ``aot_eager``
     - captured by ``Dynamo``
     - generated by ``AOTAutograd``
     - N/A
   * - ``inductor``
     - captured by ``Dynamo``
     - generated by ``AOTAutograd``
     - optimized by ``Inductor``
   * - ``...`` (many other backend options)
     - captured by ``Dynamo``
     - generated by ``AOTAutograd``
     - optimized by custom implementations
