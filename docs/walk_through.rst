A Walk Through Example of ``torch.compile``
===========================================

In this tutorial, we will learn how does PyTorch compiler work for the following code:

.. code-block:: python

    import torch

    @torch.compile
    def modified_sigmoid(x, y):
        x = 1.0 / (torch.exp(-x) + 5)
        if x.mean() > 0.5:
            x = x / 1.1
        return x * y

    shape_10_inputs = torch.randn(10, requires_grad=True), torch.randn(10, requires_grad=True)
    shape_8_inputs = torch.randn(8, requires_grad=True), torch.randn(8, requires_grad=True)
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

That's it! You can call it your fist Just-In-Time compiler, although it is `compiled` by your brain rather than an automated program.