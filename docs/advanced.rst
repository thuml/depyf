Advanced Usage of ``depyf``
==========================================

Enhance PyTorch Logging
----------------------------
Beyond two simple context manager ``depyf.prepare_debug()`` and ``depyf.debug()``, you can also add a single line ``import depyf; depyf.install()`` to enable the decompilation of PyTorch bytecode.

Then, run the code with the environment variable ``export TORCH_LOGS="+dynamo,guards,bytecode"`` to get verbose logging information. (Or you can ``export TORCH_LOGS="+bytecode"`` to focus on the bytecode only.)

In the long log output, you can see that decompiled bytecode occurs after the modified bytecode of Dynamo.

Decompile dynamically-generated Python functions
--------------------------------------------------------

``depyf`` can also be used for decompiling bytecode that is not related to PyTorch.

For example, we can use ``depyf`` to understand some dynamically generated functions from ``dataclasses``:

.. code-block:: python

    from dataclasses import dataclass
    @dataclass
    class Data:
        x: int
        y: float

    import depyf
    print(depyf.decompile(Data.__init__))
    print(depyf.decompile(Data.__eq__))

In the above example, ``@dataclass`` adds many magic methods for the class ``Data``, and we can inspect the source code of these methods:

.. code-block:: python

    def __init__(self, x, y):
        self.x = x
        self.y = y
        return None

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return (self.x, self.y) == (other.x, other.y)
        return NotImplemented

The output source code is semantically equivalent to the function, but not syntactically the same. It verbosely adds many details that are hidden in the Python code. For example, the above output code explicitly returns `None`, which is typically ignored.

By using ``depyf``, you can understand many internal details of Python.
