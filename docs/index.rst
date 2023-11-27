Welcome to the documentation of ``depyf``
==========================================

Before learning the usage of ``depyf``, we recommend reading the :doc:`walk_through` example of ``torch.compile``, so that you can understand how ``depyf`` would help you.

``depyf`` aims to address two pain points of ``torch.compile``:

- ``torch.compile`` transforms Python bytecode, but very few developers can understand Python bytecode. ``depyf`` helps to decompile the transformed bytecode back into Python source code, so that developers can understand how ``torch.compile`` transforms their code. This greatly helps users to adapt their code to ``torch.compile``, so that they can write code friendly to ``torch.compile``.
- Many functions in ``torch.compile`` are dynamically generated, which can only be run as a black box. ``depyf`` helps to dump the source code to files, and to link these functions with the source code files, so that users can use debuggers to step through these functions. This greatly helps users to understand ``torch.compile`` and debug issues like ``NaN`` during training.

Take the workflow from the walk-through example:

.. image:: _static/images/dynamo-workflow.png
  :width: 1200
  :alt: Dynamo workflow

``depyf`` helps to:

- Give a source code description of the above workflow, so that users can easily understand it.
- Generate source code for transformed bytecode and resume functions.
- Link graph computation functions with on-disk code, so that debuggers can step through the code.

The main usage of ``depyf`` involves two context managers:

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

    import depyf
    with depyf.prepare_debug(function, "./debug_dir"):
        # warmup
        for i in range(100):
            output = function(shape_10_inputs)
            output = function(shape_8_inputs)
    # the program will pause here for you to set breakpoints
    # then you can hit breakpoints when running the function
    with depyf.debug():
        output = function(shape_10_inputs)


.. toctree::
   :maxdepth: 1

   walk_through
