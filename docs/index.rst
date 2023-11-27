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

- Generate source code for transformed bytecode and resume functions.
- 

.. toctree::
   :maxdepth: 1

   walk_through
