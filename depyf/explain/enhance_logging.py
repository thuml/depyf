import types
from depyf.decompiler import decompile, DecompilationError


def pytorch_bytecode_src_hook(code: types.CodeType, new_code: types.CodeType):
    import torch
    bytecode_log = torch._logging.getArtifactLogger(
        "torch._dynamo.convert_frame", "bytecode"
    )
    import logging

    if bytecode_log.isEnabledFor(logging.DEBUG):
        try:
            decompiled_src = decompile(new_code)
            bytecode_log.debug("possible source code:")
            bytecode_log.debug(decompiled_src)
        except DecompilationError as e:
            bytecode_log.debug("Decompilation fails due to: %s", str(e))
        finally:
            bytecode_log.debug(
                "If you find the decompiled code is wrong,"
                "please submit an issue at "
                "https://github.com/thuml/depyf/issues."
            )


_handle = None


def install():
    """
    Install the bytecode hook for PyTorch, integrate into PyTorch's logging system.

    Example:

    .. code-block:: python

        import torch
        import depyf
        depyf.install()
        # anything with torch.compile
        @torch.compile
        def f(a, b):
            return a + b
        f(torch.tensor(1), torch.tensor(2))
    
    Turn on bytecode log by ``export TORCH_LOGS="+bytecode"``, and execute the script.
    We will see the decompiled source code in the log:

    .. code-block:: text

        ORIGINAL BYTECODE f test.py line 5 
        7           0 LOAD_FAST                0 (a)
                    2 LOAD_FAST                1 (b)
                    4 BINARY_ADD
                    6 RETURN_VALUE
        
        
        MODIFIED BYTECODE f test.py line 5 
        5           0 LOAD_GLOBAL              0 (__compiled_fn_1)
                    2 LOAD_FAST                0 (a)
                    4 LOAD_FAST                1 (b)
                    6 CALL_FUNCTION            2
                    8 UNPACK_SEQUENCE          1
                    10 RETURN_VALUE
        
        
        possible source code:
        def f(a, b):
            __temp_2, = __compiled_fn_1(a, b)
            return __temp_2
        
        If you find the decompiled code is wrong,please submit an issue at https://github.com/thuml/depyf/issues.
    
    To uninstall the hook, use :func:`depyf.uninstall()`.
    """
    import torch
    global _handle
    if _handle is not None:
        return
    _handle = torch._dynamo.convert_frame.register_bytecode_hook(
        pytorch_bytecode_src_hook)


def uninstall():
    """
    Uninstall the bytecode hook for PyTorch.
    Should be called after :func:`depyf.install()`.
    """
    global _handle
    if _handle is None:
        return
    _handle.remove()
    _handle = None
