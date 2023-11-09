import types
from depyf.decompiler import decompile

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
        except Exception as e:
            bytecode_log.debug("Decompilation fails due to: %s", str(e))
        finally:
            bytecode_log.debug(
                "If you find the decompiled code is wrong,"
                "please submit an issue at "
                "https://github.com/thuml/depyf/issues."
            )

_handle = None

def install():
    import torch
    global _handle
    if _handle is not None:
        return
    _handle = torch._dynamo.convert_frame.register_bytecode_hook(pytorch_bytecode_src_hook)


def uninstall():
    global _handle
    if _handle is None:
        return
    _handle.remove()
    _handle = None