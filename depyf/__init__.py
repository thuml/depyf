from typing import List, Tuple, Dict, Union, Callable, Optional
from types import CodeType

from .decompiler import Decompiler

def decompile(code: Union[CodeType, Callable]):
    """Decompile a code object or a function."""
    return Decompiler(code).decompile()


import types


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
                "https://github.com/youkaichao/depyf/issues."
            )

def install():
    """Install the hook to decompile pytorch bytecode."""
    import torch
    if getattr(torch._dynamo.config, "output_bytecode_hooks", None) is None:
        raise RuntimeError("Please upgrade pytorch to have bytecode hooks.")
    if pytorch_bytecode_src_hook in torch._dynamo.config.output_bytecode_hooks:
        return
    torch._dynamo.config.output_bytecode_hooks.append(pytorch_bytecode_src_hook)

def uninstall():
    """Uninstall the hook."""
    import torch
    if getattr(torch._dynamo.config, "output_bytecode_hooks", None) is None:
        raise RuntimeError("Please upgrade pytorch to have bytecode hooks.")
    if pytorch_bytecode_src_hook not in torch._dynamo.config.output_bytecode_hooks:
        return
    torch._dynamo.config.output_bytecode_hooks.remove(pytorch_bytecode_src_hook)

__version__ = open("VERSION.txt").read().strip()
