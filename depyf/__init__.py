from typing import List, Tuple, Dict, Union, Callable, Optional, Any
from types import CodeType
import warnings

from .decompiler import Decompiler

from .code_transform import structure_hash

def decompile(code: Union[CodeType, Callable]):
    """Decompile a code object or a function."""
    return Decompiler(code).decompile()


try:
    import torch
    torch_version = torch.__version__
    valid = ("dev" not in torch_version and torch_version >= "2.2") or ("dev" in torch_version and torch_version.split("dev")[-1] >= "20231020")
    if not valid:
        warnings.warn((
            "Please use the nightly version of PyTorch to enable bytecode hooks.\n"
            "PyTorch nightly can be installed by: `conda install pytorch-nightly::pytorch torchvision torchaudio -c pytorch-nightly`"
        ))
except ImportError:
    pass

from depyf.explain.enhance_logging import install, uninstall

import os

__version__ = open(f"{os.path.dirname(__file__)}/VERSION.txt").read().strip()

