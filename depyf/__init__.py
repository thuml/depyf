from typing import List, Tuple, Dict, Union, Callable, Optional
from types import CodeType

from .decompiler import Decompiler

def decompile(code: Union[CodeType, Callable]):
    """Decompile a code object or a function."""
    return Decompiler(code).decompile()
