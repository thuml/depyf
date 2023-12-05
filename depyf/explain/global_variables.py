import os

import torch

from torch._inductor.codecache import PyCodeCache

data = {
    "dump_src_dir": os.path.join(os.path.dirname(__file__), "dumped_src"),
    "unpatched__exec_with_source": torch.fx.graph_module._exec_with_source,
    "unpatched_load_by_key_path": PyCodeCache.load_by_key_path,
    "unpatched___call__": torch._dynamo.eval_frame.OptimizeContext.__call__,
    "optimized_functions": set(),
    "is_inside_prepare_debug": False,
}
