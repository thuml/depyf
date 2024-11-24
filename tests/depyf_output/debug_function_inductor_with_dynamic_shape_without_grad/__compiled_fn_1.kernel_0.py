# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused_abs_add_div_lt_sum_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*', 'bool*', 'float*'], '''
#include "/var/folders/vm/ssf622nn02j77t14q1j8_88w0000gn/T/torchinductor_youkaichao/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       bool* out_ptr1,
                       float* out_ptr2)
{
    {
        {
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8LL); x0+=static_cast<int64_t>(4LL))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                tmp_acc0_vec = tmp_acc0_vec + tmp0;
            }
            for(int64_t x0=static_cast<int64_t>(8LL); x0<static_cast<int64_t>(10LL); x0+=static_cast<int64_t>(2LL))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(2LL));
                tmp_acc0_vec = sum_masked_reduce(tmp_acc0_vec, tmp0, static_cast<int64_t>(2LL));
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr0[static_cast<int64_t>(0LL)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        auto tmp0 = out_ptr0[static_cast<int64_t>(0LL)];
        auto tmp1 = static_cast<float>(0.0);
        auto tmp2 = tmp0 < tmp1;
        out_ptr1[static_cast<int64_t>(0LL)] = tmp2;
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8LL); x0+=static_cast<int64_t>(4LL))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
            auto tmp1 = tmp0.abs();
            auto tmp2 = static_cast<float>(1.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp0 / tmp4;
            tmp5.store(out_ptr2 + static_cast<int64_t>(x0));
        }
        for(int64_t x0=static_cast<int64_t>(8LL); x0<static_cast<int64_t>(10LL); x0+=static_cast<int64_t>(2LL))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(2LL));
            auto tmp1 = tmp0.abs();
            auto tmp2 = static_cast<float>(1.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp0 / tmp4;
            tmp5.store(out_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(2LL));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (10, ), (1, ))
    assert_size_stride(arg1_1, (10, ), (1, ))
    buf1 = empty_strided_cpu((), (), torch.float32)
    buf2 = empty_strided_cpu((), (), torch.bool)
    buf0 = empty_strided_cpu((10, ), (1, ), torch.float32)
    cpp_fused_abs_add_div_lt_sum_0(arg1_1, arg0_1, buf1, buf2, buf0)
    del arg0_1
    del arg1_1
    return (buf0, buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((10, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((10, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
