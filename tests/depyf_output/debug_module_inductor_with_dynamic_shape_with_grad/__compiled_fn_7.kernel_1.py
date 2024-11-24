# AOT ID: ['2_backward']
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


cpp_fused_abs_add_div_mul_neg_sgn_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*', 'const int64_t'], '''
#include "/var/folders/vm/ssf622nn02j77t14q1j8_88w0000gn/T/torchinductor_youkaichao/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       const int64_t ks0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))); x0+=static_cast<int64_t>(4LL))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
            auto tmp2 = tmp1.abs();
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 + tmp4;
            auto tmp6 = tmp0 / tmp5;
            auto tmp7 = tmp0.neg();
            auto tmp8 = tmp1 / tmp5;
            auto tmp9 = tmp8 / tmp5;
            auto tmp10 = tmp7 * tmp9;
            auto tmp11 =
            [&]()
            {
                auto left = decltype(tmp1)::blendv(decltype(tmp1)(0), decltype(tmp1)(1), decltype(tmp1)(0) < tmp1);
                auto right = decltype(tmp1)::blendv(decltype(tmp1)(0), decltype(tmp1)(1), tmp1 < decltype(tmp1)(0));
                return left - right;
            }
            ()
            ;
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp6 + tmp12;
            tmp13.store(out_ptr0 + static_cast<int64_t>(x0));
        }
        for(int64_t x0=static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))); x0<static_cast<int64_t>(ks0); x0+=(static_cast<int64_t>(ks0 + ((-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))) == 0 ? 1 : static_cast<int64_t>(ks0 + ((-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))))))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + ((-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + ((-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))));
            auto tmp2 = tmp1.abs();
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 + tmp4;
            auto tmp6 = tmp0 / tmp5;
            auto tmp7 = tmp0.neg();
            auto tmp8 = tmp1 / tmp5;
            auto tmp9 = tmp8 / tmp5;
            auto tmp10 = tmp7 * tmp9;
            auto tmp11 =
            [&]()
            {
                auto left = decltype(tmp1)::blendv(decltype(tmp1)(0), decltype(tmp1)(1), decltype(tmp1)(0) < tmp1);
                auto right = decltype(tmp1)::blendv(decltype(tmp1)(0), decltype(tmp1)(1), tmp1 < decltype(tmp1)(0));
                return left - right;
            }
            ()
            ;
            auto tmp12 = tmp10 * tmp11;
            auto tmp13 = tmp6 + tmp12;
            tmp13.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + ((-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, tangents_1 = args
    args.clear()
    s0 = primals_1
    assert_size_stride(primals_2, (s0, ), (1, ))
    assert_size_stride(tangents_1, (s0, ), (1, ))
    buf0 = empty_strided_cpu((s0, ), (1, ), torch.float32)
    cpp_fused_abs_add_div_mul_neg_sgn_0(tangents_1, primals_2, buf0, s0)
    del primals_2
    del tangents_1
    return (None, buf0, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 8
    primals_2 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
