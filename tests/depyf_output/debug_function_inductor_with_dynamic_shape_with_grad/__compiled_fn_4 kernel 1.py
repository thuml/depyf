
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


cpp_fused_abs_add_div_mul_neg_sgn_0 = async_compile.cpp('''
#include "/var/folders/vm/ssf622nn02j77t14q1j8_88w0000gn/T/torchinductor_youkaichao/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       const long ks0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr1[static_cast<long>(x0)];
            auto tmp7 = in_ptr2[static_cast<long>(x0)];
            auto tmp2 = std::abs(tmp1);
            auto tmp3 = static_cast<float>(1.0);
            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
            auto tmp5 = tmp0 / tmp4;
            auto tmp6 = decltype(tmp0)(-tmp0);
            auto tmp8 = tmp7 / tmp4;
            auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
            auto tmp10 = tmp1 > 0 ? decltype(tmp1)(1) : decltype(tmp1)(0);
            auto tmp11 = tmp1 < 0 ? decltype(tmp1)(1) : decltype(tmp1)(0);
            auto tmp12 = tmp10 - tmp11;
            auto tmp13 = decltype(tmp9)(tmp9 * tmp12);
            auto tmp14 = decltype(tmp5)(tmp5 + tmp13);
            out_ptr0[static_cast<long>(x0)] = tmp14;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, div, tangents_1 = args
    args.clear()
    s0 = primals_1
    assert_size_stride(primals_2, (s0, ), (1, ))
    assert_size_stride(div, (s0, ), (1, ))
    assert_size_stride(tangents_1, (s0, ), (1, ))
    buf0 = empty((s0, ), device='cpu', dtype=torch.float32)
    cpp_fused_abs_add_div_mul_neg_sgn_0(c_void_p(tangents_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(div.data_ptr()), c_void_p(buf0.data_ptr()), c_long(s0))
    del div
    del primals_2
    del tangents_1
    return (None, buf0, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 8
    primals_2 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    div = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, div, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
