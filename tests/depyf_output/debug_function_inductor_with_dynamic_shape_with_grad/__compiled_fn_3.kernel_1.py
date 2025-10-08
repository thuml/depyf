# AOT ID: ['1_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused_abs_add_div_expand_mul_neg_sgn_sum_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'const int64_t'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       const int64_t ks0)
{
    {
        {
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(ks0); x0+=static_cast<int64_t>(4LL))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                        auto tmp2 = tmp1.abs();
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = tmp1 / tmp5;
                        auto tmp7 = tmp0 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    if(C10_UNLIKELY(x0 >= static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))) && x0 < static_cast<int64_t>(ks0)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + (-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + (-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))));
                        auto tmp2 = tmp1.abs();
                        auto tmp3 = static_cast<float>(1.0);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 + tmp4;
                        auto tmp6 = tmp1 / tmp5;
                        auto tmp7 = tmp0 * tmp6;
                        tmp_acc0_vec = sum_masked_reduce(tmp_acc0_vec, tmp7, static_cast<int64_t>(ks0 + (-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))));
                    }
                }
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float, 1>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr0[static_cast<int64_t>(0LL)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(ks0); x0+=static_cast<int64_t>(4LL))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL))))))
                {
                    auto tmp0 = out_ptr0[static_cast<int64_t>(0LL)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                    auto tmp16 = in_ptr3[static_cast<int64_t>(0LL)];
                    auto tmp3 = tmp2.abs();
                    auto tmp4 = static_cast<float>(1.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp2 / tmp6;
                    auto tmp8 = tmp1 * tmp7;
                    auto tmp9 = static_cast<float>(-1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = at::vec::Vectorized<float>(tmp0);
                    auto tmp13 = tmp12 + tmp11;
                    auto tmp15 = tmp14 * tmp10;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 + tmp17;
                    auto tmp19 = tmp1 * tmp18;
                    auto tmp20 = tmp19 / tmp6;
                    auto tmp21 = tmp19.neg();
                    auto tmp22 = tmp7 / tmp6;
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 =
                    [&]()
                    {
                        auto left = decltype(tmp2)::blendv(decltype(tmp2)(0), decltype(tmp2)(1), decltype(tmp2)(0) < tmp2);
                        auto right = decltype(tmp2)::blendv(decltype(tmp2)(0), decltype(tmp2)(1), tmp2 < decltype(tmp2)(0));
                        return left - right;
                    }
                    ()
                    ;
                    auto tmp25 = tmp23 * tmp24;
                    auto tmp26 = tmp20 + tmp25;
                    tmp13.store(out_ptr1 + static_cast<int64_t>(x0));
                    tmp26.store(out_ptr2 + static_cast<int64_t>(x0));
                }
                if(C10_UNLIKELY(x0 >= static_cast<int64_t>(4LL*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))) && x0 < static_cast<int64_t>(ks0)))
                {
                    auto tmp0 = out_ptr0[static_cast<int64_t>(0LL)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + (-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + (-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + (-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))));
                    auto tmp16 = in_ptr3[static_cast<int64_t>(0LL)];
                    auto tmp3 = tmp2.abs();
                    auto tmp4 = static_cast<float>(1.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp2 / tmp6;
                    auto tmp8 = tmp1 * tmp7;
                    auto tmp9 = static_cast<float>(-1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = at::vec::Vectorized<float>(tmp0);
                    auto tmp13 = tmp12 + tmp11;
                    auto tmp15 = tmp14 * tmp10;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 + tmp17;
                    auto tmp19 = tmp1 * tmp18;
                    auto tmp20 = tmp19 / tmp6;
                    auto tmp21 = tmp19.neg();
                    auto tmp22 = tmp7 / tmp6;
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 =
                    [&]()
                    {
                        auto left = decltype(tmp2)::blendv(decltype(tmp2)(0), decltype(tmp2)(1), decltype(tmp2)(0) < tmp2);
                        auto right = decltype(tmp2)::blendv(decltype(tmp2)(0), decltype(tmp2)(1), tmp2 < decltype(tmp2)(0));
                        return left - right;
                    }
                    ()
                    ;
                    auto tmp25 = tmp23 * tmp24;
                    auto tmp26 = tmp20 + tmp25;
                    tmp13.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + (-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))));
                    tmp26.store(out_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + (-4LL)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4LL)))));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, sum_1, tangents_1 = args
        args.clear()
        s52 = primals_1
        assert_size_stride(primals_2, (s52, ), (1, ))
        assert_size_stride(primals_3, (s52, ), (1, ))
        assert_size_stride(sum_1, (), ())
        assert_size_stride(tangents_1, (s52, ), (1, ))
        buf0 = empty_strided_cpu((), (), torch.float32)
        buf1 = empty_strided_cpu((s52, ), (1, ), torch.float32)
        buf2 = empty_strided_cpu((s52, ), (1, ), torch.float32)
        cpp_fused_abs_add_div_expand_mul_neg_sgn_sum_0(tangents_1, primals_2, primals_3, sum_1, buf0, buf1, buf2, s52)
        del buf0
        del primals_2
        del primals_3
        del sum_1
        del tangents_1
        return (None, buf2, buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 8
    primals_2 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    sum_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, sum_1, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
