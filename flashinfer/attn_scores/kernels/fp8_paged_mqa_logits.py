# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from TensorRT-LLM (Apache 2.0):
# tensorrt_llm/_torch/cute_dsl_kernels/blackwell/paged_mqa_logits/fp8_paged_mqa_logits.py
# Original copyright: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
CuTe DSL FP8 paged MQA logits kernel for Blackwell (SM100).

Architecture:
  - 384 threads: 256 math (2 WGs) + 128 specialized (2 TMA + 2 UMMA)
  - 1 TMA per KV block [128, 128], UMMA iterates 4x K=32
  - 2 warp groups process 2 KV blocks per iteration (kNumMathWarpGroups=2)
  - Q reloaded via TMA pipeline when q_idx (batch) changes
  - Persistent kernel: CTAs iterate through assigned (q_idx, kv_idx) pairs
  - Weights cached in registers: preloaded once per q_idx change (not per KV block)
  - KV Scales loaded via TMA to SMEM (separate pipeline per group, Math consumes)

Merged KV+Scale pipeline:
  - KV data and scales share a single TMA barrier per group
  - TMA loads both KV and Scale under one barrier (combined tx_count)
  - UMMA waits on merged barrier (for KV GEMM), does NOT release
  - Math waits on merged barrier (for scale read), Math releases

Fused KV layout:
  - KV data and scales stored contiguously per physical block:
    [num_phys_blocks, block_kv * (head_dim + 4)] bytes
  - Per block: [KV_all_tokens (block_kv * head_dim bytes)] [Scales (block_kv * 4 bytes)]
  - KV and Scale views are derived inside __call__ using CuTE pointer arithmetic

Scheduler:
  - schedule_meta[sm_idx] = (start_q_idx, start_kv_idx / kNumMathWarpGroups)
  - schedule_meta[sm_idx+1] = end boundary for this CTA
  - fetch_next_task pattern: each warp role independently advances (q_idx, kv_idx)
  - kv_idx in units of KV blocks, advances by kNumMathWarpGroups=2 per step

Dynamic shape support:
  - Model-constant dims (block_kv, head_dim, N, per_token) remain static for codegen
  - Runtime-varying dims (batch_size, num_phys_blocks, max_ctx, max_blocks_per_seq,
    num_ctas) are marked dynamic via mark_compact_shape_dynamic
  - Allows JIT cache reuse across different batch sizes / sequence lengths

Epilogue dtype flows (--acc_dtype / --epi_dtype):

  Flow 1: --acc_dtype fp32 --epi_dtype fp16
    Q(FP8) x K(FP8) -> MMA acc(FP32) -> TMEM(FP32)
      -> LDTM -> Reg(FP32) -> cvt FP16 -> ReLU(FP16)
      -> FMA(fma.rn.f16x2) with weights(FP16 from SMEM) -> partial sum(FP16)
      -> x scale(FP32->FP16) -> cvt output_dtype -> store logits(output_dtype)

  Flow 2: --acc_dtype fp16 --epi_dtype fp16
    Q(FP8) x K(FP8) -> MMA acc(FP16) -> TMEM(FP16, pack_16b)
      -> LDTM -> Reg(FP16) -> ReLU(FP16)
      -> FMA(fma.rn.f16x2) with weights(FP16 from SMEM) -> partial sum(FP16)
      -> x scale(FP32->FP16) -> cvt output_dtype -> store logits(output_dtype)

  --output_dtype: fp32 (default), fp16, bf16. Controls logits tensor dtype and final store conversion.
  Default: --acc_dtype fp32 --epi_dtype fp32 --output_dtype fp32
"""

from dataclasses import dataclass
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float16, Int32
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, math, vector
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ..selective_logits import _SelectiveLogitsMode

# CuTe DSL CUDA 13 validates rounding modes as string literals. The string
# form is also accepted by older wrappers, so keep it version-independent.
_RND_RN = "rn"

# Epilogue FMA unroll. NUM_W_IN_REG is the split point between the register
# and SMEM weight paths, both unrolled this wide, so it must stay a multiple
# of it -- otherwise the two paths stop tiling the subtile and the epilogue
# reads past it. Mirrors _EPI_SUBTILE_UNROLL in attn_scores.py, which applies
# the same granularity to the full logical N tile divided by num_epi_subtiles.
_EPI_UNROLL = 4
_MAX_STRIPED_SPLIT_KV = 8
_SPLIT_KV_CHOICES = (1, 2, 4, 8, 16, 32, 64)
# Max registers per thread for the epilogue weight cache. 160 is the largest
# footprint the per-slot policy below already requests at the shape it was
# tuned for (num_heads=64, next_n=4), so bounding here leaves every
# num_heads=64 configuration unchanged.
_MAX_W_CACHE_REGS = 160


@dsl_user_op
def store_global_u64_if(predicate, address, value, *, loc=None, ip=None) -> None:
    """Issue one predicated 64-bit store through a raw byte address."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(predicate).ir_value(loc=loc, ip=ip),
            cutlass.Int64(address).ir_value(loc=loc, ip=ip),
            cutlass.Uint64(value).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.b32 p, $0, 0; @p st.global.u64 [$1], $2; }",
        "r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_v4_f32(tensor, row, kv_base, stage, *, loc=None, ip=None):
    """Load four contiguous FP32 scores with one 16-byte shared-memory load."""
    elem_offset = cute.crd2idx((row, kv_base, stage), tensor.layout)
    smem_addr = (tensor.iterator + elem_offset).toint(loc=loc, ip=ip)
    result = llvm.inline_asm(
        llvm.StructType.get_literal([Int32.mlir_type] * 4),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        i32_bits_to_fp32(
            Int32(llvm.extractvalue(Int32.mlir_type, result, [idx], loc=loc, ip=ip))
        )
        for idx in range(4)
    )


@dataclass(frozen=True)
class FP8MQALogitsTiling:
    """Trace-time tiling and pipeline policy for one compiled specialization."""

    block_kv: int
    next_n: int
    num_q_stages: int
    num_kv_stages: int
    num_umma_stages: int
    num_epi_subtiles: int = 1
    remove_kv_wait_in_epilogue: bool = False
    early_tmem_copy: bool = False
    smem_subpartition_opt: bool = False


SELECTIVE_LOGITS_TILING = FP8MQALogitsTiling(
    block_kv=128,
    next_n=16,
    num_q_stages=3,
    num_kv_stages=3,
    num_umma_stages=2,
    num_epi_subtiles=8,
    remove_kv_wait_in_epilogue=True,
)


@dsl_user_op
def pack_f16x2(
    a: Float16,
    b: Float16,
    *,
    loc=None,
    ip=None,
) -> Int32:
    f16_ty = Float16.mlir_type
    i32_ty = Int32.mlir_type
    vec2_f16 = ir.VectorType.get([2], f16_ty, loc=loc)
    v = vector.from_elements(
        vec2_f16,
        (Float16(a).ir_value(loc=loc, ip=ip), Float16(b).ir_value(loc=loc, ip=ip)),
        loc=loc,
        ip=ip,
    )
    return Int32(llvm.bitcast(i32_ty, v, loc=loc, ip=ip))


@dsl_user_op
def unpack_f16x2(
    packed: Int32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float16, Float16]:
    f16_ty = Float16.mlir_type
    vec2_f16 = ir.VectorType.get([2], f16_ty, loc=loc)
    v = llvm.bitcast(vec2_f16, Int32(packed).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    r0 = Float16(
        vector.extract(v, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )
    r1 = Float16(
        vector.extract(v, dynamic_position=[], static_position=[1], loc=loc, ip=ip)
    )
    return r0, r1


@dsl_user_op
def fma_f16x2(
    a: Int32,
    b: Int32,
    c: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [
                Int32(a).ir_value(loc=loc, ip=ip),
                Int32(b).ir_value(loc=loc, ip=ip),
                Int32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def max_f16x2(
    a: Int32,
    b: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_f16x2(
    a: Int32,
    b: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    i32_ty = Int32.mlir_type
    return Int32(
        llvm.inline_asm(
            i32_ty,
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "add.f16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def select_i32(predicate, true_value, false_value, *, loc=None, ip=None):
    """Select one Int32 value without introducing divergent control flow."""
    return cutlass.Int32(
        arith.select(
            cutlass.Boolean(predicate).ir_value(loc=loc, ip=ip),
            cutlass.Int32(true_value).ir_value(loc=loc, ip=ip),
            cutlass.Int32(false_value).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fp32_to_i32_bits(value, *, loc=None, ip=None) -> Int32:
    """Return the IEEE-754 payload without changing any score bits."""
    return Int32(
        llvm.bitcast(
            Int32.mlir_type,
            cutlass.Float32(value).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def i32_bits_to_fp32(value, *, loc=None, ip=None) -> cutlass.Float32:
    """Interpret an Int32 metadata word as an IEEE-754 float."""
    return cutlass.Float32(
        llvm.bitcast(
            cutlass.Float32.mlir_type,
            Int32(value).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def abs_f32(value, *, loc=None, ip=None) -> cutlass.Float32:
    """Return the IEEE FP32 absolute value used by the score reduction."""
    return cutlass.Float32(
        math.absf(
            cutlass.Float32(value).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def make_warp_uniform_i64(value) -> cutlass.Int64:
    """Materialize one warp-uniform 64-bit value as two uniform halves."""
    bits = cutlass.Uint64(value)
    lo = cute.arch.make_warp_uniform(cutlass.Int32(bits & 0xFFFFFFFF))
    hi = cute.arch.make_warp_uniform(cutlass.Int32(bits >> 32))
    return cutlass.Int64(
        cutlass.Uint64(cutlass.Uint32(lo)) | (cutlass.Uint64(cutlass.Uint32(hi)) << 32)
    )


class FP8MQALogitsKernel:
    """FP8 paged MQA logits kernel for Blackwell (SM100).

    Each CTA processes a range of (q_idx, kv_split) pairs.
    A split = 2 consecutive KV blocks within a sequence (one per warp group).
    Q is shared between warp groups and reloaded when q_idx changes.
    """

    is_selective_logits_kernel = False
    has_store_warp = False
    uses_single_umma_producer = False
    uses_request_tiled_q = False
    uses_contiguous_kv = False
    split_kv = 1
    num_count_segments: int
    store_wg_registers: int

    def __init__(
        self,
        block_kv: int = 128,
        phys_block_kv: int = 128,
        num_heads: int = 64,
        head_dim: int = 128,
        next_n: int = 1,
        num_sms: int = 148,
        remove_kv_wait_in_epilogue: bool = False,
        early_tmem_copy: bool = False,
        smem_subpartition_opt: bool = False,
        max_kv_pipeline: bool = False,
        max_umma_pipeline: bool = False,
        num_epi_subtiles: int = 1,
        epi_dtype=cutlass.Float32,
        acc_dtype=cutlass.Float32,
        output_dtype=cutlass.Float32,
        tiling: FP8MQALogitsTiling | None = None,
    ):
        if tiling is not None:
            block_kv = tiling.block_kv
            next_n = tiling.next_n
            num_epi_subtiles = tiling.num_epi_subtiles
        self.block_kv = block_kv
        self.phys_block_kv = phys_block_kv
        self.scale_tma_tile = phys_block_kv
        self.num_blocks_per_mma = block_kv // phys_block_kv
        assert block_kv % phys_block_kv == 0, (
            f"block_kv={block_kv} must be divisible by phys_block_kv={phys_block_kv}"
        )
        assert self.num_blocks_per_mma <= 4, (
            f"num_blocks_per_mma={self.num_blocks_per_mma} exceeds max 4"
        )
        self.remove_kv_wait_in_epilogue = remove_kv_wait_in_epilogue
        self.early_tmem_copy = early_tmem_copy
        self.smem_subpartition_opt = smem_subpartition_opt
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.next_n = next_n
        self.N = next_n * num_heads
        self.num_sms = num_sms
        self.num_epi_subtiles = num_epi_subtiles
        self.epi_dtype = epi_dtype
        self.epi_bytes = 2 if epi_dtype == cutlass.Float16 else 4
        # sW stage stride padded to 128-byte SMEM alignment for TMA bulk copy.
        # Without padding, e.g. fp16 + N=32 gives 64B per stage, so stage 1
        # at +64 would be misaligned (TMA requires 128-byte aligned SMEM dest).
        w_stage_bytes = self.N * self.epi_bytes
        self.w_stage_stride = ((w_stage_bytes + 127) // 128 * 128) // self.epi_bytes
        self.output_dtype = output_dtype
        if self.N % num_epi_subtiles != 0:
            raise ValueError("N must be divisible by num_epi_subtiles")
        if num_epi_subtiles % next_n != 0 and next_n % num_epi_subtiles != 0:
            raise ValueError(
                "num_epi_subtiles must either partition each logical row "
                "or group a whole number of logical rows"
            )
        if (self.N // num_epi_subtiles) % 4 != 0:
            raise ValueError(
                "N // num_epi_subtiles must be divisible by 4 (FMA unroll granularity)"
            )
        self.num_groups = 2

        self.num_math_threads = 256
        self.num_specialized_threads = 128
        self.threads_per_cta = 384
        self.num_math_warps = 8
        self.tma_warp_base = 8
        self.umma_warp_base = 10
        self.math_wg_registers = 240
        self.specialized_wg_registers = 24
        # 3 stages for Q pipelining across batch sequences
        self.num_q_stages = tiling.num_q_stages if tiling is not None else 3

        # TMEM: 512 columns total, each group needs N columns per UMMA stage
        # max_umma_stages = 512 // (2 * N)
        TMEM_COLS = 512
        if tiling is not None:
            self.num_umma_stages = tiling.num_umma_stages
        elif max_umma_pipeline:
            self.num_umma_stages = min(2, TMEM_COLS // (2 * self.N))
        else:
            self.num_umma_stages = 1

        if tiling is not None:
            self.num_kv_stages = tiling.num_kv_stages
        elif max_kv_pipeline:
            smem_capacity = utils.get_smem_capacity_in_bytes()
            # Reserve ~1 KB for barriers and misc
            SMEM_BUDGET = smem_capacity - 1024
            # KV+Scale per stage (*2 groups):
            #   2 * (block_kv * head_dim * 1B + block_kv * 4B)
            kv_scale_per_stage = 2 * (block_kv * head_dim + block_kv * 4)
            # Q+W per stage: Q is N * head_dim * 1B, W uses padded stride
            qw_per_stage = self.N * head_dim + self.w_stage_stride * self.epi_bytes
            qw_total = qw_per_stage * self.num_q_stages
            self.num_kv_stages = (SMEM_BUDGET - qw_total) // kv_scale_per_stage
        else:
            self.num_kv_stages = 3

        # Pad SMEM to push sW/sScales into sub-partition 1 (>= 128KB),
        # avoiding sub-bank conflicts with UMMA reading sKV.
        # Layout: barriers(~256B) | sKV_0 | sKV_1 | sQ | [pad] | sW | sScales
        if self.smem_subpartition_opt:
            BOUNDARY = 128 * 1024
            used = (
                256
                + 2 * (block_kv * head_dim * self.num_kv_stages)
                + self.N * head_dim * self.num_q_stages
            )
            used = ((used + 127) // 128) * 128
            if used < BOUNDARY:
                self.smem_pad_bytes = ((BOUNDARY - used + 1023) // 1024) * 1024
            else:
                self.smem_pad_bytes = 0
        else:
            self.smem_pad_bytes = 0

        self.acc_dtype = acc_dtype
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (block_kv, self.N)

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.a_major_mode = a_major
        self.b_major_mode = b_major

        self.mma_tiler = (*self.mma_tiler_mn, 1)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype,
            b_dtype,
            a_major,
            b_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])  # 32

        # Full-K: tile K = head_dim (128), 1 TMA per block
        mma_inst_tile_k = self.head_dim // mma_inst_shape_k  # 4
        full_k = mma_inst_shape_k * mma_inst_tile_k  # 128
        self.mma_tiler = (
            self.mma_tiler_mn[0],
            self.mma_tiler_mn[1],
            full_k,
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]

        # KV SMEM: 3 stages per group, each stage holds full [128, 128]
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            a_dtype,
            self.num_kv_stages,
        )
        # Q SMEM: 1 stage, holds full [N, 128]
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            b_dtype,
            self.num_q_stages,
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.num_tmem_alloc_cols_total = (
            self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
        )
        return tiled_mma

    @cute.jit
    def _make_math_tmem_copy_and_partition(
        self,
        tCtAcc_base,
        epi_sub_mn,
        copy_atom_t2r,
        epi_tidx,
        cC,
    ):
        """Build the group-local TMEM copy view shared by both math WGs."""
        # (MMA_M, MMA_N, STAGE) -> (EPI_M, EPI_N, M, N, STAGE)
        tAcc = tCtAcc_base[((None, None), 0, 0, None)]
        tAcc_epi = cute.flat_divide(tAcc, epi_sub_mn)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
        tTR_cC = thr_copy_t2r.partition_D(cC)
        tTR_rAcc = cute.make_rmem_tensor_like(tTR_cC, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc_base, tTR_cC, tTR_rAcc

    @cute.jit
    def _store_result(
        self,
        mLogits,
        mTileMeta,
        q_idx,
        row,
        kv_pos,
        result,
        scale_val,
        next_n: cutlass.Constexpr,
    ):
        """Store one unmasked paged-decode score."""
        out_row = q_idx * next_n + row
        # Form the complete row-major offset explicitly in 64 bits. CuTe's
        # coordinate linearization may use the layout's narrower index type,
        # while large dense outputs can exceed the signed-Int32 element range.
        score_offset = cutlass.Int64(
            cutlass.Uint64(cutlass.Uint32(out_row))
            * cutlass.Uint64(mLogits.layout.stride[0])
        ) + cutlass.Int64(kv_pos)
        score_ptr = mLogits.iterator + score_offset
        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
            score_ptr[0] = self.output_dtype(result * Float16(scale_val))
        else:
            score_ptr[0] = self.output_dtype(result * scale_val)

    def _store_result_packed_causal_meta(
        self,
        mLogits,
        mIndexCounts,
        mTileMeta,
        mPolicyValues,
        q_idx,
        out_row_base,
        q_valid,
        causal_base,
        row_start,
        row_end,
        row,
        store_kv_pos,
        store_kv_valid,
        result,
        scale_val,
        next_n: cutlass.Constexpr,
    ):
        """Declare the request-tiled metadata store hook."""
        raise NotImplementedError(
            "causal tile metadata is available only on causal subclasses"
        )

    @cute.jit
    def __call__(
        self,
        kv_fused: cute.Tensor,  # Fused KV: [num_phys_blocks, block_bytes] FP8
        b: cute.Tensor,  # Q: [N, head_dim, batch_size]
        weights: cute.Tensor,  # [N, batch_size] (transposed for TMA)
        logits: cute.Tensor,  # [batch_size * next_n, max_context_len]
        block_table: cute.Tensor,  # [batch_size, max_blocks_per_seq]
        context_lens: cute.Tensor,  # [batch_size]
        schedule_meta: cute.Tensor,  # [num_sms+1, 2] int32
        num_phys_blocks: cutlass.Int32,
        batch_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._launch(
            kv_fused,
            b,
            weights,
            logits,
            block_table,
            block_table,
            context_lens,
            schedule_meta,
            context_lens,
            schedule_meta,
            schedule_meta,
            num_phys_blocks,
            batch_size,
            stream,
        )

    @cute.jit
    def _launch(
        self,
        kv_fused: cute.Tensor,
        b: cute.Tensor,
        weights: cute.Tensor,
        logits: cute.Tensor,
        block_table: cute.Tensor,
        index_counts: cute.Tensor,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        kv_scale: cute.Tensor,
        tile_meta: cute.Tensor,
        policy_values: cute.Tensor,
        num_phys_blocks: cutlass.Int32,
        batch_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # The common engine carries both the physical page table and selective
        # output counters. The public dense-logits path supplies its block table
        # in both slots; compile-time selective hooks are the only users of the
        # separate counter tensor.
        # Derive KV and Scale views from fused buffer using CuTE ops.
        # Fused layout per physical block: [KV data (phys_block_kv*head_dim)] [Scales (phys_block_kv*4)]
        # Recast fused buffer to FP8 (same 1-byte elements, needed for MMA type inference)
        kv_fp8 = cute.recast_tensor(kv_fused, cutlass.Float8E4M3FN)

        # Q (b) was passed as uint8 to work around DLPack's lack of float8 support;
        # recast back to FP8 so MMA type inference and TMA descriptors are correct.
        b = cute.recast_tensor(b, cutlass.Float8E4M3FN)

        # Read the real per-block stride (bytes = FP8 elements) from the input.
        # When KV is the indexer K-cache pool view, the pool is laid out as
        # [num_blocks, num_layers, kvFactor, blockSize], so dim-0 stride =
        # num_layers * kvFactor * phys_block_bytes (not phys_block_bytes).
        # Using the input stride keeps both contiguous test path and the
        # strided prod path correct. FP8 = 1 byte per element, so the byte
        # stride is the same as the element stride after recast.
        kv_block_stride = kv_fused.layout.stride[0]

        if cutlass.const_expr(self.uses_contiguous_kv):
            kv_layout = cute.make_layout(
                (num_phys_blocks, self.head_dim, 1),
                stride=(kv_block_stride, 1, 0),
            )
            a = cute.make_tensor(kv_fp8.iterator, kv_layout)
            scales = kv_scale
        else:
            # KV view: [phys_block_kv, head_dim, num_phys_blocks] FP8
            # Each TMA loads one physical block; multiple TMAs fill a compute tile.
            kv_layout = cute.make_layout(
                (self.phys_block_kv, self.head_dim, num_phys_blocks),
                stride=(self.head_dim, 1, kv_block_stride),
            )
            a = cute.make_tensor(kv_fp8.iterator, kv_layout)
            # Scale view: offset pointer to scale region, recast FP8 -> Float32
            # [phys_block_kv, num_phys_blocks] float32 (after recast)
            scale_fp8_layout = cute.make_layout(
                (self.phys_block_kv * 4, num_phys_blocks),
                stride=(1, kv_block_stride),
            )
            fused_block_layout = cute.make_layout(
                (
                    self.phys_block_kv * (self.head_dim + 4),
                    num_phys_blocks,
                ),
                stride=(1, kv_block_stride),
            )
            fused_blocks = cute.make_tensor(kv_fp8.iterator, fused_block_layout)
            scale_storage = cute.domain_offset(
                (self.phys_block_kv * self.head_dim, 0), fused_blocks
            )
            scale_fp8 = cute.make_tensor(
                scale_storage.iterator,
                scale_fp8_layout,
            )
            scales = cute.recast_tensor(scale_fp8, cutlass.Float32)

        a_dtype = a.element_type
        b_dtype = b.element_type
        a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()

        tiled_mma = self._setup_mma(a_dtype, b_dtype, a_major, b_major)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA for KV (A); fmha_decode_paged pattern.
        # Build a TMA SMEM layout via tiled_divide on the full compute-tile
        # layout, then select to drop trivial K dim. Atom uses mode [0] as
        # single-tile SMEM layout and (phys, head) as cta_tiler.
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        if cutlass.const_expr(self.uses_contiguous_kv):
            a_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id
            )
            tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
                a_op,
                a,
                self.a_smem_layout_staged,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            self.a_tma_view_layout = self.a_smem_layout_staged
        else:
            self.a_tma_view_layout = sm100_utils.make_smem_layout(
                cute.nvgpu.OperandMajorMode.K,
                (self.block_kv, self.head_dim),
                a_dtype,
                self.num_kv_stages,
            )
            # ((tile_M, tile_K), rest_M, rest_K, stages) -> drop trivial rest_K
            self.a_tma_view_layout = cute.tiled_divide(
                self.a_tma_view_layout, (self.phys_block_kv, self.head_dim)
            )
            # ((tile_M, tile_K), rest_M=num_sub_blocks, stages)
            self.a_tma_view_layout = cute.select(self.a_tma_view_layout, mode=[0, 1, 3])
            # atom SMEM = single-tile (mode 0)
            tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
                tma_load_op,
                a,
                self.a_tma_view_layout[0],
                (self.phys_block_kv, self.head_dim),
            )

        # TMA for Q (B); full K=128, L dim = batch_size (unchanged)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA for Weights; [N, batch_size], tile [N], L=batch_size
        # The selective-logits path keeps [query, head] weights and rebases each
        # request-owned query tile directly; paged decode retains [N, batch].
        self.w_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages),
            stride=(1, self.w_stage_stride),
        )
        if cutlass.const_expr(self.uses_request_tiled_q):
            w_smem_per_stage = cute.make_layout(
                (self.next_n, self.num_heads), stride=(self.num_heads, 1)
            )
            w_tma_tiler = (self.next_n, self.num_heads)
        else:
            w_smem_per_stage = cute.select(self.w_smem_layout_staged, mode=[0])
            w_tma_tiler = self.w_smem_layout_staged.shape[:1]
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op,
            weights,
            w_smem_per_stage,
            w_tma_tiler,
        )

        # TMA for Scales; [phys_block_kv, num_phys_blocks], tile [phys_block_kv]
        # SMEM holds compute_block_kv scales per stage; filled by
        # num_blocks_per_mma sub-block TMAs at consecutive offsets.
        self.s_smem_layout_staged = cute.make_layout(
            (self.block_kv, self.num_kv_stages)
        )
        scale_tma_tile = self.scale_tma_tile
        s_smem_per_subblock = cute.make_layout((scale_tma_tile,))
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op,
            scales,
            s_smem_per_subblock,
            (scale_tma_tile,),
        )

        b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
        w_copy_size = self.N * self.epi_bytes
        # Per sub-block: phys_block_kv * head_dim (KV) + phys_block_kv * 4 (scales)
        kv_tma_bytes_per_subblock = self.scale_tma_tile * self.head_dim
        scale_tma_bytes_per_subblock = self.scale_tma_tile * 4
        # Total per compute tile = num_blocks_per_mma sub-blocks
        self.num_kv_scale_tma_bytes = self.num_blocks_per_mma * (
            kv_tma_bytes_per_subblock + scale_tma_bytes_per_subblock
        )
        # Q + Weights share barrier (like DeepGEMM)
        self.num_q_tma_bytes = b_copy_size * atom_thr_size + w_copy_size

        num_ctas = self.num_sms

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 4]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.num_q_stages * 2]
            umma_mbar: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 4]
            store_mbar: cute.struct.MemRange[cutlass.Int64, 8]
            tmem_holding_buf: cutlass.Int32

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_s,
            tma_tensor_s,
            logits,
            block_table,
            index_counts,
            context_lens,
            schedule_meta,
            tile_meta,
            policy_values,
            batch_size,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.w_smem_layout_staged,
            self.s_smem_layout_staged,
            self.a_tma_view_layout,
            self.epi_tile,
            SharedStorage,
        ).launch(
            grid=(1, 1, num_ctas),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.jit
    def _run_tma_warp_0(
        self,
        NUM_BLOCKS_PER_MMA,
        NUM_MATH_WG,
        a_mcast_mask,
        b_mcast_mask,
        batch_size,
        block_kv_val,
        end_kv_idx,
        end_q_idx,
        has_work,
        kv_pipeline_0,
        mA_mkl,
        mB_nkl,
        mBlockTable,
        mContextLens,
        mS_tma,
        mTileMeta,
        mW_tma,
        next_kv_idx,
        next_num_kv,
        next_q_idx,
        q_idx,
        q_pipeline,
        sKV_0,
        sQ,
        sScales_0,
        sW,
        tAgA_0,
        tAsA_0,
        tBgB,
        tBsB,
        tSgS_0,
        tSsS_0,
        tWgW,
        tWsW,
        thr_mma,
        tidx,
        tma_atom_a,
        tma_atom_b,
        tma_atom_s,
        tma_atom_w,
    ):
        """Run the Q and first KV/scale TMA producer role."""
        cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
        q_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_q_stages
        )
        kv_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )

        # TMA warp 0: loads Q (prefetch) + KV for group 0
        lane_idx = tidx % 32

        # Block table prefetch: 32 lanes cache block indices,
        # distributed via shuffle. Each lane holds num_blocks_per_mma
        # physical block indices per compute tile.
        cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
        kv_blk_ptr = cutlass.Int32(32)  # force prefetch on first use

        # Prefetch the first assigned Q only; zero-work persistent CTAs do
        # not own a valid packed query tile.
        if has_work:
            q_pipeline.producer_acquire(q_prod_state)
            q_bar = q_pipeline.producer_get_barrier(q_prod_state)
            if cutlass.const_expr(self.uses_request_tiled_q):
                query_start = mTileMeta[(next_q_idx, 0)]
                shifted_b = cute.domain_offset(
                    (query_start * self.num_heads, 0, 0), mB_nkl
                )
                shifted_gB = cute.local_tile(
                    shifted_b,
                    cute.slice_(self.mma_tiler, (0, None, None)),
                    (None, None, None),
                )
                shifted_tCgB = thr_mma.partition_B(shifted_gB)
                _, shifted_tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sQ, 0, 3),
                    cute.group_modes(shifted_tCgB, 0, 3),
                )
                shifted_w = cute.domain_offset((query_start, 0), mW_tma)
                shifted_gW = cute.local_tile(
                    shifted_w,
                    (self.next_n, self.num_heads),
                    (None, None),
                )
                _, shifted_tWgW = cpasync.tma_partition(
                    tma_atom_w,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sW, 0, 2),
                    cute.group_modes(shifted_gW, 0, 2),
                )
                cute.copy(
                    tma_atom_b,
                    shifted_tBgB[(None, 0, 0, 0)],
                    tBsB[(None, q_prod_state.index)],
                    tma_bar_ptr=q_bar,
                    mcast_mask=b_mcast_mask,
                )
                cute.copy(
                    tma_atom_w,
                    shifted_tWgW[(None, 0, 0)],
                    tWsW[(None, q_prod_state.index)],
                    tma_bar_ptr=q_bar,
                )
            else:
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, next_q_idx)],
                    tBsB[(None, q_prod_state.index)],
                    tma_bar_ptr=q_bar,
                    mcast_mask=b_mcast_mask,
                )
                cute.copy(
                    tma_atom_w,
                    tWgW[(None, next_q_idx)],
                    tWsW[(None, q_prod_state.index)],
                    tma_bar_ptr=q_bar,
                )
            q_prod_state.advance()

        while has_work:
            # fetch_next_task: commit next -> current
            q_idx_old = q_idx
            q_idx = next_q_idx
            kv_idx = next_kv_idx
            num_kv = next_num_kv

            # Q prefetch: when batch changes, load Q for NEXT batch
            if q_idx != q_idx_old:
                kv_blk_ptr = cutlass.Int32(32)  # force re-prefetch
                prefetch_next = q_idx + 1
                # The task-advance blocks below skip zero-length rows, so
                # this lookahead must land on the SAME row the consumers
                # will visit next. Prefetching a raw q_idx+1 that the
                # iterators skip pushes the skipped row's Q/W into the
                # pipeline, and the row after the gap then consumes it --
                # silently wrong logits. Stage counts still match, so it
                # does not hang, which is why only an interspersed-zero
                # case exposes it (a trailing zero is masked by the
                # prefetch_next < end_q_idx guard below).
                peek_ctx = cutlass.Int32(1)  # nonzero => do not skip
                if prefetch_next < end_q_idx:
                    peek_ctx = mContextLens[prefetch_next]
                while (peek_ctx == 0) & (prefetch_next < end_q_idx):
                    prefetch_next = prefetch_next + 1
                    if prefetch_next < end_q_idx:
                        peek_ctx = mContextLens[prefetch_next]
                    else:
                        peek_ctx = cutlass.Int32(1)
                if (prefetch_next < end_q_idx) | (
                    (prefetch_next == end_q_idx) & (end_kv_idx > 0)
                ):
                    q_pipeline.producer_acquire(q_prod_state)
                    q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                    if cutlass.const_expr(self.uses_request_tiled_q):
                        query_start = mTileMeta[(prefetch_next, 0)]
                        shifted_b = cute.domain_offset(
                            (query_start * self.num_heads, 0, 0), mB_nkl
                        )
                        shifted_gB = cute.local_tile(
                            shifted_b,
                            cute.slice_(self.mma_tiler, (0, None, None)),
                            (None, None, None),
                        )
                        shifted_tCgB = thr_mma.partition_B(shifted_gB)
                        _, shifted_tBgB = cpasync.tma_partition(
                            tma_atom_b,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sQ, 0, 3),
                            cute.group_modes(shifted_tCgB, 0, 3),
                        )
                        shifted_w = cute.domain_offset((query_start, 0), mW_tma)
                        shifted_gW = cute.local_tile(
                            shifted_w,
                            (self.next_n, self.num_heads),
                            (None, None),
                        )
                        _, shifted_tWgW = cpasync.tma_partition(
                            tma_atom_w,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sW, 0, 2),
                            cute.group_modes(shifted_gW, 0, 2),
                        )
                        cute.copy(
                            tma_atom_b,
                            shifted_tBgB[(None, 0, 0, 0)],
                            tBsB[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                            mcast_mask=b_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_w,
                            shifted_tWgW[(None, 0, 0)],
                            tWsW[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                        )
                    else:
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, 0, prefetch_next)],
                            tBsB[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                            mcast_mask=b_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_w,
                            tWgW[(None, prefetch_next)],
                            tWsW[(None, q_prod_state.index)],
                            tma_bar_ptr=q_bar,
                        )
                    q_prod_state.advance()

            if cutlass.const_expr(not self.uses_contiguous_kv):
                # Block table prefetch for group 0.
                # Each lane loads num_blocks_per_mma physical block indices
                # for one compute tile (kv_idx counts compute tiles).
                # Refill the per-lane block cache every 32 compute tiles.
                if kv_blk_ptr == 32:
                    kv_blk_ptr = cutlass.Int32(0)
                    prefetch_kv = kv_idx + lane_idx * NUM_MATH_WG
                    # Load physical block indices owned by this lane.
                    if prefetch_kv < num_kv:
                        base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                        page_table_row = q_idx
                        if cutlass.const_expr(self.uses_request_tiled_q):
                            page_table_row = mTileMeta[(q_idx, 1)]
                        # Cache every physical block used by this MMA tile.
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = mBlockTable[
                                (page_table_row, base_phys + i)
                            ]
                    else:
                        # Fill inactive lanes with an address-safe block.
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = cutlass.Int32(0)
                phys_blks = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                # Broadcast the cached indices for the current tile.
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    phys_blks[i] = cute.arch.shuffle_sync(cached_blks[i], kv_blk_ptr)
                kv_blk_ptr = kv_blk_ptr + 1
                kv_pipeline_0.producer_acquire(kv_prod_state_0)
                bar = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                stage = kv_prod_state_0.index
                # Load KV + Scale for group 0: num_blocks_per_mma TMAs per tile.
                # Issue the original paged KV and scale transactions inline.
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    cute.copy(
                        tma_atom_a,
                        tAgA_0[(None, 0, 0, phys_blks[i])],
                        tAsA_0[(None, i, stage)],
                        tma_bar_ptr=bar,
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_s,
                        tSgS_0[(None, 0, phys_blks[i])],
                        tSsS_0[(None, i, stage)],
                        tma_bar_ptr=bar,
                    )
            else:
                kv_pipeline_0.producer_acquire(kv_prod_state_0)
                bar = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                stage = kv_prod_state_0.index
                safe_kv_idx = cutlass.Int32(0)
                kv_start = cutlass.Int32(0)
                if kv_idx < num_kv:
                    safe_kv_idx = kv_idx
                    request_kv_start = mTileMeta[(q_idx, 1)]
                    kv_start = request_kv_start // 4 * 4
                shifted_a = cute.domain_offset((kv_start, 0, 0), mA_mkl)
                shifted_gA = cute.local_tile(
                    shifted_a,
                    cute.slice_(self.mma_tiler, (None, 0, None)),
                    (None, None, None),
                )
                shifted_tCgA = thr_mma.partition_A(shifted_gA)
                _, shifted_tAgA = cpasync.tma_partition(
                    tma_atom_a,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sKV_0, 0, 3),
                    cute.group_modes(shifted_tCgA, 0, 3),
                )
                shifted_s = cute.domain_offset((kv_start,), mS_tma)
                shifted_gS = cute.local_tile(shifted_s, (self.block_kv,), (None,))
                _, shifted_tSgS = cpasync.tma_partition(
                    tma_atom_s,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sScales_0, 0, 1),
                    shifted_gS,
                )
                cute.copy(
                    tma_atom_a,
                    shifted_tAgA[(None, safe_kv_idx, 0, 0)],
                    tAsA_0[(None, stage)],
                    tma_bar_ptr=bar,
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_s,
                    shifted_tSgS[(None, safe_kv_idx)],
                    tSsS_0[(None, stage)],
                    tma_bar_ptr=bar,
                )
            kv_prod_state_0.advance()

            # Advance: inline fetch_next_task
            next_kv_idx = kv_idx + NUM_MATH_WG
            if next_kv_idx >= num_kv:
                next_q_idx = q_idx + 1
                next_kv_idx = 0
                if next_q_idx < batch_size:
                    next_num_kv = (
                        mContextLens[next_q_idx] + block_kv_val - 1
                    ) // block_kv_val
                # Zero-length rows get no scheduled work, but the exact-
                # coordinate termination below would still step onto them and
                # run a full pipeline turn (TMA + UMMA + epilogue). Skip them,
                # stopping at this CTA's end boundary so that boundary stays
                # reachable -- overshooting it would make the loop unable to
                # terminate. Every warp role runs this identical traversal.
                while (
                    (next_num_kv == 0)
                    & (next_q_idx < batch_size)
                    & ((next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx))
                ):
                    next_q_idx = next_q_idx + 1
                    if next_q_idx < batch_size:
                        next_num_kv = (
                            mContextLens[next_q_idx] + block_kv_val - 1
                        ) // block_kv_val
            # Update while-loop condition
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

    @cute.jit
    def _run_tma_warp_1(
        self,
        NUM_BLOCKS_PER_MMA,
        NUM_MATH_WG,
        a_mcast_mask,
        batch_size,
        block_kv_val,
        end_kv_idx,
        end_q_idx,
        has_work,
        kv_pipeline_1,
        mA_mkl,
        mBlockTable,
        mContextLens,
        mS_tma,
        mTileMeta,
        next_kv_idx,
        next_num_kv,
        next_q_idx,
        q_idx,
        sKV_1,
        sScales_1,
        tAgA_1,
        tAsA_1,
        tSgS_1,
        tSsS_1,
        thr_mma,
        tidx,
        tma_atom_a,
        tma_atom_s,
    ):
        """Run the second KV/scale TMA producer role."""
        cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
        kv_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )

        # TMA warp 1: loads KV + Scale for group 1 only
        lane_idx = tidx % 32

        # Block table prefetch for group 1
        cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
        kv_blk_ptr = cutlass.Int32(32)  # force prefetch on first use

        while has_work:
            # fetch_next_task: commit next -> current
            q_idx_old = q_idx
            q_idx = next_q_idx
            kv_idx = next_kv_idx
            num_kv = next_num_kv

            # New q_idx: force block table re-prefetch.
            if q_idx != q_idx_old:
                kv_blk_ptr = cutlass.Int32(32)

            if cutlass.const_expr(not self.uses_contiguous_kv):
                # Block table prefetch for group 1.
                # Refill the group-1 cache every 32 compute tiles.
                if kv_blk_ptr == 32:
                    kv_blk_ptr = cutlass.Int32(0)
                    prefetch_kv = kv_idx + 1 + lane_idx * NUM_MATH_WG
                    # Load physical block indices owned by this lane.
                    if prefetch_kv < num_kv:
                        base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                        page_table_row = q_idx
                        if cutlass.const_expr(self.uses_request_tiled_q):
                            page_table_row = mTileMeta[(q_idx, 1)]
                        # Cache every physical block used by this MMA tile.
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = mBlockTable[
                                (page_table_row, base_phys + i)
                            ]
                    else:
                        # Fill inactive lanes with an address-safe block.
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = cutlass.Int32(0)
                phys_blks = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                # Broadcast the cached indices for the current tile.
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    phys_blks[i] = cute.arch.shuffle_sync(cached_blks[i], kv_blk_ptr)
                kv_blk_ptr = kv_blk_ptr + 1
                kv_pipeline_1.producer_acquire(kv_prod_state_1)
                bar = kv_pipeline_1.producer_get_barrier(kv_prod_state_1)
                stage = kv_prod_state_1.index
                # Load KV + Scale for group 1: num_blocks_per_mma TMAs per tile.
                # Issue the original paged KV and scale transactions inline.
                for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                    cute.copy(
                        tma_atom_a,
                        tAgA_1[(None, 0, 0, phys_blks[i])],
                        tAsA_1[(None, i, stage)],
                        tma_bar_ptr=bar,
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_s,
                        tSgS_1[(None, 0, phys_blks[i])],
                        tSsS_1[(None, i, stage)],
                        tma_bar_ptr=bar,
                    )
            else:
                kv_pipeline_1.producer_acquire(kv_prod_state_1)
                bar = kv_pipeline_1.producer_get_barrier(kv_prod_state_1)
                stage = kv_prod_state_1.index
                group_kv_idx = kv_idx + 1
                safe_kv_idx = cutlass.Int32(0)
                kv_start = cutlass.Int32(0)
                if group_kv_idx < num_kv:
                    safe_kv_idx = group_kv_idx
                    request_kv_start = mTileMeta[(q_idx, 1)]
                    kv_start = request_kv_start // 4 * 4
                shifted_a = cute.domain_offset((kv_start, 0, 0), mA_mkl)
                shifted_gA = cute.local_tile(
                    shifted_a,
                    cute.slice_(self.mma_tiler, (None, 0, None)),
                    (None, None, None),
                )
                shifted_tCgA = thr_mma.partition_A(shifted_gA)
                _, shifted_tAgA = cpasync.tma_partition(
                    tma_atom_a,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sKV_1, 0, 3),
                    cute.group_modes(shifted_tCgA, 0, 3),
                )
                shifted_s = cute.domain_offset((kv_start,), mS_tma)
                shifted_gS = cute.local_tile(shifted_s, (self.block_kv,), (None,))
                _, shifted_tSgS = cpasync.tma_partition(
                    tma_atom_s,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sScales_1, 0, 1),
                    shifted_gS,
                )
                cute.copy(
                    tma_atom_a,
                    shifted_tAgA[(None, safe_kv_idx, 0, 0)],
                    tAsA_1[(None, stage)],
                    tma_bar_ptr=bar,
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_s,
                    shifted_tSgS[(None, safe_kv_idx)],
                    tSsS_1[(None, stage)],
                    tma_bar_ptr=bar,
                )
            kv_prod_state_1.advance()

            # Advance: inline fetch_next_task
            next_kv_idx = kv_idx + NUM_MATH_WG
            if next_kv_idx >= num_kv:
                next_q_idx = q_idx + 1
                next_kv_idx = 0
                if next_q_idx < batch_size:
                    next_num_kv = (
                        mContextLens[next_q_idx] + block_kv_val - 1
                    ) // block_kv_val
                # Zero-length rows get no scheduled work, but the exact-
                # coordinate termination below would still step onto them and
                # run a full pipeline turn (TMA + UMMA + epilogue). Skip them,
                # stopping at this CTA's end boundary so that boundary stays
                # reachable -- overshooting it would make the loop unable to
                # terminate. Every warp role runs this identical traversal.
                while (
                    (next_num_kv == 0)
                    & (next_q_idx < batch_size)
                    & ((next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx))
                ):
                    next_q_idx = next_q_idx + 1
                    if next_q_idx < batch_size:
                        next_num_kv = (
                            mContextLens[next_q_idx] + block_kv_val - 1
                        ) // block_kv_val
            # Update while-loop condition
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

    @cute.jit
    def _run_umma_warp_0(
        self,
        NUM_MATH_WG,
        batch_size,
        block_kv_val,
        end_kv_idx,
        end_q_idx,
        has_work,
        is_leader_cta,
        kv_pipeline_0,
        kv_pipeline_1,
        mContextLens,
        next_kv_idx,
        next_num_kv,
        next_q_idx,
        q_idx,
        q_pipeline,
        tCrA_0,
        tCrA_1,
        tCrB,
        tCtAcc_fake_staged,
        tiled_mma,
        tmem,
        tmem_group_1_offset,
        umma_pipeline_0,
        umma_pipeline_1,
    ):
        """Run the first UMMA producer role and optional second ring."""
        cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
        q_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        kv_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        umma_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        if cutlass.const_expr(self.uses_single_umma_producer):
            kv_cons_state_umma_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_kv_stages
            )
            umma_prod_state_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_umma_stages
            )

        # UMMA warp for group 0
        # Must wait on Q pipeline: TMA operations with different
        # barriers are NOT visibility-ordered even within the same
        # warp. KV0 barrier arriving does not guarantee Q SMEM
        # writes are visible.
        # TMEM: wait for math warp 0's allocation, retrieve pointer
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
        tCtAcc_base_1 = cute.make_tensor(
            tmem_ptr + tmem_group_1_offset, tCtAcc_fake_staged.layout
        )

        if is_leader_cta:
            num_k_blocks = cute.size(tCrA_0.shape[2])
            q_stage_0 = cutlass.Int32(0)

            while has_work:
                # fetch_next_task: commit next to current.
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                # Wait for Q pipeline when batch changes
                if q_idx != q_idx_old:
                    if q_idx_old < batch_size:
                        q_cons_state_umma_0.advance()
                    q_pipeline.consumer_wait(q_cons_state_umma_0)
                    q_stage_0 = q_cons_state_umma_0.index

                # Process KV block for group 0 (kv_idx + 0)
                # Unconditional UMMA: OOB iterations
                # compute on garbage data; results written to aligned
                # padding region in logits buffer.
                # Wait for KV before acquiring an empty TMEM stage.
                kv_ready_0 = kv_pipeline_0.consumer_try_wait(kv_cons_state_umma_0)
                umma_empty_0 = umma_pipeline_0.producer_try_acquire(umma_prod_state_0)
                kv_pipeline_0.consumer_wait(kv_cons_state_umma_0, kv_ready_0)
                umma_pipeline_0.producer_acquire(umma_prod_state_0, umma_empty_0)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                kv_stage = kv_cons_state_umma_0.index
                tCtAcc_0 = tCtAcc_base_0[(None, None, None, umma_prod_state_0.index)]
                for k_block in cutlass.range_constexpr(num_k_blocks):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc_0,
                        tCrA_0[None, None, k_block, kv_stage],
                        tCrB[None, None, k_block, q_stage_0],
                        tCtAcc_0,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                # No consumer_release here; Math WG 0 releases.
                kv_cons_state_umma_0.advance()

                umma_pipeline_0.producer_commit(umma_prod_state_0)
                umma_prod_state_0.advance()

                if cutlass.const_expr(self.uses_single_umma_producer):
                    # Issue group 1 from the same warp after group 0 has
                    # committed.  Both UMMA pipelines remain independent,
                    # so their two-stage TMEM rings and math-consumer
                    # lifetimes are unchanged; only duplicated traversal
                    # and producer-warp control are removed.
                    kv_ready_1 = kv_pipeline_1.consumer_try_wait(kv_cons_state_umma_1)
                    umma_empty_1 = umma_pipeline_1.producer_try_acquire(
                        umma_prod_state_1
                    )
                    kv_pipeline_1.consumer_wait(kv_cons_state_umma_1, kv_ready_1)
                    umma_pipeline_1.producer_acquire(umma_prod_state_1, umma_empty_1)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kv_stage_1 = kv_cons_state_umma_1.index
                    tCtAcc_1 = tCtAcc_base_1[
                        (None, None, None, umma_prod_state_1.index)
                    ]
                    for k_block in cutlass.range_constexpr(num_k_blocks):
                        cute.gemm(
                            tiled_mma,
                            tCtAcc_1,
                            tCrA_1[None, None, k_block, kv_stage_1],
                            tCrB[None, None, k_block, q_stage_0],
                            tCtAcc_1,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    kv_cons_state_umma_1.advance()
                    umma_pipeline_1.producer_commit(umma_prod_state_1)
                    umma_prod_state_1.advance()

                # Advance: inline fetch_next_task
                next_kv_idx = kv_idx + NUM_MATH_WG
                if next_kv_idx >= num_kv:
                    next_q_idx = q_idx + 1
                    next_kv_idx = 0
                    if next_q_idx < batch_size:
                        next_num_kv = (
                            mContextLens[next_q_idx] + block_kv_val - 1
                        ) // block_kv_val
                    # Zero-length rows get no scheduled work, but the exact-
                    # coordinate termination below would still step onto them and
                    # run a full pipeline turn (TMA + UMMA + epilogue). Skip them,
                    # stopping at this CTA's end boundary so that boundary stays
                    # reachable -- overshooting it would make the loop unable to
                    # terminate. Every warp role runs this identical traversal.
                    while (
                        (next_num_kv == 0)
                        & (next_q_idx < batch_size)
                        & ((next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx))
                    ):
                        next_q_idx = next_q_idx + 1
                        if next_q_idx < batch_size:
                            next_num_kv = (
                                mContextLens[next_q_idx] + block_kv_val - 1
                            ) // block_kv_val
                # Update while-loop condition
                has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

    @cute.jit
    def _run_umma_warp_1(
        self,
        NUM_MATH_WG,
        batch_size,
        block_kv_val,
        end_kv_idx,
        end_q_idx,
        has_work,
        is_leader_cta,
        kv_pipeline_1,
        mContextLens,
        next_kv_idx,
        next_num_kv,
        next_q_idx,
        q_idx,
        q_pipeline,
        tCrA_1,
        tCrB,
        tCtAcc_fake_staged,
        tiled_mma,
        tmem,
        tmem_group_1_offset,
        umma_pipeline_1,
    ):
        """Run the second UMMA producer role."""
        cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
        q_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        kv_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        umma_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )

        # UMMA warp for group 1
        # Explicitly waits on Q pipeline; critical because TMA warp 1
        # only loads KV1, not Q. Without this wait, UMMA warp 1 can
        # start GEMM before TMA warp 0 finishes loading Q into SMEM.
        # TMEM: wait for umma_warp_0's allocation, retrieve pointer
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc_base_1 = cute.make_tensor(
            tmem_ptr + tmem_group_1_offset, tCtAcc_fake_staged.layout
        )

        if is_leader_cta:
            num_k_blocks_1 = cute.size(tCrA_1.shape[2])
            q_stage_1 = cutlass.Int32(0)

            while has_work:
                # fetch_next_task: commit next to current.
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                # Wait for Q pipeline when batch changes
                if q_idx != q_idx_old:
                    if q_idx_old < batch_size:
                        q_cons_state_umma_1.advance()
                    q_pipeline.consumer_wait(q_cons_state_umma_1)
                    q_stage_1 = q_cons_state_umma_1.index

                # Process KV block for group 1 (kv_idx + 1)
                # Run UMMA unconditionally; aligned padding absorbs OOB output.
                # Wait for KV before acquiring an empty TMEM stage.
                kv_ready_1 = kv_pipeline_1.consumer_try_wait(kv_cons_state_umma_1)
                umma_empty_1 = umma_pipeline_1.producer_try_acquire(umma_prod_state_1)
                kv_pipeline_1.consumer_wait(kv_cons_state_umma_1, kv_ready_1)
                umma_pipeline_1.producer_acquire(umma_prod_state_1, umma_empty_1)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                kv_stage_1 = kv_cons_state_umma_1.index
                tCtAcc_1 = tCtAcc_base_1[(None, None, None, umma_prod_state_1.index)]
                for k_block in cutlass.range_constexpr(num_k_blocks_1):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc_1,
                        tCrA_1[None, None, k_block, kv_stage_1],
                        tCrB[None, None, k_block, q_stage_1],
                        tCtAcc_1,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                # No consumer_release here; Math WG 1 releases.
                kv_cons_state_umma_1.advance()

                umma_pipeline_1.producer_commit(umma_prod_state_1)
                umma_prod_state_1.advance()

                # Advance: inline fetch_next_task
                next_kv_idx = kv_idx + NUM_MATH_WG
                if next_kv_idx >= num_kv:
                    next_q_idx = q_idx + 1
                    next_kv_idx = 0
                    if next_q_idx < batch_size:
                        next_num_kv = (
                            mContextLens[next_q_idx] + block_kv_val - 1
                        ) // block_kv_val
                    # Zero-length rows get no scheduled work, but the exact-
                    # coordinate termination below would still step onto them and
                    # run a full pipeline turn (TMA + UMMA + epilogue). Skip them,
                    # stopping at this CTA's end boundary so that boundary stays
                    # reachable -- overshooting it would make the loop unable to
                    # terminate. Every warp role runs this identical traversal.
                    while (
                        (next_num_kv == 0)
                        & (next_q_idx < batch_size)
                        & ((next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx))
                    ):
                        next_q_idx = next_q_idx + 1
                        if next_q_idx < batch_size:
                            next_num_kv = (
                                mContextLens[next_q_idx] + block_kv_val - 1
                            ) // block_kv_val
                # Update while-loop condition
                has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

    @cute.jit
    def _run_math_warp(
        self,
        NUM_MATH_WG,
        batch_size,
        block_kv_val,
        copy_atom_t2r,
        end_kv_idx,
        end_q_idx,
        epi_sub_mn,
        has_work,
        kv_pipeline_math,
        mContextLens,
        mIndexCounts,
        mLogits,
        mPolicyValues,
        mTileMeta,
        next_kv_idx,
        next_n,
        next_num_kv,
        next_q_idx,
        num_epi_subtiles,
        num_heads,
        num_tmem_alloc_cols_total,
        q_idx,
        q_pipeline,
        sScales,
        sW,
        tCtAcc_fake_staged,
        tidx,
        tmem,
        tmem_group_1_offset,
        umma_pipeline_math,
        warpgroup_idx,
        sStoreScores=None,
        store_pipeline_math=None,
    ):
        """Run one math warpgroup epilogue and score-plane producer."""
        cute.arch.setmaxregister_increase(self.math_wg_registers)
        # Both math WGs advance their own Q state in lockstep.
        q_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )

        # TMEM: math warp 0 is the allocator; all math warps wait + retrieve
        tmem.allocate(num_tmem_alloc_cols_total)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        # Both math warpgroups execute one body. Select the group-private
        # TMEM plane with the uniform warpgroup index while retaining the
        # original two physical accumulator rings.
        math_tmem_ptr = tmem_ptr + warpgroup_idx * tmem_group_1_offset
        if cutlass.const_expr(self.acc_dtype == Float16):
            # Both group offsets are whole TMEM panels. Reassert that
            # alignment after the uniform dynamic selection so packed
            # FP16 LDTM satisfies its 8-byte verifier requirement.
            math_tmem_ptr = cute.make_ptr(
                self.acc_dtype,
                math_tmem_ptr.toint(),
                cute.AddressSpace.tmem,
                assumed_align=8,
            )
        tCtAcc_base = cute.make_tensor(math_tmem_ptr, tCtAcc_fake_staged.layout)

        epi_tidx = tidx % 128
        cC = cute.make_identity_tensor(epi_sub_mn)

        kv_cons_state_math = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        umma_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )
        if cutlass.const_expr(self.has_store_warp):
            store_prod_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 2
            )
        # Both math warpgroups execute the same reduction body.
        (
            tiled_copy_t2r,
            tTR_tAcc_base,
            tTR_cC,
            tTR_rAcc,
        ) = self._make_math_tmem_copy_and_partition(
            tCtAcc_base, epi_sub_mn, copy_atom_t2r, epi_tidx, cC
        )
        m_coord = tTR_cC[0][0]

        # Weight register cache: only first NUM_W_IN_REG per next_n slot.
        # Remaining weights read from SMEM in epilogue.
        # FP16 weights use half the regs, so we can fit
        # all heads for next_n <= 3.
        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
            MAX_NUM_W_IN_REG = 64 if next_n <= 3 else 48
            # FP16 weights pack two per register.
            MAX_W_CACHE_ELEMS = 2 * _MAX_W_CACHE_REGS
        else:
            MAX_NUM_W_IN_REG = 64 if next_n == 1 else 40 if next_n >= 4 else 52
            MAX_W_CACHE_ELEMS = _MAX_W_CACHE_REGS
        # MAX_NUM_W_IN_REG caps the per-slot count, but the register
        # footprint is the product NUM_W_IN_REG * next_n. Whenever
        # num_heads <= MAX_NUM_W_IN_REG the per-slot cap never binds,
        # and the footprint grows to num_heads * next_n == N, which the
        # API admits up to 256 -- past the register file. Bound the
        # product too. Must stay a single constexpr expression: a
        # staged `if` here makes NUM_W_IN_REG a runtime value and the
        # range_constexpr loops below fail to compile.
        NUM_W_IN_REG = max(
            _EPI_UNROLL,
            min(
                MAX_NUM_W_IN_REG,
                num_heads,
                ((MAX_W_CACHE_ELEMS // next_n) // _EPI_UNROLL) * _EPI_UNROLL,
            ),
        )
        if cutlass.const_expr(self.has_store_warp and self.N <= 128):
            # The role-local math budget is large enough to keep the
            # complete tuned Q-tile reduction weights live. Load them once
            # at the Q transition before entering the KV reduction
            # turns. Wider legal N shapes retain the bounded cache above.
            NUM_W_IN_REG = num_heads
        assert NUM_W_IN_REG % _EPI_UNROLL == 0
        w_cache = cute.make_rmem_tensor(
            cute.make_layout((next_n, NUM_W_IN_REG), stride=(NUM_W_IN_REG, 1)),
            self.epi_dtype,
        )
        q_stage_local = cutlass.Int32(0)
        if cutlass.const_expr(self.uses_request_tiled_q):
            store_aligned_prefix = cutlass.Int32(0)
            store_out_row_base = cutlass.Int32(0)
            store_q_valid = cutlass.Int32(0)
            store_causal_base = cutlass.Int32(0)
        if cutlass.const_expr(
            self.is_selective_logits_kernel and not self.has_store_warp
        ):
            store_row_starts = cute.make_rmem_tensor(next_n, cutlass.Int32)
            store_row_ends = cute.make_rmem_tensor(next_n, cutlass.Int32)
        scale_val = cutlass.Float32(0.0)

        while has_work:
            # fetch_next_task: commit next -> current
            q_idx_old = q_idx
            q_idx = next_q_idx
            kv_idx = next_kv_idx
            num_kv = next_num_kv

            if cutlass.const_expr(self.has_store_warp):
                store_pipeline_math.producer_acquire(store_prod_state)

            # Q pipeline consumer: wait for Q+Weights SMEM
            if q_idx != q_idx_old:
                if q_idx_old < batch_size:
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()
                q_pipeline.consumer_wait(q_cons_state)
                q_stage_local = q_cons_state.index
                if cutlass.const_expr(self.uses_request_tiled_q):
                    # Contiguous ingress may carry an aligned prefix;
                    # paged ingress starts at logical request column zero.
                    store_aligned_prefix = cutlass.Int32(0)
                    if cutlass.const_expr(self.uses_contiguous_kv):
                        request_kv_start = mTileMeta[(q_idx, 1)]
                        store_aligned_prefix = request_kv_start % 4
                    store_out_row_base = mTileMeta[(q_idx, 0)]
                    store_q_valid = mTileMeta[(q_idx, 2)]
                    store_causal_base = mTileMeta[(q_idx, 3)]
                if cutlass.const_expr(
                    self.is_selective_logits_kernel and not self.has_store_warp
                ):
                    for store_row in cutlass.range_constexpr(next_n):
                        store_row_starts[store_row] = mTileMeta[(q_idx, 4 + store_row)]
                        store_row_ends[store_row] = mTileMeta[
                            (q_idx, 4 + next_n + store_row)
                        ]
                # Preload first NUM_W_IN_REG weights per slot
                for t_i in cutlass.range_constexpr(next_n):
                    for w_j in cutlass.range_constexpr(NUM_W_IN_REG):
                        w_cache[(t_i, w_j)] = sW[(t_i, w_j, q_stage_local)]

            # Process this warpgroup's logical KV block. OOB results are
            # written to the aligned padding region.
            math_kv_idx = kv_idx + warpgroup_idx
            kv_pos = math_kv_idx * block_kv_val + m_coord
            if cutlass.const_expr(
                self.uses_request_tiled_q and not self.has_store_warp
            ):
                store_kv_pos = kv_pos
                store_kv_valid = cutlass.Boolean(True)
                if cutlass.const_expr(self.uses_contiguous_kv):
                    store_kv_pos = kv_pos - store_aligned_prefix
                    store_kv_valid = store_kv_pos >= 0

            if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                # Skip KV wait, relying on the matching UMMA barrier's
                # transitive visibility.
                umma_pipeline_math.consumer_wait(umma_cons_state)
            else:
                # Keep the readiness probes adjacent within the selected
                # physical pipeline, then pass their tokens unchanged.
                kv_ready_math = kv_pipeline_math.consumer_try_wait(kv_cons_state_math)
                umma_ready_math = umma_pipeline_math.consumer_try_wait(umma_cons_state)
                kv_pipeline_math.consumer_wait(kv_cons_state_math, kv_ready_math)
                sc_stage = kv_cons_state_math.index
                scale_val = sScales[(m_coord, sc_stage, warpgroup_idx)]
                umma_pipeline_math.consumer_wait(umma_cons_state, umma_ready_math)

            # --- TMEM sub-tile setup ---
            # flat_divide accumulator by sub-tile shape;
            # partition once, then loop over sub-tiles.
            tTR_tAcc = tTR_tAcc_base[
                (None, None, None, None, None, umma_cons_state.index)
            ]

            # --- First sub-tile LDTM + KV release ---
            if cutlass.const_expr(self.early_tmem_copy):
                # Issue first sub-tile LDTM early
                cute.copy(
                    tiled_copy_t2r,
                    tTR_tAcc[(None, None, None, 0, 0)],
                    tTR_rAcc,
                )
                # Scale LDS + KV release fill latency
                if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                    sc_stage = kv_cons_state_math.index
                    scale_val = sScales[(m_coord, sc_stage, warpgroup_idx)]
                if cutlass.const_expr(not self.has_store_warp):
                    kv_pipeline_math.consumer_release(kv_cons_state_math)
                    kv_cons_state_math.advance()
                cute.arch.fence_view_async_tmem_load()
            else:
                # Default: scale LDS + KV release first
                if cutlass.const_expr(self.remove_kv_wait_in_epilogue):
                    sc_stage = kv_cons_state_math.index
                    scale_val = sScales[(m_coord, sc_stage, warpgroup_idx)]
                if cutlass.const_expr(not self.has_store_warp):
                    kv_pipeline_math.consumer_release(kv_cons_state_math)
                    kv_cons_state_math.advance()
                cute.copy(
                    tiled_copy_t2r,
                    tTR_tAcc[(None, None, None, 0, 0)],
                    tTR_rAcc,
                )
                cute.arch.fence_view_async_tmem_load()

            if cutlass.const_expr(
                self.is_selective_logits_kernel and self.epi_dtype == cutlass.Float32
            ):
                scale_val = scale_val * cutlass.Float32(0.5)

            # --- Sub-tile compute loop ---
            # Each sub-tile: LDTM.xN -> fence -> load ->
            # ReLU+FMA. Breaks FMA chain (16->4 per chunk)
            # and interleaves LDTM with FP32 compute to
            # reduce ShadowPipeThrottle.
            # Partition the complete logical N tile. Recover the row
            # and head offset from each flattened N coordinate so
            # reductions and stores remain row-private.
            subtile_n = self.N // num_epi_subtiles
            if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                packed_zero = pack_f16x2(Float16(0.0), Float16(0.0))
            for epi_subtile in cutlass.range_constexpr(num_epi_subtiles):
                linear_n = epi_subtile * subtile_n
                # LDTM for sub-tiles 1..N-1
                # (sub-tile 0 handled above)
                if epi_subtile > 0:
                    if cutlass.const_expr(not self.has_store_warp):
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, 0, epi_subtile)],
                            tTR_rAcc,
                        )
                    cute.arch.fence_view_async_tmem_load()
                # Release UMMA after last LDTM+fence
                if epi_subtile == num_epi_subtiles - 1:
                    umma_pipeline_math.consumer_release(umma_cons_state)
                    umma_cons_state.advance()
                acc_vec = tTR_rAcc.load()
                if cutlass.const_expr(
                    self.has_store_warp and epi_subtile + 1 < num_epi_subtiles
                ):
                    # Reuse the fragment destination once its current
                    # SSA value is materialized, overlapping the next
                    # LDTM with this subtile's arithmetic.
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, 0, epi_subtile + 1)],
                        tTR_rAcc,
                    )
                # A wider TMEM load may contain multiple complete
                # logical rows. Keep each row's reduction state and
                # accumulator slice independent within the fragment.
                rows_per_subtile = max(1, next_n // num_epi_subtiles)
                row_n = subtile_n // rows_per_subtile
                h_base, subtile_base_t = cute.idx2crd(linear_n, (num_heads, next_n))
                for subtile_row in cutlass.range_constexpr(rows_per_subtile):
                    t = subtile_base_t + subtile_row
                    row_n_base = subtile_row * row_n
                    if cutlass.const_expr(h_base == 0):
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            ps0 = packed_zero
                            ps1 = packed_zero
                        else:
                            s0x = cutlass.Float32(0.0)
                            s0y = cutlass.Float32(0.0)
                            s1x = cutlass.Float32(0.0)
                            s1y = cutlass.Float32(0.0)
                    # Reg-path: weights from registers
                    reg_h_end = min(row_n, max(0, NUM_W_IN_REG - h_base))
                    for h in cutlass.range_constexpr(0, reg_h_end, 4):
                        n0 = row_n_base + h
                        h_g = h_base + h
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            pa01 = pack_f16x2(
                                Float16(acc_vec[n0]), Float16(acc_vec[n0 + 1])
                            )
                            pa23 = pack_f16x2(
                                Float16(acc_vec[n0 + 2]),
                                Float16(acc_vec[n0 + 3]),
                            )
                            pa01 = max_f16x2(pa01, packed_zero)
                            pa23 = max_f16x2(pa23, packed_zero)
                            pw01 = pack_f16x2(w_cache[(t, h_g)], w_cache[(t, h_g + 1)])
                            pw23 = pack_f16x2(
                                w_cache[(t, h_g + 2)], w_cache[(t, h_g + 3)]
                            )
                            ps0 = fma_f16x2(pa01, pw01, ps0)
                            ps1 = fma_f16x2(pa23, pw23, ps1)
                        else:
                            if cutlass.const_expr(self.is_selective_logits_kernel):
                                a0, a1 = cute.arch.add_packed_f32x2(
                                    (acc_vec[n0], acc_vec[n0 + 1]),
                                    (
                                        abs_f32(acc_vec[n0]),
                                        abs_f32(acc_vec[n0 + 1]),
                                    ),
                                )
                                a2, a3 = cute.arch.add_packed_f32x2(
                                    (acc_vec[n0 + 2], acc_vec[n0 + 3]),
                                    (
                                        abs_f32(acc_vec[n0 + 2]),
                                        abs_f32(acc_vec[n0 + 3]),
                                    ),
                                )
                            else:
                                a0 = cutlass.max(acc_vec[n0], cutlass.Float32(0.0))
                                a1 = cutlass.max(acc_vec[n0 + 1], cutlass.Float32(0.0))
                                a2 = cutlass.max(acc_vec[n0 + 2], cutlass.Float32(0.0))
                                a3 = cutlass.max(acc_vec[n0 + 3], cutlass.Float32(0.0))
                            w0 = w_cache[(t, h_g)]
                            w1 = w_cache[(t, h_g + 1)]
                            w2 = w_cache[(t, h_g + 2)]
                            w3 = w_cache[(t, h_g + 3)]
                            s0x, s0y = cute.arch.fma_packed_f32x2(
                                (a0, a1),
                                (w0, w1),
                                (s0x, s0y),
                                rnd=_RND_RN,
                            )
                            s1x, s1y = cute.arch.fma_packed_f32x2(
                                (a2, a3),
                                (w2, w3),
                                (s1x, s1y),
                                rnd=_RND_RN,
                            )
                    # SMEM-path: weights from shared mem
                    smem_h_start = max(0, NUM_W_IN_REG - h_base)
                    for h in cutlass.range_constexpr(smem_h_start, row_n, 4):
                        n0 = row_n_base + h
                        h_g = h_base + h
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            pa01 = pack_f16x2(
                                Float16(acc_vec[n0]), Float16(acc_vec[n0 + 1])
                            )
                            pa23 = pack_f16x2(
                                Float16(acc_vec[n0 + 2]),
                                Float16(acc_vec[n0 + 3]),
                            )
                            pa01 = max_f16x2(pa01, packed_zero)
                            pa23 = max_f16x2(pa23, packed_zero)
                            pw01 = pack_f16x2(
                                sW[(t, h_g, q_stage_local)],
                                sW[(t, h_g + 1, q_stage_local)],
                            )
                            pw23 = pack_f16x2(
                                sW[(t, h_g + 2, q_stage_local)],
                                sW[(t, h_g + 3, q_stage_local)],
                            )
                            ps0 = fma_f16x2(pa01, pw01, ps0)
                            ps1 = fma_f16x2(pa23, pw23, ps1)
                        else:
                            if cutlass.const_expr(self.is_selective_logits_kernel):
                                a0, a1 = cute.arch.add_packed_f32x2(
                                    (acc_vec[n0], acc_vec[n0 + 1]),
                                    (
                                        abs_f32(acc_vec[n0]),
                                        abs_f32(acc_vec[n0 + 1]),
                                    ),
                                )
                                a2, a3 = cute.arch.add_packed_f32x2(
                                    (acc_vec[n0 + 2], acc_vec[n0 + 3]),
                                    (
                                        abs_f32(acc_vec[n0 + 2]),
                                        abs_f32(acc_vec[n0 + 3]),
                                    ),
                                )
                            else:
                                a0 = cutlass.max(acc_vec[n0], cutlass.Float32(0.0))
                                a1 = cutlass.max(acc_vec[n0 + 1], cutlass.Float32(0.0))
                                a2 = cutlass.max(acc_vec[n0 + 2], cutlass.Float32(0.0))
                                a3 = cutlass.max(acc_vec[n0 + 3], cutlass.Float32(0.0))
                            w0 = sW[(t, h_g, q_stage_local)]
                            w1 = sW[(t, h_g + 1, q_stage_local)]
                            w2 = sW[(t, h_g + 2, q_stage_local)]
                            w3 = sW[(t, h_g + 3, q_stage_local)]
                            s0x, s0y = cute.arch.fma_packed_f32x2(
                                (a0, a1),
                                (w0, w1),
                                (s0x, s0y),
                                rnd=_RND_RN,
                            )
                            s1x, s1y = cute.arch.fma_packed_f32x2(
                                (a2, a3),
                                (w2, w3),
                                (s1x, s1y),
                                rnd=_RND_RN,
                            )
                    if cutlass.const_expr(h_base + row_n == num_heads):
                        if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                            ps_sum = add_f16x2(ps0, ps1)
                            sum_lo, sum_hi = unpack_f16x2(ps_sum)
                            result_t = sum_lo + sum_hi
                        else:
                            if cutlass.const_expr(self.is_selective_logits_kernel):
                                sum_x, sum_y = cute.arch.add_packed_f32x2(
                                    (s0x, s0y),
                                    (s1x, s1y),
                                )
                                result_t = sum_x + sum_y
                            else:
                                result_t = s0x + s0y + s1x + s1y
                        if cutlass.const_expr(self.has_store_warp):
                            published_score = cutlass.Float32(
                                result_t
                            ) * cutlass.Float32(scale_val)
                            sStoreScores[
                                (
                                    t,
                                    m_coord,
                                    store_prod_state.index,
                                    warpgroup_idx,
                                )
                            ] = published_score
                        elif cutlass.const_expr(self.uses_request_tiled_q):
                            self._store_result_packed_causal_meta(
                                mLogits,
                                mIndexCounts,
                                mTileMeta,
                                mPolicyValues,
                                q_idx,
                                store_out_row_base,
                                store_q_valid,
                                store_causal_base,
                                store_row_starts[t]
                                if cutlass.const_expr(self.is_selective_logits_kernel)
                                else cutlass.Int32(0),
                                store_row_ends[t]
                                if cutlass.const_expr(self.is_selective_logits_kernel)
                                else cutlass.Int32(0),
                                t,
                                store_kv_pos,
                                store_kv_valid,
                                result_t,
                                scale_val,
                                next_n,
                            )
                        else:
                            self._store_result(
                                mLogits,
                                mTileMeta,
                                q_idx,
                                t,
                                kv_pos,
                                result_t,
                                scale_val,
                                next_n,
                            )

            if cutlass.const_expr(self.has_store_warp):
                # Keep the KV/scale stage owned through the final scale use.
                # Releasing it after the first TMEM load allows the TMA
                # producer to overwrite the stage while the unrolled
                # epilogue still consumes scale_val.
                kv_pipeline_math.consumer_release(kv_cons_state_math)
                kv_cons_state_math.advance()
                # PipelineAsync publishes ordinary-thread SMEM writes.
                # Fence the completed score plane before the producer
                # arrivals make this stage visible to the store warpgroup.
                cute.arch.fence_view_async_shared()
                store_pipeline_math.producer_commit(store_prod_state)
                store_prod_state.advance()

            # Advance: inline fetch_next_task
            next_kv_idx = kv_idx + NUM_MATH_WG
            if next_kv_idx >= num_kv:
                next_q_idx = q_idx + 1
                next_kv_idx = 0
                if next_q_idx < batch_size:
                    next_num_kv = (
                        mContextLens[next_q_idx] + block_kv_val - 1
                    ) // block_kv_val
                # Zero-length rows get no scheduled work, but the exact-
                # coordinate termination below would still step onto them and
                # run a full pipeline turn (TMA + UMMA + epilogue). Skip them,
                # stopping at this CTA's end boundary so that boundary stays
                # reachable -- overshooting it would make the loop unable to
                # terminate. Every warp role runs this identical traversal.
                while (
                    (next_num_kv == 0)
                    & (next_q_idx < batch_size)
                    & ((next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx))
                ):
                    next_q_idx = next_q_idx + 1
                    if next_q_idx < batch_size:
                        next_num_kv = (
                            mContextLens[next_q_idx] + block_kv_val - 1
                        ) // block_kv_val
            # Update while-loop condition
            has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

        # Release the last Q stage for this math warpgroup.
        if q_idx < batch_size:
            q_pipeline.consumer_release(q_cons_state)
            q_cons_state.advance()
        if cutlass.const_expr(self.has_store_warp):
            store_pipeline_math.producer_tail(store_prod_state)

        # TMEM dealloc: math warps are allocator + last consumer
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def _run_store_warp(
        self,
        NUM_MATH_WG,
        NUM_STORE_WARPS,
        batch_size,
        block_kv_val,
        end_kv_idx,
        end_q_idx,
        has_work,
        mContextLens,
        mIndexCounts,
        mLogits,
        mPolicyValues,
        mTileMeta,
        next_kv_idx,
        next_n,
        next_num_kv,
        next_q_idx,
        q_idx,
        sStoreScores_0,
        sStoreScores_1,
        store_pipeline_0,
        store_pipeline_1,
    ):
        """Run one segmented candidate publisher warp."""
        if cutlass.const_expr(self.has_store_warp):
            cute.arch.setmaxregister_increase(self.store_wg_registers)
            store_cons_state_0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            store_cons_state_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            store_logits_addr = make_warp_uniform_i64(mLogits.iterator.toint())
            store_logits_stride = make_warp_uniform_i64(mLogits.layout.stride[0])
            store_warp = cute.arch.make_warp_uniform(
                cute.arch.warp_idx() - cutlass.Int32(12)
            )
            store_lane = cute.arch.lane_idx()
            lane_private_segment_base = store_lane * cutlass.Int32(
                self.num_count_segments // 32
            )
            num_q_groups = (next_n + NUM_STORE_WARPS - 1) // NUM_STORE_WARPS
            num_lane_cursors = (
                self.split_kv
                if cutlass.const_expr(self.split_kv > 1)
                else self.num_count_segments // 32
            )
            num_private_cursors = (
                num_lane_cursors
                if cutlass.const_expr(self.count_segment_span == 1)
                else 1
                if cutlass.const_expr(self.count_segment_span == 128)
                else 4
            )
            private_cursors = [
                [cutlass.Int32(0) for _ in range(num_private_cursors)]
                for _ in range(num_q_groups)
            ]
            cursor_q_idx = cutlass.Int32(-1)
            cached_out_row_base = cutlass.Int32(0)
            cached_q_valid = cutlass.Int32(0)
            cached_causal_base = cutlass.Int32(0)
            cached_row_logits_addrs = [cutlass.Int64(0) for _ in range(num_q_groups)]
            cached_row_logical_ends = [cutlass.Int32(0) for _ in range(num_q_groups)]
            cached_thresholds = [cutlass.Float32(0.0) for _ in range(num_q_groups)]

            while has_work:
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                if q_idx != cursor_q_idx:
                    cached_out_row_base = mTileMeta[(q_idx, 0)]
                    cached_q_valid = mTileMeta[(q_idx, 2)]
                    cached_causal_base = mTileMeta[(q_idx, 3)]
                    for q_group in cutlass.range_constexpr(num_q_groups):
                        row = store_warp + q_group * NUM_STORE_WARPS
                        if row < cached_q_valid:
                            out_row = cached_out_row_base + row
                            cached_row_logits_addrs[q_group] = (
                                store_logits_addr
                                + cutlass.Int64(
                                    cutlass.Uint64(cutlass.Uint32(out_row))
                                    * cutlass.Uint64(store_logits_stride)
                                )
                                * cutlass.Int64(8)
                            )
                            cached_row_logical_ends[q_group] = (
                                cached_causal_base + row + 1
                            )
                            cached_thresholds[q_group] = i32_bits_to_fp32(
                                mPolicyValues[out_row]
                            )
                        for cursor_idx in cutlass.range_constexpr(num_private_cursors):
                            private_cursors[q_group][cursor_idx] = cutlass.Int32(0)
                    cursor_q_idx = q_idx

                out_row_base = cached_out_row_base
                q_valid = cached_q_valid
                kv_base = store_lane * 4
                logical_base_0 = kv_idx * block_kv_val + kv_base
                logical_base_1 = (kv_idx + 1) * block_kv_val + kv_base
                split_idx = cutlass.Int32(0)
                split_end_kv = num_kv
                if cutlass.const_expr(self.split_kv > 1):
                    num_kv_pairs = (num_kv + NUM_MATH_WG - 1) // NUM_MATH_WG
                    active_splits = min(num_kv_pairs, cutlass.Int32(self.split_kv))
                    kv_pair_idx = kv_idx // NUM_MATH_WG
                    split_idx = (kv_pair_idx * active_splits) // num_kv_pairs
                    split_end_pair = cute.ceil_div(
                        (split_idx + 1) * num_kv_pairs, active_splits
                    )
                    split_end_kv = min(split_end_pair * NUM_MATH_WG, num_kv)
                store_ready_0 = store_pipeline_0.consumer_try_wait(store_cons_state_0)
                store_ready_1 = store_pipeline_1.consumer_try_wait(store_cons_state_1)
                store_pipeline_0.consumer_wait(store_cons_state_0, store_ready_0)
                if store_warp >= q_valid:
                    store_pipeline_1.consumer_wait(store_cons_state_1, store_ready_1)
                if (
                    (q_valid < next_n)
                    | (q_valid < NUM_STORE_WARPS)
                    | (kv_idx + NUM_MATH_WG >= num_kv)
                ):
                    # Peel partial-Q tiles and the final KV pair so the
                    # steady-state path carries no row/logical predicates.
                    for q_group in cutlass.range_constexpr(num_q_groups):
                        row = store_warp + q_group * NUM_STORE_WARPS
                        if (
                            cutlass.const_expr(next_n >= NUM_STORE_WARPS)
                            or row < q_valid
                        ):
                            row_logits_addr = cached_row_logits_addrs[q_group]
                            row_logical_end = cached_row_logical_ends[q_group]
                            threshold = cached_thresholds[q_group]
                            scores_0 = load_shared_v4_f32(
                                sStoreScores_0,
                                row,
                                kv_base,
                                store_cons_state_0.index,
                            )
                            if cutlass.const_expr(q_group == 0):
                                if store_warp < q_valid:
                                    store_pipeline_1.consumer_wait(
                                        store_cons_state_1, store_ready_1
                                    )
                            scores_1 = load_shared_v4_f32(
                                sStoreScores_1,
                                row,
                                kv_base,
                                store_cons_state_1.index,
                            )
                            if cutlass.const_expr(
                                next_n >= NUM_STORE_WARPS
                                and q_group == num_q_groups - 1
                            ):
                                # Both score planes are now warp-local registers;
                                # let the math warpgroups reuse their SMEM stages
                                # while this final row group is compacted.
                                store_pipeline_0.consumer_release(store_cons_state_0)
                                store_cons_state_0.advance()
                                store_pipeline_1.consumer_release(store_cons_state_1)
                                store_cons_state_1.advance()
                            if cutlass.const_expr(self.count_segment_span == 1):
                                if cutlass.const_expr(self.split_kv > 1):
                                    split_cursor = private_cursors[q_group][0]
                                    for candidate_split in cutlass.range_constexpr(
                                        1, self.split_kv
                                    ):
                                        if split_idx == candidate_split:
                                            split_cursor = private_cursors[q_group][
                                                candidate_split
                                            ]
                                    split_segment = split_idx * 32 + store_lane
                                    for vec_idx in cutlass.range_constexpr(4):
                                        split_cursor = (
                                            self._store_candidate_lane_row_pair(  # type: ignore[attr-defined]
                                                row_logits_addr,
                                                q_valid,
                                                row,
                                                row_logical_end,
                                                logical_base_0 + vec_idx,
                                                logical_base_1 + vec_idx,
                                                threshold,
                                                split_cursor,
                                                split_segment,
                                                scores_0[vec_idx],
                                                scores_1[vec_idx],
                                            )
                                        )
                                    for candidate_split in cutlass.range_constexpr(
                                        self.split_kv
                                    ):
                                        if split_idx == candidate_split:
                                            private_cursors[q_group][
                                                candidate_split
                                            ] = split_cursor
                                else:
                                    for vec_idx in cutlass.range_constexpr(4):
                                        cursor_idx = (
                                            0
                                            if cutlass.const_expr(
                                                self.num_count_segments == 32
                                            )
                                            else vec_idx // 2
                                            if cutlass.const_expr(
                                                self.num_count_segments == 64
                                            )
                                            else vec_idx
                                        )
                                        private_cursors[q_group][cursor_idx] = (
                                            self._store_candidate_lane_row_pair(  # type: ignore[attr-defined]
                                                row_logits_addr,
                                                q_valid,
                                                row,
                                                row_logical_end,
                                                logical_base_0 + vec_idx,
                                                logical_base_1 + vec_idx,
                                                threshold,
                                                private_cursors[q_group][cursor_idx],
                                                lane_private_segment_base + cursor_idx,
                                                scores_0[vec_idx],
                                                scores_1[vec_idx],
                                            )
                                        )
                            elif cutlass.const_expr(self.count_segment_span == 128):
                                pooled_cursor = private_cursors[q_group][0]
                                for vec_idx in cutlass.range_constexpr(4):
                                    pooled_cursor = self._store_candidate_warp_row_pair(  # type: ignore[attr-defined]
                                        row_logits_addr,
                                        q_valid,
                                        row,
                                        row_logical_end,
                                        logical_base_0 + vec_idx,
                                        logical_base_1 + vec_idx,
                                        threshold,
                                        pooled_cursor,
                                        split_idx,
                                        scores_0[vec_idx],
                                        scores_1[vec_idx],
                                    )
                                private_cursors[q_group][0] = pooled_cursor
                            else:
                                for vec_idx in cutlass.range_constexpr(4):
                                    private_cursors[q_group][vec_idx] = (
                                        self._store_candidate_warp_row_pair(  # type: ignore[attr-defined]
                                            row_logits_addr,
                                            q_valid,
                                            row,
                                            row_logical_end,
                                            logical_base_0 + vec_idx,
                                            logical_base_1 + vec_idx,
                                            threshold,
                                            private_cursors[q_group][vec_idx],
                                            split_idx * 4 + vec_idx,
                                            scores_0[vec_idx],
                                            scores_1[vec_idx],
                                        )
                                    )
                else:
                    for q_group in cutlass.range_constexpr(num_q_groups):
                        row = store_warp + q_group * NUM_STORE_WARPS
                        row_logits_addr = cached_row_logits_addrs[q_group]
                        threshold = cached_thresholds[q_group]
                        scores_0 = load_shared_v4_f32(
                            sStoreScores_0,
                            row,
                            kv_base,
                            store_cons_state_0.index,
                        )
                        if cutlass.const_expr(q_group == 0):
                            store_pipeline_1.consumer_wait(
                                store_cons_state_1, store_ready_1
                            )
                        scores_1 = load_shared_v4_f32(
                            sStoreScores_1,
                            row,
                            kv_base,
                            store_cons_state_1.index,
                        )
                        if cutlass.const_expr(
                            next_n >= NUM_STORE_WARPS and q_group == num_q_groups - 1
                        ):
                            # Both score planes are now warp-local registers;
                            # let the math warpgroups reuse their SMEM stages
                            # while this final row group is compacted.
                            store_pipeline_0.consumer_release(store_cons_state_0)
                            store_cons_state_0.advance()
                            store_pipeline_1.consumer_release(store_cons_state_1)
                            store_cons_state_1.advance()
                        if cutlass.const_expr(self.count_segment_span == 1):
                            if cutlass.const_expr(self.split_kv > 1):
                                split_cursor = private_cursors[q_group][0]
                                for candidate_split in cutlass.range_constexpr(
                                    1, self.split_kv
                                ):
                                    if split_idx == candidate_split:
                                        split_cursor = private_cursors[q_group][
                                            candidate_split
                                        ]
                                split_segment = split_idx * 32 + store_lane
                                for vec_idx in cutlass.range_constexpr(4):
                                    split_cursor = (
                                        self._store_candidate_lane_row_pair_full(  # type: ignore[attr-defined]
                                            row_logits_addr,
                                            logical_base_0 + vec_idx,
                                            logical_base_1 + vec_idx,
                                            threshold,
                                            split_cursor,
                                            split_segment,
                                            scores_0[vec_idx],
                                            scores_1[vec_idx],
                                        )
                                    )
                                for candidate_split in cutlass.range_constexpr(
                                    self.split_kv
                                ):
                                    if split_idx == candidate_split:
                                        private_cursors[q_group][candidate_split] = (
                                            split_cursor
                                        )
                            else:
                                for vec_idx in cutlass.range_constexpr(4):
                                    cursor_idx = (
                                        0
                                        if cutlass.const_expr(
                                            self.num_count_segments == 32
                                        )
                                        else vec_idx // 2
                                        if cutlass.const_expr(
                                            self.num_count_segments == 64
                                        )
                                        else vec_idx
                                    )
                                    private_cursors[q_group][cursor_idx] = (
                                        self._store_candidate_lane_row_pair_full(  # type: ignore[attr-defined]
                                            row_logits_addr,
                                            logical_base_0 + vec_idx,
                                            logical_base_1 + vec_idx,
                                            threshold,
                                            private_cursors[q_group][cursor_idx],
                                            lane_private_segment_base + cursor_idx,
                                            scores_0[vec_idx],
                                            scores_1[vec_idx],
                                        )
                                    )
                        elif cutlass.const_expr(self.count_segment_span == 128):
                            pooled_cursor = private_cursors[q_group][0]
                            for vec_idx in cutlass.range_constexpr(4):
                                pooled_cursor = (
                                    self._store_candidate_warp_row_pair_full(  # type: ignore[attr-defined]
                                        row_logits_addr,
                                        logical_base_0 + vec_idx,
                                        logical_base_1 + vec_idx,
                                        threshold,
                                        pooled_cursor,
                                        split_idx,
                                        scores_0[vec_idx],
                                        scores_1[vec_idx],
                                    )
                                )
                            private_cursors[q_group][0] = pooled_cursor
                        else:
                            for vec_idx in cutlass.range_constexpr(4):
                                private_cursors[q_group][vec_idx] = (
                                    self._store_candidate_warp_row_pair_full(  # type: ignore[attr-defined]
                                        row_logits_addr,
                                        logical_base_0 + vec_idx,
                                        logical_base_1 + vec_idx,
                                        threshold,
                                        private_cursors[q_group][vec_idx],
                                        split_idx * 4 + vec_idx,
                                        scores_0[vec_idx],
                                        scores_1[vec_idx],
                                    )
                                )

                publish_split = kv_idx + NUM_MATH_WG >= num_kv
                if cutlass.const_expr(self.split_kv > 1):
                    publish_split = kv_idx + NUM_MATH_WG >= split_end_kv
                if publish_split:
                    for q_group in cutlass.range_constexpr(num_q_groups):
                        row = store_warp + q_group * NUM_STORE_WARPS
                        if cutlass.const_expr(self.count_segment_span == 1):
                            publish_count = row < q_valid
                        else:
                            publish_count = (store_lane == 0) & (row < q_valid)
                        if publish_count:
                            out_row = out_row_base + row
                            if cutlass.const_expr(self.count_segment_span == 1):
                                if cutlass.const_expr(self.split_kv > 1):
                                    for candidate_split in cutlass.range_constexpr(
                                        self.split_kv
                                    ):
                                        if split_idx == candidate_split:
                                            mIndexCounts[
                                                (
                                                    out_row,
                                                    candidate_split * 32
                                                    + store_lane
                                                    + 1,
                                                )
                                            ] = private_cursors[q_group][
                                                candidate_split
                                            ]
                                else:
                                    num_lane_segments = (
                                        1
                                        if cutlass.const_expr(
                                            self.num_count_segments == 32
                                        )
                                        else 2
                                        if cutlass.const_expr(
                                            self.num_count_segments == 64
                                        )
                                        else 4
                                    )
                                    for vec_idx in cutlass.range_constexpr(
                                        num_lane_segments
                                    ):
                                        mIndexCounts[
                                            (
                                                out_row,
                                                lane_private_segment_base + vec_idx + 1,
                                            )
                                        ] = private_cursors[q_group][vec_idx]
                            elif cutlass.const_expr(self.count_segment_span == 128):
                                mIndexCounts[(out_row, split_idx + 1)] = (
                                    private_cursors[q_group][0]
                                )
                            else:
                                for vec_idx in cutlass.range_constexpr(4):
                                    mIndexCounts[
                                        (out_row, split_idx * 4 + vec_idx + 1)
                                    ] = private_cursors[q_group][vec_idx]
                        if cutlass.const_expr(self.count_segment_span == 128):
                            private_cursors[q_group][0] = cutlass.Int32(0)
                        elif cutlass.const_expr(self.count_segment_span != 1):
                            # The cursor is warp-uniform state. Every lane must
                            # reset it before this persistent CTA accepts a new
                            # split work item; lane 0 alone publishes the count.
                            for vec_idx in cutlass.range_constexpr(4):
                                private_cursors[q_group][vec_idx] = cutlass.Int32(0)
                if cutlass.const_expr(next_n < NUM_STORE_WARPS):
                    store_pipeline_0.consumer_release(store_cons_state_0)
                    store_cons_state_0.advance()
                    store_pipeline_1.consumer_release(store_cons_state_1)
                    store_cons_state_1.advance()

                next_kv_idx = kv_idx + NUM_MATH_WG
                if next_kv_idx >= num_kv:
                    next_q_idx = q_idx + 1
                    next_kv_idx = 0
                    if next_q_idx < batch_size:
                        next_num_kv = (
                            mContextLens[next_q_idx] + block_kv_val - 1
                        ) // block_kv_val
                    while (
                        (next_num_kv == 0)
                        & (next_q_idx < batch_size)
                        & ((next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx))
                    ):
                        next_q_idx = next_q_idx + 1
                        if next_q_idx < batch_size:
                            next_num_kv = (
                                mContextLens[next_q_idx] + block_kv_val - 1
                            ) // block_kv_val
                has_work = (next_q_idx != end_q_idx) | (next_kv_idx != end_kv_idx)

    @cute.jit
    def _run_reserved_warp(self):
        """Apply the specialized register budget to the reserved warp."""
        cute.arch.setmaxregister_decrease(self.specialized_wg_registers)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # KV pool
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # Q (L dim = batch_size)
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,  # Weights TMA coord tensor [N, batch_size]
        tma_atom_s: cute.CopyAtom,
        mS_tma: cute.Tensor,  # Scales TMA coord tensor [block_kv, num_phys_blocks]
        mLogits: cute.Tensor,  # [batch_size * next_n, max_context_len]
        mBlockTable: cute.Tensor,  # [batch_size, max_blocks_per_seq]
        mIndexCounts: cute.Tensor,  # selective counters or paged placeholder
        mContextLens: cute.Tensor,  # [batch_size]
        mScheduleMeta: cute.Tensor,  # [num_sms+1, 2] int32
        mTileMeta: cute.Tensor,  # selective request metadata or paged placeholder
        mPolicyValues: cute.Tensor,  # selective row policy or paged placeholder
        batch_size: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        s_smem_layout_staged: cute.Layout,
        a_tma_view_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # Warp roles (matches DeepGEMM SM100)
        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < 8
        is_tma_warp_0 = warp_idx == 8
        is_tma_warp_1 = warp_idx == 9
        is_tma_warp = is_tma_warp_0 | is_tma_warp_1
        is_umma_warp_0 = warp_idx == 10
        # With one UMMA producer, warp 10 issues both independent accumulator
        # groups back-to-back; warp 11 becomes reserved.  This stays a static
        # specialization so the legacy two-producer path is not predicated at
        # runtime.
        if cutlass.const_expr(self.uses_single_umma_producer):
            is_umma_warp_1 = cutlass.Boolean(False)
        else:
            is_umma_warp_1 = warp_idx == 11
        is_store_warp = warp_idx >= 12

        # Early schedule metadata load: issue global loads ASAP so their
        # ~200-cycle L2 latency overlaps with subsequent prologue setup
        # (SMEM alloc, TMA partition, MMA fragment creation, etc.)
        NUM_MATH_WG = 2  # kNumMathWarpGroups
        NUM_STORE_WARPS = 4
        NUM_BLOCKS_PER_MMA = self.num_blocks_per_mma
        sm_idx = bidz
        start_q = mScheduleMeta[(sm_idx, 0)]
        start_kv_half = mScheduleMeta[(sm_idx, 1)]
        end_q_idx = mScheduleMeta[(sm_idx + 1, 0)]
        end_kv_half = mScheduleMeta[(sm_idx + 1, 1)]
        # Early mContextLens load: overlap ~200-cycle L2 latency with the
        # entire prologue setup (pipelines, SMEM alloc, TMA partition, etc.)
        # Clamp to avoid OOB when start_q == batch_size (zero-work CTA sentinel).
        # Note: zero-work CTAs get a stale current_num_kv (from the last batch
        # element), but it is never used because has_work will be False.
        start_q_clamped = min(start_q, batch_size - 1)
        current_num_kv = (
            mContextLens[start_q_clamped] + self.block_kv - 1
        ) // self.block_kv

        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_s)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        block_kv_val = self.block_kv
        num_heads = self.num_heads
        next_n = self.next_n
        num_epi_subtiles = self.num_epi_subtiles

        # === Pipelines ===
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Q pipeline: TMA producer to Math consumer (8 math warps).
        # PipelineTmaAsync: consumer_release uses is_signalling_thread
        # (lane 0 per warp). 8 math warps * 1 lane-0 = 8 arrives.
        q_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.num_q_stages,
            producer_group=prod_group,
            consumer_group=q_cons_group,
            tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        # Merged KV+Scale pipelines (per-group, 3 stages each)
        # Like DeepGEMM: KV data and scales share one barrier.
        # TMA loads both under one barrier. Math is consumer (releases).
        # UMMA also waits on this barrier (for KV GEMM) but does NOT release.
        math_warps_per_group = self.num_math_warps // 2  # 4 warps
        kv_cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, math_warps_per_group
        )
        kv_pipeline_0 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar.data_ptr(),
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        kv_pipeline_1 = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar.data_ptr() + self.num_kv_stages * 2,
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        math_pipeline_ring = min(warpgroup_idx, cutlass.Int32(1))
        kv_pipeline_math = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar.data_ptr()
            + math_pipeline_ring * self.num_kv_stages * 2,
            num_stages=self.num_kv_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=self.num_kv_scale_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )

        # UMMA pipelines (per-group)
        math_threads_per_group = self.num_math_threads // 2
        umma_pipeline_0 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, math_threads_per_group
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline_1 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar.data_ptr() + self.num_umma_stages * 2,
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, math_threads_per_group
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline_math = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar.data_ptr()
            + math_pipeline_ring * self.num_umma_stages * 2,
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, math_threads_per_group
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Candidate-only score handoff.  Each math warpgroup produces one
        # query_tile x KV128 FP32 plane; the dedicated store warpgroup consumes it.
        # Two stages per math group provide the requested direct 32-KiB
        # ping-pong without involving the reserved warp 11.
        if cutlass.const_expr(self.has_store_warp):
            store_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
            store_pipeline_0 = pipeline.PipelineAsync.create(
                barrier_storage=storage.store_mbar.data_ptr(),
                num_stages=2,
                producer_group=store_group,
                consumer_group=store_group,
            )
            store_pipeline_1 = pipeline.PipelineAsync.create(
                barrier_storage=storage.store_mbar.data_ptr() + 4,
                num_stages=2,
                producer_group=store_group,
                consumer_group=store_group,
            )
            # The math body selects its existing group-local ring once through a
            # uniform pointer offset. The two store-facing wrappers above
            # still own initialization and expose both rings independently.
            store_pipeline_math = pipeline.PipelineAsync.create(
                barrier_storage=storage.store_mbar.data_ptr() + math_pipeline_ring * 4,
                num_stages=2,
                producer_group=store_group,
                consumer_group=store_group,
                defer_sync=True,
            )
        # TMEM: only Math warps (8*32=256) + UMMA warps (2*32=64) = 320 threads
        # TMA warps do NOT participate, so they can start TMA loads earlier.
        # Math warp 0 is the allocator (like fp16_gemm_3's epilogue warp 0),
        # because math warps are the last TMEM consumers (epilogue reads).
        # The TMEM allocation barrier includes only roles that retrieve the
        # allocation.  A single UMMA producer reserves warp 11, reducing the
        # participant set from ten to nine warps.
        # 10 warps: warp 0-7 (math) + warp 10-11 (umma)
        tmem_alloc_num_threads = 288 if self.uses_single_umma_producer else 320
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=tmem_alloc_num_threads
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,  # math warp 0 does alloc+free (last TMEM consumer)
            is_two_cta=False,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # SMEM allocation: per-group KV + shared Q
        sKV_0 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sKV_1 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sQ = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # Pad SMEM to push sW/sScales into sub-partition 1 (>= 128KB)
        # to avoid sub-bank conflicts with UMMA reading sKV from
        # sub-partition 0.
        if cutlass.const_expr(self.smem_pad_bytes > 0):
            _ = smem.allocate(self.smem_pad_bytes)
        # Weights SMEM: [N, num_q_stages], shared Q barrier
        sW = smem.allocate_tensor(
            element_type=self.epi_dtype,
            layout=w_smem_layout_staged,
            byte_alignment=128,
        )
        # Preserve the logical (query row, head, stage) coordinates instead of
        # flattening row/head addresses throughout the producer and epilogue.
        sW = cute.make_tensor(
            sW.iterator,
            cute.make_layout(
                (next_n, num_heads, self.num_q_stages),
                stride=(num_heads, 1, self.w_stage_stride),
            ),
        )
        # Scales SMEM: [block_kv, num_kv_stages] float32, per group
        scales_layout = cute.make_layout(
            (block_kv_val, self.num_kv_stages, 2),
            stride=(1, block_kv_val, block_kv_val * self.num_kv_stages),
        )
        sScales = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=scales_layout,
            byte_alignment=128,
        )
        sScales_0 = sScales[(None, None, 0)]
        sScales_1 = sScales[(None, None, 1)]
        if cutlass.const_expr(self.has_store_warp):
            store_score_layout = cute.make_layout(
                (next_n, block_kv_val, 2, 2),
                stride=(
                    block_kv_val,
                    1,
                    next_n * block_kv_val,
                    next_n * block_kv_val * 2,
                ),
            )
            sStoreScores = smem.allocate_tensor(
                element_type=cutlass.Float32,
                layout=store_score_layout,
                byte_alignment=128,
            )
            sStoreScores_0 = sStoreScores[(None, None, None, 0)]
            sStoreScores_1 = sStoreScores[(None, None, None, 1)]

        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        # Partition KV (A): fmha_decode_paged pattern.
        # Partition KV (A): paged decode uses physical sub-blocks; causal
        # Request-tiled ingress keeps the upstream SMEM/MMA topology while
        # rebasing Q and weights through per-tile request metadata.
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        if cutlass.const_expr(self.uses_contiguous_kv):
            tCgA = thr_mma.partition_A(
                cute.local_tile(
                    mA_mkl,
                    cute.slice_(self.mma_tiler, (None, 0, None)),
                    (None, None, None),
                )
            )
            tAsA_0, tAgA_0 = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sKV_0, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            tAsA_1, tAgA_1 = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sKV_1, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
        else:
            # SMEM view is ((tile), num_sub_blocks, stages) - built in __call__.
            # Use .outer (plain layout); swizzle is captured by sKV_0's iterator.
            sKV_0_for_tma = cute.make_tensor(sKV_0.iterator, a_tma_view_layout.outer)
            sKV_1_for_tma = cute.make_tensor(sKV_1.iterator, a_tma_view_layout.outer)
            # GMEM: local_tile by (phys, head), then group first 2 modes into tile.
            gA = cute.local_tile(
                mA_mkl,
                (self.phys_block_kv, self.head_dim),
                coord=(None, None, None),
            )
            tAsA_0, tAgA_0 = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                sKV_0_for_tma,
                cute.group_modes(gA, 0, 2),
            )
            tAsA_1, tAgA_1 = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                sKV_1_for_tma,
                cute.group_modes(gA, 0, 2),
            )

        # Partition Q (B): shared SMEM, L dim = batch_size
        # Partition Q (B): request-tiled ingress rebases query_tile x H; paged
        # decode indexes the original L=batch dimension.
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        tCgB = thr_mma.partition_B(gB_nkl)
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[(None, 0, None, None)]  # [tma, K, L]

        # Partition Weights: standalone TMA, [N, batch_size] to [N] per stage
        # Partition Weights: retain the same linear SMEM stage while causal
        # selective logits views it as [query_tile, num_heads].
        if cutlass.const_expr(self.uses_request_tiled_q):
            gW = cute.local_tile(
                mW_tma,
                (self.next_n, self.num_heads),
                (None, None),
            )
            tWsW, tWgW = cpasync.tma_partition(
                tma_atom_w,
                0,
                cute.make_layout((1,)),
                cute.group_modes(sW, 0, 2),
                cute.group_modes(gW, 0, 2),
            )
        else:
            tWsW, tWgW = cpasync.tma_partition(
                tma_atom_w,
                0,
                cute.make_layout((1,)),
                cute.group_modes(sW, 0, 2),
                cute.group_modes(mW_tma, 0, 1),
            )

        # Partition Scales: explicit sub_blocks + stages dims.
        # Layout (phys_block_kv, num_sub, stages) K-major with custom strides.
        if cutlass.const_expr(self.uses_contiguous_kv):
            gS = cute.local_tile(mS_tma, (self.block_kv,), coord=(None,))
            sScales_0_for_tma = cute.group_modes(sScales_0, 0, 1)
            sScales_1_for_tma = cute.group_modes(sScales_1, 0, 1)
        else:
            # GMEM: local_tile by phys to match atom's tile size
            gS = cute.local_tile(mS_tma, (self.scale_tma_tile,), coord=(None, None))
            s_tma_view_layout = cute.make_layout(
                (
                    self.phys_block_kv,
                    self.num_blocks_per_mma,
                    self.num_kv_stages,
                ),
                stride=(1, self.phys_block_kv, self.block_kv),
            )
            sScales_0_for_tma = cute.make_tensor(sScales_0.iterator, s_tma_view_layout)
            sScales_1_for_tma = cute.make_tensor(sScales_1.iterator, s_tma_view_layout)
        tSsS_0, tSgS_0 = cpasync.tma_partition(
            tma_atom_s,
            0,
            cute.make_layout(1),
            sScales_0_for_tma,
            gS,
        )
        tSsS_1, tSgS_1 = cpasync.tma_partition(
            tma_atom_s,
            0,
            cute.make_layout(1),
            sScales_1_for_tma,
            gS,
        )

        # MMA fragments
        tCrA_0 = tiled_mma.make_fragment_A(sKV_0)
        tCrA_1 = tiled_mma.make_fragment_A(sKV_1)
        tCrB = tiled_mma.make_fragment_B(sQ)  # shared
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # Staged acc (fp16_gemm_3 pattern): append UMMA stage dim
        # shape: (*acc_shape, STAGE); dynamic index on last dim reduces rank
        us = self.num_umma_stages
        cols = self.num_tmem_alloc_cols
        acc_shape_staged = cute.append(acc_shape, us)
        tCtAcc_fake_staged = tiled_mma.make_fragment_C(acc_shape_staged)

        # TMEM layout info (allocation deferred to UMMA/Math warp branches)
        cols_per_group = cols * us * (32 // self.acc_dtype.width)
        tmem_group_1_offset = cols_per_group
        num_tmem_alloc_cols_total = self.num_tmem_alloc_cols_total

        # Epilogue setup
        c_layout = utils.LayoutEnum.ROW_MAJOR
        epi_sub_mn = (epi_tile[0], self.N // num_epi_subtiles)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            c_layout,
            self.acc_dtype,
            self.acc_dtype,
            epi_sub_mn,
            use_2cta_instrs,
        )

        # ===== SCHEDULER: derive values from early-loaded schedule metadata =====
        end_kv_idx = end_kv_half * NUM_MATH_WG

        # Convert start to KV block units (matching DeepGEMM)
        current_q_idx = start_q
        current_kv_idx = start_kv_half * NUM_MATH_WG

        # ===== COMMON SCHEDULER STATE (before warp branches) =====
        # Each warp role independently maintains its own copy of these
        # variables (like DeepGEMM where each role creates its own scheduler).
        # Pre-fetch first task (current_num_kv loaded early above for latency hiding)
        next_q_idx = current_q_idx
        next_kv_idx = current_kv_idx
        next_num_kv = current_num_kv
        # Sentinel: no previous batch (matches DeepGEMM's q_idx = batch_size)
        q_idx = batch_size
        # While-loop termination flag (matches DeepGEMM's fetch_next_task pattern).
        # True if this CTA has work assigned (start != end in schedule_meta).
        has_work = (current_q_idx != end_q_idx) | (current_kv_idx != end_kv_idx)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ===== WARP-SPECIALIZED EXECUTION =====

        if is_tma_warp_0:
            self._run_tma_warp_0(
                NUM_BLOCKS_PER_MMA,
                NUM_MATH_WG,
                a_mcast_mask,
                b_mcast_mask,
                batch_size,
                block_kv_val,
                end_kv_idx,
                end_q_idx,
                has_work,
                kv_pipeline_0,
                mA_mkl,
                mB_nkl,
                mBlockTable,
                mContextLens,
                mS_tma,
                mTileMeta,
                mW_tma,
                next_kv_idx,
                next_num_kv,
                next_q_idx,
                q_idx,
                q_pipeline,
                sKV_0,
                sQ,
                sScales_0,
                sW,
                tAgA_0,
                tAsA_0,
                tBgB,
                tBsB,
                tSgS_0,
                tSsS_0,
                tWgW,
                tWsW,
                thr_mma,
                tidx,
                tma_atom_a,
                tma_atom_b,
                tma_atom_s,
                tma_atom_w,
            )
        elif is_tma_warp_1:
            self._run_tma_warp_1(
                NUM_BLOCKS_PER_MMA,
                NUM_MATH_WG,
                a_mcast_mask,
                batch_size,
                block_kv_val,
                end_kv_idx,
                end_q_idx,
                has_work,
                kv_pipeline_1,
                mA_mkl,
                mBlockTable,
                mContextLens,
                mS_tma,
                mTileMeta,
                next_kv_idx,
                next_num_kv,
                next_q_idx,
                q_idx,
                sKV_1,
                sScales_1,
                tAgA_1,
                tAsA_1,
                tSgS_1,
                tSsS_1,
                thr_mma,
                tidx,
                tma_atom_a,
                tma_atom_s,
            )
        elif is_umma_warp_0:
            self._run_umma_warp_0(
                NUM_MATH_WG,
                batch_size,
                block_kv_val,
                end_kv_idx,
                end_q_idx,
                has_work,
                is_leader_cta,
                kv_pipeline_0,
                kv_pipeline_1,
                mContextLens,
                next_kv_idx,
                next_num_kv,
                next_q_idx,
                q_idx,
                q_pipeline,
                tCrA_0,
                tCrA_1,
                tCrB,
                tCtAcc_fake_staged,
                tiled_mma,
                tmem,
                tmem_group_1_offset,
                umma_pipeline_0,
                umma_pipeline_1,
            )
        elif is_umma_warp_1:
            self._run_umma_warp_1(
                NUM_MATH_WG,
                batch_size,
                block_kv_val,
                end_kv_idx,
                end_q_idx,
                has_work,
                is_leader_cta,
                kv_pipeline_1,
                mContextLens,
                next_kv_idx,
                next_num_kv,
                next_q_idx,
                q_idx,
                q_pipeline,
                tCrA_1,
                tCrB,
                tCtAcc_fake_staged,
                tiled_mma,
                tmem,
                tmem_group_1_offset,
                umma_pipeline_1,
            )
        elif is_math_warp:
            if cutlass.const_expr(self.has_store_warp):
                self._run_math_warp(
                    NUM_MATH_WG,
                    batch_size,
                    block_kv_val,
                    copy_atom_t2r,
                    end_kv_idx,
                    end_q_idx,
                    epi_sub_mn,
                    has_work,
                    kv_pipeline_math,
                    mContextLens,
                    mIndexCounts,
                    mLogits,
                    mPolicyValues,
                    mTileMeta,
                    next_kv_idx,
                    next_n,
                    next_num_kv,
                    next_q_idx,
                    num_epi_subtiles,
                    num_heads,
                    num_tmem_alloc_cols_total,
                    q_idx,
                    q_pipeline,
                    sScales,
                    sW,
                    tCtAcc_fake_staged,
                    tidx,
                    tmem,
                    tmem_group_1_offset,
                    umma_pipeline_math,
                    warpgroup_idx,
                    sStoreScores,
                    store_pipeline_math,
                )
            else:
                self._run_math_warp(
                    NUM_MATH_WG,
                    batch_size,
                    block_kv_val,
                    copy_atom_t2r,
                    end_kv_idx,
                    end_q_idx,
                    epi_sub_mn,
                    has_work,
                    kv_pipeline_math,
                    mContextLens,
                    mIndexCounts,
                    mLogits,
                    mPolicyValues,
                    mTileMeta,
                    next_kv_idx,
                    next_n,
                    next_num_kv,
                    next_q_idx,
                    num_epi_subtiles,
                    num_heads,
                    num_tmem_alloc_cols_total,
                    q_idx,
                    q_pipeline,
                    sScales,
                    sW,
                    tCtAcc_fake_staged,
                    tidx,
                    tmem,
                    tmem_group_1_offset,
                    umma_pipeline_math,
                    warpgroup_idx,
                )
        elif is_store_warp:
            if cutlass.const_expr(self.has_store_warp):
                self._run_store_warp(
                    NUM_MATH_WG,
                    NUM_STORE_WARPS,
                    batch_size,
                    block_kv_val,
                    end_kv_idx,
                    end_q_idx,
                    has_work,
                    mContextLens,
                    mIndexCounts,
                    mLogits,
                    mPolicyValues,
                    mTileMeta,
                    next_kv_idx,
                    next_n,
                    next_num_kv,
                    next_q_idx,
                    q_idx,
                    sStoreScores_0,
                    sStoreScores_1,
                    store_pipeline_0,
                    store_pipeline_1,
                )
            else:
                self._run_reserved_warp()
        else:
            self._run_reserved_warp()


class _FP8MQARequestTiledKernel(FP8MQALogitsKernel):
    """Paged-ingress adapter for request-owned query tiles."""

    uses_request_tiled_q = True

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        phys_block_kv: int,
        num_sms: int,
        tiling: FP8MQALogitsTiling = SELECTIVE_LOGITS_TILING,
        epi_dtype=cutlass.Float32,
        acc_dtype=cutlass.Float32,
        output_dtype=cutlass.Float32,
    ):
        super().__init__(
            block_kv=tiling.block_kv,
            phys_block_kv=phys_block_kv,
            num_heads=num_heads,
            head_dim=head_dim,
            next_n=tiling.next_n,
            num_sms=num_sms,
            remove_kv_wait_in_epilogue=tiling.remove_kv_wait_in_epilogue,
            early_tmem_copy=tiling.early_tmem_copy,
            smem_subpartition_opt=tiling.smem_subpartition_opt,
            num_epi_subtiles=tiling.num_epi_subtiles,
            epi_dtype=epi_dtype,
            acc_dtype=acc_dtype,
            output_dtype=output_dtype,
            tiling=tiling,
        )

    @cute.jit
    def _store_result_packed_causal_meta(
        self,
        mLogits,
        mIndexCounts,
        mTileMeta,
        mPolicyValues,
        q_idx,
        out_row_base,
        q_valid,
        causal_base,
        row_start,
        row_end,
        row,
        store_kv_pos,
        store_kv_valid,
        result,
        scale_val,
        next_n: cutlass.Constexpr,
    ):
        """Store with prefix and KV-split coordinates already materialized."""
        out_row = out_row_base + row
        should_store = (
            (row < q_valid) & store_kv_valid & (store_kv_pos < causal_base + row + 1)
        )
        # Invalid lanes still construct an address inside the output view;
        # only the final global store is predicated. This removes the per-row
        # divergent causal branch without weakening any predicate.
        safe_out_row = select_i32(should_store, out_row, 0)
        safe_kv_pos = select_i32(should_store, store_kv_pos, 0)
        score_offset = cutlass.Int64(
            cutlass.Uint64(cutlass.Uint32(safe_out_row))
            * cutlass.Uint64(mLogits.layout.stride[0])
        ) + cutlass.Int64(safe_kv_pos)
        score_ptr = mLogits.iterator + score_offset
        if should_store:
            if cutlass.const_expr(self.epi_dtype == cutlass.Float16):
                score_ptr[0] = self.output_dtype(result * Float16(scale_val))
            else:
                score_ptr[0] = self.output_dtype(result * scale_val)


class FP8MQASelectiveLogitsKernel(_FP8MQARequestTiledKernel):
    """Selective-logits kernel with bounded publication.

    ``SAMPLE`` writes a compact sampled-logit plane. Candidate/repair modes
    write the packed uint64 ``(float_bits, logical_index)`` ABI. They can use
    one global logical count or fixed store-warp segment counters consumed by
    the in-place finalizer. Other modes are
    histogram-only or exact-index-only; none materializes dense logits.
    Per-row exact starts/ends live in request-tile ``tile_meta`` columns, while
    threshold/radix-prefix words stay in caller-owned row storage. Publication
    performs no host readback or metadata-copy launch.
    """

    is_selective_logits_kernel = True
    uses_single_umma_producer = True

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        phys_block_kv: int,
        num_sms: int,
        mode: _SelectiveLogitsMode,
        capacity: int,
        radix_shift: int = 0,
        candidate_uses_causal_bounds: bool = True,
        split_kv: int = 1,
        num_count_segments: int = 0,
        count_segment_span: int = 0,
        tiling: FP8MQALogitsTiling = SELECTIVE_LOGITS_TILING,
    ):
        if not isinstance(mode, _SelectiveLogitsMode):
            raise ValueError("mode must be a _SelectiveLogitsMode")
        if capacity <= 0:
            raise ValueError("selective-logits capacity must be positive")
        if mode == _SelectiveLogitsMode.RADIX_HISTOGRAM and radix_shift not in (
            0,
            8,
            16,
            24,
        ):
            raise ValueError("radix_shift must be one of {0, 8, 16, 24}")
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            phys_block_kv=phys_block_kv,
            num_sms=num_sms,
            tiling=tiling,
        )
        self.mode = mode
        self.capacity = capacity
        self.radix_shift = radix_shift
        self.candidate_uses_causal_bounds = candidate_uses_causal_bounds
        self.split_kv = split_kv
        self.num_count_segments = num_count_segments
        self.count_segment_span = count_segment_span
        self.has_store_warp = (
            mode == _SelectiveLogitsMode.CANDIDATE and num_count_segments > 0
        )
        if self.has_store_warp:
            if tiling.block_kv != 128:
                raise ValueError(
                    "dedicated candidate store warp requires KV128 compute tiles"
                )
            if num_count_segments not in (4, 8, 12, 16, 32, 64, 128):
                raise ValueError(
                    "dedicated candidate store warp requires "
                    "4, 8, 12, 16, 32, 64, or 128 segments"
                )
            if split_kv not in _SPLIT_KV_CHOICES:
                raise ValueError("candidate split_kv must be 1, 2, 4, 8, 16, 32, or 64")
            striped_split_layout = num_count_segments == 4 * split_kv
            pooled_split_layout = (
                split_kv > _MAX_STRIPED_SPLIT_KV and num_count_segments == split_kv
            )
            lane_local_layout = split_kv <= 4 and num_count_segments == 32 * split_kv
            valid_split_layout = (
                striped_split_layout or pooled_split_layout or lane_local_layout
            )
            if split_kv > 1 and not valid_split_layout:
                required_layout = (
                    f"{split_kv}"
                    if split_kv > _MAX_STRIPED_SPLIT_KV
                    else f"{4 * split_kv}"
                )
                if split_kv <= 4:
                    required_layout += f" or {32 * split_kv}"
                raise ValueError(
                    f"split_kv={split_kv} requires {required_layout} candidate segments"
                )
            # Store coordinates are request-local shared-score coordinates.
            # KV128 tiles and KV32 segments keep every vector origin
            # four-aligned. Store warps without a mapped query row still
            # participate in every pipeline wait and release.
            assert tiling.block_kv % 4 == 0
            if num_count_segments == 4:
                assert count_segment_span % 4 == 0
            self.threads_per_cta = 512
            self.math_wg_registers = 160 if num_heads == 64 else 168
            self.specialized_wg_registers = 24
            self.store_wg_registers = 168 if num_heads == 64 else 152

    @cute.jit
    def __call__(
        self,
        kv_fused: cute.Tensor,
        q: cute.Tensor,
        weights: cute.Tensor,
        values: cute.Tensor,
        block_table: cute.Tensor,
        index_counts: cute.Tensor,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        tile_meta: cute.Tensor,
        policy_values: cute.Tensor,
        num_phys_blocks: cutlass.Int32,
        num_tiles: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._launch(
            kv_fused,
            q,
            weights,
            values,
            block_table,
            index_counts,
            context_lens,
            schedule_meta,
            block_table,
            tile_meta,
            policy_values,
            num_phys_blocks,
            num_tiles,
            stream,
        )

    @cute.jit
    def _store_candidate_warp_row_pair(
        self,
        row_logits_addr,
        q_valid,
        row,
        row_logical_end,
        logical_base_0,
        logical_base_1,
        threshold,
        cursor,
        segment,
        score_0,
        score_1,
    ):
        """Append both ordered math streams for one store segment."""
        row_valid = row < q_valid
        selected_0 = (
            row_valid & (logical_base_0 < row_logical_end) & (score_0 >= threshold)
        )
        selected_1 = (
            row_valid & (logical_base_1 < row_logical_end) & (score_1 >= threshold)
        )
        selected_mask_0 = cute.arch.vote_ballot_sync(selected_0)
        selected_mask_1 = cute.arch.vote_ballot_sync(selected_1)
        lane = cute.arch.lane_idx()
        lane_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
        lane_rank_0 = cutlass.Int32(cute.arch.popc(selected_mask_0 & lane_mask))
        lane_rank_1 = cutlass.Int32(cute.arch.popc(selected_mask_1 & lane_mask))
        warp_count_0 = cutlass.Int32(cute.arch.popc(selected_mask_0))
        warp_count_1 = cutlass.Int32(cute.arch.popc(selected_mask_1))
        segment_capacity = cutlass.Int32(self.capacity // self.num_count_segments)
        segment_begin = segment * segment_capacity
        segment_end = (segment + 1) * segment_capacity
        slot_0 = segment_begin + cursor + lane_rank_0
        cursor_1 = cursor + warp_count_0
        slot_1 = segment_begin + cursor_1 + lane_rank_1
        packed_0 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_0))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_0))
        packed_1 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_1))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_1))
        store_global_u64_if(
            selected_0 & (slot_0 < segment_end),
            row_logits_addr + cutlass.Int64(slot_0) * cutlass.Int64(8),
            packed_0,
        )
        store_global_u64_if(
            selected_1 & (slot_1 < segment_end),
            row_logits_addr + cutlass.Int64(slot_1) * cutlass.Int64(8),
            packed_1,
        )
        return cursor_1 + warp_count_1

    @cute.jit
    def _store_candidate_warp_row_pair_full(
        self,
        row_logits_addr,
        logical_base_0,
        logical_base_1,
        threshold,
        cursor,
        segment,
        score_0,
        score_1,
    ):
        """Append one full KV pair without row or logical-bound predicates."""
        selected_0 = score_0 >= threshold
        selected_1 = score_1 >= threshold
        selected_mask_0 = cute.arch.vote_ballot_sync(selected_0)
        selected_mask_1 = cute.arch.vote_ballot_sync(selected_1)
        lane = cute.arch.lane_idx()
        lane_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
        lane_rank_0 = cutlass.Int32(cute.arch.popc(selected_mask_0 & lane_mask))
        lane_rank_1 = cutlass.Int32(cute.arch.popc(selected_mask_1 & lane_mask))
        warp_count_0 = cutlass.Int32(cute.arch.popc(selected_mask_0))
        warp_count_1 = cutlass.Int32(cute.arch.popc(selected_mask_1))
        segment_capacity = cutlass.Int32(self.capacity // self.num_count_segments)
        segment_begin = segment * segment_capacity
        segment_end = (segment + 1) * segment_capacity
        slot_0 = segment_begin + cursor + lane_rank_0
        cursor_1 = cursor + warp_count_0
        slot_1 = segment_begin + cursor_1 + lane_rank_1
        packed_0 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_0))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_0))
        packed_1 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_1))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_1))
        store_global_u64_if(
            selected_0 & (slot_0 < segment_end),
            row_logits_addr + cutlass.Int64(slot_0) * cutlass.Int64(8),
            packed_0,
        )
        store_global_u64_if(
            selected_1 & (slot_1 < segment_end),
            row_logits_addr + cutlass.Int64(slot_1) * cutlass.Int64(8),
            packed_1,
        )
        return cursor_1 + warp_count_1

    @cute.jit
    def _store_candidate_lane_row_pair(
        self,
        row_logits_addr,
        q_valid,
        row,
        row_logical_end,
        logical_base_0,
        logical_base_1,
        threshold,
        cursor,
        segment,
        score_0,
        score_1,
    ):
        """Append one exact KV pair into a fixed store-lane slice."""
        row_valid = row < q_valid
        selected_0 = (
            row_valid & (logical_base_0 < row_logical_end) & (score_0 >= threshold)
        )
        selected_1 = (
            row_valid & (logical_base_1 < row_logical_end) & (score_1 >= threshold)
        )
        segment_capacity = cutlass.Int32(self.capacity // self.num_count_segments)
        segment_begin = segment * segment_capacity
        segment_end = segment_begin + segment_capacity
        slot_0 = segment_begin + cursor
        cursor_1 = cursor + cutlass.Int32(selected_0)
        slot_1 = segment_begin + cursor_1
        cursor_2 = cursor_1 + cutlass.Int32(selected_1)
        packed_0 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_0))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_0))
        packed_1 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_1))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_1))
        store_global_u64_if(
            selected_0 & (slot_0 < segment_end),
            row_logits_addr + cutlass.Int64(slot_0) * cutlass.Int64(8),
            packed_0,
        )
        store_global_u64_if(
            selected_1 & (slot_1 < segment_end),
            row_logits_addr + cutlass.Int64(slot_1) * cutlass.Int64(8),
            packed_1,
        )
        return cursor_2

    @cute.jit
    def _store_candidate_lane_row_pair_full(
        self,
        row_logits_addr,
        logical_base_0,
        logical_base_1,
        threshold,
        cursor,
        segment,
        score_0,
        score_1,
    ):
        """Append one full KV pair into a fixed store-lane slice."""
        selected_0 = score_0 >= threshold
        selected_1 = score_1 >= threshold
        segment_capacity = cutlass.Int32(self.capacity // self.num_count_segments)
        segment_begin = segment * segment_capacity
        segment_end = segment_begin + segment_capacity
        slot_0 = segment_begin + cursor
        cursor_1 = cursor + cutlass.Int32(selected_0)
        slot_1 = segment_begin + cursor_1
        cursor_2 = cursor_1 + cutlass.Int32(selected_1)
        packed_0 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_0))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_0))
        packed_1 = (
            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score_1))) << 32
        ) | cutlass.Uint64(cutlass.Uint32(logical_base_1))
        store_global_u64_if(
            selected_0 & (slot_0 < segment_end),
            row_logits_addr + cutlass.Int64(slot_0) * cutlass.Int64(8),
            packed_0,
        )
        store_global_u64_if(
            selected_1 & (slot_1 < segment_end),
            row_logits_addr + cutlass.Int64(slot_1) * cutlass.Int64(8),
            packed_1,
        )
        return cursor_2

    @cute.jit
    def _ordered_key(self, score):
        bits = cutlass.Uint32(fp32_to_i32_bits(score))
        sign_mask = cutlass.Uint32(0x80000000)
        if cutlass.Int32(bits) < 0:
            sign_mask = cutlass.Uint32(0xFFFFFFFF)
        return bits ^ sign_mask

    @cute.jit
    def _store_result_packed_causal_meta(
        self,
        mLogits,
        mIndexCounts,
        mTileMeta,
        mPolicyValues,
        q_idx,
        out_row_base,
        q_valid,
        causal_base,
        row_start,
        row_end,
        row,
        store_kv_pos,
        store_kv_valid,
        result,
        scale_val,
        next_n: cutlass.Constexpr,
    ):
        out_row = out_row_base + row
        if cutlass.const_expr(
            self.mode == _SelectiveLogitsMode.CANDIDATE
            and self.candidate_uses_causal_bounds
        ):
            # K4 rows are request-owned causal tiles; retain the upstream
            # hoisted base and avoid two bound loads in the hot path.
            valid = (
                (row < q_valid)
                & store_kv_valid
                & (store_kv_pos < causal_base + row + 1)
            )
        else:
            valid = (
                (row < q_valid)
                & store_kv_valid
                & (store_kv_pos >= row_start)
                & (store_kv_pos < row_end)
            )
        score = cutlass.Float32(result) * cutlass.Float32(scale_val)

        if cutlass.const_expr(self.mode == _SelectiveLogitsMode.SAMPLE):
            if valid & (store_kv_pos < self.capacity):
                mLogits[(out_row, store_kv_pos)] = score
        elif cutlass.const_expr(self.mode == _SelectiveLogitsMode.RADIX_HISTOGRAM):
            ordered = self._ordered_key(score)
            prefix = cutlass.Uint32(mPolicyValues[out_row])
            prefix_matches = cutlass.Boolean(True)
            if cutlass.const_expr(self.radix_shift != 24):
                prefix_matches = (ordered >> (self.radix_shift + 8)) == prefix
            if valid & prefix_matches:
                radix_bin = cutlass.Int32((ordered >> self.radix_shift) & 0xFF)
                cute.arch.atomic_add(
                    cute.domain_offset((out_row, radix_bin), mIndexCounts).iterator,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
        else:
            policy_word = mPolicyValues[out_row]
            selected = cutlass.Boolean(False)
            if cutlass.const_expr(
                self.mode == _SelectiveLogitsMode.CANDIDATE
                or self.mode == _SelectiveLogitsMode.REPAIR
            ):
                selected = valid & (score >= i32_bits_to_fp32(policy_word))
            else:
                score_key = self._ordered_key(score)
                threshold_key = self._ordered_key(i32_bits_to_fp32(policy_word))
                if cutlass.const_expr(self.mode == _SelectiveLogitsMode.EXACT_GREATER):
                    selected = valid & (score_key > threshold_key)
                else:
                    selected = valid & (score_key == threshold_key)

            segment_begin = cutlass.Int32(0)
            segment_end = cutlass.Int32(self.capacity)
            slot = cutlass.Int32(0)
            if cutlass.const_expr(
                self.num_count_segments > 0
                and (
                    self.mode == _SelectiveLogitsMode.CANDIDATE
                    or self.mode == _SelectiveLogitsMode.REPAIR
                )
            ):
                # Match the segmented producer's fixed store-warp
                # segments: a 384-token split is striped across 12 warps in
                # consecutive 32-token lanes.  The retained set at overflow is
                # part of the packed consumer ABI, so contiguous KV ranges are
                # not interchangeable here.
                segment = (store_kv_pos // self.count_segment_span) % cutlass.Int32(
                    self.num_count_segments
                )
                segment_begin = segment * self.capacity // self.num_count_segments
                segment_end = (segment + 1) * self.capacity // self.num_count_segments
                if selected:
                    slot = segment_begin + cute.arch.atomic_add(
                        cute.domain_offset(
                            (out_row, segment + 1), mIndexCounts
                        ).iterator,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
            else:
                selected_mask = cute.arch.vote_ballot_sync(selected)
                warp_count = cutlass.Int32(cute.arch.popc(selected_mask))
                lane = cute.arch.lane_idx()
                lane_mask = (
                    cutlass.Uint32(1) << cutlass.Uint32(lane)
                ) - cutlass.Uint32(1)
                lane_rank = cutlass.Int32(cute.arch.popc(selected_mask & lane_mask))
                warp_base = cutlass.Int32(0)
                if lane == cutlass.Int32(0) and warp_count != cutlass.Int32(0):
                    warp_base = cute.arch.atomic_add(
                        cute.domain_offset((out_row, 0), mIndexCounts).iterator,
                        warp_count,
                        sem="relaxed",
                        scope="gpu",
                    )
                warp_base = cute.arch.shuffle_sync(warp_base, cutlass.Int32(0))
                slot = warp_base + lane_rank
            if selected:
                if slot < segment_end:
                    # The packed consumer uses request-local logical indices.
                    logical_index = store_kv_pos
                    if cutlass.const_expr(
                        self.mode == _SelectiveLogitsMode.CANDIDATE
                        or self.mode == _SelectiveLogitsMode.REPAIR
                    ):
                        packed = (
                            cutlass.Uint64(cutlass.Uint32(fp32_to_i32_bits(score)))
                            << 32
                        ) | cutlass.Uint64(cutlass.Uint32(logical_index))
                        mLogits[(out_row, slot)] = cutlass.Int64(packed)
                    else:
                        mLogits[(out_row, slot)] = logical_index
