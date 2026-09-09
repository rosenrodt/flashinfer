# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Prepared API for selective FP8 MQA logits publication.

This module provides the selective-logits producer: a request-tiled engine
consumes FlashInfer's fused paged FP8 KV representation and publishes only
bounded samples, candidates, radix histograms, or exact-completion indices.
"""

from __future__ import annotations

import bisect
import importlib.util
import math
import os
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Literal, Tuple

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.attn_scores import (
    fp8_paged_mqa_topk_trace_dispatch,
)

try:  # pragma: no cover - exercised by whichever branch this torch provides
    from torch._C import _cuda_getCurrentRawStream as _raw_stream
except ImportError:  # pragma: no cover
    _raw_stream = None

_CUTE_DSL_AVAILABLE = (
    importlib.util.find_spec("cutlass") is not None
    and importlib.util.find_spec("cutlass.cute") is not None
)

_MAX_QUERY_TILE = 16
_MIN_SHORT_QUERY_LOGICAL_N = 32
_COUNT_SEGMENTS = (4, 8, 12, 16, 32, 64, 128)
_METADATA_HEADER_COLUMNS = 4
_FP8_MMA_INST_K = 32
_EPI_UNROLL = 4
_COMPUTE_BLOCK_KV = 128
_MAX_BLOCKS_PER_MMA = 4
_DEFAULT_TOP_K = 512
_MAX_TOP_K = 512
_SAMPLE_CAPACITY = 3840
_CANDIDATE_CAPACITY = 6144
_REPAIR_CAPACITY = 8192
_TARGET_CANDIDATES = 2048
_SPLIT_KV_CHOICES = (1, 2, 4, 8, 16, 32, 64)
_MAX_STRIPED_SPLIT_KV = 8

_CandidatePublication = Literal["warp_striped", "warp_pooled", "lane_local"]
_WARP_STRIPED_PUBLICATION: Literal["warp_striped"] = "warp_striped"
_WARP_POOLED_PUBLICATION: Literal["warp_pooled"] = "warp_pooled"


@dataclass(frozen=True)
class _CandidateSchedule:
    """Shape-selected candidate kernel topology."""

    query_tile: int
    logical_n: int
    num_epi_subtiles: int
    num_umma_stages: int
    split_kv: int
    publication: _CandidatePublication
    count_segments: int


def _candidate_count_segments(split_kv: int) -> int:
    """Return striped low-split or pooled high-split publisher segments."""
    return 4 * split_kv if split_kv <= _MAX_STRIPED_SPLIT_KV else split_kv


def _select_candidate_publication(
    split_kv: int,
    *,
    lane_local: bool = False,
) -> tuple[_CandidatePublication, int]:
    """Choose the production publisher layout for a split count.

    Prefill uses lane-local publication through four splits to preserve
    independent candidate cursors in the long-running score kernel. Decode uses
    compact warp-ranked stripes, then pooled segments above eight splits.
    """
    if lane_local:
        if split_kv > 4:
            raise ValueError("lane-local publication supports at most four splits")
        return "lane_local", 32 * split_kv
    if split_kv <= _MAX_STRIPED_SPLIT_KV:
        return _WARP_STRIPED_PUBLICATION, _candidate_count_segments(split_kv)
    return _WARP_POOLED_PUBLICATION, _candidate_count_segments(split_kv)


class _SelectiveLogitsMode(IntEnum):
    """Output mode for selective FP8 MQA logits production.

    ``SAMPLE`` writes FP32 scores. ``CANDIDATE`` and ``REPAIR`` write packed
    score/index records. ``RADIX_HISTOGRAM`` and the two ``EXACT_*`` modes
    provide device-side exact TopK completion primitives.
    """

    SAMPLE = 0
    CANDIDATE = 1
    REPAIR = 2
    RADIX_HISTOGRAM = 3
    EXACT_GREATER = 4
    EXACT_EQUAL = 5


if _CUTE_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack


@dataclass(frozen=True)
class _PreparedFP8MQASelectiveLogits:
    """Address-stable metadata and specialization for selective FP8 logits.

    Instances are created by :func:`_prepare_fp8_mqa_selective_logits`. Their
    tensor addresses are retained by compiled CuTe launch arguments; use
    :func:`_refresh_fp8_mqa_selective_logits` to update values in place rather
    than replacing these tensors during CUDA graph replay.
    """

    context_lens: torch.Tensor
    tile_meta: torch.Tensor
    schedule_meta: torch.Tensor
    schedule_prefix: torch.Tensor
    rows: int
    num_phys_blocks: int
    page_size: int
    query_tile: int
    num_epi_subtiles: int
    tiles: int
    batches: int
    num_sms: int
    kv_chunk_size: int
    split_kv: int
    causal_bounds: bool
    row_end_base: int
    num_heads: int
    head_dim: int
    num_umma_stages: int = 2
    block_table_columns: int = 0
    _launches: dict[tuple[Any, ...], Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )


@dataclass
class _CachedSelectiveLogitsLaunch:
    identity: tuple[Any, ...]
    tensors: tuple[torch.Tensor, ...]
    compiled: Any
    args: tuple[Any, ...]
    count_view: torch.Tensor | None


@dataclass(frozen=True)
class _CachedFinalizeLaunch:
    identity: tuple[Any, ...]
    tensors: tuple[torch.Tensor, ...]
    compiled: Any
    args: tuple[Any, ...]


def _selective_logits_launch_identity(
    tensors: tuple[torch.Tensor, ...],
    capacity: int,
) -> tuple[Any, ...]:
    """Identify an address- and layout-stable compiled launch binding."""
    return (
        capacity,
        tuple(
            (
                id(tensor),
                tensor.data_ptr(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.dtype,
                tensor.device,
            )
            for tensor in tensors
        ),
    )


def _selective_logits_tensor_specs(
    tensors: tuple[torch.Tensor, ...],
) -> tuple[Any, ...]:
    """Return the non-address tensor contract used for launch rebinding."""
    return tuple(
        (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )
        for tensor in tensors
    )


_METADATA_COMPILED: dict[Tuple[int, ...], Any] = {}
_COMPACT_METADATA_COMPILED: dict[Tuple[int, ...], Any] = {}
_SELECTIVE_LOGITS_COMPILED: dict[Tuple[Any, ...], Any] = {}
_FINALIZE_COMPILED: dict[Tuple[int, ...], Any] = {}
_RADIX_SELECT_COMPILED: dict[Tuple[int, ...], Any] = {}
_COMBINE_EXACT_COMPILED: dict[Tuple[int, ...], Any] = {}
_SAMPLE_PACK_COMPILED: dict[Tuple[int, ...], Any] = {}
_THRESHOLD_FLOOR_COMPILED: dict[Tuple[int, ...], Any] = {}
_REPAIR_THRESHOLD_COMPILED: dict[Tuple[int, ...], Any] = {}
_PACKED_SCORES_COMPILED: dict[Tuple[int, ...], Any] = {}
_PACKED_TOPK_COMPILED: dict[Tuple[int, ...], Any] = {}
_PACKED_GATHER_COMPILED: dict[Tuple[int, ...], Any] = {}
_REPAIR_SCATTER_COMPILED: dict[Tuple[int, ...], Any] = {}
_REPAIR_RESOLVE_COMPILED: dict[Tuple[int, ...], Any] = {}
_REPAIR_ROW_COMPACT_COMPILED: dict[Tuple[int, ...], Any] = {}
_REPAIR_ROW_PACK_COMPILED: dict[Tuple[int, ...], Any] = {}
_COMPACT_REPAIR_RESOLVE_COMPILED: dict[Tuple[int, ...], Any] = {}
_COMPACT_COMBINE_EXACT_COMPILED: dict[Tuple[int, ...], Any] = {}
_SAMPLE_THRESHOLD_COMPILED: dict[Tuple[int, ...], Any] = {}
_STREAM_CACHE: dict[Tuple[int, int], Any] = {}


def _byte_range(tensor: torch.Tensor) -> Tuple[int, int]:
    """Return the occupied positive-stride byte interval for alias checks."""
    elements = 1
    if tensor.numel() != 0:
        elements += sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
    begin = tensor.data_ptr()
    return begin, begin + elements * tensor.element_size()


def _overlap(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _select_candidate_split_kv(
    q_lengths: list[int],
    kv_lengths: list[int],
    *,
    query_tile: int,
    num_sms: int,
) -> int:
    """Choose the smallest production split with the best SM-wave utilization."""
    best_split = 1
    best_work_items = 0
    best_wave_slots = 1
    base_work_items = sum(
        math.ceil(q_length / query_tile)
        for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True)
        if q_length > 0 and kv_length > 0
    )
    # Higher split counts are useful only while request tiles underfill one
    # device wave. Once the unsplit grid fills a wave, keep the established
    # compact-layout choice and avoid multiplying finalization work.
    decode_shaped = max(q_lengths, default=0) <= query_tile
    split_candidates: tuple[int, ...]
    if decode_shaped and base_work_items == 1:
        split_candidates = _SPLIT_KV_CHOICES
    elif decode_shaped and base_work_items < num_sms:
        split_candidates = _SPLIT_KV_CHOICES[:4]
    else:
        split_candidates = _SPLIT_KV_CHOICES[:2]
    for split_kv in split_candidates:
        work_items = sum(
            math.ceil(q_length / query_tile)
            * min(split_kv, math.ceil(kv_length / (2 * _COMPUTE_BLOCK_KV)))
            for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True)
            if q_length > 0 and kv_length > 0
        )
        wave_slots = math.ceil(work_items / num_sms) * num_sms
        if work_items * best_wave_slots > best_work_items * wave_slots:
            best_split = split_kv
            best_work_items = work_items
            best_wave_slots = wave_slots
    return best_split


def _require_aligned(tensors: tuple[torch.Tensor, ...]) -> None:
    if any(tensor.data_ptr() % 16 != 0 for tensor in tensors):
        raise ValueError("selective-logits tensor bases must be 16-byte aligned")


def _validate_specialization(
    device: torch.device,
    num_heads: int,
    head_dim: int,
    *,
    max_query_tile: int = _MAX_QUERY_TILE,
) -> Any:
    """Choose the largest legal request tile for the paged FP8 kernel shape."""
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError("num_heads must be a positive integer")
    if not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError("head_dim must be a positive integer")
    if head_dim % _FP8_MMA_INST_K:
        raise ValueError(
            f"head_dim must be a multiple of {_FP8_MMA_INST_K}; got {head_dim}"
        )
    if not isinstance(max_query_tile, int) or not 0 < max_query_tile <= _MAX_QUERY_TILE:
        raise ValueError(f"max_query_tile must be in [1, {_MAX_QUERY_TILE}]")
    from .kernels.fp8_paged_mqa_logits import SELECTIVE_LOGITS_TILING

    properties = torch.cuda.get_device_properties(device)
    smem_limit = max(
        int(getattr(properties, "shared_memory_per_block_optin", 0)),
        int(getattr(properties, "sharedMemPerBlockOptin", 0)),
        int(getattr(properties, "shared_memory_per_block", 0)),
    )
    # Preserve the tuned two-stage policy wherever it fits. Fall back to one
    # UMMA stage only for shapes in the full paged kernel's N=256 envelope.
    for num_umma_stages in (SELECTIVE_LOGITS_TILING.num_umma_stages, 1):
        tmem_limit = 512 // (2 * num_umma_stages)
        for query_tile in range(max_query_tile, 0, -1):
            logical_n = query_tile * num_heads
            if logical_n < 8 or logical_n > tmem_limit or logical_n % 8:
                continue
            num_epi_subtiles = next(
                (
                    subtiles
                    for subtiles in (8, 4, 2, 1)
                    if logical_n % subtiles == 0
                    and (logical_n // subtiles) % _EPI_UNROLL == 0
                    and (subtiles % query_tile == 0 or query_tile % subtiles == 0)
                ),
                None,
            )
            if num_epi_subtiles is None:
                continue
            w_stage_bytes = ((logical_n * 4 + 127) // 128) * 128
            kv_scale_bytes = (
                2
                * (
                    SELECTIVE_LOGITS_TILING.block_kv * head_dim
                    + SELECTIVE_LOGITS_TILING.block_kv * 4
                )
                * SELECTIVE_LOGITS_TILING.num_kv_stages
            )
            q_weight_bytes = (
                logical_n * head_dim + w_stage_bytes
            ) * SELECTIVE_LOGITS_TILING.num_q_stages
            store_score_bytes = (
                query_tile * SELECTIVE_LOGITS_TILING.block_kv * 2 * 2 * 4
            )
            required_smem = kv_scale_bytes + q_weight_bytes + store_score_bytes + 1024
            if required_smem <= smem_limit:
                return replace(
                    SELECTIVE_LOGITS_TILING,
                    next_n=query_tile,
                    num_epi_subtiles=num_epi_subtiles,
                    num_umma_stages=num_umma_stages,
                )
    raise ValueError(
        f"no legal selective FP8 query tile for num_heads={num_heads}, "
        f"head_dim={head_dim} within N<={tmem_limit} and {smem_limit} bytes SMEM"
    )


def _select_request_tiling(
    device: torch.device,
    num_heads: int,
    head_dim: int,
    q_lengths: list[int],
) -> Any:
    """Select a request tile without padding short-query work to Q16."""
    min_query_tile = math.ceil(_MIN_SHORT_QUERY_LOGICAL_N / num_heads)
    max_query_tile = min(
        _MAX_QUERY_TILE,
        max(max(q_lengths, default=1), min_query_tile),
    )
    return _validate_specialization(
        device,
        num_heads,
        head_dim,
        max_query_tile=max_query_tile,
    )


def _select_candidate_schedule(
    device: torch.device,
    num_heads: int,
    head_dim: int,
    q_lengths: list[int],
    kv_lengths: list[int],
    *,
    num_sms: int,
) -> _CandidateSchedule:
    """Choose all candidate-kernel scheduling knobs from one cost boundary."""
    tiling = _select_request_tiling(device, num_heads, head_dim, q_lengths)
    split_kv = _select_candidate_split_kv(
        q_lengths,
        kv_lengths,
        query_tile=tiling.next_n,
        num_sms=num_sms,
    )
    publication, count_segments = _select_candidate_publication(
        split_kv,
        lane_local=max(q_lengths, default=0) > tiling.next_n,
    )
    return _CandidateSchedule(
        query_tile=tiling.next_n,
        logical_n=tiling.next_n * num_heads,
        num_epi_subtiles=tiling.num_epi_subtiles,
        num_umma_stages=tiling.num_umma_stages,
        split_kv=split_kv,
        publication=publication,
        count_segments=count_segments,
    )


def _candidate_schedule_matches_prepared(
    schedule: _CandidateSchedule,
    prepared: _PreparedFP8MQASelectiveLogits,
) -> bool:
    """Check that preparation preserved every compiled topology knob."""
    return (
        schedule.query_tile == prepared.query_tile
        and schedule.logical_n == prepared.query_tile * prepared.num_heads
        and schedule.num_epi_subtiles == prepared.num_epi_subtiles
        and schedule.num_umma_stages == prepared.num_umma_stages
        and schedule.split_kv == prepared.split_kv
    )


def _validate_prepared(
    prepared: _PreparedFP8MQASelectiveLogits,
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Validate the address-stable metadata ABI before entering CuTe."""
    max_tiles = prepared.tiles
    metadata = (
        prepared.context_lens,
        prepared.tile_meta,
        prepared.schedule_meta,
        prepared.schedule_prefix,
    )
    if (
        prepared.rows != q.shape[0]
        or prepared.num_phys_blocks != kv_fused.shape[0]
        or prepared.page_size != kv_fused.shape[1]
        or prepared.batches != block_table.shape[0]
    ):
        raise ValueError("prepared metadata is stale for the supplied q/KV layout")
    if torch.cuda.get_device_capability(q.device)[0] != 10:
        raise ValueError("selective FP8 logits requires an SM100-class device")
    if (
        prepared.num_sms
        > torch.cuda.get_device_properties(q.device).multi_processor_count
    ):
        raise ValueError("prepared metadata is stale for the current CUDA device")
    if any(
        tensor.device != q.device or not tensor.is_contiguous() for tensor in metadata
    ):
        raise ValueError("prepared metadata must be contiguous on the input device")
    if (
        prepared.context_lens.dtype != torch.int32
        or prepared.context_lens.shape != (max_tiles,)
        or prepared.tile_meta.dtype != torch.int32
        or prepared.tile_meta.shape
        != (max_tiles, _METADATA_HEADER_COLUMNS + 2 * prepared.query_tile)
        or prepared.schedule_meta.dtype != torch.int32
        or prepared.schedule_meta.shape != (prepared.num_sms + 1, 2)
        or prepared.schedule_prefix.dtype != torch.int32
        or prepared.schedule_prefix.shape != (max_tiles,)
    ):
        raise ValueError("prepared metadata has an invalid selective-logits ABI")
    _require_aligned(metadata)
    return metadata


def _current_stream(device: torch.device):
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    raw_stream = (
        _raw_stream(device_index)
        if _raw_stream is not None
        else torch.cuda.current_stream(device_index).cuda_stream
    )
    key = (device_index, raw_stream)
    stream = _STREAM_CACHE.get(key)
    if stream is None:
        stream = cuda.CUstream(raw_stream)
        _STREAM_CACHE[key] = stream
    return stream


def _require_cute() -> None:
    if not _CUTE_DSL_AVAILABLE:
        raise RuntimeError("selective FP8 logits requires nvidia-cutlass-dsl")


def _validate_prefixes(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> None:
    tensors = (q, kv_fused, weights, cu_q, cu_kv, block_table)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("selective-logits inputs must be torch tensors")
    if q.dtype != torch.float8_e4m3fn:
        raise ValueError("q must be float8_e4m3fn")
    if kv_fused.dtype != torch.uint8:
        raise ValueError("kv_fused must be uint8")
    if weights.dtype != torch.float32:
        raise ValueError("weights must be float32")
    if tuple(q.shape[1:]) != (num_heads, head_dim) or q.dim() != 3:
        raise ValueError(f"q must have shape [Q, {num_heads}, {head_dim}]")
    if (
        kv_fused.dim() != 4
        or kv_fused.shape[2] != 1
        or kv_fused.shape[3] != head_dim + 4
    ):
        raise ValueError(
            f"kv_fused must have shape [num_blocks, page_size, 1, {head_dim + 4}]"
        )
    page_size = kv_fused.shape[1]
    if (
        page_size <= 0
        or _COMPUTE_BLOCK_KV % page_size
        or _COMPUTE_BLOCK_KV // page_size > _MAX_BLOCKS_PER_MMA
    ):
        raise ValueError("page_size must be one of 32, 64, or 128")
    if tuple(weights.shape) != (q.shape[0], num_heads):
        raise ValueError(f"weights must have shape [Q, {num_heads}]")
    if cu_q.dtype != torch.int32 or cu_kv.dtype != torch.int32:
        raise ValueError("cu_q and cu_kv must be int32")
    if cu_q.dim() != 1 or cu_q.shape != cu_kv.shape or cu_q.shape[0] < 2:
        raise ValueError("cu_q and cu_kv must have matching shape [B+1]")
    if (
        block_table.dtype != torch.int32
        or block_table.dim() != 2
        or block_table.shape[0] != cu_q.shape[0] - 1
    ):
        raise ValueError("block_table must be int32 [B, max_pages_per_request]")
    device = q.device
    if not q.is_cuda or any(tensor.device != device for tensor in tensors[1:]):
        raise ValueError(
            "all selective-logits inputs must be CUDA tensors on one device"
        )
    if torch.cuda.get_device_capability(device)[0] != 10:
        raise ValueError("selective FP8 logits requires an SM100-class device")
    kv_flat = kv_fused.flatten(1)
    if not (
        q.is_contiguous()
        and weights.is_contiguous()
        and cu_q.is_contiguous()
        and cu_kv.is_contiguous()
        and block_table.is_contiguous()
        and kv_flat.data_ptr() == kv_fused.data_ptr()
        and kv_flat.stride(1) == 1
    ):
        raise ValueError(
            "selective-logits inputs require contiguous q/weights/prefixes/page table "
            "and contiguous bytes within each KV page"
        )
    _require_aligned(tensors)


def _validate_device_metadata_values(
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    *,
    rows: int,
    page_size: int,
    num_phys_blocks: int,
    block_table: torch.Tensor | None = None,
    block_table_columns: int = 0,
    causal_bounds: bool,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
    row_end_base: int = 0,
) -> None:
    """Perform opt-in synchronized checks for trusted device metadata."""
    if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") in ("0", ""):
        return
    if torch.cuda.is_current_stream_capturing():
        return

    q_prefix = cu_q.tolist()
    kv_prefix = cu_kv.tolist()
    if q_prefix[0] != 0 or kv_prefix[0] != 0:
        raise ValueError("cu_q and cu_kv must start at zero")
    if q_prefix[-1] != rows:
        raise ValueError(f"cu_q[-1] must equal q.shape[0] ({rows})")
    if any(left > right for left, right in zip(q_prefix, q_prefix[1:], strict=False)):
        raise ValueError("cu_q must be monotonically nondecreasing")
    if any(left > right for left, right in zip(kv_prefix, kv_prefix[1:], strict=False)):
        raise ValueError("cu_kv must be monotonically nondecreasing")

    pages_per_tile = _COMPUTE_BLOCK_KV // page_size
    required_columns = 0
    for batch, (q_begin, q_end, kv_begin, kv_end) in enumerate(
        zip(q_prefix, q_prefix[1:], kv_prefix, kv_prefix[1:], strict=False)
    ):
        q_len = q_end - q_begin
        kv_len = kv_end - kv_begin
        if causal_bounds and q_len > kv_len:
            raise ValueError(
                f"causal request {batch} has q length {q_len} greater than "
                f"kv length {kv_len}"
            )
        need = math.ceil(kv_len / _COMPUTE_BLOCK_KV) * pages_per_tile
        required_columns = max(required_columns, need)

    available_columns = (
        block_table.shape[1] if block_table is not None else block_table_columns
    )
    if available_columns < required_columns:
        raise ValueError(
            f"block_table has {available_columns} columns but selective logits "
            f"requires at least {required_columns} for these KV prefixes"
        )

    if block_table is not None and required_columns:
        table = block_table[:, :required_columns].tolist()
        for batch, (kv_begin, kv_end) in enumerate(
            zip(kv_prefix, kv_prefix[1:], strict=False)
        ):
            need = math.ceil((kv_end - kv_begin) / _COMPUTE_BLOCK_KV) * pages_per_tile
            if any(not 0 <= index < num_phys_blocks for index in table[batch][:need]):
                raise ValueError(
                    f"block_table row {batch} contains a physical block index "
                    f"outside [0, {num_phys_blocks})"
                )

    if row_starts is None and row_ends is None:
        return
    assert row_starts is not None and row_ends is not None
    starts = row_starts.tolist()
    ends = row_ends.tolist()
    for batch, (q_begin, q_end, kv_begin, kv_end) in enumerate(
        zip(q_prefix, q_prefix[1:], kv_prefix, kv_prefix[1:], strict=False)
    ):
        kv_len = kv_end - kv_begin
        for row in range(q_begin, q_end):
            start = starts[row]
            end = ends[row]
            if end == 0:
                if start != 0:
                    raise ValueError(
                        f"inactive explicit row {row} must use start=end=0"
                    )
                continue
            effective_end = end + row_end_base
            if start < 0 or start > effective_end or effective_end > kv_len:
                raise ValueError(
                    f"explicit row {row} must satisfy 0 <= start <= "
                    f"end + row_end_base <= request {batch} KV length ({kv_len})"
                )


def _launch_metadata(
    prepared: _PreparedFP8MQASelectiveLogits,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    *,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
) -> None:
    """Refresh caller-owned metadata with no allocation or host readback."""
    device = prepared.context_lens.device
    if (
        cu_q.dtype != torch.int32
        or cu_kv.dtype != torch.int32
        or cu_q.shape != (prepared.batches + 1,)
        or cu_kv.shape != cu_q.shape
        or cu_q.device != device
        or cu_kv.device != device
        or not cu_q.is_contiguous()
        or not cu_kv.is_contiguous()
    ):
        raise ValueError(
            "cu_q and cu_kv must be contiguous int32 [prepared.batches+1] "
            "on the prepared device"
        )
    explicit_bounds = row_starts is not None or row_ends is not None
    if explicit_bounds == prepared.causal_bounds:
        expected = (
            "explicit row bounds" if not prepared.causal_bounds else "causal bounds"
        )
        raise ValueError(f"prepared metadata requires {expected}")
    if explicit_bounds:
        if row_starts is None or row_ends is None:
            raise ValueError("row_starts and row_ends must be provided together")
        if (
            row_starts.dtype != torch.int32
            or row_ends.dtype != torch.int32
            or row_starts.shape != (prepared.rows,)
            or row_ends.shape != (prepared.rows,)
            or row_starts.device != device
            or row_ends.device != device
            or not row_starts.is_contiguous()
            or not row_ends.is_contiguous()
        ):
            raise ValueError(
                "row_starts and row_ends must be contiguous int32 [Q] on q.device"
            )
        _require_aligned((cu_q, cu_kv, row_starts, row_ends))
    else:
        _require_aligned((cu_q, cu_kv))

    _validate_device_metadata_values(
        cu_q,
        cu_kv,
        rows=prepared.rows,
        page_size=prepared.page_size,
        num_phys_blocks=prepared.num_phys_blocks,
        block_table_columns=prepared.block_table_columns,
        causal_bounds=prepared.causal_bounds,
        row_starts=row_starts,
        row_ends=row_ends,
        row_end_base=prepared.row_end_base,
    )

    base_args = (
        from_dlpack(cu_q, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(cu_kv, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
    )
    output_args = (
        from_dlpack(prepared.context_lens, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(prepared.tile_meta, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        ),
        from_dlpack(prepared.schedule_meta, assumed_align=16),
        from_dlpack(
            prepared.schedule_prefix, assumed_align=16
        ).mark_compact_shape_dynamic(mode=0, divisibility=1),
        cutlass.Int32(prepared.batches),
    )
    args: tuple[Any, ...]
    if explicit_bounds:
        args = (
            *base_args,
            from_dlpack(row_starts, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(row_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            cutlass.Int32(prepared.row_end_base),
            *output_args,
        )
    else:
        args = (*base_args, *output_args)
    stream = _current_stream(device)
    capability = torch.cuda.get_device_capability(device)
    key = (
        *capability,
        prepared.num_sms,
        prepared.query_tile,
        prepared.kv_chunk_size,
        prepared.split_kv,
        int(explicit_bounds),
    )
    compiled = _METADATA_COMPILED.get(key)
    if compiled is None:
        from .kernels.selective_logits_metadata import (
            SelectiveLogitsExplicitMetadataScheduleKernel,
            SelectiveLogitsMetadataScheduleKernel,
        )

        metadata_kernel = (
            SelectiveLogitsExplicitMetadataScheduleKernel(
                prepared.num_sms,
                query_tile=prepared.query_tile,
                kv_chunk_size=prepared.kv_chunk_size,
                split_kv=prepared.split_kv,
                paged=True,
            )
            if explicit_bounds
            else SelectiveLogitsMetadataScheduleKernel(
                prepared.num_sms,
                query_tile=prepared.query_tile,
                kv_chunk_size=prepared.kv_chunk_size,
                split_kv=prepared.split_kv,
                paged=True,
            )
        )
        compiled = cute.compile(metadata_kernel, *args, stream)
        _METADATA_COMPILED[key] = compiled
    compiled(*args, stream)


def _launch_compact_metadata(
    prepared: _PreparedFP8MQASelectiveLogits,
    cu_q: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    row_flags: torch.Tensor,
) -> None:
    """Build a fixed-capacity schedule containing only flagged query tiles."""
    if prepared.causal_bounds:
        raise ValueError("compact repair metadata requires explicit preparation")
    args = (
        from_dlpack(cu_q, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(row_starts, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(row_ends, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(row_flags, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(prepared.context_lens, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(prepared.tile_meta, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        ),
        from_dlpack(prepared.schedule_meta, assumed_align=16),
        from_dlpack(
            prepared.schedule_prefix, assumed_align=16
        ).mark_compact_shape_dynamic(mode=0, divisibility=1),
        cutlass.Int32(prepared.batches),
        cutlass.Int32(prepared.rows),
    )
    stream = _current_stream(prepared.context_lens.device)
    capability = torch.cuda.get_device_capability(prepared.context_lens.device)
    key = (
        *capability,
        prepared.num_sms,
        prepared.kv_chunk_size,
        prepared.split_kv,
    )
    compiled = _COMPACT_METADATA_COMPILED.get(key)
    if compiled is None:
        from .kernels.selective_logits_metadata import (
            SelectiveLogitsCompactMetadataScheduleKernel,
        )

        compiled = cute.compile(
            SelectiveLogitsCompactMetadataScheduleKernel(
                prepared.num_sms,
                query_tile=prepared.query_tile,
                kv_chunk_size=prepared.kv_chunk_size,
                split_kv=prepared.split_kv,
                paged=True,
            ),
            *args,
            stream,
        )
        _COMPACT_METADATA_COMPILED[key] = compiled
    compiled(*args, stream)


def _prepare_fp8_mqa_selective_logits(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    block_table: torch.Tensor,
    *,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
    row_end_base: int = 0,
    num_sms: int | None = None,
    kv_chunk_size: int = 0,
    split_kv: int = 1,
    num_heads: int = 8,
    head_dim: int = 128,
    query_tile: int | None = None,
) -> _PreparedFP8MQASelectiveLogits:
    """Allocate and build request-owned selective scheduling metadata.

    ``kv_fused`` and ``block_table`` use the same paged FP8 representation and
    padded block-table extent as :func:`fp8_paged_mqa_logits`; table entries may
    map logical pages to arbitrary valid physical blocks. ``num_heads`` and
    ``head_dim`` define the prepared problem shape and become compile-time
    kernel configuration. The request tile is selected automatically from those
    constraints. Optional bounds are trusted device metadata in request-local
    coordinates; callers maintain
    ``0 <= start <= end <= request_kv_length``. For compact rectangular repair
    rows, ``row_end_base`` is added to every nonzero end; zero remains the
    inactive-row sentinel. Supplying explicit bounds is required for compact
    sampled KV and arbitrary compact repair rows.
    ``num_sms`` may cap an exceptional-path launch, and
    ``kv_chunk_size`` partitions each nonempty query tile into fixed-size work
    items for modes whose global atomics make that ownership safe. Candidate
    publication instead uses split-K-like ``split_kv`` ownership. Each active
    split receives a pair-aligned KV range. Splits through eight use four
    warp-ranked publisher stripes per owner; higher splits pool those stripes
    into one publisher segment per owner. Short ranges may activate fewer
    splits than requested.
    Use :func:`_refresh_fp8_mqa_selective_logits` inside replay when these device-side
    prefixes or bounds change.

    Args:
        q: Contiguous FP8 E4M3 query tensor ``[Q, num_heads, head_dim]``.
        kv_fused: Contiguous paged byte tensor
            ``[num_blocks, page_size, 1, head_dim + 4]``. Each page stores FP8
            keys followed by one FP32 scale per token.
        weights: Contiguous FP32 per-head weights ``[Q, num_heads]``.
        cu_q: Contiguous int32 CUDA query prefix sums ``[B + 1]``.
        cu_kv: Contiguous int32 CUDA logical KV prefix sums ``[B + 1]``.
        block_table: Contiguous int32 CUDA physical page indices
            ``[B, max_pages]``.
        row_starts: Optional contiguous int32 request-local inclusive starts
            ``[Q]``. Must be provided together with ``row_ends``.
        row_ends: Optional contiguous int32 request-local exclusive ends
            ``[Q]``. Zero denotes an inactive row.
        row_end_base: Value added to every nonzero explicit row end.
        num_sms: Optional persistent CTA count. Defaults to the device SM count.
        kv_chunk_size: Fixed token extent for atomic output modes. Zero disables
            chunking; nonzero values must be positive multiples of 256.
        split_kv: Candidate-mode split count. Supported values are 1, 2, 4,
            8, 16, 32, and 64.
        num_heads: Query head count, specialized at compile time.
        head_dim: Head dimension, specialized at compile time and divisible by
            32.
        query_tile: Optional internal request-tile specialization. Public plans
            select this from their per-request query lengths.

    Returns:
        Prepared metadata whose storage must remain alive while launches or
        captured graphs use it.

    Note:
        Prefix values and page indices live on the device and are trusted by
        default. ``cu_q`` and ``cu_kv`` must start at zero, be monotonic, and
        end at ``Q`` and the logical KV total respectively. Causal requests
        require ``q_len <= kv_len``. The page table must contain enough padded
        entries for every 128-token compute tile, and every entry the kernel
        may read must be in ``[0, num_blocks)``. Explicit rows require
        ``0 <= start <= end + row_end_base <= request_kv_len``; inactive rows
        use ``start == end == 0``. Set ``FLASHINFER_VALIDATE_INPUTS=1`` for
        synchronized development-time checks outside CUDA graph capture.
    """
    _require_cute()
    if not isinstance(row_end_base, int) or row_end_base < 0:
        raise ValueError("row_end_base must be a nonnegative integer")
    if row_end_base and row_starts is None and row_ends is None:
        raise ValueError("row_end_base requires explicit row bounds")
    _validate_prefixes(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    _validate_device_metadata_values(
        cu_q,
        cu_kv,
        rows=q.shape[0],
        page_size=kv_fused.shape[1],
        num_phys_blocks=kv_fused.shape[0],
        block_table=block_table,
        causal_bounds=row_starts is None and row_ends is None,
        row_starts=row_starts,
        row_ends=row_ends,
        row_end_base=row_end_base,
    )
    tiling = _validate_specialization(
        q.device,
        num_heads,
        head_dim,
        max_query_tile=query_tile if query_tile is not None else _MAX_QUERY_TILE,
    )
    if query_tile is not None and tiling.next_n != query_tile:
        raise ValueError(
            f"query_tile {query_tile} is not legal for H={num_heads}, D={head_dim}"
        )
    rows = q.shape[0]
    batches = cu_q.shape[0] - 1
    if rows <= 0 or kv_fused.shape[0] <= 0:
        raise ValueError("selective-logits q and kv lengths must be positive")
    query_tile = tiling.next_n
    max_tiles = (rows + query_tile - 1) // query_tile + batches - 1
    device_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    if num_sms is None:
        num_sms = device_sms
    if not isinstance(num_sms, int) or not 0 < num_sms <= device_sms:
        raise ValueError(f"num_sms must be in [1, {device_sms}]")
    if not isinstance(kv_chunk_size, int) or kv_chunk_size < 0 or kv_chunk_size % 256:
        raise ValueError("kv_chunk_size must be zero or a positive multiple of 256")
    if split_kv not in _SPLIT_KV_CHOICES:
        raise ValueError("split_kv must be 1, 2, 4, 8, 16, 32, or 64")
    if kv_chunk_size and split_kv != 1:
        raise ValueError("kv_chunk_size and split_kv > 1 are mutually exclusive")
    prepared = _PreparedFP8MQASelectiveLogits(
        context_lens=torch.empty(max_tiles, dtype=torch.int32, device=q.device),
        tile_meta=torch.zeros(
            (max_tiles, _METADATA_HEADER_COLUMNS + 2 * query_tile),
            dtype=torch.int32,
            device=q.device,
        ),
        schedule_meta=torch.empty((num_sms + 1, 2), dtype=torch.int32, device=q.device),
        schedule_prefix=torch.empty(max_tiles, dtype=torch.int32, device=q.device),
        rows=rows,
        num_phys_blocks=kv_fused.shape[0],
        page_size=kv_fused.shape[1],
        query_tile=query_tile,
        num_epi_subtiles=tiling.num_epi_subtiles,
        tiles=max_tiles,
        batches=batches,
        num_sms=num_sms,
        kv_chunk_size=kv_chunk_size,
        split_kv=split_kv,
        causal_bounds=row_starts is None and row_ends is None,
        row_end_base=row_end_base,
        num_heads=num_heads,
        head_dim=head_dim,
        num_umma_stages=tiling.num_umma_stages,
        block_table_columns=block_table.shape[1],
    )
    _launch_metadata(
        prepared,
        cu_q,
        cu_kv,
        row_starts=row_starts,
        row_ends=row_ends,
    )
    return prepared


def _refresh_fp8_mqa_selective_logits(
    prepared: _PreparedFP8MQASelectiveLogits,
    cu_q: torch.Tensor,
    cu_kv: torch.Tensor,
    *,
    row_starts: torch.Tensor | None = None,
    row_ends: torch.Tensor | None = None,
) -> None:
    """Refresh address-stable request metadata on the current CUDA stream.

    Args:
        prepared: Object returned by :func:`_prepare_fp8_mqa_selective_logits`.
        cu_q: Updated contiguous int32 CUDA query prefixes ``[B + 1]``.
        cu_kv: Updated contiguous int32 CUDA logical KV prefixes ``[B + 1]``.
        row_starts: Updated explicit inclusive starts ``[Q]``, when the
            prepared object uses explicit bounds.
        row_ends: Updated explicit exclusive ends ``[Q]``, when the prepared
            object uses explicit bounds.

    The prepared object fixes tensor shapes, page-table width, bounds mode,
    and specialization. Refresh changes values only and is safe to capture in
    a CUDA graph; synchronized opt-in input checks are skipped during capture.
    """
    _require_cute()
    _launch_metadata(
        prepared,
        cu_q,
        cu_kv,
        row_starts=row_starts,
        row_ends=row_ends,
    )


def _runtime_args(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    values: torch.Tensor,
    index_counts: torch.Tensor,
    prepared: _PreparedFP8MQASelectiveLogits,
    policy_words: torch.Tensor,
) -> Tuple[Any, ...]:
    q_view = q.reshape(q.shape[0] * prepared.num_heads, prepared.head_dim).unsqueeze(-1)
    kv_flat = kv_fused.flatten(1)
    return (
        from_dlpack(kv_flat, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(
            q_view.view(torch.uint8), assumed_align=16
        ).mark_compact_shape_dynamic(
            mode=0,
            stride_order=q_view.dim_order(),
            divisibility=prepared.num_heads,
        ),
        from_dlpack(weights, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(values, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(block_table, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(index_counts, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(prepared.context_lens, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(prepared.schedule_meta, assumed_align=16),
        from_dlpack(prepared.tile_meta, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        ),
        from_dlpack(policy_words, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
    )


def _fp8_mqa_selective_logits(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    prepared: _PreparedFP8MQASelectiveLogits,
    *,
    mode: _SelectiveLogitsMode,
    capacity: int,
    values: torch.Tensor,
    index_counts: torch.Tensor,
    policy_values: torch.Tensor | None = None,
    radix_shift: int = 0,
    _compile_only: bool = False,
) -> None:
    """Run one explicitly selected logits phase into caller-owned storage.

    ``kv_fused`` and ``block_table`` follow :func:`fp8_paged_mqa_logits`, and
    this function never dispatches to that full-logits API based on row length.

    Candidate/repair use packed int64 ``values[Q, capacity]``. A one-column
    count tensor publishes directly. The cooperative path uses four striped
    segments per split-KV owner through split 8, then one pooled segment per
    owner for split 16/32/64. Legacy lane-local publication remains accepted
    through the split-1 Qx33, split-2 Qx65, and split-4 Qx129 layouts. All
    segmented forms require
    :func:`_finalize_fp8_mqa_packed_candidates` to compact values and publish
    either the exact count or the packed ``capacity + 1`` overflow sentinel in
    column zero. Exact completion writes int32 indices and one logical count.
    Radix uses ``index_counts[Q, 256]``. Sample uses float32
    ``values[Q, capacity]``.
    Clearing outputs is stream ordered and therefore participates in replay.

    Args:
        q: Contiguous FP8 E4M3 queries matching ``prepared``.
        kv_fused: Fused paged FP8 KV storage matching ``prepared``.
        weights: Contiguous FP32 per-head weights ``[Q, num_heads]``.
        block_table: Contiguous int32 physical-page table compatible with the
            prefixes used to refresh ``prepared``.
        prepared: Address-stable metadata and compile-time specialization.
        mode: Explicit output phase; this API performs no length-based routing.
        capacity: Number of elements available in each output row.
        values: Caller-owned output storage with dtype/layout determined by
            ``mode``.
        index_counts: Caller-owned count or histogram storage with the layout
            described above.
        policy_values: Required threshold or radix-prefix words for every mode
            except ``SAMPLE``.
        radix_shift: Byte shift for ``RADIX_HISTOGRAM``; one of 0, 8, 16, 24.

    This stateful prepare/run interface intentionally owns its compile cache
    and retained DLPack bindings instead of using the stateless
    ``flashinfer_api`` wrapper.
    """
    _require_cute()
    launch_tensors: tuple[torch.Tensor, ...] = (
        q,
        kv_fused,
        weights,
        block_table,
        values,
        index_counts,
    )
    if policy_values is not None:
        launch_tensors = (*launch_tensors, policy_values)
    if isinstance(prepared, _PreparedFP8MQASelectiveLogits):
        # Cached CuTe arguments retain these metadata addresses alongside the
        # ordinary inputs and outputs. Include their structural identity so a
        # caller-visible storage rebind rebuilds the DLPack launch binding.
        launch_tensors = (
            *launch_tensors,
            prepared.context_lens,
            prepared.tile_meta,
            prepared.schedule_meta,
        )
    cached_required_words = 0
    if isinstance(index_counts, torch.Tensor) and index_counts.dim() == 2:
        if mode == _SelectiveLogitsMode.RADIX_HISTOGRAM:
            cached_required_words = 256
        elif mode in (_SelectiveLogitsMode.CANDIDATE, _SelectiveLogitsMode.REPAIR):
            cached_required_words = index_counts.shape[1]
        else:
            cached_required_words = 1
    cache_key = (
        mode,
        capacity,
        radix_shift,
        cached_required_words,
    )
    if (
        isinstance(mode, _SelectiveLogitsMode)
        and isinstance(prepared, _PreparedFP8MQASelectiveLogits)
        and all(isinstance(tensor, torch.Tensor) for tensor in launch_tensors)
    ):
        identity = _selective_logits_launch_identity(
            launch_tensors,
            capacity,
        )
        cached = prepared._launches.get(cache_key)
        if cached is not None and cached.identity == identity:
            validation_enabled = os.environ.get(
                "FLASHINFER_VALIDATE_INPUTS", "0"
            ) not in ("0", "")
            if not validation_enabled or torch.cuda.is_current_stream_capturing():
                if _compile_only:
                    return
                if cached.count_view is not None:
                    cached.count_view.zero_()
                cached.compiled(*cached.args, _current_stream(q.device))
                return
    if q.dtype != torch.float8_e4m3fn or kv_fused.dtype != torch.uint8:
        raise ValueError("q must be float8_e4m3fn and kv_fused must be uint8")
    if weights.dtype != torch.float32:
        raise ValueError("weights must be float32")
    if q.dim() != 3 or tuple(q.shape[1:]) != (
        prepared.num_heads,
        prepared.head_dim,
    ):
        raise ValueError(
            f"q must have shape [Q, {prepared.num_heads}, {prepared.head_dim}]"
        )
    if (
        kv_fused.dim() != 4
        or kv_fused.shape[0] != prepared.num_phys_blocks
        or kv_fused.shape[1] != prepared.page_size
        or kv_fused.shape[2] != 1
        or kv_fused.shape[3] != prepared.head_dim + 4
    ):
        raise ValueError("kv_fused shape does not match prepared paged KV")
    if (
        block_table.dtype != torch.int32
        or block_table.dim() != 2
        or block_table.shape[0] != prepared.batches
        or block_table.shape[1] < prepared.block_table_columns
    ):
        raise ValueError("block_table must be int32 [B, max_pages_per_request]")
    if tuple(weights.shape) != (q.shape[0], prepared.num_heads):
        raise ValueError(f"weights must have shape [Q, {prepared.num_heads}]")
    if not q.is_cuda or any(
        tensor.device != q.device for tensor in (kv_fused, weights, block_table)
    ):
        raise ValueError(
            "q, kv_fused, weights, and block_table must share one CUDA device"
        )
    kv_flat = kv_fused.flatten(1)
    if not (
        q.is_contiguous()
        and weights.is_contiguous()
        and block_table.is_contiguous()
        and kv_flat.data_ptr() == kv_fused.data_ptr()
        and kv_flat.stride(1) == 1
    ):
        raise ValueError(
            "q, weights, block_table, and bytes within each KV page must be contiguous"
        )
    if (
        os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") not in ("0", "")
        and not torch.cuda.is_current_stream_capturing()
        and block_table.numel()
    ):
        block_min, block_max = torch.aminmax(block_table)
        if int(block_min) < 0 or int(block_max) >= prepared.num_phys_blocks:
            raise ValueError(
                "block_table contains a physical block index outside the prepared "
                f"range [0, {prepared.num_phys_blocks})"
            )
    if not isinstance(mode, _SelectiveLogitsMode):
        raise ValueError("mode must be a _SelectiveLogitsMode")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if mode == _SelectiveLogitsMode.RADIX_HISTOGRAM and radix_shift not in (
        0,
        8,
        16,
        24,
    ):
        raise ValueError("radix_shift must be one of {0, 8, 16, 24}")
    metadata = _validate_prepared(prepared, q, kv_fused, block_table)
    padded_rows = (
        (prepared.rows + prepared.query_tile - 1) // prepared.query_tile
    ) * prepared.query_tile
    if mode == _SelectiveLogitsMode.SAMPLE:
        values_dtype = torch.float32
    elif mode in (_SelectiveLogitsMode.CANDIDATE, _SelectiveLogitsMode.REPAIR):
        values_dtype = torch.int64
    else:
        values_dtype = torch.int32
    if values.dtype != values_dtype or values.device != q.device:
        raise ValueError(
            f"values must be caller-owned {values_dtype} storage on q.device"
        )
    if values.dim() != 2 or values.shape[0] < padded_rows or values.shape[1] < capacity:
        raise ValueError("values storage is smaller than [padded_Q, capacity]")
    if index_counts.dtype != torch.int32 or index_counts.device != q.device:
        raise ValueError("index_counts must be caller-owned int32 storage on q.device")
    if mode == _SelectiveLogitsMode.RADIX_HISTOGRAM:
        required_words = 256
    elif mode in (_SelectiveLogitsMode.CANDIDATE, _SelectiveLogitsMode.REPAIR):
        if index_counts.dim() != 2 or index_counts.shape[1] not in (
            1,
            *tuple(segments + 1 for segments in _COUNT_SEGMENTS),
        ):
            raise ValueError(
                "candidate counts must use one global word or "
                "4/8/12/16/32/64/128 segments"
            )
        required_words = index_counts.shape[1]
    else:
        required_words = 1
    if index_counts.dim() != 2 or index_counts.shape[0] < padded_rows:
        raise ValueError("index_counts has too few rows")
    if index_counts.shape[1] < required_words:
        raise ValueError("index_counts storage is too narrow for the selected mode")
    num_count_segments = (
        required_words - 1
        if mode in (_SelectiveLogitsMode.CANDIDATE, _SelectiveLogitsMode.REPAIR)
        else 0
    )
    if (
        prepared.kv_chunk_size
        and mode == _SelectiveLogitsMode.CANDIDATE
        and num_count_segments > 0
    ):
        raise ValueError(
            "segmented candidate store publication requires exclusive query tiles"
        )
    striped_split_layout = (
        prepared.split_kv <= _MAX_STRIPED_SPLIT_KV
        and num_count_segments == 4 * prepared.split_kv
    )
    pooled_split_layout = (
        prepared.split_kv > _MAX_STRIPED_SPLIT_KV
        and num_count_segments == prepared.split_kv
    )
    lane_local_layout = (
        prepared.split_kv <= 4 and num_count_segments == 32 * prepared.split_kv
    )
    valid_split_layout = (
        striped_split_layout or pooled_split_layout or lane_local_layout
    )
    if num_count_segments > 0 and (
        mode != _SelectiveLogitsMode.CANDIDATE or not valid_split_layout
    ):
        required_layout = f"Qx{_candidate_count_segments(prepared.split_kv) + 1}"
        if prepared.split_kv <= 4:
            required_layout += f" or Qx{32 * prepared.split_kv + 1}"
        raise ValueError(
            f"split_kv={prepared.split_kv} candidate publication requires "
            f"{required_layout} candidate counts"
        )
    if num_count_segments and capacity % num_count_segments:
        raise ValueError(
            "segmented candidate capacity must be divisible by its segment count: "
            f"mode={mode}, capacity={capacity}, segments={num_count_segments}"
        )
    lane_local_layout = mode == _SelectiveLogitsMode.CANDIDATE and lane_local_layout
    if lane_local_layout and (capacity // num_count_segments) % (
        256 // num_count_segments
    ):
        raise ValueError(
            "lane-local candidates require per-slice capacity divisible by "
            "their cooperating-thread count"
        )
    count_segment_span = 0
    if (
        mode in (_SelectiveLogitsMode.CANDIDATE, _SelectiveLogitsMode.REPAIR)
        and required_words > 1
    ):
        # Each store warp consumes one fixed 32-token slice from each math
        # stream and appends both into the same warp-ranked segment. The
        # lane-local layouts instead assign one fixed slice to every store-lane
        # vector or adjacent vector pair.
        count_segment_span = (
            1 if lane_local_layout else 128 if pooled_split_layout else 32
        )
    if (
        mode == _SelectiveLogitsMode.CANDIDATE
        and required_words > 1
        and num_count_segments not in (4, 8, 12, 16, 32, 64, 128)
    ):
        raise ValueError(
            "candidate publication requires 4, 8, 12, 16, 32, 64, or 128 segments"
        )
    if not values.is_contiguous() or not index_counts.is_contiguous():
        raise ValueError("selective-logits outputs must be contiguous")
    output_ranges = (_byte_range(values), _byte_range(index_counts))
    input_ranges = tuple(
        _byte_range(tensor) for tensor in (q, kv_fused, weights, block_table)
    )
    metadata_ranges = tuple(_byte_range(tensor) for tensor in metadata)
    if _overlap(*output_ranges) or any(
        _overlap(output_range, protected_range)
        for output_range in output_ranges
        for protected_range in (*input_ranges, *metadata_ranges)
    ):
        raise ValueError(
            "selective-logits outputs must not alias each other, inputs, or prepared metadata"
        )

    if mode != _SelectiveLogitsMode.SAMPLE:
        if policy_values is None or policy_values.shape != (prepared.rows,):
            raise ValueError("this mode requires one policy value per Q row")
        if policy_values.device != q.device or not policy_values.is_contiguous():
            raise ValueError("policy_values must be contiguous on q.device")
        if any(_overlap(_byte_range(policy_values), item) for item in output_ranges):
            raise ValueError("policy_values must not alias selective-logits outputs")
        if any(_overlap(_byte_range(policy_values), item) for item in metadata_ranges):
            raise ValueError("policy_values must not alias prepared metadata")
        if mode == _SelectiveLogitsMode.RADIX_HISTOGRAM:
            if policy_values.dtype not in (torch.int32, torch.uint32):
                raise ValueError("radix policy_values must contain uint32 prefix bits")
            policy_words = policy_values.view(torch.int32)
        else:
            if policy_values.dtype != torch.float32:
                raise ValueError("threshold policy_values must be float32")
            policy_words = policy_values.view(torch.int32)
    else:
        # Sample never reads policy words; retain a valid address without
        # introducing a dummy allocation into the captureable launch path.
        policy_words = prepared.context_lens

    _require_aligned(
        (
            q,
            kv_fused,
            weights,
            block_table,
            values,
            index_counts,
            policy_words,
            *metadata,
        )
    )
    if mode != _SelectiveLogitsMode.SAMPLE:
        index_counts[:, :required_words].zero_()

    args = (
        *_runtime_args(
            q,
            kv_fused,
            weights,
            block_table,
            values,
            index_counts,
            prepared,
            policy_words,
        ),
        cutlass.Int32(kv_fused.shape[0]),
        cutlass.Int32(prepared.context_lens.shape[0]),
    )
    stream = _current_stream(q.device)
    capability = torch.cuda.get_device_capability(q.device)
    key = (
        capability[0] * 10 + capability[1],
        prepared.num_sms,
        mode,
        capacity,
        required_words,
        count_segment_span,
        radix_shift,
        prepared.causal_bounds,
        prepared.split_kv,
        prepared.num_heads,
        prepared.head_dim,
        prepared.page_size,
        prepared.query_tile,
        prepared.num_epi_subtiles,
        prepared.num_umma_stages,
    )
    compiled = _SELECTIVE_LOGITS_COMPILED.get(key)
    if compiled is None:
        from .kernels.fp8_paged_mqa_logits import (
            FP8MQASelectiveLogitsKernel,
            SELECTIVE_LOGITS_TILING,
        )

        compiled = cute.compile(
            FP8MQASelectiveLogitsKernel(
                num_heads=prepared.num_heads,
                head_dim=prepared.head_dim,
                phys_block_kv=prepared.page_size,
                num_sms=prepared.num_sms,
                mode=mode,
                capacity=capacity,
                radix_shift=radix_shift,
                candidate_uses_causal_bounds=prepared.causal_bounds,
                split_kv=prepared.split_kv,
                num_count_segments=(
                    required_words - 1
                    if mode
                    in (_SelectiveLogitsMode.CANDIDATE, _SelectiveLogitsMode.REPAIR)
                    else 0
                ),
                count_segment_span=count_segment_span,
                tiling=replace(
                    SELECTIVE_LOGITS_TILING,
                    next_n=prepared.query_tile,
                    num_epi_subtiles=prepared.num_epi_subtiles,
                    num_umma_stages=prepared.num_umma_stages,
                ),
            ),
            *args,
            stream,
        )
        _SELECTIVE_LOGITS_COMPILED[key] = compiled
    if not _compile_only:
        compiled(*args, stream)
    prepared._launches[cache_key] = _CachedSelectiveLogitsLaunch(
        identity=_selective_logits_launch_identity(
            launch_tensors,
            capacity,
        ),
        tensors=launch_tensors,
        compiled=compiled,
        args=args,
        count_view=(
            index_counts[:, :required_words]
            if mode != _SelectiveLogitsMode.SAMPLE
            else None
        ),
    )


def _finalize_fp8_mqa_packed_candidates(
    index_counts: torch.Tensor,
    values: torch.Tensor,
    lengths: torch.Tensor,
    repair_flags: torch.Tensor,
    *,
    rows: int,
    capacity: int,
    top_k: int = _DEFAULT_TOP_K,
    _compile_only: bool = False,
) -> None:
    """Compact segmented candidates and publish exact count/repair metadata.

    Args:
        index_counts: Contiguous int32 CUDA counts with one logical count plus
            4, 8, 12, 32, 64, or 128 segment counters.
        values: Contiguous int64 packed candidate rows ``[padded_Q, capacity]``.
        lengths: Contiguous int32 CUDA output lengths with at least ``rows``
            elements.
        repair_flags: Contiguous int32 CUDA output flags with at least ``rows``
            elements.
        rows: Number of logical rows to finalize.
        capacity: Candidate capacity per row; divisible by the segment count.
        top_k: Minimum retained candidates required before exact repair.

    The function compacts retained segment prefixes in place. Column zero is
    set to the exact logical count, or ``capacity + 1`` when any segment
    overflows; ``repair_flags`` also marks underfilled rows.
    """
    _require_cute()
    tensors = (index_counts, values, lengths, repair_flags)
    if all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        identity = (rows, top_k, *_selective_logits_launch_identity(tensors, capacity))
        cached = getattr(values, "_flashinfer_packed_finalize_launch", None)
        if cached is not None and cached.identity == identity:
            if _compile_only:
                return
            cached.compiled(*cached.args, _current_stream(index_counts.device))
            return
    if rows <= 0 or capacity <= 0:
        raise ValueError("rows and capacity must be positive")
    if not isinstance(top_k, int) or not 0 < top_k <= _MAX_TOP_K:
        raise ValueError(f"top_k must be an integer in [1, {_MAX_TOP_K}]")
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError(
            "candidate values, counts, lengths, and flags must be contiguous CUDA"
        )
    if (
        index_counts.dtype != torch.int32
        or values.dtype != torch.int64
        or lengths.dtype != torch.int32
        or repair_flags.dtype != torch.int32
    ):
        raise ValueError("candidate finalizer requires int64 values and int32 metadata")
    if not all(tensor.device == index_counts.device for tensor in tensors[1:]):
        raise ValueError("candidate finalizer tensors must share one device")
    if (
        index_counts.dim() != 2
        or index_counts.shape[0] < rows
        or index_counts.shape[1]
        not in (1, *tuple(segments + 1 for segments in _COUNT_SEGMENTS))
        or values.dim() != 2
        or values.shape[0] < rows
        or values.shape[1] < capacity
        or lengths.dim() != 1
        or lengths.numel() < rows
        or repair_flags.dim() != 1
        or repair_flags.numel() < rows
    ):
        raise ValueError(
            "candidate finalizer storage is smaller than the requested rows"
        )
    num_segments = index_counts.shape[1] - 1
    if num_segments and capacity % num_segments:
        raise ValueError(
            "segmented candidate capacity must be divisible by its segment count"
        )
    if num_segments in (32, 64, 128) and (capacity // num_segments) % (
        256 // num_segments
    ):
        raise ValueError(
            "lane-local candidates require per-slice capacity divisible by "
            "their cooperating-thread count"
        )
    ranges = tuple(_byte_range(tensor) for tensor in tensors)
    if any(
        _overlap(ranges[left], ranges[right])
        for left in range(len(ranges))
        for right in range(left + 1, len(ranges))
    ):
        raise ValueError("candidate finalizer tensors must not alias")
    _require_aligned(tensors)

    args = (
        from_dlpack(index_counts, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(values, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(lengths, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(repair_flags, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        cutlass.Int32(rows),
        cutlass.Int32(capacity),
    )
    stream = _current_stream(index_counts.device)
    capability = torch.cuda.get_device_capability(index_counts.device)
    key = (*capability, capacity, num_segments, top_k)
    compiled = _FINALIZE_COMPILED.get(key)
    if compiled is None:
        from .kernels.selective_logits_metadata import SelectiveLogitsFinalizeKernel

        compiled = cute.compile(
            SelectiveLogitsFinalizeKernel(capacity, num_segments, top_k), *args, stream
        )
        _FINALIZE_COMPILED[key] = compiled
    if not _compile_only:
        compiled(*args, stream)
    values._flashinfer_packed_finalize_launch = _CachedFinalizeLaunch(
        identity=(rows, top_k, *_selective_logits_launch_identity(tensors, capacity)),
        tensors=tensors,
        compiled=compiled,
        args=args,
    )


class _FP8PagedMQASelectiveTopKEngine:
    """Compute exact TopK indices without exposing selective score phases.

    The wrapper owns sampled thresholding, candidate publication, TopK, and
    exact device repair. Callers see only the standard ``plan``/``run``
    lifecycle and final indices.

    ``run`` returns stable packed int32 storage with shape ``[total_q, top_k]``;
    ``cu_q`` maps rows back to requests. The caller explicitly selects this
    wrapper instead of :func:`fp8_paged_mqa_logits`; no sequence-length-based
    fallback is performed.

    Calls on one wrapper must not overlap. Synchronize before changing streams
    or calling :meth:`plan` again; replanning invalidates captured graphs.
    """

    def __init__(self) -> None:
        self._candidate_schedule: _CandidateSchedule | None = None
        self._prepared: _PreparedFP8MQASelectiveLogits | None = None
        self._repair_prepared: _PreparedFP8MQASelectiveLogits | None = None
        self._candidate_prepared: _PreparedFP8MQASelectiveLogits | None = None
        self._sample_prepared: _PreparedFP8MQASelectiveLogits | None = None
        self._rows = 0
        self._batches = 0
        self._top_k = _DEFAULT_TOP_K
        self._padded_rows = 0
        self._kv_rows = 0
        self._cu_q: torch.Tensor | None = None
        self._cu_kv: torch.Tensor | None = None
        self._needs_sampling = False
        self._compact_repair = False
        self._repair_q: torch.Tensor | None = None
        self._repair_weights: torch.Tensor | None = None
        self._repair_cu_q: torch.Tensor | None = None
        self._repair_row_ids: torch.Tensor | None = None
        self._repair_active_count: torch.Tensor | None = None
        self._second_row_ids: torch.Tensor | None = None
        self._second_active_count: torch.Tensor | None = None
        self._compact_repair_flags: torch.Tensor | None = None
        self._sample_count = 0
        self._sample_indices: torch.Tensor | None = None
        self._sample_request_ids: torch.Tensor | None = None
        self._sample_request_starts: torch.Tensor | None = None
        self._sample_ends: torch.Tensor | None = None
        self._sample_fused: torch.Tensor | None = None
        self._sample_block_table: torch.Tensor | None = None
        self._sample_scores: torch.Tensor | None = None
        self._sample_scratch: torch.Tensor | None = None
        self._sample_top_scores: torch.Tensor | None = None
        self._sample_top_positions: torch.Tensor | None = None
        self._sample_rank_indices: torch.Tensor | None = None
        self._sample_ranks: torch.Tensor | None = None
        self._thresholds: torch.Tensor | None = None
        self._selected_bins: torch.Tensor | None = None
        self._candidate_values: torch.Tensor | None = None
        self._candidate_counts: torch.Tensor | None = None
        self._candidate_lengths: torch.Tensor | None = None
        self._repair_flags: torch.Tensor | None = None
        self._repair_mask: torch.Tensor | None = None
        self._row_ends: torch.Tensor | None = None
        self._repair_starts: torch.Tensor | None = None
        self._repair_ends: torch.Tensor | None = None
        self._repair_thresholds: torch.Tensor | None = None
        self._packed_repair_starts: torch.Tensor | None = None
        self._packed_repair_ends: torch.Tensor | None = None
        self._packed_repair_thresholds: torch.Tensor | None = None
        self._repair_counts: torch.Tensor | None = None
        self._repair_lengths: torch.Tensor | None = None
        self._effective_lengths: torch.Tensor | None = None
        self._second_repair_flags: torch.Tensor | None = None
        self._zero_repair_mask: torch.Tensor | None = None
        self._zero_repair_flags: torch.Tensor | None = None
        self._nonzero_threshold_mask: torch.Tensor | None = None
        self._zero_repair_ends: torch.Tensor | None = None
        self._topk_row_states: torch.Tensor | None = None
        self._repair_scores: torch.Tensor | None = None
        self._repair_topk_positions_i32: torch.Tensor | None = None
        self._repair_selected: torch.Tensor | None = None
        self._repair_topk_compiled: Any = None
        self._histograms: torch.Tensor | None = None
        self._prefixes: torch.Tensor | None = None
        self._ranks: torch.Tensor | None = None
        self._threshold_bits: torch.Tensor | None = None
        self._radix_values: torch.Tensor | None = None
        self._greater_indices: torch.Tensor | None = None
        self._greater_counts: torch.Tensor | None = None
        self._equal_indices: torch.Tensor | None = None
        self._equal_counts: torch.Tensor | None = None
        self._selected: torch.Tensor | None = None
        self._exact_error: torch.Tensor | None = None
        self._radix_select_launch: Any = None
        self._combine_exact_launch: Any = None
        self._zero_combine_exact_launch: Any = None
        self._sample_pack_compiled: Any = None
        self._threshold_floor_launch: Any = None
        self._repair_threshold_launch: Any = None
        self._packed_scores_launch: Any = None
        self._packed_topk_launch: Any = None
        self._repair_gather_launch: Any = None
        self._repair_scatter_launch: Any = None
        self._repair_resolve_launch: Any = None
        self._repair_row_compact_launch: Any = None
        self._repair_row_pack_launch: Any = None
        self._second_repair_row_pack_launch: Any = None
        self._compact_repair_resolve_launch: Any = None
        self._compact_combine_exact_launch: Any = None
        self._sample_threshold_launch: Any = None
        self._cuda_graph: torch.cuda.CUDAGraph | None = None
        self._graph_identity: tuple[Any, ...] | None = None

    def _try_replan_in_place(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        cu_q: torch.Tensor,
        cu_kv: torch.Tensor,
        block_table: torch.Tensor,
        *,
        q_prefix: list[int],
        kv_prefix: list[int],
        top_k: int,
        num_heads: int,
        head_dim: int,
        num_sms: int | None,
    ) -> bool:
        """Refresh a topology-compatible plan without replacing its storage."""
        if self._graph_identity is None:
            return False
        # Graph identity is published last and commits every reusable plan field.
        assert self._candidate_prepared is not None
        assert self._repair_prepared is not None
        assert self._prepared is not None
        assert self._sample_prepared is not None
        assert self._sample_indices is not None
        assert self._sample_request_ids is not None
        assert self._sample_request_starts is not None
        assert self._sample_block_table is not None
        assert self._sample_fused is not None
        assert self._sample_ends is not None
        assert self._sample_ranks is not None
        assert self._sample_rank_indices is not None
        assert self._sample_top_scores is not None
        assert self._row_ends is not None
        assert self._candidate_counts is not None
        if (
            q.shape[0] != self._rows
            or len(q_prefix) - 1 != self._batches
            or top_k != self._top_k
            or num_heads != self._candidate_prepared.num_heads
            or head_dim != self._candidate_prepared.head_dim
            or _selective_logits_tensor_specs((q, kv_fused, weights, block_table))
            != tuple(entry[2:] for entry in self._graph_identity[1])
        ):
            return False

        device_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        requested_num_sms = device_sms if num_sms is None else num_sms
        if requested_num_sms != self._candidate_prepared.num_sms:
            return False

        logical_ends_host: list[int] = []
        q_lengths: list[int] = []
        kv_lengths: list[int] = []
        for q_begin, q_end, kv_begin, kv_end in zip(
            q_prefix[:-1],
            q_prefix[1:],
            kv_prefix[:-1],
            kv_prefix[1:],
            strict=True,
        ):
            q_len = q_end - q_begin
            kv_len = kv_end - kv_begin
            if kv_len < q_len:
                return False
            q_lengths.append(q_len)
            kv_lengths.append(kv_len)
            logical_ends_host.extend(range(kv_len - q_len + 1, kv_len + 1))
        if len(logical_ends_host) != self._rows or min(logical_ends_host) <= 0:
            return False

        candidate_schedule = _select_candidate_schedule(
            q.device,
            num_heads,
            head_dim,
            q_lengths,
            kv_lengths,
            num_sms=requested_num_sms,
        )
        if candidate_schedule != self._candidate_schedule:
            return False
        if not _candidate_schedule_matches_prepared(
            candidate_schedule, self._candidate_prepared
        ):
            return False
        if candidate_schedule.count_segments + 1 != self._candidate_counts.shape[1]:
            return False
        needs_sampling = any(
            kv_length > _CANDIDATE_CAPACITY for kv_length in kv_lengths
        )
        if needs_sampling != self._needs_sampling:
            return False

        sample_indices: list[int] = []
        sample_request_ids: list[int] = []
        sample_request_starts = [0]
        sample_ends_host: list[int] = []
        sample_pages_per_request: list[int] = []
        sample_ranks: list[int] = []
        logical_row = 0
        target_candidates = min(_TARGET_CANDIDATES, max(512, 4 * top_k))
        for request, kv_length in enumerate(kv_lengths):
            sample_count = min(kv_length, _SAMPLE_CAPACITY)
            request_indices = [
                index * (kv_length - 1) // max(sample_count - 1, 1)
                for index in range(sample_count)
            ]
            sample_indices.extend(request_indices)
            sample_request_ids.extend([request] * sample_count)
            sample_request_starts.append(len(sample_indices))
            sample_pages_per_request.append(math.ceil(sample_count / _COMPUTE_BLOCK_KV))
            q_len = q_lengths[request]
            for row_end in logical_ends_host[logical_row : logical_row + q_len]:
                valid = bisect.bisect_left(request_indices, row_end)
                sample_ends_host.append(valid)
                sample_ranks.append(
                    max(1, min(valid, math.ceil(target_candidates * valid / row_end)))
                )
            logical_row += q_len

        max_sample_pages = max(sample_pages_per_request, default=0)
        max_sample_rank = max(sample_ranks)
        if (
            len(sample_indices) != self._sample_indices.numel()
            or len(sample_request_ids) != self._sample_request_ids.numel()
            or len(sample_request_starts) != self._sample_request_starts.numel()
            or max_sample_pages > self._sample_block_table.shape[1]
            or sum(sample_pages_per_request) > self._sample_fused.shape[0]
            or max_sample_rank > self._sample_top_scores.shape[1]
        ):
            return False

        new_graph_identity = _selective_logits_launch_identity(
            (q, kv_fused, weights, block_table), 0
        )
        # A failed refresh must not leave a callable mixture of plan generations.
        self._graph_identity = None
        self._cuda_graph = None
        self._cu_q = cu_q
        self._cu_kv = cu_kv
        self._kv_rows = kv_prefix[-1]
        self._row_ends.copy_(
            torch.tensor(logical_ends_host, dtype=torch.int32, device=q.device)
        )
        self._sample_indices.copy_(
            torch.tensor(sample_indices, dtype=torch.int32, device=q.device)
        )
        self._sample_request_ids.copy_(
            torch.tensor(sample_request_ids, dtype=torch.int32, device=q.device)
        )
        self._sample_request_starts.copy_(
            torch.tensor(sample_request_starts, dtype=torch.int32, device=q.device)
        )
        self._sample_ends.copy_(
            torch.tensor(sample_ends_host, dtype=torch.int32, device=q.device)
        )
        self._sample_ranks.copy_(
            torch.tensor(sample_ranks, dtype=torch.int32, device=q.device)
        )
        self._sample_rank_indices.copy_(
            torch.tensor(sample_ranks, dtype=torch.int64, device=q.device)
            .sub_(1)
            .unsqueeze(1)
        )
        self._sample_block_table.zero_()
        page_begin = 0
        sample_table_host = [
            [0] * self._sample_block_table.shape[1] for _ in range(self._batches)
        ]
        for request, page_count in enumerate(sample_pages_per_request):
            sample_table_host[request][:page_count] = range(
                page_begin, page_begin + page_count
            )
            page_begin += page_count
        self._sample_block_table.copy_(
            torch.tensor(sample_table_host, dtype=torch.int32, device=q.device)
        )
        self._sample_count = len(sample_indices)

        _refresh_fp8_mqa_selective_logits(self._candidate_prepared, cu_q, cu_kv)
        sample_cu_kv = torch.tensor(
            sample_request_starts, dtype=torch.int32, device=q.device
        )
        sample_starts = torch.zeros_like(self._sample_ends)
        _refresh_fp8_mqa_selective_logits(
            self._sample_prepared,
            cu_q,
            sample_cu_kv,
            row_starts=sample_starts,
            row_ends=self._sample_ends,
        )
        if self._compact_repair:
            assert self._repair_cu_q is not None
            assert self._packed_repair_starts is not None
            assert self._packed_repair_ends is not None
            repair_cu_q = self._repair_cu_q
            repair_starts = self._packed_repair_starts
            repair_ends = self._packed_repair_ends
        else:
            assert self._repair_starts is not None
            assert self._repair_ends is not None
            repair_cu_q = cu_q
            repair_starts = self._repair_starts
            repair_ends = self._repair_ends
        _refresh_fp8_mqa_selective_logits(
            self._repair_prepared,
            repair_cu_q,
            cu_kv,
            row_starts=repair_starts,
            row_ends=repair_ends,
        )
        _refresh_fp8_mqa_selective_logits(
            self._prepared,
            repair_cu_q,
            cu_kv,
            row_starts=repair_starts,
            row_ends=repair_ends,
        )

        if self._compact_repair:
            self._compile_compact_repair_launches(q, weights)
        self._compile_sample_pack(q, kv_fused, block_table)
        self._precompile_score_launches(q, kv_fused, weights, block_table)

        torch.cuda.synchronize(q.device)
        new_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(new_graph):
            self._run_impl(q, kv_fused, weights, block_table)
        self._cuda_graph = new_graph
        self._graph_identity = new_graph_identity
        return True

    def plan(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        cu_q: torch.Tensor,
        cu_kv: torch.Tensor,
        block_table: torch.Tensor,
        *,
        top_k: int = _DEFAULT_TOP_K,
        num_heads: int = 8,
        head_dim: int = 128,
        num_sms: int | None = None,
    ) -> None:
        """Prepare and precompile one exact TopK problem.

        Q and KV extents size plan-owned storage but are runtime kernel
        metadata, not compilation keys. Compiled objects specialize only on
        architecture and tiling/code-generation options such as H/D, page
        size, query tile, publication layout, and pipeline stages.

        Args:
            q: Contiguous FP8 E4M3 query tensor
                ``[total_q, num_heads, head_dim]``.
            kv_fused: Fused paged FP8 KV values and FP32 scales with shape
                ``[num_pages, page_size, 1, head_dim + 4]``.
            weights: Contiguous FP32 head weights ``[total_q, num_heads]``.
            cu_q: Contiguous int32 CUDA query prefix sums ``[batch + 1]``.
            cu_kv: Contiguous int32 CUDA KV prefix sums ``[batch + 1]``.
            block_table: Contiguous int32 physical-page map
                ``[batch, max_pages]``.
            top_k: Number of request-local indices returned per query row.
                Any integer from 1 through 512 is supported.
            num_heads: Query-head count specialized by the scoring kernel.
            head_dim: Head dimension specialized by the scoring kernel.
            num_sms: Optional persistent-CTA count. The device SM count is
                used when omitted.

        This method allocates stable intermediate storage, reads the two prefix
        tensors on the host, and compiles every CuTe kernel used by :meth:`run`.
        Replanning the same tensor geometry with a different compatible KV
        length refreshes metadata in place and reuses both storage and compiled
        kernels. A sampling-boundary or ``split_kv`` topology change rebuilds
        the plan. Call this method outside CUDA graph capture and rebuild the
        plan when input storage, shapes, layouts, device, or specialization
        parameters change.
        """
        _require_cute()
        if not isinstance(top_k, int) or not 0 < top_k <= _MAX_TOP_K:
            raise ValueError(f"top_k must be an integer in [1, {_MAX_TOP_K}]")
        if cu_q.numel() < 2 or cu_kv.numel() != cu_q.numel():
            raise ValueError("cu_q and cu_kv must have matching [batch+1] extents")
        q_prefix = cu_q.tolist()
        kv_prefix = cu_kv.tolist()
        if q_prefix[0] != 0 or kv_prefix[0] != 0:
            raise ValueError("cu_q and cu_kv must start at zero")
        if q_prefix[-1] != q.shape[0]:
            raise ValueError("cu_q[-1] must equal q.shape[0]")
        if any(
            left > right for left, right in zip(q_prefix, q_prefix[1:], strict=False)
        ):
            raise ValueError("cu_q must be monotonically nondecreasing")
        if any(
            left > right for left, right in zip(kv_prefix, kv_prefix[1:], strict=False)
        ):
            raise ValueError("cu_kv must be monotonically nondecreasing")
        if block_table.shape[0] != len(q_prefix) - 1:
            raise ValueError("block_table must have one row per request")
        if self._try_replan_in_place(
            q,
            kv_fused,
            weights,
            cu_q,
            cu_kv,
            block_table,
            q_prefix=q_prefix,
            kv_prefix=kv_prefix,
            top_k=top_k,
            num_heads=num_heads,
            head_dim=head_dim,
            num_sms=num_sms,
        ):
            return
        # A rebuild cannot leave the prior graph callable with replacement fields.
        self._graph_identity = None
        self._cuda_graph = None
        self._candidate_schedule = None
        self._rows = q.shape[0]
        self._batches = len(q_prefix) - 1
        self._top_k = top_k
        self._kv_rows = kv_prefix[-1]
        self._cu_q = cu_q
        self._cu_kv = cu_kv
        logical_ends_host: list[int] = []
        q_lengths: list[int] = []
        kv_lengths: list[int] = []
        for q_begin, q_end, kv_begin, kv_end in zip(
            q_prefix[:-1],
            q_prefix[1:],
            kv_prefix[:-1],
            kv_prefix[1:],
            strict=True,
        ):
            q_len = q_end - q_begin
            kv_len = kv_end - kv_begin
            if kv_len < q_len:
                raise ValueError(
                    "causal selective TopK requires every KV length >= its Q length"
                )
            q_lengths.append(q_len)
            kv_lengths.append(kv_len)
            logical_ends_host.extend(range(kv_len - q_len + 1, kv_len + 1))
        if not logical_ends_host or min(logical_ends_host) <= 0:
            raise ValueError("every query row must have at least one causal KV row")
        self._padded_rows = (
            (self._rows + _MAX_QUERY_TILE - 1) // _MAX_QUERY_TILE * _MAX_QUERY_TILE
        )
        device = q.device
        self._row_ends = torch.tensor(
            logical_ends_host, device=device, dtype=torch.int32
        )
        self._repair_starts = torch.zeros(self._rows, dtype=torch.int32, device=device)
        self._repair_ends = torch.zeros_like(self._repair_starts)
        # The compact repair scorer packs full Q16/H8 work items. Other
        # supported query tiles keep the generic exact-repair path.
        self._compact_repair = self._batches == 1 and num_heads == 8 and head_dim == 128
        repair_q = q
        repair_weights = weights
        repair_cu_q = cu_q
        repair_starts = self._repair_starts
        repair_ends = self._repair_ends
        if self._compact_repair:
            self._repair_q = torch.empty_like(q)
            self._repair_weights = torch.empty_like(weights)
            self._repair_cu_q = torch.tensor(
                [0, self._rows], dtype=torch.int32, device=device
            )
            self._repair_row_ids = torch.empty(
                self._rows, dtype=torch.int32, device=device
            )
            self._repair_active_count = torch.zeros(1, dtype=torch.int32, device=device)
            self._second_row_ids = torch.empty_like(self._repair_row_ids)
            self._second_active_count = torch.zeros_like(self._repair_active_count)
            self._packed_repair_starts = torch.zeros_like(self._repair_starts)
            self._packed_repair_ends = torch.zeros_like(self._repair_ends)
            self._packed_repair_thresholds = torch.empty(
                self._rows, dtype=torch.float32, device=device
            )
            self._compact_repair_flags = torch.empty(
                self._rows, dtype=torch.int32, device=device
            )
            repair_q = self._repair_q
            repair_weights = self._repair_weights
            repair_cu_q = self._repair_cu_q
            repair_starts = self._packed_repair_starts
            repair_ends = self._packed_repair_ends

        requested_num_sms = num_sms
        device_sms = torch.cuda.get_device_properties(device).multi_processor_count
        if requested_num_sms is None:
            requested_num_sms = device_sms
        if (
            not isinstance(requested_num_sms, int)
            or requested_num_sms <= 0
            or requested_num_sms > device_sms
        ):
            raise ValueError(f"num_sms must be in [1, {device_sms}]")
        candidate_schedule = _select_candidate_schedule(
            device,
            num_heads,
            head_dim,
            q_lengths,
            kv_lengths,
            num_sms=requested_num_sms,
        )
        query_tile = candidate_schedule.query_tile
        split_kv = candidate_schedule.split_kv
        self._candidate_schedule = candidate_schedule
        self._candidate_prepared = _prepare_fp8_mqa_selective_logits(
            q,
            kv_fused,
            weights,
            cu_q,
            cu_kv,
            block_table,
            num_sms=num_sms,
            split_kv=split_kv,
            num_heads=num_heads,
            head_dim=head_dim,
            query_tile=query_tile,
        )
        if not _candidate_schedule_matches_prepared(
            candidate_schedule, self._candidate_prepared
        ):
            raise RuntimeError("candidate preparation changed the selected topology")
        self._repair_prepared = _prepare_fp8_mqa_selective_logits(
            repair_q,
            kv_fused,
            repair_weights,
            repair_cu_q,
            cu_kv,
            block_table,
            row_starts=repair_starts,
            row_ends=repair_ends,
            num_sms=num_sms,
            kv_chunk_size=256,
            num_heads=num_heads,
            head_dim=head_dim,
            query_tile=query_tile,
        )
        self._prepared = _prepare_fp8_mqa_selective_logits(
            repair_q,
            kv_fused,
            repair_weights,
            repair_cu_q,
            cu_kv,
            block_table,
            row_starts=repair_starts,
            row_ends=repair_ends,
            num_sms=requested_num_sms,
            kv_chunk_size=256,
            num_heads=num_heads,
            head_dim=head_dim,
            query_tile=query_tile,
        )
        self._padded_rows = (
            (self._rows + self._prepared.query_tile - 1)
            // self._prepared.query_tile
            * self._prepared.query_tile
        )
        self._needs_sampling = any(
            kv_length > _CANDIDATE_CAPACITY for kv_length in kv_lengths
        )
        sample_indices: list[int] = []
        sample_request_ids: list[int] = []
        sample_request_starts = [0]
        sample_ends_host: list[int] = []
        sample_pages_per_request: list[int] = []
        logical_row = 0
        target_candidates = min(_TARGET_CANDIDATES, max(512, 4 * top_k))
        sample_ranks: list[int] = []
        for request, kv_length in enumerate(kv_lengths):
            sample_count = min(kv_length, _SAMPLE_CAPACITY)
            request_indices = [
                index * (kv_length - 1) // max(sample_count - 1, 1)
                for index in range(sample_count)
            ]
            sample_indices.extend(request_indices)
            sample_request_ids.extend([request] * sample_count)
            sample_request_starts.append(len(sample_indices))
            sample_pages_per_request.append(math.ceil(sample_count / _COMPUTE_BLOCK_KV))
            q_len = q_prefix[request + 1] - q_prefix[request]
            for row_end in logical_ends_host[logical_row : logical_row + q_len]:
                valid = bisect.bisect_left(request_indices, row_end)
                sample_ends_host.append(valid)
                sample_ranks.append(
                    max(
                        1,
                        min(
                            valid,
                            math.ceil(target_candidates * valid / row_end),
                        ),
                    )
                )
            logical_row += q_len
        self._sample_count = len(sample_indices)
        self._sample_indices = torch.tensor(
            sample_indices, dtype=torch.int32, device=device
        )
        self._sample_request_ids = torch.tensor(
            sample_request_ids, dtype=torch.int32, device=device
        )
        self._sample_request_starts = torch.tensor(
            sample_request_starts, dtype=torch.int32, device=device
        )
        total_sample_pages = sum(sample_pages_per_request)
        self._sample_fused = torch.zeros(
            (total_sample_pages, _COMPUTE_BLOCK_KV, 1, head_dim + 4),
            dtype=torch.uint8,
            device=device,
        )
        max_sample_pages = max(sample_pages_per_request, default=0)
        sample_table_host = [[0] * max_sample_pages for _ in range(self._batches)]
        page_begin = 0
        for request, page_count in enumerate(sample_pages_per_request):
            sample_table_host[request][:page_count] = range(
                page_begin, page_begin + page_count
            )
            page_begin += page_count
        self._sample_block_table = torch.tensor(
            sample_table_host, dtype=torch.int32, device=device
        )
        sample_ends = torch.tensor(sample_ends_host, dtype=torch.int32, device=device)
        self._sample_ends = sample_ends
        sample_starts = torch.zeros_like(sample_ends)
        sample_cu_kv = torch.tensor(
            sample_request_starts, dtype=torch.int32, device=device
        )
        self._sample_prepared = _prepare_fp8_mqa_selective_logits(
            q,
            self._sample_fused,
            weights,
            cu_q,
            sample_cu_kv,
            self._sample_block_table,
            row_starts=sample_starts,
            row_ends=sample_ends,
            num_sms=num_sms,
            num_heads=num_heads,
            head_dim=head_dim,
            query_tile=query_tile,
        )
        self._sample_scores = torch.full(
            (self._padded_rows, _SAMPLE_CAPACITY),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        self._sample_scratch = torch.empty(
            (self._padded_rows, 1), dtype=torch.int32, device=device
        )
        max_sample_rank = max(sample_ranks)
        self._sample_top_scores = torch.empty(
            (self._rows, max_sample_rank), dtype=torch.float32, device=device
        )
        self._sample_top_positions = torch.empty(
            (self._rows, max_sample_rank), dtype=torch.int64, device=device
        )
        self._sample_rank_indices = (
            torch.tensor(sample_ranks, dtype=torch.int64, device=device)
            .sub_(1)
            .unsqueeze(1)
        )
        self._sample_ranks = torch.tensor(
            sample_ranks, dtype=torch.int32, device=device
        )
        self._thresholds = torch.empty(self._rows, dtype=torch.float32, device=device)
        self._selected_bins = torch.empty(self._rows, dtype=torch.int32, device=device)

        count_words = 1 + candidate_schedule.count_segments
        self._candidate_values = torch.empty(
            (self._padded_rows, _REPAIR_CAPACITY),
            dtype=torch.int64,
            device=device,
        )
        self._candidate_counts = torch.empty(
            (self._padded_rows, count_words), dtype=torch.int32, device=device
        )
        self._candidate_lengths = torch.empty(
            self._rows, dtype=torch.int32, device=device
        )
        self._repair_flags = torch.empty_like(self._candidate_lengths)
        self._repair_mask = torch.empty(self._rows, dtype=torch.bool, device=device)
        self._repair_thresholds = torch.empty_like(self._thresholds)
        self._repair_counts = torch.empty(
            (self._padded_rows, 1), dtype=torch.int32, device=device
        )
        self._repair_lengths = torch.empty_like(self._candidate_lengths)
        self._effective_lengths = torch.empty_like(self._candidate_lengths)
        self._second_repair_flags = torch.empty_like(self._candidate_lengths)
        self._zero_repair_mask = torch.empty(
            self._rows, dtype=torch.bool, device=device
        )
        self._zero_repair_flags = torch.empty_like(self._candidate_lengths)
        self._nonzero_threshold_mask = torch.empty_like(self._zero_repair_mask)
        self._zero_repair_ends = torch.empty_like(self._candidate_lengths)
        self._repair_scores = (
            self._candidate_values[: self._rows]
            .view(torch.float32)
            .as_strided(
                (self._rows, _REPAIR_CAPACITY),
                (_REPAIR_CAPACITY * 2, 2),
                storage_offset=1,
            )
        )
        self._repair_topk_positions_i32 = torch.empty(
            (self._rows, top_k), dtype=torch.int32, device=device
        )
        self._repair_selected = torch.empty_like(self._repair_topk_positions_i32)
        from ..topk_varlen.topk_varlen import (
            _RADIX_STATE_SIZE,
            _compile_radix,
            _radix_get_chunk_config,
            get_device_sm_count,
            get_shared_bytes_per_block_optin,
            torch_to_cutlass_dtype,
        )

        topk_num_sms = get_device_sm_count(device)
        repair_topk_ctas, repair_topk_chunk = _radix_get_chunk_config(
            _REPAIR_CAPACITY,
            torch.float32,
            self._rows,
            topk_num_sms,
            get_shared_bytes_per_block_optin(device),
        )
        topk_groups = max(1, topk_num_sms // repair_topk_ctas)
        self._topk_row_states = torch.zeros(
            (topk_groups, _RADIX_STATE_SIZE), dtype=torch.int32, device=device
        )
        self._repair_topk_compiled = _compile_radix(
            torch_to_cutlass_dtype(torch.float32),
            top_k,
            1,
            1,
            _REPAIR_CAPACITY,
            False,
            repair_topk_ctas,
            repair_topk_chunk,
            topk_num_sms,
            2,
            _REPAIR_CAPACITY * 2,
        )
        self._histograms = torch.empty(
            (self._padded_rows, 256), dtype=torch.int32, device=device
        )
        self._prefixes = torch.empty(self._rows, dtype=torch.int32, device=device)
        self._ranks = torch.empty(self._rows, dtype=torch.int32, device=device)
        self._threshold_bits = torch.empty(self._rows, dtype=torch.int32, device=device)
        self._radix_values = torch.empty(
            (self._padded_rows, 1), dtype=torch.int32, device=device
        )
        self._greater_indices = torch.empty(
            (self._padded_rows, top_k), dtype=torch.int32, device=device
        )
        self._greater_counts = torch.empty(
            (self._padded_rows, 1), dtype=torch.int32, device=device
        )
        self._equal_indices = torch.empty_like(self._greater_indices)
        self._equal_counts = torch.empty_like(self._greater_counts)
        self._selected = torch.empty(
            (self._rows, top_k), dtype=torch.int32, device=device
        )
        self._exact_error = torch.empty(1, dtype=torch.int32, device=device)
        self._compile_auxiliary_launches()
        if self._compact_repair:
            self._compile_compact_repair_launches(q, weights)
        self._compile_sample_pack(q, kv_fused, block_table)
        self._precompile_score_launches(q, kv_fused, weights, block_table)
        if not self._compact_repair:
            _launch_compact_metadata(
                self._repair_prepared,
                self._cu_q,
                self._repair_starts,
                self._repair_ends,
                self._repair_flags,
            )
            _launch_compact_metadata(
                self._prepared,
                self._cu_q,
                self._repair_starts,
                self._repair_ends,
                self._second_repair_flags,
            )
        torch.cuda.synchronize(device)
        new_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(new_graph):
            self._run_impl(q, kv_fused, weights, block_table)
        new_graph_identity = _selective_logits_launch_identity(
            (q, kv_fused, weights, block_table), 0
        )
        self._cuda_graph = new_graph
        self._graph_identity = new_graph_identity

    def _compile_sample_pack(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        block_table: torch.Tensor,
    ) -> None:
        assert self._prepared is not None
        assert self._sample_indices is not None
        assert self._sample_request_ids is not None
        assert self._sample_request_starts is not None
        assert self._sample_fused is not None
        assert self._sample_block_table is not None
        from .kernels.selective_logits_metadata import SelectiveLogitsSamplePackKernel

        source = kv_fused.flatten(1)
        destination = self._sample_fused.flatten(1)
        args = (
            from_dlpack(source, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(block_table, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._sample_block_table, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(
                self._sample_request_ids, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._sample_request_starts, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._sample_indices, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(destination, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            cutlass.Int32(self._sample_count),
        )
        stream = _current_stream(q.device)
        capability = torch.cuda.get_device_capability(q.device)
        key = (
            *capability,
            self._prepared.head_dim,
            self._prepared.page_size,
            _COMPUTE_BLOCK_KV,
        )
        compiled = _SAMPLE_PACK_COMPILED.get(key)
        if compiled is None:
            compiled = cute.compile(
                SelectiveLogitsSamplePackKernel(
                    self._prepared.head_dim,
                    self._prepared.page_size,
                    _COMPUTE_BLOCK_KV,
                ),
                *args,
                stream,
            )
            _SAMPLE_PACK_COMPILED[key] = compiled
        self._sample_pack_compiled = compiled

    def _compile_compact_repair_launches(
        self, q: torch.Tensor, weights: torch.Tensor
    ) -> None:
        """Compile and bind the batch-one dense repair-row path."""
        assert self._prepared is not None
        assert self._repair_flags is not None
        assert self._repair_thresholds is not None
        assert self._row_ends is not None
        assert self._repair_row_ids is not None
        assert self._repair_active_count is not None
        assert self._second_row_ids is not None
        assert self._second_active_count is not None
        assert self._repair_q is not None
        assert self._repair_weights is not None
        assert self._packed_repair_ends is not None
        assert self._packed_repair_thresholds is not None
        assert self._compact_repair_flags is not None
        assert self._candidate_values is not None
        assert self._repair_lengths is not None
        assert self._repair_topk_positions_i32 is not None
        assert self._selected is not None
        assert self._ranks is not None
        assert self._second_repair_flags is not None
        assert self._exact_error is not None
        assert self._greater_indices is not None
        assert self._greater_counts is not None
        assert self._equal_indices is not None
        assert self._equal_counts is not None
        from .kernels.selective_logits_metadata import (
            SelectiveCompactRepairResolveKernel,
            SelectiveCompactTopKCombineExactKernel,
            SelectiveRepairRowCompactKernel,
            SelectiveRepairRowPackKernel,
        )

        stream = _current_stream(q.device)
        capability = torch.cuda.get_device_capability(q.device)

        compact_args = (
            from_dlpack(
                self._repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_thresholds, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._row_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._repair_row_ids, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._packed_repair_thresholds, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._packed_repair_ends, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_active_count, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            cutlass.Int32(self._rows),
        )
        compact_key = (*capability, 256)
        compact_compiled = _REPAIR_ROW_COMPACT_COMPILED.get(compact_key)
        if compact_compiled is None:
            compact_compiled = cute.compile(
                SelectiveRepairRowCompactKernel(), *compact_args, stream
            )
            _REPAIR_ROW_COMPACT_COMPILED[compact_key] = compact_compiled
        self._repair_row_compact_launch = (compact_compiled, compact_args)

        q_bytes = q.view(torch.uint8).reshape(-1)
        packed_q_bytes = self._repair_q.view(torch.uint8).reshape(-1)
        pack_args = (
            from_dlpack(q_bytes, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                weights.reshape(-1), assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_row_ids, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_active_count, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(packed_q_bytes, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._repair_weights.reshape(-1), assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            cutlass.Int32(self._rows),
        )
        pack_key = (
            *capability,
            self._prepared.num_heads,
            self._prepared.head_dim,
            self._prepared.num_sms,
        )
        pack_compiled = _REPAIR_ROW_PACK_COMPILED.get(pack_key)
        if pack_compiled is None:
            pack_compiled = cute.compile(
                SelectiveRepairRowPackKernel(
                    self._prepared.num_heads,
                    self._prepared.head_dim,
                    self._prepared.num_sms,
                ),
                *pack_args,
                stream,
            )
            _REPAIR_ROW_PACK_COMPILED[pack_key] = pack_compiled
        self._repair_row_pack_launch = (pack_compiled, pack_args)
        second_pack_args = (
            *pack_args[:2],
            from_dlpack(
                self._second_row_ids, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._second_active_count, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            *pack_args[4:],
        )
        self._second_repair_row_pack_launch = (pack_compiled, second_pack_args)

        resolve_args = (
            from_dlpack(self._candidate_values, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(
                self._repair_lengths, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_topk_positions_i32, assumed_align=16
            ).mark_layout_dynamic(leading_dim=1),
            from_dlpack(
                self._repair_row_ids, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_active_count, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._compact_repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._row_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._selected, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(
                self._second_row_ids, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._packed_repair_ends, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._ranks, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._second_repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._second_active_count, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._exact_error, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            cutlass.Int32(self._rows),
        )
        resolve_key = (*capability, self._top_k, _REPAIR_CAPACITY, 256)
        resolve_compiled = _COMPACT_REPAIR_RESOLVE_COMPILED.get(resolve_key)
        if resolve_compiled is None:
            resolve_compiled = cute.compile(
                SelectiveCompactRepairResolveKernel(self._top_k, _REPAIR_CAPACITY),
                *resolve_args,
                stream,
            )
            _COMPACT_REPAIR_RESOLVE_COMPILED[resolve_key] = resolve_compiled
        self._compact_repair_resolve_launch = (resolve_compiled, resolve_args)

        combine_args = (
            from_dlpack(self._greater_indices, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._greater_counts, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._equal_indices, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._equal_counts, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._ranks, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._packed_repair_ends, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._second_row_ids, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._second_active_count, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._selected, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._exact_error, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            cutlass.Int32(self._rows),
        )
        combine_key = (*capability, self._top_k, 256)
        combine_compiled = _COMPACT_COMBINE_EXACT_COMPILED.get(combine_key)
        if combine_compiled is None:
            combine_compiled = cute.compile(
                SelectiveCompactTopKCombineExactKernel(self._top_k),
                *combine_args,
                stream,
            )
            _COMPACT_COMBINE_EXACT_COMPILED[combine_key] = combine_compiled
        self._compact_combine_exact_launch = (combine_compiled, combine_args)

    def _precompile_score_launches(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
    ) -> None:
        """Compile and bind every score phase needed by :meth:`run`."""
        assert self._prepared is not None
        assert self._repair_prepared is not None
        assert self._candidate_prepared is not None
        assert self._sample_prepared is not None
        assert self._histograms is not None
        assert self._prefixes is not None
        assert self._radix_values is not None
        assert self._greater_indices is not None
        assert self._greater_counts is not None
        assert self._equal_indices is not None
        assert self._equal_counts is not None
        assert self._repair_flags is not None
        repair_q = self._repair_q if self._compact_repair else q
        repair_weights = self._repair_weights if self._compact_repair else weights
        assert repair_q is not None
        assert repair_weights is not None
        assert self._repair_thresholds is not None
        assert self._repair_counts is not None
        assert self._repair_lengths is not None
        assert self._effective_lengths is not None
        assert self._second_repair_flags is not None
        assert self._zero_repair_mask is not None
        assert self._zero_repair_flags is not None
        assert self._nonzero_threshold_mask is not None
        assert self._zero_repair_ends is not None
        assert self._row_ends is not None
        assert self._threshold_bits is not None
        assert self._thresholds is not None
        assert self._selected_bins is not None
        assert self._candidate_counts is not None
        assert self._repair_flags is not None
        assert self._row_ends is not None
        assert self._repair_thresholds is not None
        assert self._repair_ends is not None
        assert self._second_repair_flags is not None
        assert self._sample_scores is not None
        assert self._sample_scratch is not None
        assert self._sample_fused is not None
        assert self._sample_block_table is not None
        assert self._thresholds is not None
        assert self._candidate_values is not None
        assert self._candidate_counts is not None
        assert self._candidate_lengths is not None
        assert self._repair_flags is not None
        _fp8_mqa_selective_logits(
            q,
            self._sample_fused,
            weights,
            self._sample_block_table,
            self._sample_prepared,
            mode=_SelectiveLogitsMode.SAMPLE,
            capacity=_SAMPLE_CAPACITY,
            values=self._sample_scores,
            index_counts=self._sample_scratch,
            _compile_only=True,
        )
        _fp8_mqa_selective_logits(
            q,
            kv_fused,
            weights,
            block_table,
            self._candidate_prepared,
            mode=_SelectiveLogitsMode.CANDIDATE,
            capacity=_CANDIDATE_CAPACITY,
            values=self._candidate_values,
            index_counts=self._candidate_counts,
            policy_values=self._thresholds,
            _compile_only=True,
        )
        _finalize_fp8_mqa_packed_candidates(
            self._candidate_counts,
            self._candidate_values,
            self._candidate_lengths,
            self._repair_flags,
            rows=self._rows,
            capacity=_CANDIDATE_CAPACITY,
            top_k=self._top_k,
            _compile_only=True,
        )
        _fp8_mqa_selective_logits(
            repair_q,
            kv_fused,
            repair_weights,
            block_table,
            self._repair_prepared,
            mode=_SelectiveLogitsMode.REPAIR,
            capacity=_REPAIR_CAPACITY,
            values=self._candidate_values,
            index_counts=self._repair_counts,
            policy_values=(
                self._packed_repair_thresholds
                if self._compact_repair
                else self._repair_thresholds
            ),
            _compile_only=True,
        )
        _finalize_fp8_mqa_packed_candidates(
            self._repair_counts,
            self._candidate_values,
            self._repair_lengths,
            (
                self._compact_repair_flags
                if self._compact_repair
                else self._second_repair_flags
            ),
            rows=self._rows,
            capacity=_REPAIR_CAPACITY,
            top_k=self._top_k,
            _compile_only=True,
        )
        for shift in (24, 16, 8, 0):
            _fp8_mqa_selective_logits(
                repair_q,
                kv_fused,
                repair_weights,
                block_table,
                self._prepared,
                mode=_SelectiveLogitsMode.RADIX_HISTOGRAM,
                capacity=1,
                values=self._radix_values,
                index_counts=self._histograms,
                policy_values=self._prefixes,
                radix_shift=shift,
                _compile_only=True,
            )
        thresholds = self._threshold_bits.view(torch.float32)
        for mode, values, counts in (
            (
                _SelectiveLogitsMode.EXACT_GREATER,
                self._greater_indices,
                self._greater_counts,
            ),
            (
                _SelectiveLogitsMode.EXACT_EQUAL,
                self._equal_indices,
                self._equal_counts,
            ),
        ):
            _fp8_mqa_selective_logits(
                repair_q,
                kv_fused,
                repair_weights,
                block_table,
                self._prepared,
                mode=mode,
                capacity=self._top_k,
                values=values,
                index_counts=counts,
                policy_values=thresholds,
                _compile_only=True,
            )

    def _compile_auxiliary_launches(self) -> None:
        assert self._histograms is not None
        assert self._prefixes is not None
        assert self._ranks is not None
        assert self._threshold_bits is not None
        assert self._thresholds is not None
        assert self._selected_bins is not None
        assert self._greater_indices is not None
        assert self._greater_counts is not None
        assert self._equal_indices is not None
        assert self._equal_counts is not None
        assert self._selected is not None
        assert self._exact_error is not None
        assert self._sample_ranks is not None
        assert self._sample_ends is not None
        assert self._sample_prepared is not None
        from .kernels.selective_logits_metadata import (
            SelectivePackedGatherKernel,
            SelectivePackedTopKKernel,
            SelectiveRepairThresholdKernel,
            SelectiveRepairScatterKernel,
            SelectiveRepairResolveKernel,
            SelectiveSampleThresholdKernel,
            SelectiveTopKCombineExactKernel,
            SelectiveTopKRadixSelectKernel,
            SelectiveThresholdFloorKernel,
        )

        stream = _current_stream(self._histograms.device)
        capability = torch.cuda.get_device_capability(self._histograms.device)
        sample_threshold_args = (
            from_dlpack(self._sample_scores, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._sample_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._sample_ranks, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._row_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._thresholds, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._selected_bins, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            cutlass.Int32(self._rows),
        )
        sample_threshold_key = (*capability, self._top_k, _SAMPLE_CAPACITY, 256)
        sample_threshold_compiled = _SAMPLE_THRESHOLD_COMPILED.get(sample_threshold_key)
        if sample_threshold_compiled is None:
            sample_threshold_compiled = cute.compile(
                SelectiveSampleThresholdKernel(self._top_k),
                *sample_threshold_args,
                stream,
            )
            _SAMPLE_THRESHOLD_COMPILED[sample_threshold_key] = sample_threshold_compiled
        self._sample_threshold_launch = (
            sample_threshold_compiled,
            sample_threshold_args,
        )

        packed_topk_args = (
            from_dlpack(self._candidate_values, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(
                self._candidate_lengths, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._selected, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            cutlass.Int32(self._rows),
        )
        packed_topk_key = (*capability, self._top_k, _CANDIDATE_CAPACITY, 512)
        packed_topk_compiled = _PACKED_TOPK_COMPILED.get(packed_topk_key)
        if packed_topk_compiled is None:
            packed_topk_compiled = cute.compile(
                SelectivePackedTopKKernel(self._top_k, _CANDIDATE_CAPACITY),
                *packed_topk_args,
                stream,
            )
            _PACKED_TOPK_COMPILED[packed_topk_key] = packed_topk_compiled
        self._packed_topk_launch = (packed_topk_compiled, packed_topk_args)

        repair_gather_args = (
            from_dlpack(self._candidate_values, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(
                self._repair_lengths, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_topk_positions_i32, assumed_align=16
            ).mark_layout_dynamic(leading_dim=1),
            from_dlpack(self._repair_selected, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            cutlass.Int32(self._rows),
        )
        repair_gather_key = (*capability, self._top_k, _REPAIR_CAPACITY, 256)
        repair_gather_compiled = _PACKED_GATHER_COMPILED.get(repair_gather_key)
        if repair_gather_compiled is None:
            repair_gather_compiled = cute.compile(
                SelectivePackedGatherKernel(self._top_k, _REPAIR_CAPACITY),
                *repair_gather_args,
                stream,
            )
            _PACKED_GATHER_COMPILED[repair_gather_key] = repair_gather_compiled
        self._repair_gather_launch = (repair_gather_compiled, repair_gather_args)

        repair_scatter_args = (
            from_dlpack(self._repair_mask, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._repair_selected, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._selected, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            cutlass.Int32(self._rows),
        )
        repair_scatter_key = (*capability, self._top_k, 256)
        repair_scatter_compiled = _REPAIR_SCATTER_COMPILED.get(repair_scatter_key)
        if repair_scatter_compiled is None:
            repair_scatter_compiled = cute.compile(
                SelectiveRepairScatterKernel(self._top_k),
                *repair_scatter_args,
                stream,
            )
            _REPAIR_SCATTER_COMPILED[repair_scatter_key] = repair_scatter_compiled
        self._repair_scatter_launch = (
            repair_scatter_compiled,
            repair_scatter_args,
        )

        repair_resolve_args = (
            from_dlpack(self._candidate_values, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(
                self._repair_lengths, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._repair_topk_positions_i32, assumed_align=16
            ).mark_layout_dynamic(leading_dim=1),
            from_dlpack(
                self._repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._second_repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._row_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._selected, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._repair_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._ranks, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._prefixes, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._exact_error, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            cutlass.Int32(self._rows),
        )
        repair_resolve_key = (
            *capability,
            self._top_k,
            _REPAIR_CAPACITY,
            256,
        )
        repair_resolve_compiled = _REPAIR_RESOLVE_COMPILED.get(repair_resolve_key)
        if repair_resolve_compiled is None:
            repair_resolve_compiled = cute.compile(
                SelectiveRepairResolveKernel(self._top_k, _REPAIR_CAPACITY),
                *repair_resolve_args,
                stream,
            )
            _REPAIR_RESOLVE_COMPILED[repair_resolve_key] = repair_resolve_compiled
        self._repair_resolve_launch = (
            repair_resolve_compiled,
            repair_resolve_args,
        )

        radix_args = (
            from_dlpack(self._histograms, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._prefixes, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._ranks, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._threshold_bits, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            cutlass.Int32(self._rows),
            cutlass.Int32(24),
        )
        radix_key = (*capability, 256)
        radix_compiled = _RADIX_SELECT_COMPILED.get(radix_key)
        if radix_compiled is None:
            radix_compiled = cute.compile(
                SelectiveTopKRadixSelectKernel(), *radix_args, stream
            )
            _RADIX_SELECT_COMPILED[radix_key] = radix_compiled
        self._radix_select_launch = (radix_compiled, radix_args)

        threshold_args = (
            from_dlpack(self._thresholds, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._thresholds, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            cutlass.Int32(self._rows),
        )
        threshold_key = (*capability, 256)
        threshold_compiled = _THRESHOLD_FLOOR_COMPILED.get(threshold_key)
        if threshold_compiled is None:
            threshold_compiled = cute.compile(
                SelectiveThresholdFloorKernel(), *threshold_args, stream
            )
            _THRESHOLD_FLOOR_COMPILED[threshold_key] = threshold_compiled
        self._threshold_floor_launch = (threshold_compiled, threshold_args)

        repair_threshold_args = (
            from_dlpack(self._candidate_counts, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(
                self._repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._thresholds, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._selected_bins, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._row_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._repair_thresholds, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._repair_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            cutlass.Int32(self._rows),
        )
        repair_threshold_key = (*capability, self._top_k, 8, 256)
        repair_threshold_compiled = _REPAIR_THRESHOLD_COMPILED.get(repair_threshold_key)
        if repair_threshold_compiled is None:
            repair_threshold_compiled = cute.compile(
                SelectiveRepairThresholdKernel(self._top_k),
                *repair_threshold_args,
                stream,
            )
            _REPAIR_THRESHOLD_COMPILED[repair_threshold_key] = repair_threshold_compiled
        self._repair_threshold_launch = (
            repair_threshold_compiled,
            repair_threshold_args,
        )

        combine_args = (
            from_dlpack(self._greater_indices, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._greater_counts, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._equal_indices, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._equal_counts, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._ranks, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(
                self._second_repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(self._row_ends, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            from_dlpack(self._selected, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(self._exact_error, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, divisibility=1
            ),
            cutlass.Int32(self._rows),
        )
        combine_key = (*capability, self._top_k, 256)
        combine_compiled = _COMBINE_EXACT_COMPILED.get(combine_key)
        if combine_compiled is None:
            combine_compiled = cute.compile(
                SelectiveTopKCombineExactKernel(self._top_k), *combine_args, stream
            )
            _COMBINE_EXACT_COMPILED[combine_key] = combine_compiled
        self._combine_exact_launch = (combine_compiled, combine_args)
        zero_combine_args = (
            *combine_args[:5],
            from_dlpack(
                self._zero_repair_flags, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            from_dlpack(
                self._zero_repair_ends, assumed_align=16
            ).mark_compact_shape_dynamic(mode=0, divisibility=1),
            *combine_args[7:],
        )
        self._zero_combine_exact_launch = (combine_compiled, zero_combine_args)

    def _run_impl(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Enqueue the complete selective path and return stable TopK indices.

        Sampling, thresholding, candidate publication/finalization, TopK, and
        device-flagged exact repair all execute on the caller's current CUDA
        stream. No public phase selection or host data-dependent branch is
        involved.

        Args:
            q: Query tensor matching the storage and layout supplied to
                :meth:`plan`.
            kv_fused: Fused paged KV tensor matching :meth:`plan`.
            weights: Head weights matching :meth:`plan`.
            block_table: Physical-page map matching :meth:`plan`.

        Returns:
            Stable plan-owned int32 storage with shape ``[total_q, top_k]``.
            Each row contains request-local KV indices, and ``cu_q`` maps rows
            to requests. A row with fewer than ``top_k`` causal keys retains
            every available key and pads the suffix with request-local key
            zero. A later call overwrites the same storage.

        ``run`` performs no CuTe compilation or transient CUDA allocation after
        a successful plan and is safe to capture and replay in a CUDA graph.
        Calls on this wrapper must not overlap.
        """
        if (
            self._prepared is None
            or self._repair_prepared is None
            or self._candidate_prepared is None
        ):
            raise RuntimeError("plan must be called before run")
        assert self._sample_prepared is not None
        assert self._cu_q is not None
        assert self._cu_kv is not None
        assert self._sample_indices is not None
        assert self._sample_request_ids is not None
        assert self._sample_request_starts is not None
        assert self._sample_fused is not None
        assert self._sample_block_table is not None
        assert self._sample_scores is not None
        assert self._sample_scratch is not None
        assert self._sample_top_scores is not None
        assert self._sample_top_positions is not None
        assert self._sample_rank_indices is not None
        assert self._sample_ranks is not None
        assert self._thresholds is not None
        assert self._candidate_values is not None
        assert self._candidate_counts is not None
        assert self._candidate_lengths is not None
        assert self._repair_flags is not None
        assert self._repair_mask is not None
        assert self._repair_starts is not None
        assert self._repair_ends is not None
        assert self._repair_thresholds is not None
        assert self._repair_counts is not None
        assert self._repair_lengths is not None
        assert self._effective_lengths is not None
        assert self._second_repair_flags is not None
        assert self._row_ends is not None
        assert self._topk_row_states is not None
        assert self._repair_scores is not None
        assert self._repair_topk_positions_i32 is not None
        assert self._repair_selected is not None
        assert self._repair_topk_compiled is not None
        assert self._packed_topk_launch is not None
        assert self._repair_gather_launch is not None
        assert self._repair_scatter_launch is not None
        assert self._repair_resolve_launch is not None
        assert self._sample_threshold_launch is not None
        assert self._histograms is not None
        assert self._prefixes is not None
        assert self._ranks is not None
        assert self._threshold_bits is not None
        assert self._radix_values is not None
        assert self._greater_indices is not None
        assert self._greater_counts is not None
        assert self._equal_indices is not None
        assert self._equal_counts is not None
        assert self._selected is not None
        assert self._exact_error is not None
        assert self._zero_repair_mask is not None
        assert self._zero_repair_flags is not None
        assert self._nonzero_threshold_mask is not None
        assert self._zero_repair_ends is not None
        assert self._zero_combine_exact_launch is not None

        _validate_prepared(self._candidate_prepared, q, kv_fused, block_table)
        if (
            tuple(weights.shape) != (self._rows, self._candidate_prepared.num_heads)
            or weights.dtype != torch.float32
            or weights.device != q.device
            or not weights.is_contiguous()
        ):
            raise ValueError(
                "weights must be contiguous FP32 storage matching the prepared problem"
            )

        stream = _current_stream(q.device)
        if self._needs_sampling:
            source = kv_fused.flatten(1)
            destination = self._sample_fused.flatten(1)
            self._sample_scores.fill_(float("-inf"))
            self._sample_pack_compiled(
                from_dlpack(source, assumed_align=16).mark_layout_dynamic(
                    leading_dim=1
                ),
                from_dlpack(block_table, assumed_align=16).mark_layout_dynamic(
                    leading_dim=1
                ),
                from_dlpack(
                    self._sample_block_table, assumed_align=16
                ).mark_layout_dynamic(leading_dim=1),
                from_dlpack(
                    self._sample_request_ids, assumed_align=16
                ).mark_compact_shape_dynamic(mode=0, divisibility=1),
                from_dlpack(
                    self._sample_request_starts, assumed_align=16
                ).mark_compact_shape_dynamic(mode=0, divisibility=1),
                from_dlpack(
                    self._sample_indices, assumed_align=16
                ).mark_compact_shape_dynamic(mode=0, divisibility=1),
                from_dlpack(destination, assumed_align=16).mark_layout_dynamic(
                    leading_dim=1
                ),
                cutlass.Int32(self._sample_count),
                stream,
            )
            _fp8_mqa_selective_logits(
                q,
                self._sample_fused,
                weights,
                self._sample_block_table,
                self._sample_prepared,
                mode=_SelectiveLogitsMode.SAMPLE,
                capacity=_SAMPLE_CAPACITY,
                values=self._sample_scores,
                index_counts=self._sample_scratch,
            )
            sample_threshold_compiled, sample_threshold_args = (
                self._sample_threshold_launch
            )
            sample_threshold_compiled(*sample_threshold_args, stream)
        else:
            self._thresholds.fill_(float("-inf"))

        _fp8_mqa_selective_logits(
            q,
            kv_fused,
            weights,
            block_table,
            self._candidate_prepared,
            mode=_SelectiveLogitsMode.CANDIDATE,
            capacity=_CANDIDATE_CAPACITY,
            values=self._candidate_values,
            index_counts=self._candidate_counts,
            policy_values=self._thresholds,
        )
        _finalize_fp8_mqa_packed_candidates(
            self._candidate_counts,
            self._candidate_values,
            self._candidate_lengths,
            self._repair_flags,
            rows=self._rows,
            capacity=_CANDIDATE_CAPACITY,
            top_k=self._top_k,
        )
        packed_topk_compiled, packed_topk_args = self._packed_topk_launch
        packed_topk_compiled(*packed_topk_args, stream)

        torch.ne(self._repair_flags, 0, out=self._repair_mask)
        repair_threshold_compiled, repair_threshold_args = self._repair_threshold_launch
        repair_threshold_compiled(*repair_threshold_args, stream)
        if self._compact_repair:
            assert self._repair_active_count is not None
            assert self._packed_repair_ends is not None
            assert self._repair_row_compact_launch is not None
            assert self._repair_row_pack_launch is not None
            assert self._repair_cu_q is not None
            assert self._repair_q is not None
            assert self._repair_weights is not None
            assert self._packed_repair_starts is not None
            assert self._packed_repair_thresholds is not None
            self._repair_active_count.zero_()
            self._packed_repair_ends.zero_()
            compact_compiled, compact_args = self._repair_row_compact_launch
            compact_compiled(*compact_args, stream)
            pack_compiled, pack_args = self._repair_row_pack_launch
            pack_compiled(*pack_args, stream)
            _refresh_fp8_mqa_selective_logits(
                self._repair_prepared,
                self._repair_cu_q,
                self._cu_kv,
                row_starts=self._packed_repair_starts,
                row_ends=self._packed_repair_ends,
            )
            repair_q = self._repair_q
            repair_weights = self._repair_weights
            repair_thresholds = self._packed_repair_thresholds
        else:
            _refresh_fp8_mqa_selective_logits(
                self._repair_prepared,
                self._cu_q,
                self._cu_kv,
                row_starts=self._repair_starts,
                row_ends=self._repair_ends,
            )
            repair_q = q
            repair_weights = weights
            repair_thresholds = self._repair_thresholds
        _fp8_mqa_selective_logits(
            repair_q,
            kv_fused,
            repair_weights,
            block_table,
            self._repair_prepared,
            mode=_SelectiveLogitsMode.REPAIR,
            capacity=_REPAIR_CAPACITY,
            values=self._candidate_values,
            index_counts=self._repair_counts,
            policy_values=repair_thresholds,
        )
        _finalize_fp8_mqa_packed_candidates(
            self._repair_counts,
            self._candidate_values,
            self._repair_lengths,
            (
                self._compact_repair_flags
                if self._compact_repair
                else self._second_repair_flags
            ),
            rows=self._rows,
            capacity=_REPAIR_CAPACITY,
            top_k=self._top_k,
        )
        self._repair_topk_compiled(
            self._repair_scores,
            self._topk_row_states,
            self._repair_lengths,
            self._repair_topk_positions_i32,
            None,
        )
        if self._compact_repair:
            assert self._second_active_count is not None
            assert self._second_repair_row_pack_launch is not None
            assert self._compact_repair_resolve_launch is not None
            self._second_active_count.zero_()
            self._second_repair_flags.zero_()
            self._packed_repair_ends.zero_()
            self._exact_error.zero_()
            resolve_compiled, resolve_args = self._compact_repair_resolve_launch
            resolve_compiled(*resolve_args, stream)
            second_pack_compiled, second_pack_args = self._second_repair_row_pack_launch
            second_pack_compiled(*second_pack_args, stream)
            _refresh_fp8_mqa_selective_logits(
                self._prepared,
                self._repair_cu_q,
                self._cu_kv,
                row_starts=self._packed_repair_starts,
                row_ends=self._packed_repair_ends,
            )
        else:
            repair_resolve_compiled, repair_resolve_args = self._repair_resolve_launch
            repair_resolve_compiled(*repair_resolve_args, stream)
            _refresh_fp8_mqa_selective_logits(
                self._prepared,
                self._cu_q,
                self._cu_kv,
                row_starts=self._repair_starts,
                row_ends=self._repair_ends,
            )
        radix_compiled, radix_args = self._radix_select_launch
        for shift in (24, 16, 8, 0):
            _fp8_mqa_selective_logits(
                repair_q,
                kv_fused,
                repair_weights,
                block_table,
                self._prepared,
                mode=_SelectiveLogitsMode.RADIX_HISTOGRAM,
                capacity=1,
                values=self._radix_values,
                index_counts=self._histograms,
                policy_values=self._prefixes,
                radix_shift=shift,
            )
            radix_compiled(*radix_args[:-1], cutlass.Int32(shift), stream)

        thresholds = self._threshold_bits.view(torch.float32)
        _fp8_mqa_selective_logits(
            repair_q,
            kv_fused,
            repair_weights,
            block_table,
            self._prepared,
            mode=_SelectiveLogitsMode.EXACT_GREATER,
            capacity=self._top_k,
            values=self._greater_indices,
            index_counts=self._greater_counts,
            policy_values=thresholds,
        )
        _fp8_mqa_selective_logits(
            repair_q,
            kv_fused,
            repair_weights,
            block_table,
            self._prepared,
            mode=_SelectiveLogitsMode.EXACT_EQUAL,
            capacity=self._top_k,
            values=self._equal_indices,
            index_counts=self._equal_counts,
            policy_values=thresholds,
        )
        if self._compact_repair:
            assert self._compact_combine_exact_launch is not None
            combine_compiled, combine_args = self._compact_combine_exact_launch
        else:
            combine_compiled, combine_args = self._combine_exact_launch
        combine_compiled(*combine_args, stream)

        return self._selected

    def _run_selective(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Run the complete selective path and return stable TopK indices.

        Eager calls replay the address-stable CUDA graph prepared by
        :meth:`plan`. During an enclosing CUDA-graph capture, the individual
        operations are emitted directly so the wrapper remains composable.
        """
        if self._cuda_graph is None or self._graph_identity is None:
            raise RuntimeError("plan must be called before run")
        identity = _selective_logits_launch_identity(
            (q, kv_fused, weights, block_table), 0
        )
        if identity != self._graph_identity:
            raise ValueError("run tensors must match the address-stable plan")
        if torch.cuda.is_current_stream_capturing():
            return self._run_impl(q, kv_fused, weights, block_table)
        self._cuda_graph.replay()
        assert self._selected is not None
        return self._selected


class FP8PagedMQATopKWrapper(_FP8PagedMQASelectiveTopKEngine):
    """Compute exact paged FP8 MQA TopK with an explicit scoring strategy.

    ``strategy="full"`` materializes the production paged logits and applies
    exact variable-length TopK. ``strategy="selective"`` uses bounded
    candidate publication and exact repair. The strategy is explicit and
    remains fixed for the wrapper lifetime; no sequence-length routing occurs.

    Both strategies expose the same ``plan``/``run`` lifecycle and return
    stable request-local int32 indices with shape ``[total_q, top_k]``.
    """

    def __init__(self, *, strategy: Literal["full", "selective"]) -> None:
        if strategy not in ("full", "selective"):
            raise ValueError("strategy must be 'full' or 'selective'")
        super().__init__()
        self._strategy = strategy
        self._full_groups: list[
            tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        self._full_scores: torch.Tensor | None = None
        self._full_row_ends: torch.Tensor | None = None
        self._full_cuda_graph: torch.cuda.CUDAGraph | None = None
        self._full_graph_identity: tuple[Any, ...] | None = None
        self._full_max_context_len = 0
        self._full_impl: Any = None

    @property
    def strategy(self) -> Literal["full", "selective"]:
        """The caller-selected scoring strategy."""
        return self._strategy

    def _plan_full(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        cu_q: torch.Tensor,
        cu_kv: torch.Tensor,
        block_table: torch.Tensor,
        *,
        top_k: int,
        num_heads: int,
        head_dim: int,
        num_sms: int | None,
    ) -> None:
        """Prepare full paged logits and exact per-row TopK."""
        from ..topk_varlen.topk_varlen import top_k_varlen
        from .attn_scores import (
            compute_paged_mqa_logits_schedule,
            fp8_paged_mqa_logits,
            padded_context_len,
        )

        if num_sms is not None:
            raise ValueError("num_sms is only supported by strategy='selective'")
        if not isinstance(top_k, int) or not 0 < top_k <= _MAX_TOP_K:
            raise ValueError(f"top_k must be an integer in [1, {_MAX_TOP_K}]")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if not isinstance(head_dim, int) or head_dim <= 0 or head_dim % _FP8_MMA_INST_K:
            raise ValueError(
                f"head_dim must be a positive multiple of {_FP8_MMA_INST_K}"
            )
        if q.dim() != 3 or tuple(q.shape[1:]) != (num_heads, head_dim):
            raise ValueError(
                "q must have shape [total_q, num_heads, head_dim] matching "
                "num_heads and head_dim"
            )
        if weights.shape != (q.shape[0], num_heads):
            raise ValueError("weights must have shape [total_q, num_heads]")
        if cu_q.numel() < 2 or cu_kv.numel() != cu_q.numel():
            raise ValueError("cu_q and cu_kv must have matching [batch+1] extents")
        if block_table.dim() != 2 or block_table.shape[0] != cu_q.numel() - 1:
            raise ValueError("block_table must have one row per request")

        q_prefix = cu_q.tolist()
        kv_prefix = cu_kv.tolist()
        if q_prefix[0] != 0 or kv_prefix[0] != 0:
            raise ValueError("cu_q and cu_kv must start at zero")
        if q_prefix[-1] != q.shape[0]:
            raise ValueError("cu_q[-1] must equal q.shape[0]")
        if any(a > b for a, b in zip(q_prefix, q_prefix[1:], strict=False)):
            raise ValueError("cu_q must be monotonically nondecreasing")
        if any(a > b for a, b in zip(kv_prefix, kv_prefix[1:], strict=False)):
            raise ValueError("cu_kv must be monotonically nondecreasing")

        row_ends: list[int] = []
        chunks: list[tuple[int, int, int, int]] = []
        max_next_n = _COMPUTE_BLOCK_KV // num_heads
        if max_next_n < 1:
            raise ValueError("num_heads exceeds the full paged-logits N envelope")
        for request, (q_begin, q_end, kv_begin, kv_end) in enumerate(
            zip(
                q_prefix[:-1],
                q_prefix[1:],
                kv_prefix[:-1],
                kv_prefix[1:],
                strict=True,
            )
        ):
            q_len = q_end - q_begin
            kv_len = kv_end - kv_begin
            if q_len <= 0 or kv_len < q_len:
                raise ValueError(
                    "full causal TopK requires nonempty Q and every KV length >= Q"
                )
            first_end = kv_len - q_len + 1
            row_ends.extend(range(first_end, kv_len + 1))
            chunk_begin = q_begin
            while chunk_begin < q_end:
                next_n = min(max_next_n, q_end - chunk_begin)
                chunk_end = chunk_begin + next_n
                context_end = first_end + (chunk_end - q_begin) - 1
                chunks.append((chunk_begin, chunk_end, request, context_end))
                chunk_begin = chunk_end

        # Invalidate before replacing any storage; a failed rebuild must not
        # replay a graph bound to the previous generation's output buffers.
        self._full_graph_identity = None
        self._full_cuda_graph = None
        self._full_impl = None
        self._rows = q.shape[0]
        self._batches = len(q_prefix) - 1
        self._top_k = top_k
        self._kv_rows = kv_prefix[-1]
        self._cu_q = cu_q
        self._cu_kv = cu_kv
        self._full_max_context_len = max(row_ends)
        self._full_row_ends = torch.tensor(row_ends, dtype=torch.int32, device=q.device)
        score_columns = max(padded_context_len(self._full_max_context_len), self._top_k)
        if self._top_k == _DEFAULT_TOP_K:
            # Keep the native selector on its streaming, spill-free path. Logical
            # row lengths still exclude padding from score selection.
            score_columns = max(score_columns, 8192)
        self._full_scores = torch.empty(
            (self._rows, score_columns),
            dtype=torch.float32,
            device=q.device,
        )
        self._selected = torch.empty(
            (self._rows, top_k), dtype=torch.int32, device=q.device
        )

        self._full_groups = []
        group_begin = 0
        while group_begin < len(chunks):
            q_begin, _, _, _ = chunks[group_begin]
            next_n = chunks[group_begin][1] - q_begin
            group_end = group_begin + 1
            while group_end < len(chunks):
                prev = chunks[group_end - 1]
                current = chunks[group_end]
                if current[0] != prev[1] or current[1] - current[0] != next_n:
                    break
                group_end += 1
            selected_chunks = chunks[group_begin:group_end]
            q_end = selected_chunks[-1][1]
            context_lens = torch.tensor(
                [chunk[3] for chunk in selected_chunks],
                dtype=torch.int32,
                device=q.device,
            )
            request_ids = torch.tensor(
                [chunk[2] for chunk in selected_chunks],
                dtype=torch.int64,
                device=block_table.device,
            )
            group_block_table = block_table.index_select(0, request_ids).contiguous()
            schedule = compute_paged_mqa_logits_schedule(context_lens, device=q.device)
            self._full_groups.append(
                (
                    q_begin,
                    q_end,
                    next_n,
                    context_lens,
                    group_block_table,
                    schedule,
                    request_ids,
                )
            )
            group_begin = group_end

        full_scores = self._full_scores
        full_row_ends = self._full_row_ends
        selected = self._selected
        full_groups = tuple(self._full_groups)
        full_max_context_len = self._full_max_context_len
        selected_top_k = self._top_k
        full_hints = None
        if selected_top_k == _DEFAULT_TOP_K:
            # In-range, evenly spaced hints enable the native self-sampling
            # selector. They steer sampling, never determine the exact result.
            # Use int64 during construction to avoid overflowing long row ends.
            hint_columns = torch.arange(
                selected_top_k, dtype=torch.int64, device=q.device
            )
            full_hints = (
                full_row_ends[:, None] * hint_columns[None, :] // selected_top_k
            ).to(torch.int32)

        def full_impl() -> torch.Tensor:
            for (
                q_begin,
                q_end,
                next_n,
                context_lens,
                group_block_table,
                schedule,
                request_ids,
            ) in full_groups:
                # Page mappings are per-run payload, including graph replay.
                torch.index_select(block_table, 0, request_ids, out=group_block_table)
                batch = (q_end - q_begin) // next_n
                fp8_paged_mqa_logits(
                    q[q_begin:q_end].view(batch, next_n, num_heads, head_dim),
                    kv_fused,
                    weights[q_begin:q_end],
                    context_lens,
                    group_block_table,
                    full_max_context_len,
                    schedule_meta=schedule,
                    out=full_scores[q_begin:q_end],
                )
            # Consume causal lengths directly and write the stable int32 output;
            # a dense torch.topk would scan padding and require an index copy.
            top_k_varlen(
                full_scores,
                full_row_ends,
                selected_top_k,
                pre_idx=full_hints,
                out_indices=selected,
            )
            # Keep the fixed-width public contract for rows shorter than TopK:
            # retain every available key and fill unused slots with local key 0.
            selected.clamp_min_(0)
            return selected

        new_identity = _selective_logits_launch_identity(
            (q, kv_fused, weights, block_table), 0
        )
        full_impl()
        torch.cuda.synchronize(q.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            full_impl()
        self._full_impl = full_impl
        self._full_cuda_graph = graph
        self._full_graph_identity = new_identity

    def plan(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        cu_q: torch.Tensor,
        cu_kv: torch.Tensor,
        block_table: torch.Tensor,
        *,
        top_k: int = _DEFAULT_TOP_K,
        num_heads: int = 8,
        head_dim: int = 128,
        num_sms: int | None = None,
    ) -> None:
        """Prepare exact causal TopK on Blackwell SM100/SM103 outside graph capture.

        Args:
            q: Contiguous CUDA FP8 E4M3 queries ``[total_q, num_heads, head_dim]``.
            kv_fused: Contiguous CUDA uint8 storage
                ``[num_pages, page_size, 1, head_dim + 4]``. Each physical page
                contains all FP8 keys followed by one FP32 scale per token;
                the bytes are not interleaved per token. Page size is 32, 64,
                or 128, with sufficient table padding for 128-token tiles.
            weights: Contiguous CUDA FP32 weights ``[total_q, num_heads]``.
            cu_q: Contiguous CUDA int32 query prefix sums ``[batch + 1]``.
            cu_kv: Contiguous CUDA int32 KV prefix sums ``[batch + 1]``.
                Prefixes start at zero and are nondecreasing; each request
                must have at least as many KV tokens as query rows.
            block_table: Contiguous CUDA int32 physical page indices
                ``[batch, max_pages]``, with valid entries for padded tiles.
            top_k: Integer output width from 1 through 512, default 512.
            num_heads: Scoring head count, default 8; must admit a legal
                paged-logits compute tile for the selected strategy.
            head_dim: Head dimension, a positive multiple of 32, default 128;
                the selected tile must fit device shared memory.
            num_sms: Selective-only persistent CTA count, from 1 through the
                device SM count. None uses the device count. Full rejects it.

        All tensors must share one device. Planning reads prefixes on the host
        and allocates/precompiles run resources. Full replans replace storage;
        selective replans reuse storage when geometry and topology permit.
        Replan after changing prefixes, tensor storage/layout, or specialization,
        and recapture any enclosing graph after replanning. A failed rebuild
        invalidates run until a successful plan; early validation failures can
        leave the prior plan usable. Synchronize prior work before replanning.
        """
        if self._strategy == "selective":
            super().plan(
                q,
                kv_fused,
                weights,
                cu_q,
                cu_kv,
                block_table,
                top_k=top_k,
                num_heads=num_heads,
                head_dim=head_dim,
                num_sms=num_sms,
            )
            return
        self._plan_full(
            q,
            kv_fused,
            weights,
            cu_q,
            cu_kv,
            block_table,
            top_k=top_k,
            num_heads=num_heads,
            head_dim=head_dim,
            num_sms=num_sms,
        )

    @flashinfer_api(trace=fp8_paged_mqa_topk_trace_dispatch)
    def run(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Return plan-owned int32 request-local indices ``[total_q, top_k]``.

        Args:
            q: Query tensor with the storage, shape and layout passed to plan.
            kv_fused: Fused KV tensor with the storage and layout passed to plan.
            weights: FP32 weights with the storage and layout passed to plan.
            block_table: Page map with the storage and layout passed to plan.

        Input contents, including page mappings, may change in place between
        calls. Prefix changes require replanning. Output is overwritten on the
        next run; clone it to retain a result. Ordering is unspecified, ties may
        select equivalent keys, and underfilled rows retain all causal keys
        with remaining slots filled by request-local key zero.

        A successful plan is required. Eager calls replay the prepared graph;
        outer CUDA graph capture emits the operations into that graph. Calls
        on one wrapper must not overlap; synchronize before switching streams
        or replanning. Input storage/layout changes raise ValueError.
        """
        if self._strategy == "selective":
            return self._run_selective(q, kv_fused, weights, block_table)
        if self._full_graph_identity is None or self._full_cuda_graph is None:
            raise RuntimeError("plan must be called before run")
        identity = _selective_logits_launch_identity(
            (q, kv_fused, weights, block_table), 0
        )
        if identity != self._full_graph_identity:
            raise ValueError("run tensors must match the address-stable plan")
        if torch.cuda.is_current_stream_capturing():
            return self._full_impl()
        self._full_cuda_graph.replay()
        assert self._selected is not None
        return self._selected


__all__ = ["FP8PagedMQATopKWrapper"]
