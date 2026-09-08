"""Correctness tests for the selective-logits API."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from flashinfer.attn_scores.selective_logits import (
    _select_candidate_split_kv,
    _select_candidate_publication,
    _SelectiveLogitsMode,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("cutlass") is None
    or torch.cuda.get_device_capability()[0] != 10,
    reason="selective FP8 logits production requires an SM100-class CUDA device",
)

_PAGED_INPUT_CACHE: dict[
    tuple[torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]
] = {}


@pytest.mark.parametrize(
    ("q_rows", "query_tile", "expected_split"),
    ((4096, 16, 1), (8192, 16, 2), (4096, 2, 1), (8192, 2, 1)),
)
def test_candidate_split_kv_minimizes_wave_quantization(
    q_rows: int, query_tile: int, expected_split: int
):
    assert (
        _select_candidate_split_kv(
            [q_rows],
            [262144],
            query_tile=query_tile,
            num_sms=148,
        )
        == expected_split
    )


@pytest.mark.parametrize(
    ("batch_size", "expected_split"), ((1, 64), (32, 4), (64, 2), (148, 1))
)
def test_candidate_split_kv_fills_decode_wave(batch_size: int, expected_split: int):
    assert (
        _select_candidate_split_kv(
            [1] * batch_size,
            [1048576] * batch_size,
            query_tile=4,
            num_sms=148,
        )
        == expected_split
    )


@pytest.mark.parametrize(
    ("kv_length", "expected_split"), ((4096, 16), (8192, 32), (16384, 64))
)
def test_single_request_split_kv_stops_at_available_kv_pairs(
    kv_length: int, expected_split: int
):
    assert (
        _select_candidate_split_kv([1], [kv_length], query_tile=4, num_sms=148)
        == expected_split
    )


@pytest.mark.parametrize(
    ("split_kv", "expected_publication", "expected_segments"),
    (
        (1, "warp_striped", 4),
        (8, "warp_striped", 32),
        (16, "warp_pooled", 16),
        (32, "warp_pooled", 32),
        (64, "warp_pooled", 64),
    ),
)
def test_candidate_publication_is_selected_with_split_kv(
    split_kv: int, expected_publication: str, expected_segments: int
):
    assert _select_candidate_publication(split_kv) == (
        expected_publication,
        expected_segments,
    )


@pytest.mark.parametrize("split_kv", (1, 2, 4))
def test_candidate_prefill_publication_uses_lane_local_segments(split_kv):
    assert _select_candidate_publication(split_kv, lane_local=True) == (
        "lane_local",
        32 * split_kv,
    )


def test_candidate_publication_rejects_layout_from_another_split():
    inputs = _inputs(q_rows=1, kv_rows=128)
    prepared = _prepare(inputs, split_kv=1)
    values = torch.empty((16, 6144), device="cuda", dtype=torch.int64)
    split_two_lane_counts = torch.empty((16, 65), device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="split_kv=1 candidate publication requires"):
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            6144,
            values,
            split_two_lane_counts,
            prepared,
            inputs,
            torch.full((1,), float("-inf"), device="cuda"),
        )


def test_sample_pack_kernel_uses_64_bit_source_coordinates():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.attn_scores.kernels.selective_logits_metadata import (
        SelectiveLogitsSamplePackKernel,
    )
    from flashinfer.attn_scores.selective_logits import _current_stream

    head_dim = 128
    page_size = 64
    sample_page_size = 128
    source_page_bytes = page_size * (head_dim + 4)
    first_block_past_i32_byte_offset = (2**31 // source_page_bytes) + 1

    source = torch.empty(
        (first_block_past_i32_byte_offset + 1, source_page_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    expected_kv = torch.arange(head_dim, dtype=torch.uint8, device="cuda")
    expected_scale = torch.tensor([0.125], dtype=torch.float32, device="cuda").view(
        torch.uint8
    )
    source[first_block_past_i32_byte_offset, :head_dim] = expected_kv
    source[
        first_block_past_i32_byte_offset,
        page_size * head_dim : page_size * head_dim + 4,
    ] = expected_scale

    destination = torch.full(
        (1, sample_page_size * (head_dim + 4)),
        0xCD,
        dtype=torch.uint8,
        device="cuda",
    )
    block_table = torch.tensor(
        [[first_block_past_i32_byte_offset]], dtype=torch.int32, device="cuda"
    )
    sample_block_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    sample_request_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    sample_request_starts = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    sample_indices = torch.zeros(1, dtype=torch.int32, device="cuda")
    stream = _current_stream(source.device)
    args = (
        from_dlpack(source, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(block_table, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(sample_block_table, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        ),
        from_dlpack(sample_request_ids, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(sample_request_starts, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(sample_indices, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(destination, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        cutlass.Int32(1),
    )
    compiled = cute.compile(
        SelectiveLogitsSamplePackKernel(head_dim, page_size, sample_page_size),
        *args,
        stream,
    )
    compiled(*args, stream)
    torch.cuda.synchronize()

    torch.testing.assert_close(destination[0, :head_dim], expected_kv)
    scale_offset = sample_page_size * head_dim
    torch.testing.assert_close(
        destination[0, scale_offset : scale_offset + 4], expected_scale
    )


def test_phase_engine_is_not_exported_as_public_api():
    import flashinfer
    import flashinfer.attn_scores

    private_names = (
        "PreparedFP8MQASelectiveLogits",
        "SelectiveLogitsMode",
        "finalize_fp8_mqa_packed_candidates",
        "fp8_mqa_selective_logits",
        "prepare_fp8_mqa_selective_logits",
        "refresh_fp8_mqa_selective_logits",
    )
    for namespace in (flashinfer, flashinfer.attn_scores):
        assert all(not hasattr(namespace, name) for name in private_names)
        assert hasattr(namespace, "FP8PagedMQATopKWrapper")


def test_unified_wrapper_requires_explicit_supported_strategy():
    from flashinfer import FP8PagedMQATopKWrapper

    assert FP8PagedMQATopKWrapper(strategy="full").strategy == "full"
    assert FP8PagedMQATopKWrapper(strategy="selective").strategy == "selective"
    with pytest.raises(ValueError, match="strategy must be 'full' or 'selective'"):
        FP8PagedMQATopKWrapper(strategy="auto")


def _inputs(
    q_rows: int = 17,
    kv_rows: int = 387,
    num_heads: int = 8,
    head_dim: int = 128,
):
    torch.manual_seed(1)
    device = "cuda"
    q = (torch.randn(q_rows, num_heads, head_dim, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    kv_f32 = torch.randn(kv_rows, head_dim, device=device) * 0.1
    amax = kv_f32.abs().amax(dim=1).clamp_min(1e-6)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    kv = (kv_f32 / scale[:, None]).to(torch.float8_e4m3fn)
    weights = torch.randn(q_rows, num_heads, device=device, dtype=torch.float32)
    if q_rows == 1:
        cu_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
        cu_kv = torch.tensor([0, kv_rows], device=device, dtype=torch.int32)
    else:
        first_kv = min(132, kv_rows // 2)
        cu_q = torch.tensor([0, 1, q_rows], device=device, dtype=torch.int32)
        cu_kv = torch.tensor([0, first_kv, kv_rows], device=device, dtype=torch.int32)
    return q, kv, scale, weights, cu_q, cu_kv


def _valid_lengths(cu_q: torch.Tensor, cu_kv: torch.Tensor) -> list[int]:
    q_prefix = cu_q.cpu().tolist()
    kv_prefix = cu_kv.cpu().tolist()
    result = []
    for batch in range(len(q_prefix) - 1):
        q_len = q_prefix[batch + 1] - q_prefix[batch]
        kv_len = kv_prefix[batch + 1] - kv_prefix[batch]
        result.extend(kv_len - q_len + row + 1 for row in range(q_len))
    return result


def _fuse_paged_kv(
    kv: torch.Tensor, scale: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """Pack contiguous KV and scales for the public paged-logits API."""
    head_dim = kv.shape[1]
    num_blocks = kv.shape[0] // block_size
    fused = torch.zeros(
        num_blocks,
        block_size * (head_dim + 4),
        device=kv.device,
        dtype=torch.uint8,
    )
    kv_blocks = kv.view(num_blocks, block_size, head_dim)
    scale_blocks = scale.view(num_blocks, block_size)
    scale_offset = block_size * head_dim
    for block in range(num_blocks):
        fused[block, :scale_offset] = kv_blocks[block].view(torch.uint8).reshape(-1)
        fused[block, scale_offset:] = (
            scale_blocks[block].contiguous().view(torch.uint8).reshape(-1)
        )
    return fused.view(num_blocks, block_size, 1, head_dim + 4)


def _paged_inputs(inputs, block_size: int = 128):
    """Build reusable request-local pages with a reversed physical mapping."""
    q, kv, scale, weights, cu_q, cu_kv = inputs
    key = (kv, scale, block_size)
    cached = _PAGED_INPUT_CACHE.get(key)
    if cached is not None:
        return cached
    q_prefix = cu_q.cpu().tolist()
    kv_prefix = cu_kv.cpu().tolist()
    pages_per_compute_tile = 128 // block_size
    max_pages = max(
        ((kv_prefix[i + 1] - kv_prefix[i] + 127) // 128 * pages_per_compute_tile)
        for i in range(len(kv_prefix) - 1)
    )
    table = torch.zeros(
        (len(q_prefix) - 1, max_pages), device=kv.device, dtype=torch.int32
    )
    page_values = []
    page_scales = []
    for batch in range(len(q_prefix) - 1):
        begin, end = kv_prefix[batch : batch + 2]
        for local_page, offset in enumerate(range(begin, end, block_size)):
            valid = min(block_size, end - offset)
            page = torch.zeros(
                (block_size, kv.shape[1]), device=kv.device, dtype=kv.dtype
            )
            page_scale = torch.ones(block_size, device=scale.device, dtype=scale.dtype)
            page[:valid] = kv[offset : offset + valid]
            page_scale[:valid] = scale[offset : offset + valid]
            table[batch, local_page] = len(page_values)
            page_values.append(page)
            page_scales.append(page_scale)
    paged_kv = torch.stack(page_values)
    paged_scale = torch.stack(page_scales)
    fused = _fuse_paged_kv(
        paged_kv.reshape(-1, kv.shape[1]), paged_scale.reshape(-1), block_size
    )
    # Exercise a non-identity physical layout in every selective test while
    # preserving each request's logical page order through the block table.
    fused = fused.flip(0).contiguous()
    table = fused.shape[0] - 1 - table
    _PAGED_INPUT_CACHE[key] = (fused, table)
    return fused, table


def _prepare(inputs, **kwargs):
    from flashinfer.attn_scores.selective_logits import (
        _prepare_fp8_mqa_selective_logits,
    )

    q, _kv, _scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs)
    return _prepare_fp8_mqa_selective_logits(
        q, kv_fused, weights, cu_q, cu_kv, block_table, **kwargs
    )


def _oracle_rows(inputs) -> list[torch.Tensor]:
    """Evaluate the reference per-head-ReLU FP32 score definition."""
    q, kv, scale, weights, cu_q, cu_kv = inputs
    q_prefix = cu_q.cpu().tolist()
    kv_prefix = cu_kv.cpu().tolist()
    result = []
    for batch in range(len(q_prefix) - 1):
        q_begin, q_end = q_prefix[batch : batch + 2]
        kv_begin, kv_end = kv_prefix[batch : batch + 2]
        q_len = q_end - q_begin
        kv_len = kv_end - kv_begin
        for local_row in range(q_len):
            valid = kv_len - q_len + local_row + 1
            dots = torch.matmul(
                q[q_begin + local_row].float(),
                kv[kv_begin : kv_begin + valid].float().T,
            )
            weighted = (torch.relu(dots) * weights[q_begin + local_row, :, None]).sum(
                dim=0
            )
            result.append(weighted * scale[kv_begin : kv_begin + valid])
    return result


def _launch(
    mode,
    capacity,
    values,
    index_counts,
    prepared,
    inputs,
    policy_values=None,
    radix_shift=0,
):
    from flashinfer.attn_scores.selective_logits import _fp8_mqa_selective_logits

    q, _kv, _scale, weights, _cu_q, _cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs)
    _fp8_mqa_selective_logits(
        q,
        kv_fused,
        weights,
        block_table,
        prepared,
        mode=mode,
        capacity=capacity,
        values=values,
        index_counts=index_counts,
        policy_values=policy_values,
        radix_shift=radix_shift,
    )


def test_selective_sample_matches_public_paged_logits_gemm():
    """Prove both logits implementations agree on identical causal inputs."""
    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(7)
    q_rows, kv_rows = 16, 256
    q = (torch.randn(q_rows, 8, 128, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    kv_f32 = torch.randn(kv_rows, 128, device="cuda") * 0.1
    amax = kv_f32.abs().amax(dim=1).clamp_min(1e-6)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    kv = (kv_f32 / scale[:, None]).to(torch.float8_e4m3fn)
    weights = torch.randn(q_rows, 8, device="cuda", dtype=torch.float32)
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)

    prepared = _prepare(inputs)
    selective = torch.full((q_rows, kv_rows), float("nan"), device="cuda")
    unused_counts = torch.empty((q_rows, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.SAMPLE,
        kv_rows,
        selective,
        unused_counts,
        prepared,
        inputs,
    )

    paged = fp8_paged_mqa_logits(
        q.unsqueeze(0),
        _fuse_paged_kv(kv, scale),
        weights,
        torch.tensor([kv_rows], device="cuda", dtype=torch.int32),
        torch.arange(2, device="cuda", dtype=torch.int32).unsqueeze(0),
        kv_rows,
    )
    torch.cuda.synchronize()

    for row in range(q_rows):
        causal_end = kv_rows - q_rows + row + 1
        torch.testing.assert_close(
            selective[row, :causal_end],
            paged[row, :causal_end],
            atol=5e-5,
            rtol=1e-5,
        )
        assert torch.isnan(selective[row, causal_end:]).all()


def test_selective_h64_reads_arbitrary_physical_page_mapping():
    """Exercise the selective route directly with H64 and reordered pages."""
    from flashinfer.attn_scores.selective_logits import (
        _fp8_mqa_selective_logits,
        _prepare_fp8_mqa_selective_logits,
    )

    inputs = _inputs(q_rows=4, kv_rows=256, num_heads=64, head_dim=128)
    q, _kv, _scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs, block_size=64)
    prepared = _prepare_fp8_mqa_selective_logits(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_heads=64,
        head_dim=128,
    )
    assert prepared.query_tile == 2
    values = torch.full((4, 256), float("nan"), device="cuda")
    counts = torch.empty((4, 1), device="cuda", dtype=torch.int32)
    _fp8_mqa_selective_logits(
        q,
        kv_fused,
        weights,
        block_table,
        prepared,
        mode=_SelectiveLogitsMode.SAMPLE,
        capacity=256,
        values=values,
        index_counts=counts,
    )
    torch.cuda.synchronize()
    for row, expected in enumerate(_oracle_rows(inputs)):
        torch.testing.assert_close(
            values[row, : expected.numel()], expected, atol=5e-5, rtol=1e-5
        )
        assert torch.isnan(values[row, expected.numel() :]).all()

    candidates = torch.empty((4, 256), device="cuda", dtype=torch.int64)
    candidate_counts = torch.empty((4, 1), device="cuda", dtype=torch.int32)
    thresholds = torch.full((4,), float("-inf"), device="cuda")
    _fp8_mqa_selective_logits(
        q,
        kv_fused,
        weights,
        block_table,
        prepared,
        mode=_SelectiveLogitsMode.CANDIDATE,
        capacity=256,
        values=candidates,
        index_counts=candidate_counts,
        policy_values=thresholds,
    )
    torch.cuda.synchronize()
    for row, valid in enumerate(_valid_lengths(cu_q, cu_kv)):
        assert int(candidate_counts[row, 0]) == valid
        indices = (candidates[row, :valid] & 0xFFFFFFFF).sort().values
        torch.testing.assert_close(
            indices, torch.arange(valid, device="cuda", dtype=torch.int64)
        )


def test_selective_full_chain_matches_ordinary_logits_topk():
    """Use selective scoring as a complete causal logits-to-TopK stand-in."""
    from flashinfer import fp8_paged_mqa_logits
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(q_rows=16, kv_rows=1024)
    cu_q = torch.tensor([0, q.shape[0]], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    ordinary = fp8_paged_mqa_logits(
        q.unsqueeze(0),
        _fuse_paged_kv(kv, scale),
        weights,
        torch.tensor([kv.shape[0]], device="cuda", dtype=torch.int32),
        torch.arange(kv.shape[0] // 128, device="cuda", dtype=torch.int32).unsqueeze(0),
        kv.shape[0],
    )

    sample_prepared = _prepare(inputs)
    sampled = torch.full_like(ordinary, float("nan"))
    unused_counts = torch.empty((q.shape[0], 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.SAMPLE,
        kv.shape[0],
        sampled,
        unused_counts,
        sample_prepared,
        inputs,
    )
    torch.cuda.synchronize()

    valid_lengths = _valid_lengths(cu_q, cu_kv)
    thresholds = torch.empty(q.shape[0], device="cuda", dtype=torch.float32)
    ordinary_topk_scores = []
    for row, valid in enumerate(valid_lengths):
        torch.testing.assert_close(
            sampled[row, :valid], ordinary[row, :valid], atol=5e-5, rtol=1e-5
        )
        assert torch.isnan(sampled[row, valid:]).all()
        topk_scores = torch.topk(ordinary[row, :valid], 512).values
        ordinary_topk_scores.append(topk_scores)
        thresholds[row] = torch.topk(sampled[row, :valid], 512).values[-1]

    candidate_prepared = _prepare(inputs, num_sms=4, split_kv=4)
    candidate_capacity = 256
    candidates = torch.empty(
        (q.shape[0], candidate_capacity), device="cuda", dtype=torch.int64
    )
    candidate_counts = torch.empty((q.shape[0], 129), device="cuda", dtype=torch.int32)
    candidate_lengths = torch.empty(q.shape[0], device="cuda", dtype=torch.int32)
    repair_flags = torch.empty_like(candidate_lengths)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        candidate_capacity,
        candidates,
        candidate_counts,
        candidate_prepared,
        inputs,
        thresholds,
    )
    _finalize_fp8_mqa_packed_candidates(
        candidate_counts,
        candidates,
        candidate_lengths,
        repair_flags,
        rows=q.shape[0],
        capacity=candidate_capacity,
    )
    torch.cuda.synchronize()
    assert (repair_flags == 1).all()
    assert (candidate_lengths <= candidate_capacity).all()

    repair_prepared = _prepare(
        inputs,
        num_sms=4,
        kv_chunk_size=256,
    )
    repaired = torch.empty((q.shape[0], kv.shape[0]), device="cuda", dtype=torch.int64)
    repaired_counts = torch.empty((q.shape[0], 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.REPAIR,
        kv.shape[0],
        repaired,
        repaired_counts,
        repair_prepared,
        inputs,
        thresholds,
    )
    torch.cuda.synchronize()

    # Exercise the exact-completion phases as one chain rather than only as
    # isolated mode tests. The sampled TopK threshold is the K7 target key.
    threshold_bits = thresholds.view(torch.int32)
    ordered_thresholds = (
        threshold_bits
        ^ torch.where(
            threshold_bits < 0,
            torch.full_like(threshold_bits, -1),
            torch.full_like(threshold_bits, -2147483648),
        )
    ).to(torch.int64) & 0xFFFFFFFF
    radix_values = torch.empty((q.shape[0], 1), device="cuda", dtype=torch.int32)
    radix_histograms = torch.empty((q.shape[0], 256), device="cuda", dtype=torch.int32)
    for shift in (24, 16, 8, 0):
        prefixes = torch.zeros_like(threshold_bits)
        if shift != 24:
            prefixes = (ordered_thresholds >> (shift + 8)).to(torch.int32)
        _launch(
            _SelectiveLogitsMode.RADIX_HISTOGRAM,
            1,
            radix_values,
            radix_histograms,
            repair_prepared,
            inputs,
            prefixes,
            radix_shift=shift,
        )
        torch.cuda.synchronize()
        for row, valid in enumerate(valid_lengths):
            score_bits = sampled[row, :valid].view(torch.int32)
            ordered_scores = (
                score_bits
                ^ torch.where(
                    score_bits < 0,
                    torch.full_like(score_bits, -1),
                    torch.full_like(score_bits, -2147483648),
                )
            ).to(torch.int64) & 0xFFFFFFFF
            selected_scores = ordered_scores
            if shift != 24:
                selected_scores = ordered_scores[
                    (ordered_scores >> (shift + 8))
                    == (ordered_thresholds[row] >> (shift + 8))
                ]
            expected_histogram = torch.bincount(
                ((selected_scores >> shift) & 0xFF).long(), minlength=256
            )
            torch.testing.assert_close(
                radix_histograms[row], expected_histogram.to(torch.int32)
            )

    exact_values = torch.empty(
        (q.shape[0], kv.shape[0]), device="cuda", dtype=torch.int32
    )
    exact_counts = torch.empty((q.shape[0], 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.EXACT_GREATER,
        kv.shape[0],
        exact_values,
        exact_counts,
        repair_prepared,
        inputs,
        thresholds,
    )
    torch.cuda.synchronize()
    greater_rows = [
        exact_values[row, : int(exact_counts[row, 0])].clone()
        for row in range(q.shape[0])
    ]
    _launch(
        _SelectiveLogitsMode.EXACT_EQUAL,
        kv.shape[0],
        exact_values,
        exact_counts,
        repair_prepared,
        inputs,
        thresholds,
    )
    torch.cuda.synchronize()

    downstream_values = torch.linspace(
        -1.0,
        1.0,
        kv.shape[0] * 4,
        device="cuda",
        dtype=torch.float32,
    ).reshape(kv.shape[0], 4)

    for row, valid in enumerate(valid_lengths):
        count = int(repaired_counts[row, 0])
        assert count >= 512
        packed = repaired[row, :count]
        indices = (packed & 0xFFFFFFFF).to(torch.int64)
        assert ((indices >= 0) & (indices < valid)).all()
        scores = (packed >> 32).to(torch.int32).view(torch.float32)
        selective_topk_scores = torch.topk(scores, 512).values
        torch.testing.assert_close(
            selective_topk_scores.sort().values,
            ordinary_topk_scores[row].sort().values,
            atol=5e-5,
            rtol=1e-5,
        )

        greater = greater_rows[row]
        equal_count = int(exact_counts[row, 0])
        equal = exact_values[row, :equal_count].sort().values
        needed_equal = 512 - greater.numel()
        assert 0 <= needed_equal <= equal.numel()
        final_indices = torch.cat((greater, equal[:needed_equal])).to(torch.int64)
        assert final_indices.numel() == 512
        assert ((final_indices >= 0) & (final_indices < valid)).all()
        final_scores = sampled[row, final_indices]
        torch.testing.assert_close(
            final_scores.sort().values,
            ordinary_topk_scores[row].sort().values,
            atol=5e-5,
            rtol=1e-5,
        )

        # A downstream sparse consumer sees only selected indices and scores.
        # Replacing dense logits with the selective chain must preserve that
        # gathered weighted reduction.
        selective_output = (
            torch.softmax(final_scores, dim=0) @ downstream_values[final_indices]
        )
        ordinary_selected_scores = ordinary[row, final_indices]
        ordinary_output = (
            torch.softmax(ordinary_selected_scores, dim=0)
            @ downstream_values[final_indices]
        )
        torch.testing.assert_close(selective_output, ordinary_output)


def test_unified_wrapper_matches_ordinary_logits_topk():
    """The public wrapper returns final indices without exposing phase modes."""
    from flashinfer import (
        FP8PagedMQATopKWrapper,
        fp8_paged_mqa_logits,
    )

    # Exercise enough persistent Q tiles to wrap and reuse every staged
    # producer/consumer resource across successive launches.
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(q_rows=1024, kv_rows=2048)
    cu_q = torch.tensor([0, q.shape[0]], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    kv_fused = _fuse_paged_kv(kv, scale)
    block_table = torch.arange(
        kv.shape[0] // 128, device="cuda", dtype=torch.int32
    ).unsqueeze(0)
    ordinary_batch = q.shape[0] // 16
    ordinary_q = q.reshape(ordinary_batch, 16, q.shape[1], q.shape[2])
    ordinary_block_table = block_table.expand(ordinary_batch, -1).contiguous()
    ordinary = fp8_paged_mqa_logits(
        ordinary_q,
        kv_fused,
        weights,
        torch.full((ordinary_batch,), kv.shape[0], device="cuda", dtype=torch.int32),
        ordinary_block_table,
        kv.shape[0],
    ).reshape(q.shape[0], -1)

    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(q, kv_fused, weights, cu_q, cu_kv, block_table, num_sms=4)
    selected = wrapper.run(q, kv_fused, weights, block_table)
    torch.cuda.synchronize()
    first_records = [
        wrapper._candidate_values[row, : int(wrapper._candidate_lengths[row])]
        .sort()
        .values.clone()
        for row in range(q.shape[0])
    ]

    wrapper.run(q, kv_fused, weights, block_table)
    torch.cuda.synchronize()
    for row, expected_records in enumerate(first_records):
        actual_records = (
            wrapper._candidate_values[row, : int(wrapper._candidate_lengths[row])]
            .sort()
            .values
        )
        assert torch.equal(actual_records, expected_records)

    assert selected.shape == (q.shape[0], 512)
    assert selected.dtype == torch.int32
    assert int(wrapper._exact_error.item()) == 0
    for row, valid in enumerate(_valid_lengths(cu_q, cu_kv)):
        actual_indices = selected[row].long()
        assert ((actual_indices >= 0) & (actual_indices < valid)).all()
        actual_scores = ordinary[row, actual_indices]
        expected_scores = torch.topk(ordinary[row, :valid], 512).values
        torch.testing.assert_close(
            actual_scores.sort().values,
            expected_scores.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize("top_k", [8, 32, 128, 512])
def test_unified_wrapper_full_strategy_ragged_causal_topk(top_k):
    """Full and selective strategies share the exact public result contract."""
    from flashinfer import FP8PagedMQATopKWrapper

    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(q_rows=5, kv_rows=2048)
    cu_q = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, 896, 2048], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)

    wrapper = FP8PagedMQATopKWrapper(strategy="full")
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        top_k=top_k,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
    )
    first = wrapper.run(q, kv_fused, weights, block_table)
    second = wrapper.run(q, kv_fused, weights, block_table)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, kv_fused, weights, block_table)
    graph.replay()
    torch.cuda.synchronize()

    assert first.data_ptr() == second.data_ptr()
    assert first.data_ptr() == captured.data_ptr()
    assert first.shape == (q.shape[0], top_k)
    assert first.dtype == torch.int32
    for row, scores in enumerate(oracle):
        indices = first[row].long()
        assert ((indices >= 0) & (indices < scores.numel())).all()
        torch.testing.assert_close(
            scores[indices].sort().values,
            torch.topk(scores, top_k).values.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize("top_k", [1, 8, 17, 32, 128, 511, 512])
def test_unified_wrapper_ragged_batch_and_top_k(top_k):
    """Public TopK supports ragged requests and all reviewed K values."""
    from flashinfer import FP8PagedMQATopKWrapper

    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(q_rows=5, kv_rows=16384)
    cu_q = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, 7168, 16384], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)

    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        top_k=top_k,
        num_sms=4,
    )
    selected = wrapper.run(q, kv_fused, weights, block_table).long()
    torch.cuda.synchronize()

    assert selected.shape == (q.shape[0], top_k)
    assert int(wrapper._exact_error.item()) == 0
    for row, scores in enumerate(oracle):
        indices = selected[row]
        assert ((indices >= 0) & (indices < scores.numel())).all()
        torch.testing.assert_close(
            scores[indices].sort().values,
            torch.topk(scores, top_k).values.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


def test_unified_wrapper_ragged_batch_repair_cuda_graph():
    """Ragged batch metadata and exact repair remain graph capture-safe."""
    from flashinfer import FP8PagedMQATopKWrapper

    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(q_rows=5, kv_rows=16384)
    q.zero_()
    kv.zero_()
    scale.fill_(1.0)
    cu_q = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, 7168, 16384], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)

    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        top_k=32,
        num_sms=4,
    )
    eager = wrapper.run(q, kv_fused, weights, block_table)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, kv_fused, weights, block_table)
    graph.replay()
    torch.cuda.synchronize()

    assert eager.data_ptr() == captured.data_ptr()
    assert (wrapper._repair_flags != 0).all()
    assert int(wrapper._exact_error.item()) == 0
    for row, valid in enumerate(_valid_lengths(cu_q, cu_kv)):
        assert ((captured[row] >= 0) & (captured[row] < valid)).all()


@pytest.mark.parametrize("strategy", ["full", "selective"])
def test_unified_wrapper_underfilled_causal_rows(strategy):
    """Rows shorter than TopK retain every key and pad with valid key zero."""
    from flashinfer import FP8PagedMQATopKWrapper

    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(q_rows=5, kv_rows=6)
    cu_q = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, 2, 6], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)

    wrapper = FP8PagedMQATopKWrapper(strategy=strategy)
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        top_k=512,
        num_sms=4 if strategy == "selective" else None,
    )
    selected = wrapper.run(q, kv_fused, weights, block_table).long()
    torch.cuda.synchronize()

    if strategy == "selective":
        assert int(wrapper._exact_error.item()) == 0
    for row, valid in enumerate(_valid_lengths(cu_q, cu_kv)):
        indices = selected[row]
        assert ((indices >= 0) & (indices < valid)).all()
        assert torch.equal(
            torch.unique(indices).cpu(), torch.arange(valid, dtype=torch.int64)
        )


def test_unified_wrapper_compact_repair_underfilled_causal_rows():
    """Batch-one H8/D128 compact repair accepts complete short K6 rows."""
    from flashinfer import FP8PagedMQATopKWrapper

    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(q_rows=16, kv_rows=128)
    cu_q = torch.tensor([0, 16], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)

    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        top_k=512,
        num_sms=4,
    )
    selected = wrapper.run(q, kv_fused, weights, block_table).long()
    torch.cuda.synchronize()

    assert wrapper._compact_repair
    assert int(wrapper._exact_error.item()) == 0
    assert int(wrapper._second_active_count.item()) == 0
    for row, valid in enumerate(_valid_lengths(cu_q, cu_kv)):
        indices = selected[row]
        assert ((indices >= 0) & (indices < valid)).all()
        assert torch.equal(
            torch.unique(indices).cpu(), torch.arange(valid, dtype=torch.int64)
        )


@pytest.mark.parametrize(
    ("num_heads", "query_tile"),
    [(128, 1), (64, 2), (32, 4), (8, 16)],
)
def test_unified_wrapper_segmented_publication_query_tiles(num_heads, query_tile):
    """Q1/Q2/Q4/Q16 use the same segmented public candidate path."""
    from flashinfer import (
        FP8PagedMQATopKWrapper,
        fp8_paged_mqa_logits,
    )

    q_rows = query_tile + 1
    kv_rows = 131072
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=q_rows, kv_rows=kv_rows, num_heads=num_heads, head_dim=128
    )
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    kv_fused = _fuse_paged_kv(kv, scale)
    block_table = torch.arange(
        kv_rows // 128 - 1, -1, -1, device="cuda", dtype=torch.int32
    ).unsqueeze(0)
    ordinary = fp8_paged_mqa_logits(
        q.unsqueeze(0),
        kv_fused,
        weights,
        torch.tensor([kv_rows], device="cuda", dtype=torch.int32),
        block_table,
        kv_rows,
    )

    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_heads=num_heads,
        head_dim=128,
        num_sms=2,
    )
    assert wrapper._candidate_prepared.query_tile == query_tile
    assert wrapper._candidate_schedule.publication == "lane_local"
    assert wrapper._candidate_counts.shape[1] == 33
    selected = wrapper.run(q, kv_fused, weights, block_table).long()
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    if wrapper._padded_rows > q_rows:
        assert (wrapper._candidate_counts[q_rows:] == 0).all()
    for row in range(q_rows):
        row_end = kv_rows - q_rows + row + 1
        assert ((selected[row] >= 0) & (selected[row] < row_end)).all()
        actual_scores = ordinary[row, selected[row]]
        expected_scores = torch.topk(ordinary[row, :row_end], 512).values
        torch.testing.assert_close(
            actual_scores.sort().values,
            expected_scores.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize(
    ("num_heads", "q_rows", "expected_split"),
    [(8, 32, 1), (8, 64, 2), (64, 6, 1), (64, 12, 1)],
)
def test_unified_wrapper_selects_split_kv_from_sm_waves(
    num_heads, q_rows, expected_split
):
    """The public plan selects the smallest layout with best SM-wave use."""
    from flashinfer import FP8PagedMQATopKWrapper

    kv_rows = 4096
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=q_rows, kv_rows=kv_rows, num_heads=num_heads, head_dim=128
    )
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)

    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_heads=num_heads,
        head_dim=128,
        num_sms=3,
    )
    assert wrapper._candidate_prepared.split_kv == expected_split
    assert (
        wrapper._candidate_schedule.query_tile == wrapper._candidate_prepared.query_tile
    )
    assert wrapper._candidate_schedule.split_kv == expected_split
    assert wrapper._candidate_schedule.publication == "lane_local"
    assert wrapper._candidate_counts.shape[1] == 1 + 32 * expected_split
    selected = wrapper.run(q, kv_fused, weights, block_table).long()
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    for row, scores in enumerate(oracle):
        assert ((selected[row] >= 0) & (selected[row] < scores.numel())).all()
        torch.testing.assert_close(
            scores[selected[row]].sort().values,
            torch.topk(scores, 512).values.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize(
    ("num_heads", "kv_rows", "top_k", "expected_split", "force_repair"),
    (
        (8, 4096, 8, 16, False),
        (64, 8192, 17, 32, True),
        (8, 16384, 512, 64, False),
    ),
)
def test_unified_wrapper_automatic_high_split_cuda_graph(
    num_heads, kv_rows, top_k, expected_split, force_repair
):
    """Public decode planning exercises every pooled split and exact repair."""
    from flashinfer import FP8PagedMQATopKWrapper

    inputs = _inputs(q_rows=1, kv_rows=kv_rows, num_heads=num_heads, head_dim=128)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    if force_repair:
        q.zero_()
        kv.zero_()
        scale.fill_(1.0)
    kv_fused, block_table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)

    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        top_k=top_k,
        num_heads=num_heads,
        head_dim=128,
        num_sms=148,
    )
    assert wrapper._candidate_schedule.split_kv == expected_split
    assert wrapper._candidate_schedule.publication == "warp_pooled"
    assert wrapper._candidate_counts.shape[1] == expected_split + 1

    eager = wrapper.run(q, kv_fused, weights, block_table)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, kv_fused, weights, block_table)
    graph.replay()
    torch.cuda.synchronize()

    assert eager.data_ptr() == captured.data_ptr()
    assert int(wrapper._exact_error.item()) == 0
    if force_repair:
        assert (wrapper._repair_flags != 0).all()
    indices = captured[0].long()
    scores = oracle[0]
    assert ((indices >= 0) & (indices < scores.numel())).all()
    torch.testing.assert_close(
        scores[indices].sort().values,
        torch.topk(scores, top_k).values.sort().values,
        atol=5e-5,
        rtol=1e-5,
    )


def test_unified_wrapper_decode_uses_n32_request_tile():
    """Packed H8 Q1 requests avoid padding candidate scoring to MMA N128."""
    from flashinfer import FP8PagedMQATopKWrapper

    batches = 4
    kv_per_request = 2048
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=batches, kv_rows=batches * kv_per_request
    )
    cu_q = torch.arange(batches + 1, device="cuda", dtype=torch.int32)
    cu_kv = cu_q * kv_per_request
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)

    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_sms=4,
    )
    assert wrapper._candidate_prepared.query_tile == 4
    selected = wrapper.run(q, kv_fused, weights, block_table).long()
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    for row, scores in enumerate(oracle):
        torch.testing.assert_close(
            scores[selected[row]].sort().values,
            torch.topk(scores, 512).values.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


def test_unified_wrapper_split_kv_handles_heterogeneous_batch():
    """Ragged request costs share one automatic candidate publication plan."""
    from flashinfer import FP8PagedMQATopKWrapper

    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(q_rows=64, kv_rows=6144)
    cu_q = torch.tensor([0, 16, 64], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, 2048, 6144], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)

    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_sms=3,
    )
    assert wrapper._candidate_prepared.split_kv == 2
    assert wrapper._candidate_schedule.publication == "lane_local"
    assert wrapper._candidate_counts.shape[1] == 65
    selected = wrapper.run(q, kv_fused, weights, block_table).long()
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    for row, scores in enumerate(oracle):
        indices = selected[row]
        assert ((indices >= 0) & (indices < scores.numel())).all()
        torch.testing.assert_close(
            scores[indices].sort().values,
            torch.topk(scores, 512).values.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


def test_unified_wrapper_q2_segmented_overflow_repair_cuda_graph():
    """Inactive store warps retain lifecycle while Q2 repairs exact overflow."""
    from flashinfer import FP8PagedMQATopKWrapper

    q_rows = 3
    kv_rows = 131072
    num_heads = 64
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=q_rows, kv_rows=kv_rows, num_heads=num_heads, head_dim=128
    )
    q.zero_()
    kv.zero_()
    scale.fill_(1.0)
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    kv_fused = _fuse_paged_kv(kv, scale)
    block_table = torch.arange(
        kv_rows // 128 - 1, -1, -1, device="cuda", dtype=torch.int32
    ).unsqueeze(0)

    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_heads=num_heads,
        head_dim=128,
        num_sms=4,
    )
    eager = wrapper.run(q, kv_fused, weights, block_table)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, kv_fused, weights, block_table)
    graph.replay()
    torch.cuda.synchronize()

    assert eager.data_ptr() == captured.data_ptr()
    assert (wrapper._repair_flags != 0).all()
    assert int(wrapper._exact_error.item()) == 0
    assert (wrapper._candidate_counts[q_rows:] == 0).all()
    row_ends = torch.arange(
        kv_rows - q_rows + 1, kv_rows + 1, device="cuda", dtype=torch.int32
    )
    selected = captured
    assert ((selected >= 0) & (selected < row_ends[:, None])).all()


def test_unified_wrapper_precompiles_and_reuses_across_sequence_lengths():
    """Q and KV extents are runtime values, not CuTe specialization keys."""
    from flashinfer import FP8PagedMQATopKWrapper
    from flashinfer.attn_scores.selective_logits import (
        _COMBINE_EXACT_COMPILED,
        _FINALIZE_COMPILED,
        _METADATA_COMPILED,
        _RADIX_SELECT_COMPILED,
        _SAMPLE_PACK_COMPILED,
        _SELECTIVE_LOGITS_COMPILED,
        _THRESHOLD_FLOOR_COMPILED,
    )

    def make_problem(q_rows: int, kv_rows: int):
        q, kv, scale, weights, _cu_q, _cu_kv = _inputs(q_rows=q_rows, kv_rows=kv_rows)
        cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
        cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
        fused = _fuse_paged_kv(kv, scale)
        table = torch.arange(
            kv_rows // 128, device="cuda", dtype=torch.int32
        ).unsqueeze(0)
        return q, fused, weights, cu_q, cu_kv, table

    first = make_problem(16, 1024)
    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(*first, num_sms=1)
    compiled_after_plan = tuple(
        len(cache)
        for cache in (
            _METADATA_COMPILED,
            _SELECTIVE_LOGITS_COMPILED,
            _RADIX_SELECT_COMPILED,
            _COMBINE_EXACT_COMPILED,
            _FINALIZE_COMPILED,
            _SAMPLE_PACK_COMPILED,
            _THRESHOLD_FLOOR_COMPILED,
        )
    )
    wrapper.run(first[0], first[1], first[2], first[5])
    assert compiled_after_plan == tuple(
        len(cache)
        for cache in (
            _METADATA_COMPILED,
            _SELECTIVE_LOGITS_COMPILED,
            _RADIX_SELECT_COMPILED,
            _COMBINE_EXACT_COMPILED,
            _FINALIZE_COMPILED,
            _SAMPLE_PACK_COMPILED,
            _THRESHOLD_FLOOR_COMPILED,
        )
    )

    second = make_problem(32, 2048)
    second_wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    second_wrapper.plan(*second, num_sms=1)
    second_wrapper.run(second[0], second[1], second[2], second[5])
    torch.cuda.synchronize()
    assert compiled_after_plan == tuple(
        len(cache)
        for cache in (
            _METADATA_COMPILED,
            _SELECTIVE_LOGITS_COMPILED,
            _RADIX_SELECT_COMPILED,
            _COMBINE_EXACT_COMPILED,
            _FINALIZE_COMPILED,
            _SAMPLE_PACK_COMPILED,
            _THRESHOLD_FLOOR_COMPILED,
        )
    )


def test_unified_wrapper_replans_kv_length_in_place():
    """A compatible KV-length re-plan preserves storage and compiled kernels."""
    from flashinfer import FP8PagedMQATopKWrapper
    from flashinfer.attn_scores.selective_logits import (
        _COMBINE_EXACT_COMPILED,
        _FINALIZE_COMPILED,
        _METADATA_COMPILED,
        _RADIX_SELECT_COMPILED,
        _SAMPLE_PACK_COMPILED,
        _SELECTIVE_LOGITS_COMPILED,
        _THRESHOLD_FLOOR_COMPILED,
    )

    q_rows = 16
    initial_kv_rows = 12288
    replanned_kv_rows = 16384
    allocated_kv_rows = replanned_kv_rows
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=q_rows, kv_rows=allocated_kv_rows
    )
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    initial_cu_kv = torch.tensor([0, initial_kv_rows], device="cuda", dtype=torch.int32)
    replanned_cu_kv = torch.tensor(
        [0, replanned_kv_rows], device="cuda", dtype=torch.int32
    )
    kv_fused = _fuse_paged_kv(kv, scale)
    block_table = torch.arange(
        allocated_kv_rows // 128, device="cuda", dtype=torch.int32
    ).unsqueeze(0)

    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        initial_cu_kv,
        block_table,
        num_sms=4,
    )
    storage_ptrs = {
        "candidate_values": wrapper._candidate_values.data_ptr(),
        "candidate_counts": wrapper._candidate_counts.data_ptr(),
        "sample_fused": wrapper._sample_fused.data_ptr(),
        "row_ends": wrapper._row_ends.data_ptr(),
        "selected": wrapper._selected.data_ptr(),
        "candidate_context": wrapper._candidate_prepared.context_lens.data_ptr(),
        "candidate_tiles": wrapper._candidate_prepared.tile_meta.data_ptr(),
        "candidate_schedule": wrapper._candidate_prepared.schedule_meta.data_ptr(),
    }
    caches = (
        _METADATA_COMPILED,
        _SELECTIVE_LOGITS_COMPILED,
        _RADIX_SELECT_COMPILED,
        _COMBINE_EXACT_COMPILED,
        _FINALIZE_COMPILED,
        _SAMPLE_PACK_COMPILED,
        _THRESHOLD_FLOOR_COMPILED,
    )
    compiled_counts = tuple(len(cache) for cache in caches)
    first_graph = wrapper._cuda_graph
    page_order = torch.arange(
        allocated_kv_rows // 128, device="cuda", dtype=torch.int32
    ).flip(0)
    block_table.copy_(page_order.unsqueeze(0))

    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        replanned_cu_kv,
        block_table,
        num_sms=4,
    )
    selected = wrapper.run(q, kv_fused, weights, block_table).long()
    torch.cuda.synchronize()

    assert wrapper._cuda_graph is not first_graph
    assert compiled_counts == tuple(len(cache) for cache in caches)
    assert storage_ptrs == {
        "candidate_values": wrapper._candidate_values.data_ptr(),
        "candidate_counts": wrapper._candidate_counts.data_ptr(),
        "sample_fused": wrapper._sample_fused.data_ptr(),
        "row_ends": wrapper._row_ends.data_ptr(),
        "selected": wrapper._selected.data_ptr(),
        "candidate_context": wrapper._candidate_prepared.context_lens.data_ptr(),
        "candidate_tiles": wrapper._candidate_prepared.tile_meta.data_ptr(),
        "candidate_schedule": wrapper._candidate_prepared.schedule_meta.data_ptr(),
    }
    oracle = _oracle_rows(
        (
            q,
            kv.view(-1, 128, kv.shape[1])[page_order.long()].reshape(-1, kv.shape[1])[
                :replanned_kv_rows
            ],
            scale.view(-1, 128)[page_order.long()].reshape(-1)[:replanned_kv_rows],
            weights,
            cu_q,
            replanned_cu_kv,
        )
    )
    assert int(wrapper._exact_error.item()) == 0
    for row, scores in enumerate(oracle):
        indices = selected[row]
        assert ((indices >= 0) & (indices < scores.numel())).all()
        torch.testing.assert_close(
            scores[indices].sort().values,
            torch.topk(scores, 512).values.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


def test_unified_wrapper_rebinds_compatible_input_addresses():
    """Same-geometry payload tensors rebind without replacing plan storage."""
    from flashinfer import FP8PagedMQATopKWrapper

    q_rows = 16
    kv_rows = 8192
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(q_rows=q_rows, kv_rows=kv_rows)
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    kv_fused = _fuse_paged_kv(kv, scale)
    block_table = torch.arange(
        kv_rows // 128, device="cuda", dtype=torch.int32
    ).unsqueeze(0)
    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(q, kv_fused, weights, cu_q, cu_kv, block_table, num_sms=4)
    storage_ptrs = (
        wrapper._candidate_values.data_ptr(),
        wrapper._sample_fused.data_ptr(),
        wrapper._selected.data_ptr(),
    )
    first_graph = wrapper._cuda_graph

    rebound = tuple(tensor.clone() for tensor in (q, kv_fused, weights, block_table))
    wrapper.plan(
        rebound[0],
        rebound[1],
        rebound[2],
        cu_q,
        cu_kv,
        rebound[3],
        num_sms=4,
    )
    selected = wrapper.run(*rebound).long()
    torch.cuda.synchronize()

    assert wrapper._cuda_graph is not first_graph
    assert storage_ptrs == (
        wrapper._candidate_values.data_ptr(),
        wrapper._sample_fused.data_ptr(),
        wrapper._selected.data_ptr(),
    )
    oracle = _oracle_rows((rebound[0], kv, scale, rebound[2], cu_q, cu_kv))
    assert int(wrapper._exact_error.item()) == 0
    for row, scores in enumerate(oracle):
        indices = selected[row]
        torch.testing.assert_close(
            scores[indices].sort().values,
            torch.topk(scores, 512).values.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


def test_unified_wrapper_failed_replan_invalidates_generation(monkeypatch):
    """A failed in-place refresh cannot replay either mixed generation."""
    from flashinfer import FP8PagedMQATopKWrapper

    q_rows = 16
    initial_kv_rows = 8192
    replanned_kv_rows = 12288
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=q_rows, kv_rows=replanned_kv_rows
    )
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    initial_cu_kv = torch.tensor([0, initial_kv_rows], device="cuda", dtype=torch.int32)
    replanned_cu_kv = torch.tensor(
        [0, replanned_kv_rows], device="cuda", dtype=torch.int32
    )
    kv_fused = _fuse_paged_kv(kv, scale)
    block_table = torch.arange(
        replanned_kv_rows // 128, device="cuda", dtype=torch.int32
    ).unsqueeze(0)

    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        initial_cu_kv,
        block_table,
        num_sms=4,
    )
    compile_score_launches = wrapper._precompile_score_launches

    def fail_rebind(*_args, **_kwargs):
        raise RuntimeError("injected launch-rebind failure")

    monkeypatch.setattr(wrapper, "_precompile_score_launches", fail_rebind)
    with pytest.raises(RuntimeError, match="injected launch-rebind failure"):
        wrapper.plan(
            q,
            kv_fused,
            weights,
            cu_q,
            replanned_cu_kv,
            block_table,
            num_sms=4,
        )
    assert wrapper._graph_identity is None
    assert wrapper._cuda_graph is None
    with pytest.raises(RuntimeError, match="plan must be called before run"):
        wrapper.run(q, kv_fused, weights, block_table)

    monkeypatch.setattr(wrapper, "_precompile_score_launches", compile_score_launches)
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        replanned_cu_kv,
        block_table,
        num_sms=4,
    )
    selected = wrapper.run(q, kv_fused, weights, block_table)
    torch.cuda.synchronize()
    assert int(wrapper._exact_error.item()) == 0
    assert ((selected >= 0) & (selected < replanned_kv_rows)).all()


@pytest.mark.parametrize("force_ties", [False, True])
def test_unified_wrapper_sampled_common_and_exact_repair(force_ties):
    """The public run owns sampling and fail-closed repair on arbitrary pages."""
    from flashinfer import (
        FP8PagedMQATopKWrapper,
        fp8_paged_mqa_logits,
    )

    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(q_rows=16, kv_rows=8192)
    if force_ties:
        q.zero_()
        kv.zero_()
        scale.fill_(1.0)
    cu_q = torch.tensor([0, q.shape[0]], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)
    ordinary = fp8_paged_mqa_logits(
        q.unsqueeze(0),
        kv_fused,
        weights,
        torch.tensor([kv.shape[0]], device="cuda", dtype=torch.int32),
        block_table,
        kv.shape[0],
    )

    wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    wrapper.plan(q, kv_fused, weights, cu_q, cu_kv, block_table, num_sms=4)
    selected = wrapper.run(q, kv_fused, weights, block_table)
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    if force_ties:
        assert (wrapper._repair_flags != 0).all()
    for row, valid in enumerate(_valid_lengths(cu_q, cu_kv)):
        actual_indices = selected[row].long()
        assert ((actual_indices >= 0) & (actual_indices < valid)).all()
        actual_scores = ordinary[row, actual_indices]
        expected_scores = torch.topk(ordinary[row, :valid], 512).values
        torch.testing.assert_close(
            actual_scores.sort().values,
            expected_scores.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


def test_sample_candidate_overflow_and_exact_modes():
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs()
    q, kv, scale, weights, cu_q, cu_kv = inputs
    immutable = tuple(tensor.clone() for tensor in (q, kv, scale, weights))
    prepared = _prepare(inputs)
    rows = q.shape[0]
    width = 257
    sampled = torch.full((32, width), float("nan"), device="cuda")
    unused = torch.empty((32, width + 1), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="_SelectiveLogitsMode"):
        _launch("sample", width, sampled, unused, prepared, inputs)
    _launch(_SelectiveLogitsMode.SAMPLE, width, sampled, unused, prepared, inputs)
    assert prepared._launches
    cache_key = (_SelectiveLogitsMode.SAMPLE, width, 0, 1)
    cached_sample_launch = prepared._launches[cache_key]
    unused.zero_()
    _launch(_SelectiveLogitsMode.SAMPLE, width, sampled, unused, prepared, inputs)
    assert prepared._launches[cache_key] is cached_sample_launch
    torch.cuda.synchronize()

    valid_lengths = _valid_lengths(cu_q, cu_kv)
    assert valid_lengths[0] == 132
    assert valid_lengths[-1] == 255
    assert torch.isfinite(sampled[0, :132]).all()
    assert torch.isfinite(sampled[rows - 1, :255]).all()
    assert torch.isnan(sampled[0, 132:]).all()
    for row, expected in enumerate(_oracle_rows(inputs)):
        torch.testing.assert_close(
            sampled[row, : expected.numel()], expected, atol=5e-5, rtol=1e-5
        )

    thresholds = torch.stack(
        [sampled[row, valid_lengths[row] // 2] for row in range(rows)]
    )
    capacity = width
    values = torch.empty((32, capacity), device="cuda", dtype=torch.int64)
    index_counts = torch.empty((32, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        index_counts,
        prepared,
        inputs,
        thresholds,
    )
    torch.cuda.synchronize()
    candidate_cache_key = (_SelectiveLogitsMode.CANDIDATE, capacity, 0, 1)
    cached_candidate_launch = prepared._launches[candidate_cache_key]
    thresholds.add_(0.0)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        index_counts,
        prepared,
        inputs,
        thresholds,
    )
    assert prepared._launches[candidate_cache_key] is cached_candidate_launch
    torch.cuda.synchronize()
    for row, valid in enumerate(valid_lengths):
        expected = torch.nonzero(sampled[row, :valid] >= thresholds[row]).flatten()
        count = int(index_counts[row, 0])
        packed = values[row, :count]
        actual_indices = (packed & 0xFFFFFFFF).to(torch.int32)
        order = actual_indices.argsort()
        actual_indices = actual_indices[order]
        assert count == expected.numel(), (
            f"row {row}: published {count}, expected {expected.numel()}, "
            f"unique={actual_indices.unique().numel()}"
        )
        torch.testing.assert_close(
            actual_indices, expected.to(torch.int32).sort().values
        )
        actual_score_bits = ((packed[order] >> 32) & 0xFFFFFFFF).to(torch.int32)
        expected_score_bits = sampled[row, actual_indices.long()].view(torch.int32)
        torch.testing.assert_close(actual_score_bits, expected_score_bits)

    tiny_values = torch.full((32, 5), -7, device="cuda", dtype=torch.int64)
    tiny_indices = torch.full((32, 1), -11, device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.REPAIR,
        3,
        tiny_values,
        tiny_indices,
        prepared,
        inputs,
        thresholds,
    )
    torch.cuda.synchronize()
    assert torch.equal(tiny_indices[:rows, 0], index_counts[:rows, 0])
    assert (tiny_indices[:rows, 0] > 3).any()
    assert (tiny_values[:, 3:] == -7).all()
    lengths = torch.full((rows + 2,), -13, device="cuda", dtype=torch.int32)
    repair_flags = torch.full((rows + 2,), -17, device="cuda", dtype=torch.int32)
    _finalize_fp8_mqa_packed_candidates(
        tiny_indices,
        tiny_values,
        lengths,
        repair_flags,
        rows=rows,
        capacity=3,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(lengths[:rows], tiny_indices[:rows, 0].clamp(max=3))
    assert (repair_flags[:rows] == 1).all()
    assert (lengths[rows:] == -13).all()
    assert (repair_flags[rows:] == -17).all()

    exact_indices = torch.empty((32, 1), device="cuda", dtype=torch.int32)
    exact_values = torch.empty((32, width), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.EXACT_EQUAL,
        width,
        exact_values,
        exact_indices,
        prepared,
        inputs,
        thresholds,
    )
    torch.cuda.synchronize()
    for row, valid in enumerate(valid_lengths):
        expected = torch.nonzero(sampled[row, :valid] == thresholds[row]).flatten()
        count = int(exact_indices[row, 0])
        actual = exact_values[row, :count].sort().values
        torch.testing.assert_close(actual, expected.to(torch.int32).sort().values)

    _launch(
        _SelectiveLogitsMode.EXACT_GREATER,
        width,
        exact_values,
        exact_indices,
        prepared,
        inputs,
        thresholds,
    )
    torch.cuda.synchronize()
    for row, valid in enumerate(valid_lengths):
        expected = torch.nonzero(sampled[row, :valid] > thresholds[row]).flatten()
        count = int(exact_indices[row, 0])
        actual = exact_values[row, :count].sort().values
        torch.testing.assert_close(actual, expected.to(torch.int32).sort().values)

    with pytest.raises(ValueError, match="too narrow"):
        _launch(
            _SelectiveLogitsMode.SAMPLE,
            width,
            sampled,
            torch.empty((32, 0), device="cuda", dtype=torch.int32),
            prepared,
            inputs,
        )
    for before, after in zip(immutable, (q, kv, scale, weights), strict=True):
        torch.testing.assert_close(after, before)


@pytest.mark.parametrize("shift", [24, 16, 8, 0])
def test_radix_histogram_matches_sampled_score_bits(shift):
    inputs = _inputs(q_rows=1, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs)
    sampled = torch.empty((16, 130), device="cuda", dtype=torch.float32)
    unused = torch.empty((16, 256), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 130, sampled, unused, prepared, inputs)
    torch.cuda.synchronize()

    bits = sampled[0].view(torch.int32)
    ordered_i32 = bits ^ torch.where(
        bits < 0,
        torch.full_like(bits, -1),
        torch.full_like(bits, -2147483648),
    )
    ordered = ordered_i32.to(torch.int64) & 0xFFFFFFFF
    if shift == 24:
        prefix = torch.zeros(1, device="cuda", dtype=torch.int32)
        selected = ordered
    else:
        prefix = (ordered[0] >> (shift + 8)).to(torch.int32).reshape(1)
        selected = ordered[(ordered >> (shift + 8)) == prefix[0]]
    expected = torch.bincount(((selected >> shift) & 0xFF).long(), minlength=256)

    hist = torch.empty((16, 256), device="cuda", dtype=torch.int32)
    dummy = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.RADIX_HISTOGRAM,
        1,
        dummy,
        hist,
        prepared,
        inputs,
        prefix,
        radix_shift=shift,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(hist[0], expected.to(torch.int32))


def test_explicit_nonzero_row_bounds_predicate_every_store():
    inputs = _inputs(q_rows=1, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    starts = torch.tensor([17], device="cuda", dtype=torch.int32)
    ends = torch.tensor([119], device="cuda", dtype=torch.int32)
    prepared = _prepare(
        inputs,
        row_starts=starts,
        row_ends=ends,
    )
    sampled = torch.full((16, 130), float("nan"), device="cuda")
    unused = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 130, sampled, unused, prepared, inputs)
    torch.cuda.synchronize()
    assert torch.isnan(sampled[0, :17]).all()
    assert torch.isfinite(sampled[0, 17:119]).all()
    assert torch.isnan(sampled[0, 119:]).all()


def test_explicit_atomic_modes_split_one_q16_tile_across_ctas():
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=1, kv_rows=1024)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    starts = torch.zeros(1, device="cuda", dtype=torch.int32)
    ends = torch.full((1,), 1024, device="cuda", dtype=torch.int32)
    prepared = _prepare(
        inputs,
        row_starts=starts,
        row_ends=ends,
        num_sms=4,
        kv_chunk_size=256,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        prepared.schedule_meta.cpu(),
        torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0]], dtype=torch.int32),
    )

    capacity = 1024
    values = torch.empty((16, capacity), device="cuda", dtype=torch.int64)
    counts = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    lengths = torch.empty(1, device="cuda", dtype=torch.int32)
    flags = torch.empty(1, device="cuda", dtype=torch.int32)
    threshold = torch.full((1,), float("-inf"), device="cuda")
    _launch(
        _SelectiveLogitsMode.REPAIR,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        threshold,
    )
    _finalize_fp8_mqa_packed_candidates(
        counts,
        values,
        lengths,
        flags,
        rows=1,
        capacity=capacity,
    )
    torch.cuda.synchronize()
    assert int(lengths[0]) == capacity
    actual_indices = (values[0, :capacity] & 0xFFFFFFFF).sort().values
    torch.testing.assert_close(
        actual_indices, torch.arange(capacity, device="cuda", dtype=torch.int64)
    )

    segmented_counts = torch.empty((16, 5), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="requires exclusive query tiles"):
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            segmented_counts,
            prepared,
            inputs,
            threshold,
        )


def test_split_kv_two_uses_disjoint_qx9_segments():
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=17, kv_rows=1025)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(
        inputs,
        num_sms=4,
        split_kv=2,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        prepared.schedule_meta.cpu(),
        torch.tensor([[0, 0], [1, 0], [1, 2], [3, 0], [3, 0]], dtype=torch.int32),
    )

    capacity = 6144
    values = torch.empty((32, capacity), device="cuda", dtype=torch.int64)
    counts = torch.empty((32, 9), device="cuda", dtype=torch.int32)
    lengths = torch.empty(17, device="cuda", dtype=torch.int32)
    flags = torch.empty(17, device="cuda", dtype=torch.int32)
    threshold = torch.full((17,), float("-inf"), device="cuda")
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        threshold,
    )
    raw_counts = counts[:17, 1:].clone()
    _finalize_fp8_mqa_packed_candidates(
        counts, values, lengths, flags, rows=17, capacity=capacity
    )
    torch.cuda.synchronize()

    expected_lengths = torch.tensor(
        _valid_lengths(cu_q, cu_kv), device="cuda", dtype=torch.int32
    )
    torch.testing.assert_close(lengths, expected_lengths)
    torch.testing.assert_close(
        raw_counts.sum(dim=1, dtype=torch.int32), expected_lengths
    )
    assert (raw_counts[:, :4].sum(dim=1) > 0).all()
    assert raw_counts[0, 4:].sum() == 0
    assert (raw_counts[1:, 4:].sum(dim=1) > 0).all()
    for row in range(17):
        count = int(lengths[row])
        actual_indices = (values[row, :count] & 0xFFFFFFFF).sort().values
        torch.testing.assert_close(
            actual_indices, torch.arange(count, device="cuda", dtype=torch.int64)
        )

    overflow_capacity = 256
    overflow_values = torch.empty(
        (32, overflow_capacity), device="cuda", dtype=torch.int64
    )
    overflow_counts = torch.empty((32, 9), device="cuda", dtype=torch.int32)
    overflow_lengths = torch.empty(17, device="cuda", dtype=torch.int32)
    overflow_flags = torch.empty(17, device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        overflow_capacity,
        overflow_values,
        overflow_counts,
        prepared,
        inputs,
        threshold,
    )
    raw_overflow_counts = overflow_counts[:17, 1:].clone()
    _finalize_fp8_mqa_packed_candidates(
        overflow_counts,
        overflow_values,
        overflow_lengths,
        overflow_flags,
        rows=17,
        capacity=overflow_capacity,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        raw_overflow_counts.sum(dim=1, dtype=torch.int32), expected_lengths
    )
    # The one-pair request intentionally stays on split 0, so only that
    # split's half of the provisional capacity is available before repair.
    expected_overflow_lengths = expected_lengths.clamp(max=overflow_capacity)
    expected_overflow_lengths[0] = overflow_capacity // 2
    torch.testing.assert_close(overflow_lengths, expected_overflow_lengths)
    assert (overflow_flags == 1).all()

    qx5_counts = torch.empty((32, 5), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="requires Qx9"):
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            qx5_counts,
            prepared,
            inputs,
            threshold,
        )


def test_split_kv_four_uses_disjoint_qx17_segments():
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=17, kv_rows=1025)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(
        inputs,
        num_sms=4,
        split_kv=4,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        prepared.schedule_meta.cpu(),
        torch.tensor([[0, 0], [1, 1], [1, 2], [1, 3], [3, 0]], dtype=torch.int32),
    )

    capacity = 8192
    values = torch.empty((32, capacity), device="cuda", dtype=torch.int64)
    counts = torch.empty((32, 17), device="cuda", dtype=torch.int32)
    lengths = torch.empty(17, device="cuda", dtype=torch.int32)
    flags = torch.empty(17, device="cuda", dtype=torch.int32)
    threshold = torch.full((17,), float("-inf"), device="cuda")
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        threshold,
    )
    raw_counts = counts[:17, 1:].clone()
    _finalize_fp8_mqa_packed_candidates(
        counts, values, lengths, flags, rows=17, capacity=capacity
    )
    torch.cuda.synchronize()

    expected_lengths = torch.tensor(
        _valid_lengths(cu_q, cu_kv), device="cuda", dtype=torch.int32
    )
    torch.testing.assert_close(lengths, expected_lengths)
    torch.testing.assert_close(
        raw_counts.sum(dim=1, dtype=torch.int32), expected_lengths
    )
    assert (raw_counts[:, :4].sum(dim=1) > 0).all()
    assert raw_counts[0, 4:].sum() == 0
    for split_idx in range(1, 4):
        split_counts = raw_counts[1:, split_idx * 4 : (split_idx + 1) * 4]
        assert (split_counts.sum(dim=1) > 0).all()
    for row in range(17):
        count = int(lengths[row])
        actual_indices = (values[row, :count] & 0xFFFFFFFF).sort().values
        torch.testing.assert_close(
            actual_indices, torch.arange(count, device="cuda", dtype=torch.int64)
        )

    overflow_capacity = 512
    overflow_values = torch.empty(
        (32, overflow_capacity), device="cuda", dtype=torch.int64
    )
    overflow_counts = torch.empty((32, 17), device="cuda", dtype=torch.int32)
    overflow_lengths = torch.empty(17, device="cuda", dtype=torch.int32)
    overflow_flags = torch.empty(17, device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        overflow_capacity,
        overflow_values,
        overflow_counts,
        prepared,
        inputs,
        threshold,
    )
    raw_overflow_counts = overflow_counts[:17, 1:].clone()
    _finalize_fp8_mqa_packed_candidates(
        overflow_counts,
        overflow_values,
        overflow_lengths,
        overflow_flags,
        rows=17,
        capacity=overflow_capacity,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        raw_overflow_counts.sum(dim=1, dtype=torch.int32), expected_lengths
    )
    per_split_capacity = overflow_capacity // 4
    expected_overflow_lengths = torch.empty_like(expected_lengths)
    for row, logical_length in enumerate(expected_lengths.tolist()):
        num_pairs = (logical_length + 255) // 256
        active_splits = min(num_pairs, 4)
        retained = 0
        for split_idx in range(active_splits):
            split_start = (split_idx * num_pairs + active_splits - 1) // active_splits
            split_end = (
                (split_idx + 1) * num_pairs + active_splits - 1
            ) // active_splits
            split_tokens = max(
                min(split_end * 256, logical_length) - split_start * 256,
                0,
            )
            retained += min(split_tokens, per_split_capacity)
        expected_overflow_lengths[row] = retained
    torch.testing.assert_close(overflow_lengths, expected_overflow_lengths)
    assert (overflow_flags == 1).all()

    qx9_counts = torch.empty((32, 9), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="requires Qx17"):
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            qx9_counts,
            prepared,
            inputs,
            threshold,
        )

    three_split_inputs = _inputs(q_rows=1, kv_rows=641)
    prepared3 = _prepare(three_split_inputs, num_sms=4, split_kv=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        prepared3.schedule_meta.cpu(),
        torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 0]], dtype=torch.int32),
    )
    three_split_values = torch.empty((16, capacity), device="cuda", dtype=torch.int64)
    three_split_counts = torch.empty((16, 17), device="cuda", dtype=torch.int32)
    three_split_lengths = torch.empty(1, device="cuda", dtype=torch.int32)
    three_split_flags = torch.empty(1, device="cuda", dtype=torch.int32)
    three_split_threshold = torch.full((1,), float("-inf"), device="cuda")
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        three_split_values,
        three_split_counts,
        prepared3,
        three_split_inputs,
        three_split_threshold,
    )
    three_split_raw_counts = three_split_counts[0, 1:].clone()
    _finalize_fp8_mqa_packed_candidates(
        three_split_counts,
        three_split_values,
        three_split_lengths,
        three_split_flags,
        rows=1,
        capacity=capacity,
    )
    torch.cuda.synchronize()
    assert int(three_split_lengths[0]) == 641
    assert all(
        int(three_split_raw_counts[split_idx * 4 : (split_idx + 1) * 4].sum()) > 0
        for split_idx in range(3)
    )
    assert int(three_split_raw_counts[12:].sum()) == 0
    actual_indices = (three_split_values[0, :641] & 0xFFFFFFFF).sort().values
    torch.testing.assert_close(
        actual_indices, torch.arange(641, device="cuda", dtype=torch.int64)
    )


def test_split_kv_eight_uses_disjoint_qx33_warp_segments():
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=1, kv_rows=2049)
    _q, _kv, _scale, _weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs, num_sms=8, split_kv=8)
    capacity = 6144
    values = torch.empty((16, capacity), device="cuda", dtype=torch.int64)
    counts = torch.empty((16, 33), device="cuda", dtype=torch.int32)
    lengths = torch.empty(1, device="cuda", dtype=torch.int32)
    flags = torch.empty(1, device="cuda", dtype=torch.int32)
    threshold = torch.full((1,), float("-inf"), device="cuda")

    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        threshold,
    )
    raw_counts = counts[0, 1:].clone()
    _finalize_fp8_mqa_packed_candidates(
        counts, values, lengths, flags, rows=1, capacity=capacity
    )
    torch.cuda.synchronize()

    expected_length = _valid_lengths(cu_q, cu_kv)[0]
    assert int(raw_counts.sum()) == expected_length
    assert all(
        int(raw_counts[split_idx * 4 : (split_idx + 1) * 4].sum()) > 0
        for split_idx in range(8)
    )
    assert int(lengths[0]) == expected_length
    actual_indices = (values[0, :expected_length] & 0xFFFFFFFF).sort().values
    torch.testing.assert_close(
        actual_indices,
        torch.arange(expected_length, device="cuda", dtype=torch.int64),
    )


@pytest.mark.parametrize("split_kv", (16, 32, 64))
def test_high_split_kv_pools_one_ballot_segment_per_owner(split_kv: int):
    inputs = _inputs(q_rows=1, kv_rows=16385)
    prepared = _prepare(inputs, num_sms=64, split_kv=split_kv)
    capacity = 32768
    values = torch.empty((16, capacity), device="cuda", dtype=torch.int64)
    counts = torch.empty((16, split_kv + 1), device="cuda", dtype=torch.int32)
    threshold = torch.full((1,), float("-inf"), device="cuda")

    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        threshold,
    )
    torch.cuda.synchronize()

    raw_counts = counts[0, 1:].to(torch.int64)
    assert (raw_counts > 0).all()
    assert int(raw_counts.sum()) == 16385
    segment_capacity = capacity // split_kv
    published = torch.cat(
        [
            values[0, segment * segment_capacity :][: int(raw_counts[segment].item())]
            for segment in range(split_kv)
        ]
    )
    actual_indices = (published & 0xFFFFFFFF).sort().values
    torch.testing.assert_close(
        actual_indices,
        torch.arange(16385, device="cuda", dtype=torch.int64),
    )


def test_split_kv_eight_resets_warp_cursors_between_persistent_work_items():
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    batches = 32
    kv_per_request = 2049
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=batches, kv_rows=batches * kv_per_request
    )
    cu_q = torch.arange(batches + 1, device="cuda", dtype=torch.int32)
    cu_kv = cu_q * kv_per_request
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    prepared = _prepare(inputs, num_sms=16, split_kv=8)
    capacity = 6144
    values = torch.empty((batches, capacity), device="cuda", dtype=torch.int64)
    counts = torch.empty((batches, 33), device="cuda", dtype=torch.int32)
    lengths = torch.empty(batches, device="cuda", dtype=torch.int32)
    flags = torch.empty(batches, device="cuda", dtype=torch.int32)
    threshold = torch.full((batches,), float("-inf"), device="cuda")

    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        threshold,
    )
    _finalize_fp8_mqa_packed_candidates(
        counts, values, lengths, flags, rows=batches, capacity=capacity
    )
    torch.cuda.synchronize()

    assert (lengths == kv_per_request).all()
    expected = torch.arange(kv_per_request, device="cuda", dtype=torch.int64)
    for row in range(batches):
        actual = (values[row, :kv_per_request] & 0xFFFFFFFF).sort().values
        torch.testing.assert_close(actual, expected)


def test_explicit_row_end_base_preserves_inactive_tail():
    inputs = _inputs(q_rows=4, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    starts = torch.zeros(4, device="cuda", dtype=torch.int32)
    ends = torch.tensor([1, 2, 0, 0], device="cuda", dtype=torch.int32)
    prepared = _prepare(
        inputs,
        row_starts=starts,
        row_ends=ends,
        row_end_base=64,
    )
    sampled = torch.full((16, 130), float("nan"), device="cuda")
    unused = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 130, sampled, unused, prepared, inputs)
    torch.cuda.synchronize()

    assert torch.isfinite(sampled[0, :65]).all()
    assert torch.isnan(sampled[0, 65:]).all()
    assert torch.isfinite(sampled[1, :66]).all()
    assert torch.isnan(sampled[1, 66:]).all()
    assert torch.isnan(sampled[2:4]).all()


def test_all_partial_q16_request_tails_and_packed_origins():
    torch.manual_seed(2)
    q_lengths = torch.arange(1, 16, device="cuda", dtype=torch.int32)
    kv_lengths = torch.full_like(q_lengths, 260)
    kv_lengths[-1] = 257
    cu_q = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), q_lengths))
    cu_kv = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), kv_lengths))
    cu_q.cumsum_(0)
    cu_kv.cumsum_(0)
    rows = int(q_lengths.sum())
    kv_rows = int(kv_lengths.sum())
    q = (torch.randn(rows, 8, 128, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    kv_f32 = torch.randn(kv_rows, 128, device="cuda") * 0.1
    amax = kv_f32.abs().amax(dim=1).clamp_min(1e-6)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    kv = (kv_f32 / scale[:, None]).to(torch.float8_e4m3fn)
    weights = torch.randn(rows, 8, device="cuda", dtype=torch.float32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    prepared = _prepare(inputs)
    sampled = torch.full((128, 260), float("nan"), device="cuda")
    counts = torch.empty((128, 1), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 260, sampled, counts, prepared, inputs)
    torch.cuda.synchronize()

    for row, expected in enumerate(_oracle_rows(inputs)):
        torch.testing.assert_close(
            sampled[row, : expected.numel()], expected, atol=5e-5, rtol=1e-5
        )
        assert torch.isnan(sampled[row, expected.numel() :]).all()


def test_cuda_graph_refreshes_explicit_bounds_without_reallocation():
    from flashinfer.attn_scores.selective_logits import (
        _refresh_fp8_mqa_selective_logits,
    )

    inputs = _inputs(q_rows=1, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    starts = torch.tensor([17], device="cuda", dtype=torch.int32)
    ends = torch.tensor([119], device="cuda", dtype=torch.int32)
    prepared = _prepare(
        inputs,
        row_starts=starts,
        row_ends=ends,
    )
    sampled = torch.empty((16, 130), device="cuda", dtype=torch.float32)
    unused = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 130, sampled, unused, prepared, inputs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampled.fill_(float("nan"))
        _refresh_fp8_mqa_selective_logits(
            prepared,
            cu_q,
            cu_kv,
            row_starts=starts,
            row_ends=ends,
        )
        _launch(_SelectiveLogitsMode.SAMPLE, 130, sampled, unused, prepared, inputs)

    starts.fill_(23)
    ends.fill_(77)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isnan(sampled[0, :23]).all()
    assert torch.isfinite(sampled[0, 23:77]).all()
    assert torch.isnan(sampled[0, 77:]).all()


def test_cuda_graph_refreshes_row_end_base_and_inactive_tail():
    from flashinfer.attn_scores.selective_logits import (
        _refresh_fp8_mqa_selective_logits,
    )

    inputs = _inputs(q_rows=2, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    starts = torch.zeros(2, device="cuda", dtype=torch.int32)
    ends = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    prepared = _prepare(
        inputs,
        row_starts=starts,
        row_ends=ends,
        row_end_base=64,
    )
    sampled = torch.empty((16, 130), device="cuda", dtype=torch.float32)
    unused = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 130, sampled, unused, prepared, inputs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampled.fill_(float("nan"))
        _refresh_fp8_mqa_selective_logits(
            prepared,
            cu_q,
            cu_kv,
            row_starts=starts,
            row_ends=ends,
        )
        _launch(_SelectiveLogitsMode.SAMPLE, 130, sampled, unused, prepared, inputs)

    ends[0] = 0
    ends[1] = 3
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isnan(sampled[0]).all()
    assert torch.isfinite(sampled[1, :67]).all()
    assert torch.isnan(sampled[1, 67:]).all()


def test_cuda_graph_refreshes_request_prefixes_without_reallocation():
    from flashinfer.attn_scores.selective_logits import (
        _refresh_fp8_mqa_selective_logits,
    )

    inputs = _inputs()
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs)
    sampled = torch.empty((32, 257), device="cuda", dtype=torch.float32)
    counts = torch.empty((32, 1), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 257, sampled, counts, prepared, inputs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampled.fill_(float("nan"))
        _refresh_fp8_mqa_selective_logits(prepared, cu_q, cu_kv)
        _launch(_SelectiveLogitsMode.SAMPLE, 257, sampled, counts, prepared, inputs)

    cu_q[1] = 2
    graph.replay()
    torch.cuda.synchronize()
    for row, expected in enumerate(_oracle_rows(inputs)):
        torch.testing.assert_close(
            sampled[row, : expected.numel()], expected, atol=5e-5, rtol=1e-5
        )
        assert torch.isnan(sampled[row, expected.numel() :]).all()


def test_zero_ties_use_bit_exact_equal_policy():
    inputs = list(_inputs(q_rows=1, kv_rows=130))
    inputs[3].zero_()
    inputs = tuple(inputs)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs)
    values = torch.empty((16, 130), device="cuda", dtype=torch.int32)
    indices = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    positive_zero = torch.zeros(1, device="cuda", dtype=torch.float32)
    _launch(
        _SelectiveLogitsMode.EXACT_EQUAL,
        130,
        values,
        indices,
        prepared,
        inputs,
        positive_zero,
    )
    torch.cuda.synchronize()
    assert int(indices[0, 0]) == 130

    negative_zero = torch.full((1,), -0.0, device="cuda", dtype=torch.float32)
    _launch(
        _SelectiveLogitsMode.EXACT_EQUAL,
        130,
        values,
        indices,
        prepared,
        inputs,
        negative_zero,
    )
    torch.cuda.synchronize()
    assert int(indices[0, 0]) == 0


def test_cuda_graph_replays_into_caller_owned_buffers():
    inputs = _inputs(q_rows=1, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs)
    threshold = torch.zeros(1, device="cuda", dtype=torch.float32)
    values = torch.empty((16, 130), device="cuda", dtype=torch.int64)
    index_counts = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        130,
        values,
        index_counts,
        prepared,
        inputs,
        threshold,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            130,
            values,
            index_counts,
            prepared,
            inputs,
            threshold,
        )
    first = None
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        count = int(index_counts[0, 0])
        current = values[0, :count].clone()
        if first is None:
            first = current
        else:
            torch.testing.assert_close(current.sort().values, first.sort().values)
    threshold.fill_(float("inf"))
    graph.replay()
    torch.cuda.synchronize()
    assert int(index_counts[0, 0]) == 0
    threshold.zero_()
    graph.replay()
    torch.cuda.synchronize()
    restored_count = int(index_counts[0, 0])
    torch.testing.assert_close(
        values[0, :restored_count].sort().values,
        first.sort().values,
    )


@pytest.mark.parametrize(
    "metadata_name", ["context_lens", "tile_meta", "schedule_meta"]
)
def test_cached_launch_rebinds_prepared_metadata_storage(metadata_name):
    from flashinfer.attn_scores.selective_logits import (
        _refresh_fp8_mqa_selective_logits,
    )

    inputs = _inputs(q_rows=1, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs)
    values = torch.empty((16, 130), device="cuda", dtype=torch.float32)
    counts = torch.empty((16, 1), device="cuda", dtype=torch.int32)

    _launch(_SelectiveLogitsMode.SAMPLE, 130, values, counts, prepared, inputs)
    torch.cuda.synchronize()
    cache_key = (_SelectiveLogitsMode.SAMPLE, 130, 0, 1)
    cached = prepared._launches[cache_key]

    metadata = getattr(prepared, metadata_name)
    replacement = torch.empty_like(metadata)
    assert replacement.data_ptr() != metadata.data_ptr()
    metadata.set_(replacement)
    _refresh_fp8_mqa_selective_logits(prepared, cu_q, cu_kv)
    _launch(_SelectiveLogitsMode.SAMPLE, 130, values, counts, prepared, inputs)
    torch.cuda.synchronize()

    assert prepared._launches[cache_key] is not cached
    expected = _oracle_rows(inputs)[0]
    torch.testing.assert_close(
        values[0, : expected.numel()], expected, atol=5e-5, rtol=1e-5
    )


def test_segmented_candidate_lane_local_cursors_preserve_complete_rows():
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=16, kv_rows=257)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs)
    torch.cuda.synchronize()
    boundaries = prepared.schedule_meta.cpu()
    assert torch.equal(boundaries[:, 1], torch.zeros_like(boundaries[:, 1]))
    assert int(boundaries[0, 0]) == 0
    assert int(boundaries[-1, 0]) == prepared.tiles
    assert torch.all(boundaries[1:, 0] >= boundaries[:-1, 0])
    threshold = torch.full((16,), float("-inf"), device="cuda", dtype=torch.float32)
    capacity = 6120
    segmented_values = torch.empty((16, capacity), device="cuda", dtype=torch.int64)
    segmented_counts = torch.empty((16, 5), device="cuda", dtype=torch.int32)
    lengths = torch.empty(16, device="cuda", dtype=torch.int32)
    flags = torch.empty(16, device="cuda", dtype=torch.int32)

    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        segmented_values,
        segmented_counts,
        prepared,
        inputs,
        threshold,
    )
    _finalize_fp8_mqa_packed_candidates(
        segmented_counts,
        segmented_values,
        lengths,
        flags,
        rows=16,
        capacity=capacity,
    )
    torch.cuda.synchronize()

    expected_lengths = torch.tensor(
        _valid_lengths(cu_q, cu_kv), device="cuda", dtype=torch.int32
    )
    torch.testing.assert_close(lengths, expected_lengths)
    assert (flags == 1).all()
    for row in range(16):
        count = int(lengths[row])
        actual_indices = (segmented_values[row, :count] & 0xFFFFFFFF).sort().values
        torch.testing.assert_close(
            actual_indices, torch.arange(count, device="cuda", dtype=torch.int64)
        )

    overflow_capacity = 24
    overflow_values = torch.empty(
        (16, overflow_capacity), device="cuda", dtype=torch.int64
    )
    overflow_counts = torch.empty((16, 5), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        overflow_capacity,
        overflow_values,
        overflow_counts,
        prepared,
        inputs,
        threshold,
    )
    raw_overflow_counts = overflow_counts.clone()
    _finalize_fp8_mqa_packed_candidates(
        overflow_counts,
        overflow_values,
        lengths,
        flags,
        rows=16,
        capacity=overflow_capacity,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        raw_overflow_counts[:, 1:].sum(dim=1, dtype=torch.int32), expected_lengths
    )
    assert (raw_overflow_counts[:, 1:] > overflow_capacity // 4).any(dim=1).all()
    assert (overflow_counts[:, 0] == overflow_capacity + 1).all()
    torch.testing.assert_close(
        lengths,
        raw_overflow_counts[:, 1:]
        .clamp(max=overflow_capacity // 4)
        .sum(dim=1, dtype=torch.int32),
    )
    assert (lengths <= overflow_capacity).all()
    assert (flags == 1).all()


@pytest.mark.parametrize(("num_segments", "split_kv"), [(32, 1), (64, 2), (128, 4)])
def test_lane_segmented_candidates_preserve_partial_q_and_tail_rows(
    num_segments, split_kv
):
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=17, kv_rows=1025)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs, split_kv=split_kv)
    threshold = torch.full((17,), float("-inf"), device="cuda", dtype=torch.float32)
    expected_lengths = torch.tensor(
        _valid_lengths(cu_q, cu_kv), device="cuda", dtype=torch.int32
    )
    lengths = torch.empty(17, device="cuda", dtype=torch.int32)
    flags = torch.empty(17, device="cuda", dtype=torch.int32)

    capacity = 6144
    values = torch.empty((32, capacity), device="cuda", dtype=torch.int64)
    counts = torch.empty((32, num_segments + 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        threshold,
    )
    _finalize_fp8_mqa_packed_candidates(
        counts, values, lengths, flags, rows=17, capacity=capacity
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(lengths, expected_lengths)
    for row in range(17):
        count = int(lengths[row])
        actual_indices = (values[row, :count] & 0xFFFFFFFF).sort().values
        torch.testing.assert_close(
            actual_indices, torch.arange(count, device="cuda", dtype=torch.int64)
        )

    overflow_capacity = 256
    overflow_values = torch.empty(
        (32, overflow_capacity), device="cuda", dtype=torch.int64
    )
    overflow_counts = torch.empty(
        (32, num_segments + 1), device="cuda", dtype=torch.int32
    )
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        overflow_capacity,
        overflow_values,
        overflow_counts,
        prepared,
        inputs,
        threshold,
    )
    raw_overflow_counts = overflow_counts[:17].clone()
    _finalize_fp8_mqa_packed_candidates(
        overflow_counts,
        overflow_values,
        lengths,
        flags,
        rows=17,
        capacity=overflow_capacity,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        raw_overflow_counts[:, 1:].sum(dim=1, dtype=torch.int32), expected_lengths
    )
    segment_capacity = overflow_capacity // num_segments
    segment_overflow = (raw_overflow_counts[:, 1:] > segment_capacity).any(dim=1)
    assert segment_overflow.any()
    expected_published = torch.where(
        segment_overflow,
        torch.full_like(expected_lengths, overflow_capacity + 1),
        expected_lengths,
    )
    torch.testing.assert_close(overflow_counts[:17, 0], expected_published)
    torch.testing.assert_close(
        lengths,
        raw_overflow_counts[:, 1:]
        .clamp(max=segment_capacity)
        .sum(dim=1, dtype=torch.int32),
    )
    assert (flags == 1).all()


def test_adapter_rejects_unsafe_layouts_before_launch():
    from flashinfer.attn_scores.selective_logits import (
        _prepare_fp8_mqa_selective_logits,
    )

    q, kv, scale, weights, cu_q, cu_kv = _inputs(q_rows=1, kv_rows=130)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)
    raw = torch.empty(kv_fused.numel() + 1, device="cuda", dtype=torch.uint8)[1:]
    raw.copy_(kv_fused.reshape(-1))
    misaligned_kv = raw.view_as(kv_fused)
    with pytest.raises(ValueError, match="16-byte aligned"):
        _prepare_fp8_mqa_selective_logits(
            q, misaligned_kv, weights, cu_q, cu_kv, block_table
        )

    strided_weights = torch.empty((1, 16), device="cuda", dtype=torch.float32)[:, ::2]
    strided_weights.copy_(weights)
    with pytest.raises(ValueError, match="contiguous"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, strided_weights, cu_q, cu_kv, block_table
        )

    with pytest.raises(ValueError, match="split_kv must be 1, 2, 4, 8, 16, 32, or 64"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, weights, cu_q, cu_kv, block_table, split_kv=3
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _prepare_fp8_mqa_selective_logits(
            q,
            kv_fused,
            weights,
            cu_q,
            cu_kv,
            block_table,
            kv_chunk_size=256,
            split_kv=2,
        )

    q, kv, scale, weights, cu_q, _cu_kv = _inputs(q_rows=17, kv_rows=387)
    cu_kv = torch.tensor([0, 130, 387], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    prepared = _prepare(inputs)
    assert prepared.batches == 2


def test_prepared_problem_shape_configuration_reaches_compiled_kernel():
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=17, kv_rows=387, num_heads=4, head_dim=64)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(
        inputs,
        num_heads=4,
        head_dim=64,
    )
    assert prepared.num_heads == 4
    assert prepared.head_dim == 64

    values = torch.empty((32, 387), device="cuda", dtype=torch.float32)
    counts = torch.empty((32, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.SAMPLE,
        387,
        values,
        counts,
        prepared,
        inputs,
    )
    torch.cuda.synchronize()
    expected_rows = _oracle_rows(inputs)
    for row, expected in enumerate(expected_rows):
        torch.testing.assert_close(values[row, : expected.numel()], expected)

    capacity = 1024
    packed = torch.empty((32, capacity), device="cuda", dtype=torch.int64)
    segmented_counts = torch.empty((32, 33), device="cuda", dtype=torch.int32)
    thresholds = torch.full((q.shape[0],), float("-inf"), device="cuda")
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        packed,
        segmented_counts,
        prepared,
        inputs,
        thresholds,
    )
    raw_counts = segmented_counts.clone()
    lengths = torch.empty(q.shape[0], device="cuda", dtype=torch.int32)
    flags = torch.empty_like(lengths)
    _finalize_fp8_mqa_packed_candidates(
        segmented_counts,
        packed,
        lengths,
        flags,
        rows=q.shape[0],
        capacity=capacity,
    )
    torch.cuda.synchronize()
    valid_lengths = torch.tensor(
        _valid_lengths(cu_q, cu_kv), device="cuda", dtype=torch.int32
    )
    torch.testing.assert_close(
        raw_counts[: q.shape[0], 1:].sum(dim=1), valid_lengths.to(torch.int64)
    )
    torch.testing.assert_close(lengths, valid_lengths)
    for row, valid in enumerate(valid_lengths.tolist()):
        row_packed = packed[row, :valid]
        actual_indices = (row_packed & 0xFFFFFFFF).to(torch.int32)
        order = actual_indices.argsort()
        torch.testing.assert_close(
            actual_indices[order],
            torch.arange(valid, device="cuda", dtype=torch.int32),
        )
        actual_score_bits = ((row_packed[order] >> 32) & 0xFFFFFFFF).to(torch.int32)
        torch.testing.assert_close(
            actual_score_bits,
            values[row, :valid].view(torch.int32),
        )


def test_prepared_problem_shape_supports_logical_n_256():
    inputs = _inputs(q_rows=1, kv_rows=128, num_heads=256, head_dim=32)
    q, _kv, _scale, _weights, _cu_q, _cu_kv = inputs
    prepared = _prepare(inputs, num_heads=256, head_dim=32)
    assert prepared.query_tile == 1
    assert prepared.num_umma_stages == 1

    values = torch.empty((1, 128), device="cuda", dtype=torch.float32)
    counts = torch.empty((1, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.SAMPLE,
        128,
        values,
        counts,
        prepared,
        inputs,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(values[0], _oracle_rows(inputs)[0])


def test_metadata_schedule_has_no_fixed_tile_limit():
    """Runtime global scratch supports schedules beyond the old 8192-tile cap."""
    q_rows = 8193
    inputs = _inputs(q_rows=q_rows, kv_rows=8320, num_heads=256, head_dim=32)
    q, kv, scale, weights, _cu_q, _cu_kv = inputs
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)

    prepared = _prepare(inputs, num_heads=256, head_dim=32, num_sms=4)
    torch.cuda.synchronize()

    assert prepared.query_tile == 1
    assert prepared.tiles == q_rows
    assert prepared.schedule_prefix.shape == (q_rows,)
    assert int(prepared.schedule_prefix[-1]) == q_rows
    assert torch.equal(
        prepared.schedule_meta[-1].cpu(), torch.tensor([q_rows, 0], dtype=torch.int32)
    )


def test_prepared_problem_shape_rejects_unsupported_specializations():
    inputs = _inputs(q_rows=1, kv_rows=128, num_heads=4, head_dim=96)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    with pytest.raises(ValueError, match="head_dim must be a multiple"):
        _prepare(
            (
                q[:, :, :65].contiguous(),
                kv[:, :65].contiguous(),
                scale,
                weights,
                cu_q,
                cu_kv,
            ),
            num_heads=4,
            head_dim=65,
        )

    inputs = _inputs(q_rows=1, kv_rows=128, num_heads=257, head_dim=32)
    with pytest.raises(ValueError, match="no legal selective FP8 query tile"):
        _prepare(inputs, num_heads=257, head_dim=32)


def test_opt_in_validation_rejects_invalid_device_metadata(monkeypatch):
    from flashinfer.attn_scores.selective_logits import (
        _prepare_fp8_mqa_selective_logits,
    )

    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    inputs = _inputs(q_rows=17, kv_rows=387)
    q, _kv, _scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs, block_size=32)

    bad_cu_q = cu_q.clone()
    bad_cu_q[-1] = q.shape[0] - 1
    with pytest.raises(ValueError, match=r"cu_q\[-1\]"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, weights, bad_cu_q, cu_kv, block_table
        )

    bad_cu_kv = cu_kv.clone()
    bad_cu_kv[1] = -1
    with pytest.raises(ValueError, match="monotonically"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, weights, cu_q, bad_cu_kv, block_table
        )

    short_causal_kv = cu_kv.clone()
    short_causal_kv[1] = 0
    with pytest.raises(ValueError, match="greater than"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, weights, cu_q, short_causal_kv, block_table
        )

    too_short = block_table[:, :-1].contiguous()
    with pytest.raises(ValueError, match="requires at least"):
        _prepare_fp8_mqa_selective_logits(q, kv_fused, weights, cu_q, cu_kv, too_short)

    bad_table = block_table.clone()
    bad_table[0, 0] = kv_fused.shape[0]
    with pytest.raises(ValueError, match="physical block index"):
        _prepare_fp8_mqa_selective_logits(q, kv_fused, weights, cu_q, cu_kv, bad_table)

    single = _inputs(q_rows=1, kv_rows=128)
    q1, _kv1, _scale1, weights1, cu_q1, cu_kv1 = single
    fused1, table1 = _paged_inputs(single)
    starts = torch.zeros(1, device="cuda", dtype=torch.int32)
    ends = torch.full((1,), 129, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="explicit row"):
        _prepare_fp8_mqa_selective_logits(
            q1,
            fused1,
            weights1,
            cu_q1,
            cu_kv1,
            table1,
            row_starts=starts,
            row_ends=ends,
        )


@pytest.mark.parametrize("num_segments", [4, 8, 12, 16, 32, 64, 128])
def test_segmented_candidate_finalizer_compacts_in_place(num_segments):
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    capacity = (
        512
        if num_segments in (32, 64)
        else 256
        if num_segments == 128
        else 48
        if num_segments == 16
        else 24
    )
    segment_capacity = capacity // num_segments
    values = torch.full((1, capacity), -1, device="cuda", dtype=torch.int64)
    values[0, ::segment_capacity] = torch.arange(
        num_segments, device="cuda", dtype=torch.int64
    )
    counts = torch.ones((1, num_segments + 1), device="cuda", dtype=torch.int32)
    counts[:, 0].zero_()
    lengths = torch.empty(1, device="cuda", dtype=torch.int32)
    flags = torch.empty(1, device="cuda", dtype=torch.int32)
    _finalize_fp8_mqa_packed_candidates(
        counts,
        values,
        lengths,
        flags,
        rows=1,
        capacity=capacity,
    )
    torch.cuda.synchronize()
    assert int(counts[0, 0]) == num_segments
    assert int(lengths[0]) == num_segments
    assert int(flags[0]) == 1
    torch.testing.assert_close(
        values[0, :num_segments],
        torch.arange(num_segments, device="cuda", dtype=torch.int64),
    )

    counts[:, 1:].fill_(segment_capacity + 1)
    _finalize_fp8_mqa_packed_candidates(
        counts,
        values,
        lengths,
        flags,
        rows=1,
        capacity=capacity,
    )
    torch.cuda.synchronize()
    assert int(counts[0, 0]) == capacity + 1
    assert int(lengths[0]) == capacity
    assert int(flags[0]) == 1

    with pytest.raises(ValueError, match="divisible by its segment count"):
        _finalize_fp8_mqa_packed_candidates(
            counts,
            values,
            lengths,
            flags,
            rows=1,
            capacity=capacity - 1,
        )
