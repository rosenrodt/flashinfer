#!/usr/bin/env python3
"""Compare complete graph-replayed FlashInfer FP8 logits-to-TopK512 routes."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROWS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
EXECUTION_SOURCE_PATHS = (
    "benchmarks/bench_fp8_mqa_selective_logits.py",
    "flashinfer/attn_scores/attn_scores.py",
    "flashinfer/attn_scores/kernels/schedule_kernel.py",
    "flashinfer/attn_scores/kernels/fp8_paged_mqa_logits.py",
    "flashinfer/attn_scores/kernels/selective_logits_metadata.py",
    "flashinfer/attn_scores/selective_logits.py",
    "flashinfer/topk_varlen/kernels/radix_topk.py",
    "flashinfer/topk_varlen/topk_varlen.py",
)


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def _summarize(samples: list[float]) -> dict[str, float]:
    return {
        "median_us": statistics.median(samples),
        "p10_us": _percentile(samples, 0.10),
        "p90_us": _percentile(samples, 0.90),
    }


def _record_event(call) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


def _source_provenance(repository: Path) -> dict:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.splitlines()
    tracked_dirty = (
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"],
            cwd=repository,
            check=False,
            timeout=5,
        ).returncode
        != 0
    )
    source_sha256 = {
        relative_path: hashlib.sha256(
            (repository / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in EXECUTION_SOURCE_PATHS
    }
    return {
        "git_status_porcelain": status,
        "tracked_worktree_clean": not tracked_dirty,
        "source_sha256": source_sha256,
    }


def _abba_events(call_a, call_b, warmups: int, repeats: int):
    """Measure identical-input routes in alternating ABBA/BAAB blocks."""
    if repeats <= 0 or repeats % 2:
        raise ValueError("ABBA measurement requires a positive even repeat count")
    for repeat in range(warmups):
        first, second = (call_a, call_b) if repeat % 2 == 0 else (call_b, call_a)
        first()
        second()
    torch.cuda.synchronize()

    samples_a: list[float] = []
    samples_b: list[float] = []
    for block in range(repeats // 2):
        order = (
            (("a", call_a), ("b", call_b), ("b", call_b), ("a", call_a))
            if block % 2 == 0
            else (("b", call_b), ("a", call_a), ("a", call_a), ("b", call_b))
        )
        for label, call in order:
            sample = _record_event(call)
            (samples_a if label == "a" else samples_b).append(sample)
    return samples_a, samples_b


def _quantize(rows: int, columns: int, *, generator: torch.Generator):
    source = (
        torch.randn(
            (rows, columns), device="cuda", dtype=torch.float32, generator=generator
        )
        * 0.1
    )
    amax = source.abs().amax(dim=1).clamp_min(1e-6)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    return (source / scale[:, None]).to(torch.float8_e4m3fn), scale


def _fuse_paged_kv(
    kv: torch.Tensor, scale: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """Pack contiguous KV and scales for the paged-logits comparison."""
    head_dim = kv.shape[1]
    valid_rows = kv.shape[0]
    num_blocks = math.ceil(valid_rows / block_size)
    padded_rows = num_blocks * block_size
    if padded_rows != valid_rows:
        padded_kv = torch.zeros(
            (padded_rows, head_dim), device=kv.device, dtype=kv.dtype
        )
        padded_scale = torch.ones(padded_rows, device=scale.device, dtype=scale.dtype)
        padded_kv[:valid_rows] = kv
        padded_scale[:valid_rows] = scale
        kv, scale = padded_kv, padded_scale
    scale_offset = block_size * head_dim
    fused = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=kv.device,
        dtype=torch.uint8,
    )
    fused[:, :scale_offset] = (
        kv.view(num_blocks, block_size, head_dim)
        .view(torch.uint8)
        .reshape(num_blocks, scale_offset)
    )
    fused[:, scale_offset:] = (
        scale.view(num_blocks, block_size)
        .contiguous()
        .view(torch.uint8)
        .reshape(num_blocks, block_size * 4)
    )
    return fused.view(num_blocks, block_size, 1, head_dim + 4)


def run_row(kv_len: int, args) -> dict:
    """Benchmark the two public QK-to-TopK512 routes on identical inputs."""
    from flashinfer import FP8PagedMQATopKWrapper

    q_len = args.q
    if kv_len < q_len:
        raise ValueError("kv length must be at least q length for causal prefixes")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    q, _q_scale = _quantize(q_len * args.num_heads, args.head_dim, generator=generator)
    q = q.reshape(q_len, args.num_heads, args.head_dim)
    kv, kv_scale = _quantize(kv_len, args.head_dim, generator=generator)
    weights = torch.randn(
        (q_len, args.num_heads),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    cu_q = torch.tensor([0, q_len], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_len], device="cuda", dtype=torch.int32)
    row_ends = (
        torch.arange(q_len, device="cuda", dtype=torch.int32) + kv_len - q_len + 1
    )
    kv_fused = _fuse_paged_kv(kv, kv_scale, args.page_size).flip(0).contiguous()
    num_blocks = kv_fused.shape[0]
    selective_block_table = torch.arange(
        num_blocks - 1, -1, -1, device="cuda", dtype=torch.int32
    )[None, :].contiguous()

    full_wrapper = FP8PagedMQATopKWrapper(strategy="full")
    full_wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        selective_block_table,
        top_k=512,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    )
    selective_wrapper = FP8PagedMQATopKWrapper(strategy="selective")
    selective_wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        selective_block_table,
        top_k=512,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    )

    def full_chain_call():
        full_wrapper.run(q, kv_fused, weights, selective_block_table)

    def selective_chain_call():
        selective_wrapper.run(q, kv_fused, weights, selective_block_table)

    full_selected = full_wrapper.run(q, kv_fused, weights, selective_block_table)
    selected = selective_wrapper.run(q, kv_fused, weights, selective_block_table)
    torch.cuda.synchronize()
    assert full_wrapper._full_scores is not None
    full_logits = full_wrapper._full_scores[:, :kv_len]
    exact_error = int(selective_wrapper._exact_error.item())
    if exact_error != 0:
        raise AssertionError(
            f"selective exact completion failed: exact_error={exact_error}"
        )
    repair_rows = int(torch.count_nonzero(selective_wrapper._repair_flags).item())
    second_repair_rows = int(
        torch.count_nonzero(selective_wrapper._second_repair_flags).item()
    )
    full_selected_rows = full_selected.to(torch.int64)
    selected_rows = selected.to(torch.int64)
    full_selected_valid = (full_selected_rows >= 0) & (
        full_selected_rows < row_ends[:, None]
    )
    selected_valid = (selected_rows >= 0) & (selected_rows < row_ends[:, None])
    if not full_selected_valid.all():
        raise AssertionError("full TopK returned a non-causal index")
    if not selected_valid.all():
        first_bad_row = int(torch.nonzero(~selected_valid, as_tuple=False)[0, 0])
        bad_values = selected_rows[first_bad_row][~selected_valid[first_bad_row]]
        raise AssertionError(
            "selective TopK returned a non-causal index: "
            f"row={first_bad_row}, row_end={int(row_ends[first_bad_row])}, "
            f"bad_min={int(bad_values.min())}, bad_max={int(bad_values.max())}, "
            f"bad_count={bad_values.numel()}"
        )
    for row, row_end in enumerate(row_ends.tolist()):
        if row_end <= 512:
            # The fixed-width consumer ABI pads underfilled rows with key zero.
            # At or below TopK, every causal key must still be retained.
            torch.testing.assert_close(
                torch.unique(full_selected_rows[row]).cpu(),
                torch.arange(row_end, dtype=torch.int64),
            )
            torch.testing.assert_close(
                torch.unique(selected_rows[row]).cpu(),
                torch.arange(row_end, dtype=torch.int64),
            )
        else:
            selected_scores = full_logits[row, selected_rows[row]]
            full_selected_scores = full_logits[row, full_selected_rows[row]]
            try:
                torch.testing.assert_close(
                    selected_scores.sort().values,
                    full_selected_scores.sort().values,
                    atol=1e-4,
                    rtol=1e-4,
                )
            except AssertionError as error:
                raise AssertionError(
                    f"TopK score mismatch at row={row}, row_end={row_end}: {error}"
                ) from error

    full_us, selective_us = _abba_events(
        full_chain_call,
        selective_chain_call,
        args.warmups,
        args.repeats,
    )
    return {
        "bs": 1,
        "q": q_len,
        "kv": kv_len,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "page_size": args.page_size,
        "route_label": "public-full-paged-topk-vs-public-selective-topk",
        "execution_mode": "cuda-graph-replay",
        "candidate_split_kv": selective_wrapper._candidate_prepared.split_kv,
        "full_paged_logits_to_topk512": {
            **_summarize(full_us),
            "raw_us": full_us,
        },
        "selective_logits_to_topk512": {
            **_summarize(selective_us),
            "raw_us": selective_us,
        },
        "fi_speedup": statistics.median(full_us) / statistics.median(selective_us),
        "correctness": {
            "topk512_score_multisets_match": True,
            "invalid_causal_scores_consumed": 0,
            "exact_error": exact_error,
            "repair_rows": repair_rows,
            "second_repair_rows": second_repair_rows,
            "physical_page_mapping": "reverse permutation",
        },
        "protocol": {
            "warmups_per_route": args.warmups,
            "samples_per_route": args.repeats,
            "order": "alternating ABBA/BAAB blocks",
            "inner": 1,
            "seed": args.seed,
            "cache_flush": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, default=4096)
    parser.add_argument("--kv", type=int, nargs="*", default=ROWS)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.q <= 0:
        raise ValueError("q must be positive")
    if args.repeats <= 0 or args.repeats % 2:
        raise ValueError("repeats must be a positive even number for ABBA measurement")
    if args.num_heads <= 0 or args.head_dim <= 0:
        raise ValueError("num-heads and head-dim must be positive")
    if args.kv != ROWS:
        print("warning: production KV subset requested")
    importlib.import_module("flashinfer.attn_scores.kernels.fp8_paged_mqa_logits")
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    result = {
        "source": {
            "git_commit": commit or None,
            "command": sys.argv,
            **_source_provenance(repository),
        },
        "device": {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
        },
        "rows": [run_row(kv_len, args) for kv_len in args.kv],
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
