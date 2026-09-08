# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""GPU metadata builder for the PR-4365-derived selective-logits kernel."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass._mlir.dialects import llvm
from cutlass.utils.smem_allocator import SmemAllocator

_KV_PAIR_TOKENS = 256


class SelectiveLogitsSamplePackKernel:
    """Gather uniformly sampled logical KV rows into fixed paged storage."""

    def __init__(self, head_dim: int, page_size: int, sample_page_size: int = 128):
        self.head_dim = head_dim
        self.page_size = page_size
        self.sample_page_size = sample_page_size

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        block_table: cute.Tensor,
        sample_block_table: cute.Tensor,
        sample_request_ids: cute.Tensor,
        sample_request_starts: cute.Tensor,
        sample_indices: cute.Tensor,
        destination: cute.Tensor,
        sample_count: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        items = sample_count * (self.head_dim + 4)
        self.kernel(
            source,
            block_table,
            sample_block_table,
            sample_request_ids,
            sample_request_starts,
            sample_indices,
            destination,
            sample_count,
        ).launch(
            grid=(cute.ceil_div(items, 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        block_table: cute.Tensor,
        sample_block_table: cute.Tensor,
        sample_request_ids: cute.Tensor,
        sample_request_starts: cute.Tensor,
        sample_indices: cute.Tensor,
        destination: cute.Tensor,
        sample_count: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        item = block_idx * 256 + tidx
        row_width = cutlass.const_expr(self.head_dim + 4)
        if item < sample_count * row_width:
            sample_row = item // row_width
            component = item % row_width
            request = sample_request_ids[sample_row]
            logical_row = sample_indices[sample_row]
            destination_logical_row = sample_row - sample_request_starts[request]
            # Keep the tensor coordinates wide. Large ragged batches can place
            # later physical pages beyond a 2 GiB byte offset even though each
            # page-local row and component fit in 32 bits.
            source_block = cutlass.Int64(
                block_table[(request, logical_row // self.page_size)]
            )
            source_row = cutlass.Int64(logical_row % self.page_size)
            destination_block = cutlass.Int64(
                sample_block_table[
                    (request, destination_logical_row // self.sample_page_size)
                ]
            )
            destination_row = cutlass.Int64(
                destination_logical_row % self.sample_page_size
            )
            source_offset = cutlass.Int64(0)
            destination_offset = cutlass.Int64(0)
            if component < self.head_dim:
                source_offset = source_row * self.head_dim + component
                destination_offset = destination_row * self.head_dim + component
            else:
                scale_byte = cutlass.Int64(component - self.head_dim)
                source_offset = (
                    self.page_size * self.head_dim + source_row * 4 + scale_byte
                )
                destination_offset = (
                    self.sample_page_size * self.head_dim
                    + destination_row * 4
                    + scale_byte
                )
            source_linear = source_block * cutlass.Int64(
                source.layout.stride[0]
            ) + source_offset * cutlass.Int64(source.layout.stride[1])
            destination_linear = destination_block * cutlass.Int64(
                destination.layout.stride[0]
            ) + destination_offset * cutlass.Int64(destination.layout.stride[1])
            (destination.iterator + destination_linear).store(
                (source.iterator + source_linear).load()
            )


class SelectiveThresholdFloorKernel:
    """Round sampled order statistics down to a conservative FP16 bin edge."""

    @cute.jit
    def __call__(
        self,
        selected: cute.Tensor,
        thresholds: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(selected, thresholds, rows).launch(
            grid=(cute.ceil_div(rows, 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        selected: cute.Tensor,
        thresholds: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        row = block_idx * 256 + tidx
        if row < rows:
            half = cutlass.Float16(selected[row])
            bits = cutlass.Uint16(
                llvm.bitcast(cutlass.Uint16.mlir_type, half.ir_value())
            )
            ordered = cutlass.Uint16(0)
            if bits & cutlass.Uint16(0x8000):
                ordered = bits
            else:
                ordered = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
            ordered = (ordered & cutlass.Uint16(0xFFF0)) | cutlass.Uint16(0x000F)
            if ordered & cutlass.Uint16(0x8000):
                bits = ordered
            else:
                bits = (ordered ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
            threshold = llvm.bitcast(cutlass.Float16.mlir_type, bits.ir_value())
            thresholds[row] = cutlass.Float32(threshold)


class SelectiveSampleThresholdKernel:
    """Select conservative sampled thresholds with one CTA per query row."""

    _BINS = 2048
    _THREADS = 256

    def __init__(self, top_k: int):
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        samples: cute.Tensor,
        valid_samples: cute.Tensor,
        ranks: cute.Tensor,
        row_ends: cute.Tensor,
        thresholds: cute.Tensor,
        selected_bins: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            samples,
            valid_samples,
            ranks,
            row_ends,
            thresholds,
            selected_bins,
            rows,
        ).launch(
            grid=(rows, 1, 1),
            block=(self._THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        samples: cute.Tensor,
        valid_samples: cute.Tensor,
        ranks: cute.Tensor,
        row_ends: cute.Tensor,
        thresholds: cute.Tensor,
        selected_bins: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32

        smem = SmemAllocator()
        histogram = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((self._BINS,), order=(0,)),
            byte_alignment=16,
        )
        warp_totals = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((8,), order=(0,)),
            byte_alignment=16,
        )
        warp_bases = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((8,), order=(0,)),
            byte_alignment=16,
        )

        for item in cutlass.range_constexpr(self._BINS // self._THREADS):
            histogram[tidx + item * self._THREADS] = cutlass.Int32(0)
        cute.arch.sync_threads()

        count = valid_samples[row]
        sample = tidx
        while sample < count:
            value = samples[(row, sample)]
            half = cutlass.Float16(value)
            bits = cutlass.Uint16(
                llvm.bitcast(cutlass.Uint16.mlir_type, half.ir_value())
            )
            key = bits
            if not (bits & cutlass.Uint16(0x8000)):
                key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
            radix_bin = cutlass.Int32(key >> 5)
            cute.arch.atomic_add(
                histogram.iterator + radix_bin,
                cutlass.Int32(1),
                sem="relaxed",
                scope="cta",
            )
            sample = sample + self._THREADS
        cute.arch.sync_threads()

        interval_sum = cutlass.Int32(0)
        first_bin = tidx * (self._BINS // self._THREADS)
        for offset in cutlass.range_constexpr(self._BINS // self._THREADS):
            interval_sum = interval_sum + histogram[first_bin + offset]
        inclusive = interval_sum
        for step in cutlass.range_constexpr(5):
            offset = 1 << step
            prior = cute.arch.shuffle_sync_up(
                inclusive, offset, mask=0xFFFFFFFF, mask_and_clamp=0
            )
            if lane >= offset:
                inclusive = inclusive + prior
        if lane == 31:
            warp_totals[warp] = inclusive
        cute.arch.sync_threads()

        if warp == 0:
            own = cutlass.Int32(0)
            if lane < 8:
                own = warp_totals[lane]
            prefix = own
            for step in cutlass.range_constexpr(5):
                offset = 1 << step
                prior = cute.arch.shuffle_sync_up(
                    prefix, offset, mask=0xFFFFFFFF, mask_and_clamp=0
                )
                if lane >= offset:
                    prefix = prefix + prior
            if lane < 8:
                warp_bases[lane] = prefix - own
        cute.arch.sync_threads()

        if (row_ends[row] < self.top_k) | (count == 0):
            if tidx == 0:
                thresholds[row] = cutlass.Float32(float("-inf"))
                selected_bins[row] = cutlass.Int32(self._BINS - 1)
        else:
            target = ranks[row]
            before = warp_bases[warp] + inclusive - interval_sum
            if (before < target) & (before + interval_sum >= target):
                accumulated = before
                selected = first_bin
                found = cutlass.Boolean(False)
                for offset in cutlass.range_constexpr(self._BINS // self._THREADS):
                    accumulated = accumulated + histogram[first_bin + offset]
                    if (not found) & (accumulated >= target):
                        selected = first_bin + offset
                        found = cutlass.Boolean(True)
                selected_bins[row] = selected
                key = cutlass.Uint16(min(0x7FFF, (selected << 5) + 31))
                bits = key
                if not (key & cutlass.Uint16(0x8000)):
                    bits = (key ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
                threshold = llvm.bitcast(cutlass.Float16.mlir_type, bits.ir_value())
                thresholds[row] = cutlass.Float32(threshold)


class SelectiveRepairThresholdKernel:
    """Prepare bounded first-repair ranges and conservative thresholds."""

    def __init__(self, top_k: int, underflow_bin_step: int = 8):
        self.top_k = top_k
        self.underflow_bin_step = underflow_bin_step

    @cute.jit
    def __call__(
        self,
        candidate_counts: cute.Tensor,
        repair_flags: cute.Tensor,
        thresholds: cute.Tensor,
        selected_bins: cute.Tensor,
        row_ends: cute.Tensor,
        repair_thresholds: cute.Tensor,
        repair_ends: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            candidate_counts,
            repair_flags,
            thresholds,
            selected_bins,
            row_ends,
            repair_thresholds,
            repair_ends,
            rows,
        ).launch(
            grid=(cute.ceil_div(rows, 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        candidate_counts: cute.Tensor,
        repair_flags: cute.Tensor,
        thresholds: cute.Tensor,
        selected_bins: cute.Tensor,
        row_ends: cute.Tensor,
        repair_thresholds: cute.Tensor,
        repair_ends: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        row = block_idx * 256 + tidx
        if row < rows:
            flagged = repair_flags[row] != 0
            threshold = thresholds[row]
            if flagged:
                if row_ends[row] <= self.top_k:
                    # An underfilled causal row needs every valid key. Make
                    # the bounded K6 pass exact instead of refining a sample
                    # threshold that can legitimately omit low scores.
                    threshold = cutlass.Float32(float("-inf"))
                elif candidate_counts[(row, 0)] < self.top_k:
                    refined = min(
                        cutlass.Int32(2047),
                        selected_bins[row] + self.underflow_bin_step,
                    )
                    repair_key = cutlass.Uint16((refined << 5) + 31)
                    repair_bits = repair_key
                    if not (repair_key & cutlass.Uint16(0x8000)):
                        repair_bits = (
                            repair_key ^ cutlass.Uint16(0xFFFF)
                        ) & cutlass.Uint16(0x7FFF)
                    repair_half = llvm.bitcast(
                        cutlass.Float16.mlir_type, repair_bits.ir_value()
                    )
                    threshold = cutlass.Float32(repair_half)
            repair_thresholds[row] = threshold
            repair_ends[row] = row_ends[row] if flagged else cutlass.Int32(0)


class SelectivePackedScoresKernel:
    """Materialize valid score payloads from packed candidate records."""

    def __init__(self, capacity: int, threads: int = 256):
        self.capacity = capacity
        self.threads = threads

    @cute.jit
    def __call__(
        self,
        packed_values: cute.Tensor,
        lengths: cute.Tensor,
        scores: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(packed_values, lengths, scores, rows).launch(
            grid=(cute.ceil_div(rows * self.capacity, self.threads), 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        packed_values: cute.Tensor,
        lengths: cute.Tensor,
        scores: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        item = block_idx * self.threads + tidx
        if item < rows * self.capacity:
            row = item // self.capacity
            column = item % self.capacity
            score = cutlass.Float32(float("-inf"))
            if column < lengths[row]:
                packed = cutlass.Uint64(packed_values[(row, column)])
                score_bits = cutlass.Int32(packed >> 32)
                score = llvm.bitcast(cutlass.Float32.mlir_type, score_bits.ir_value())
            scores[(row, column)] = score


class SelectivePackedTopKKernel:
    """Select exact TopK indices directly from packed candidate records."""

    _THREADS = 512
    _COARSE_BINS = 1024

    def __init__(self, top_k: int, capacity: int):
        self.top_k = top_k
        self.capacity = capacity

    @cute.jit
    def __call__(
        self,
        packed_values: cute.Tensor,
        lengths: cute.Tensor,
        selected: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(packed_values, lengths, selected, rows).launch(
            grid=(rows, 1, 1),
            block=(self._THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def _ordered_f32(self, packed):
        bits = cutlass.Uint32(cutlass.Uint64(packed) >> 32)
        ordered = bits
        if not (bits & cutlass.Uint32(0x80000000)):
            ordered = (bits ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)
        return ordered

    @cute.jit
    def _coarse_bin(self, packed):
        bits = cutlass.Int32(cutlass.Uint64(packed) >> 32)
        value = llvm.bitcast(cutlass.Float32.mlir_type, bits.ir_value())
        half = cutlass.Float16(value)
        half_bits = cutlass.Uint16(
            llvm.bitcast(cutlass.Uint16.mlir_type, half.ir_value())
        )
        key = half_bits
        if not (half_bits & cutlass.Uint16(0x8000)):
            key = (half_bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
        return cutlass.Int32(key >> 6)

    @cute.jit
    def _index(self, packed):
        return cutlass.Int32(cutlass.Uint64(packed) & cutlass.Uint64(0xFFFFFFFF))

    @cute.kernel
    def kernel(
        self,
        packed_values: cute.Tensor,
        lengths: cute.Tensor,
        selected: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32
        cute.arch.griddepcontrol_wait()

        smem = SmemAllocator()
        histogram = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((self._COARSE_BINS,), order=(0,)),
            byte_alignment=128,
        )
        filtered_keys = smem.allocate_tensor(
            element_type=cutlass.Uint32,
            layout=cute.make_ordered_layout((self.capacity,), order=(0,)),
            byte_alignment=128,
        )
        filtered_indices = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((self.capacity,), order=(0,)),
            byte_alignment=128,
        )
        warp_totals = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((16,), order=(0,)),
            byte_alignment=64,
        )
        warp_bases = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((16,), order=(0,)),
            byte_alignment=64,
        )
        scalars = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((8,), order=(0,)),
            byte_alignment=32,
        )
        length = min(lengths[row], cutlass.Int32(self.capacity))
        if length <= self.top_k:
            rank = tidx
            while rank < self.top_k:
                index = cutlass.Int32(0)
                if rank < length:
                    index = self._index(packed_values[(row, rank)])
                selected[(row, rank)] = index
                rank = rank + self._THREADS
        else:
            for item in cutlass.range_constexpr(self._COARSE_BINS // self._THREADS):
                histogram[tidx + item * self._THREADS] = cutlass.Int32(0)
            cute.arch.sync_threads()

            item = tidx
            while item < length:
                packed = packed_values[(row, item)]
                coarse_bin = self._coarse_bin(packed)
                cute.arch.atomic_add(
                    histogram.iterator + coarse_bin,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="cta",
                )
                item = item + self._THREADS
            cute.arch.sync_threads()

            interval_sum = cutlass.Int32(0)
            first_bin = tidx * (self._COARSE_BINS // self._THREADS)
            for offset in cutlass.range_constexpr(self._COARSE_BINS // self._THREADS):
                interval_sum = interval_sum + histogram[first_bin + offset]
            inclusive = interval_sum
            for step in cutlass.range_constexpr(5):
                offset = 1 << step
                prior = cute.arch.shuffle_sync_up(
                    inclusive, offset, mask=0xFFFFFFFF, mask_and_clamp=0
                )
                if lane >= offset:
                    inclusive = inclusive + prior
            if lane == 31:
                warp_totals[warp] = inclusive
            cute.arch.sync_threads()
            if warp == 0:
                own = warp_totals[lane] if lane < 16 else cutlass.Int32(0)
                prefix_sum = own
                for step in cutlass.range_constexpr(5):
                    offset = 1 << step
                    prior = cute.arch.shuffle_sync_up(
                        prefix_sum, offset, mask=0xFFFFFFFF, mask_and_clamp=0
                    )
                    if lane >= offset:
                        prefix_sum = prefix_sum + prior
                if lane < 16:
                    warp_bases[lane] = prefix_sum - own
            cute.arch.sync_threads()

            before = warp_bases[warp] + inclusive - interval_sum
            if (before < self.top_k) & (before + interval_sum >= self.top_k):
                accumulated = before
                selected_bin = first_bin
                selected_before = before
                found = cutlass.Boolean(False)
                for offset in cutlass.range_constexpr(
                    self._COARSE_BINS // self._THREADS
                ):
                    count = histogram[first_bin + offset]
                    if (not found) & (accumulated + count >= self.top_k):
                        selected_bin = first_bin + offset
                        selected_before = accumulated
                        found = cutlass.Boolean(True)
                    accumulated = accumulated + count
                scalars[0] = selected_bin
                scalars[1] = selected_before
            if tidx < 3:
                scalars[tidx + 2] = cutlass.Int32(0)
            cute.arch.sync_threads()

            selected_coarse = scalars[0]
            item = tidx
            while item < length:
                packed = packed_values[(row, item)]
                coarse_bin = self._coarse_bin(packed)
                if coarse_bin < selected_coarse:
                    slot = cute.arch.atomic_add(
                        scalars.iterator + 4,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    if slot < self.top_k:
                        selected[(row, slot)] = self._index(packed)
                elif coarse_bin == selected_coarse:
                    slot = cute.arch.atomic_add(
                        scalars.iterator + 2,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    if slot < self.capacity:
                        filtered_keys[slot] = self._ordered_f32(packed)
                        filtered_indices[slot] = self._index(packed)
                item = item + self._THREADS
            cute.arch.sync_threads()

            filtered_count = scalars[2]
            remaining = cutlass.Int32(self.top_k) - scalars[4]
            prefix = cutlass.Uint32(0)
            for radix_round in cutlass.range_constexpr(4):
                shift = 24 - radix_round * 8
                if tidx < 256:
                    histogram[tidx] = cutlass.Int32(0)
                cute.arch.sync_threads()
                item = tidx
                while item < filtered_count:
                    key = filtered_keys[item]
                    matches = cutlass.Boolean(True)
                    if cutlass.const_expr(radix_round > 0):
                        matches = (key >> (shift + 8)) == (prefix >> (shift + 8))
                    if matches:
                        radix_bin = cutlass.Int32((key >> shift) & 0xFF)
                        cute.arch.atomic_add(
                            histogram.iterator + radix_bin,
                            cutlass.Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                    item = item + self._THREADS
                cute.arch.sync_threads()

                own = histogram[tidx] if tidx < 256 else cutlass.Int32(0)
                inclusive = own
                for step in cutlass.range_constexpr(5):
                    offset = 1 << step
                    prior = cute.arch.shuffle_sync_up(
                        inclusive, offset, mask=0xFFFFFFFF, mask_and_clamp=0
                    )
                    if lane >= offset:
                        inclusive = inclusive + prior
                if lane == 31:
                    warp_totals[warp] = inclusive
                cute.arch.sync_threads()
                if warp == 0:
                    warp_own = warp_totals[lane] if lane < 16 else cutlass.Int32(0)
                    warp_prefix = warp_own
                    for step in cutlass.range_constexpr(5):
                        offset = 1 << step
                        prior = cute.arch.shuffle_sync_up(
                            warp_prefix, offset, mask=0xFFFFFFFF, mask_and_clamp=0
                        )
                        if lane >= offset:
                            warp_prefix = warp_prefix + prior
                    if lane < 16:
                        warp_bases[lane] = warp_prefix - warp_own
                cute.arch.sync_threads()
                bin_before = warp_bases[warp] + inclusive - own
                if (bin_before < remaining) & (bin_before + own >= remaining):
                    scalars[5] = tidx
                    scalars[6] = remaining - bin_before
                cute.arch.sync_threads()
                selected_bin = scalars[5]
                remaining = scalars[6]
                prefix = prefix | (cutlass.Uint32(selected_bin) << shift)

            if tidx == 0:
                scalars[3] = scalars[4]
            cute.arch.sync_threads()
            item = tidx
            while item < filtered_count:
                if filtered_keys[item] < prefix:
                    slot = cute.arch.atomic_add(
                        scalars.iterator + 3,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    if slot < self.top_k:
                        selected[(row, slot)] = filtered_indices[item]
                item = item + self._THREADS
            cute.arch.sync_threads()
            item = tidx
            while item < filtered_count:
                if filtered_keys[item] == prefix:
                    slot = cute.arch.atomic_add(
                        scalars.iterator + 3,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    if slot < self.top_k:
                        selected[(row, slot)] = filtered_indices[item]
                item = item + self._THREADS


class SelectivePackedGatherKernel:
    """Gather selected indices directly from packed candidate records."""

    def __init__(self, top_k: int, capacity: int, threads: int = 256):
        self.top_k = top_k
        self.capacity = capacity
        self.threads = threads

    @cute.jit
    def __call__(
        self,
        packed_values: cute.Tensor,
        lengths: cute.Tensor,
        positions: cute.Tensor,
        selected: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(packed_values, lengths, positions, selected, rows).launch(
            grid=(rows, cute.ceil_div(self.top_k, self.threads), 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        packed_values: cute.Tensor,
        lengths: cute.Tensor,
        positions: cute.Tensor,
        selected: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, rank_block, _ = cute.arch.block_idx()
        rank = rank_block * self.threads + tidx
        if (row < rows) & (rank < self.top_k):
            index = cutlass.Int32(0)
            if rank < lengths[row]:
                position = positions[(row, rank)]
                if (position >= 0) & (position < self.capacity):
                    packed = cutlass.Uint64(packed_values[(row, position)])
                    index = cutlass.Int32(packed & cutlass.Uint64(0xFFFFFFFF))
            (selected.iterator + row * self.top_k + rank).store(index)


class SelectiveRepairScatterKernel:
    """Replace provisional TopK rows whose first repair ran."""

    def __init__(self, top_k: int, threads: int = 256):
        self.top_k = top_k
        self.threads = threads

    @cute.jit
    def __call__(
        self,
        repair_flags: cute.Tensor,
        repaired: cute.Tensor,
        selected: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(repair_flags, repaired, selected, rows).launch(
            grid=(rows, cute.ceil_div(self.top_k, self.threads), 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        repair_flags: cute.Tensor,
        repaired: cute.Tensor,
        selected: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, rank_block, _ = cute.arch.block_idx()
        rank = rank_block * self.threads + tidx
        if (row < rows) & (rank < self.top_k):
            if repair_flags[row] != 0:
                index = (repaired.iterator + row * self.top_k + rank).load()
                (selected.iterator + row * self.top_k + rank).store(index)


class SelectiveRepairResolveKernel:
    """Resolve first repair and prepare the remaining exact-repair rows."""

    def __init__(self, top_k: int, capacity: int, threads: int = 256):
        self.top_k = top_k
        self.capacity = capacity
        self.threads = threads

    @cute.jit
    def __call__(
        self,
        packed_values: cute.Tensor,
        repair_lengths: cute.Tensor,
        positions: cute.Tensor,
        repair_flags: cute.Tensor,
        second_repair_flags: cute.Tensor,
        row_ends: cute.Tensor,
        selected: cute.Tensor,
        repair_ends: cute.Tensor,
        ranks: cute.Tensor,
        prefixes: cute.Tensor,
        exact_error: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            packed_values,
            repair_lengths,
            positions,
            repair_flags,
            second_repair_flags,
            row_ends,
            selected,
            repair_ends,
            ranks,
            prefixes,
            exact_error,
            rows,
        ).launch(
            grid=(rows, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        packed_values: cute.Tensor,
        repair_lengths: cute.Tensor,
        positions: cute.Tensor,
        repair_flags: cute.Tensor,
        second_repair_flags: cute.Tensor,
        row_ends: cute.Tensor,
        selected: cute.Tensor,
        repair_ends: cute.Tensor,
        ranks: cute.Tensor,
        prefixes: cute.Tensor,
        exact_error: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        first_repair = repair_flags[row] != 0
        exact_repair = first_repair & (second_repair_flags[row] != 0)
        successful_repair = first_repair & (not exact_repair)

        for item in cutlass.range_constexpr(cute.ceil_div(self.top_k, self.threads)):
            rank = item * self.threads + tidx
            if successful_repair & (rank < self.top_k):
                index = cutlass.Int32(0)
                if rank < repair_lengths[row]:
                    position = positions[(row, rank)]
                    if (position >= 0) & (position < self.capacity):
                        packed = cutlass.Uint64(packed_values[(row, position)])
                        index = cutlass.Int32(packed & cutlass.Uint64(0xFFFFFFFF))
                selected[(row, rank)] = index

        if tidx == 0:
            second_repair_flags[row] = cutlass.Int32(exact_repair)
            repair_ends[row] = row_ends[row] if exact_repair else cutlass.Int32(0)
            ranks[row] = (
                min(row_ends[row], cutlass.Int32(self.top_k))
                if exact_repair
                else cutlass.Int32(0)
            )
            prefixes[row] = cutlass.Int32(0)
            if row == 0:
                exact_error[0] = cutlass.Int32(0)


class SelectiveTopKRadixSelectKernel:
    """Advance one exact FP32 radix-selection byte for every query row."""

    def __init__(self, threads: int = 256):
        self.threads = threads

    @cute.jit
    def __call__(
        self,
        histograms: cute.Tensor,
        prefixes: cute.Tensor,
        ranks: cute.Tensor,
        threshold_bits: cute.Tensor,
        rows: cutlass.Int32,
        shift: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(histograms, prefixes, ranks, threshold_bits, rows, shift).launch(
            grid=(cute.ceil_div(rows, self.threads), 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        histograms: cute.Tensor,
        prefixes: cute.Tensor,
        ranks: cute.Tensor,
        threshold_bits: cute.Tensor,
        rows: cutlass.Int32,
        shift: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        row = block_idx * self.threads + tidx
        if row < rows:
            rank = ranks[row]
            if rank > 0:
                greater = cutlass.Int32(0)
                selected_bin = cutlass.Int32(0)
                found = cutlass.Boolean(False)
                for offset in cutlass.range_constexpr(256):
                    radix_bin = cutlass.Int32(255 - offset)
                    count = histograms[(row, radix_bin)]
                    if (not found) & (greater + count >= rank):
                        selected_bin = radix_bin
                        found = cutlass.Boolean(True)
                    if not found:
                        greater = greater + count

                prefix = (cutlass.Uint32(prefixes[row]) << 8) | cutlass.Uint32(
                    selected_bin
                )
                prefixes[row] = cutlass.Int32(prefix)
                ranks[row] = rank - greater
                if shift == 0:
                    # Invert the monotonic key used by the score kernel without
                    # changing the selected FP32 payload.
                    mask = cutlass.Uint32(0xFFFFFFFF)
                    if cutlass.Int32(prefix) < 0:
                        mask = cutlass.Uint32(0x80000000)
                    threshold_bits[row] = cutlass.Int32(prefix ^ mask)


class SelectiveTopKCombineExactKernel:
    """Replace flagged candidate rows with exact greater/equal TopK."""

    def __init__(self, top_k: int):
        assert 0 < top_k <= 512
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        greater_indices: cute.Tensor,
        greater_counts: cute.Tensor,
        equal_indices: cute.Tensor,
        equal_counts: cute.Tensor,
        ranks: cute.Tensor,
        repair_flags: cute.Tensor,
        row_ends: cute.Tensor,
        output: cute.Tensor,
        error_count: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            greater_indices,
            greater_counts,
            equal_indices,
            equal_counts,
            ranks,
            repair_flags,
            row_ends,
            output,
            error_count,
            rows,
        ).launch(grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        greater_indices: cute.Tensor,
        greater_counts: cute.Tensor,
        equal_indices: cute.Tensor,
        equal_counts: cute.Tensor,
        ranks: cute.Tensor,
        repair_flags: cute.Tensor,
        row_ends: cute.Tensor,
        output: cute.Tensor,
        error_count: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        if repair_flags[row] != 0:
            greater = greater_counts[(row, 0)]
            needed_equal = ranks[row]
            required = min(row_ends[row], cutlass.Int32(self.top_k))
            valid = (greater == required - needed_equal) & (
                equal_counts[(row, 0)] >= needed_equal
            )
            for item_group in cutlass.range_constexpr((self.top_k + 255) // 256):
                column = tidx + item_group * 256
                if column < self.top_k:
                    if valid:
                        if column < required:
                            if column < greater:
                                output[(row, column)] = greater_indices[(row, column)]
                            else:
                                output[(row, column)] = equal_indices[
                                    (row, column - greater)
                                ]
                        else:
                            # Preserve the fixed-width consumer contract for
                            # causal rows shorter than TopK. Request-local key
                            # zero is valid for every non-empty causal row.
                            output[(row, column)] = cutlass.Int32(0)
                    else:
                        output[(row, column)] = cutlass.Int32(-1)
            if (tidx == 0) & (not valid):
                cute.arch.atomic_add(
                    error_count.iterator,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )


class SelectiveLogitsFinalizeKernel:
    """Convert logical candidate counts into packed length/repair flags."""

    def __init__(
        self,
        capacity: int,
        num_segments: int,
        top_k: int = 512,
        threads: int = 256,
    ):
        assert capacity > 0 and num_segments >= 0 and 0 < top_k <= 512
        self.capacity = capacity
        self.num_segments = num_segments
        self.top_k = top_k
        self.threads = threads

    @cute.jit
    def __call__(
        self,
        counts: cute.Tensor,
        values: cute.Tensor,
        lengths: cute.Tensor,
        flags: cute.Tensor,
        rows: cutlass.Int32,
        capacity: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.num_segments == 0):
            self.kernel(counts, values, lengths, flags, rows, capacity).launch(
                grid=(cute.ceil_div(rows, self.threads), 1, 1),
                block=(self.threads, 1, 1),
                stream=stream,
                use_pdl=True,
            )
        elif cutlass.const_expr(self.num_segments == 64):
            self.lane_segmented_kernel(
                counts, values, lengths, flags, rows, capacity
            ).launch(
                grid=(rows, 1, 1),
                block=(512, 1, 1),
                stream=stream,
                use_pdl=True,
            )
        else:
            self.segmented_kernel(
                counts, values, lengths, flags, rows, capacity
            ).launch(
                grid=(rows, 1, 1),
                block=(self.threads, 1, 1),
                stream=stream,
                use_pdl=True,
            )

    @cute.kernel
    def kernel(
        self,
        counts: cute.Tensor,
        values: cute.Tensor,
        lengths: cute.Tensor,
        flags: cute.Tensor,
        rows: cutlass.Int32,
        capacity: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        row = block_idx * self.threads + tidx
        if row < rows:
            total = counts[(row, 0)]
            lengths[row] = min(total, capacity)
            flags[row] = cutlass.Int32((total < self.top_k) | (total > capacity))
        cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def segmented_kernel(
        self,
        counts: cute.Tensor,
        values: cute.Tensor,
        lengths: cute.Tensor,
        flags: cute.Tensor,
        rows: cutlass.Int32,
        capacity: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        segment_capacity = cutlass.const_expr(self.capacity // self.num_segments)

        smem = SmemAllocator()
        staged_values = smem.allocate_tensor(
            element_type=cutlass.Int64,
            layout=cute.make_ordered_layout((segment_capacity,), order=(0,)),
            byte_alignment=16,
        )
        segment_meta = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((2 * self.num_segments + 3,), order=(0,)),
            byte_alignment=16,
        )

        if tidx == 0:
            total = cutlass.Int32(0)
            retained = cutlass.Int32(0)
            overflow = cutlass.Boolean(False)
            for segment in cutlass.range_constexpr(self.num_segments):
                count = counts[(row, segment + 1)]
                keep = min(count, segment_capacity)
                segment_meta[2 * segment] = retained
                segment_meta[2 * segment + 1] = keep
                retained = retained + keep
                total = total + count
                overflow = overflow | (count > segment_capacity)
            segment_meta[2 * self.num_segments] = total
            segment_meta[2 * self.num_segments + 1] = retained
            segment_meta[2 * self.num_segments + 2] = cutlass.Int32(overflow)
        cute.arch.sync_threads()

        # Stage one segment at a time so an earlier destination range cannot
        # overwrite a later source element in the same in-place move.
        for segment in cutlass.range_constexpr(self.num_segments):
            retained = segment_meta[2 * segment]
            keep = segment_meta[2 * segment + 1]
            begin = cutlass.Int32(segment * segment_capacity)
            item = tidx
            while item < keep:
                staged_values[item] = values[(row, begin + item)]
                item = item + self.threads
            cute.arch.sync_threads()
            item = tidx
            while item < keep:
                values[(row, retained + item)] = staged_values[item]
                item = item + self.threads
            cute.arch.sync_threads()

        if tidx == 0:
            total = segment_meta[2 * self.num_segments]
            retained = segment_meta[2 * self.num_segments + 1]
            overflow = cutlass.Boolean(segment_meta[2 * self.num_segments + 2])
            # Preserve the packed consumer ABI: segment overflow is
            # represented by capacity + 1, while an in-capacity row keeps its
            # exact logical count.
            published = total
            if overflow:
                published = cutlass.Int32(self.capacity + 1)
            counts[(row, 0)] = published
            lengths[row] = retained
            flags[row] = cutlass.Int32(
                (total < self.top_k) | (total > capacity) | overflow
            )
        cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def lane_segmented_kernel(
        self,
        counts: cute.Tensor,
        values: cute.Tensor,
        lengths: cute.Tensor,
        flags: cute.Tensor,
        rows: cutlass.Int32,
        capacity: cutlass.Int32,
    ):
        """Compact fixed store-lane vector slices through shared memory."""
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        segment_capacity = cutlass.const_expr(self.capacity // self.num_segments)
        finalize_threads = cutlass.const_expr(512)
        num_warps = cutlass.const_expr(finalize_threads // 32)
        threads_per_segment = cutlass.const_expr(finalize_threads // self.num_segments)

        smem = SmemAllocator()
        warp_meta = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((4 * num_warps + 4,), order=(0,)),
            byte_alignment=16,
        )

        segment = tidx // threads_per_segment
        segment_thread = tidx % threads_per_segment
        count = counts[(row, segment + 1)]
        keep = min(count, segment_capacity)
        begin = segment * segment_capacity
        items_per_thread = cutlass.const_expr(
            (segment_capacity + threads_per_segment - 1) // threads_per_segment
        )
        staged_values = cute.make_rmem_tensor((items_per_thread,), cutlass.Int64)
        for item_group in cutlass.range_constexpr(items_per_thread):
            item = threads_per_segment * item_group + segment_thread
            staged_values[item_group] = cutlass.Int64(0)
            if item < keep:
                staged_values[item_group] = values[(row, begin + item)]

        lane = tidx % 32
        warp = tidx // 32
        inclusive_keep = cutlass.Int32(0)
        inclusive_count = cutlass.Int32(0)
        if segment_thread == 0:
            inclusive_keep = keep
            inclusive_count = count
        for step in cutlass.range_constexpr(5):
            offset = 1 << step
            shuffled_keep = cute.arch.shuffle_sync_up(
                inclusive_keep, offset, mask=0xFFFFFFFF, mask_and_clamp=0
            )
            shuffled_count = cute.arch.shuffle_sync_up(
                inclusive_count, offset, mask=0xFFFFFFFF, mask_and_clamp=0
            )
            if lane >= offset:
                inclusive_keep = inclusive_keep + shuffled_keep
                inclusive_count = inclusive_count + shuffled_count
        overflow_mask = cute.arch.vote_ballot_sync(
            (segment_thread == 0) & (count > segment_capacity)
        )
        if lane == 31:
            warp_meta[warp] = inclusive_keep
            warp_meta[num_warps + warp] = inclusive_count
            warp_meta[2 * num_warps + warp] = cutlass.Int32(overflow_mask != 0)
        cute.arch.sync_threads()

        if tidx == 0:
            total = cutlass.Int32(0)
            retained_total = cutlass.Int32(0)
            overflow = cutlass.Boolean(False)
            for warp_idx in cutlass.range_constexpr(num_warps):
                warp_meta[3 * num_warps + warp_idx] = retained_total
                retained_total = retained_total + warp_meta[warp_idx]
                total = total + warp_meta[num_warps + warp_idx]
                overflow = overflow | cutlass.Boolean(
                    warp_meta[2 * num_warps + warp_idx]
                )
            warp_meta[4 * num_warps] = total
            warp_meta[4 * num_warps + 1] = retained_total
            warp_meta[4 * num_warps + 2] = cutlass.Int32(overflow)
        cute.arch.sync_threads()

        retained = warp_meta[3 * num_warps + warp] + inclusive_keep - keep
        for item_group in cutlass.range_constexpr(items_per_thread):
            item = threads_per_segment * item_group + segment_thread
            if item < keep:
                values[(row, retained + item)] = staged_values[item_group]

        if tidx == 0:
            total = warp_meta[4 * num_warps]
            retained_total = warp_meta[4 * num_warps + 1]
            overflow = cutlass.Boolean(warp_meta[4 * num_warps + 2])
            published = total
            if overflow:
                published = cutlass.Int32(self.capacity + 1)
            counts[(row, 0)] = published
            lengths[row] = retained_total
            flags[row] = cutlass.Int32(
                (total < self.top_k) | (total > capacity) | overflow
            )
        cute.arch.griddepcontrol_launch_dependents()


class SelectiveLogitsMetadataScheduleKernel:
    """Build request-owned causal tiles and the split-balanced CTA schedule."""

    def __init__(
        self,
        num_sms: int,
        query_tile: int = 16,
        kv_chunk_size: int = 0,
        split_kv: int = 1,
        paged: bool = False,
    ):
        self.num_sms = num_sms
        self.query_tile = query_tile
        self.kv_chunk_size = kv_chunk_size
        self.split_kv = split_kv
        self.paged = paged

    @cute.jit
    def __call__(
        self,
        cu_q: cute.Tensor,
        cu_kv: cute.Tensor,
        context_lens: cute.Tensor,
        tile_meta: cute.Tensor,
        schedule_meta: cute.Tensor,
        schedule_prefix: cute.Tensor,
        num_batches: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            cu_q,
            cu_kv,
            context_lens,
            tile_meta,
            schedule_meta,
            schedule_prefix,
            num_batches,
        ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        cu_q: cute.Tensor,
        cu_kv: cute.Tensor,
        context_lens: cute.Tensor,
        tile_meta: cute.Tensor,
        schedule_meta: cute.Tensor,
        schedule_prefix: cute.Tensor,
        num_batches: cutlass.Int32,
    ):
        query_tile = cutlass.const_expr(self.query_tile)
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        num_tiles = context_lens.shape[0]

        # All CTA threads build causal tile records in parallel.
        tile_idx = tidx
        while tile_idx < num_tiles:
            context_len = cutlass.Int32(0)
            if tile_idx < num_tiles:
                owner = cutlass.Int32(0)
                owner_tile_begin = cutlass.Int32(0)
                q_begin = cu_q[0]
                q_end = cu_q[1]
                q_len = q_end - q_begin
                owner_tile_end = cute.ceil_div(q_len, query_tile)

                # Walk monotonically to the request owning this fixed tile slot.
                while (owner + 1 < num_batches) & (tile_idx >= owner_tile_end):
                    owner_tile_begin = owner_tile_end
                    owner = owner + 1
                    q_begin = cu_q[owner]
                    q_end = cu_q[owner + 1]
                    q_len = q_end - q_begin
                    owner_tile_end = owner_tile_begin + cute.ceil_div(q_len, query_tile)

                active = tile_idx < owner_tile_end
                q_start = cutlass.Int32(0)
                kv_start = cutlass.Int32(0)
                q_valid = cutlass.Int32(0)
                causal_base = cutlass.Int32(0)
                if active:
                    local_tile = tile_idx - owner_tile_begin
                    q_start = q_begin + local_tile * query_tile
                    q_valid = min(q_end - q_start, query_tile)
                    kv_start = cu_kv[owner]
                    kv_len = cu_kv[owner + 1] - kv_start
                    causal_base = kv_len - q_len + local_tile * query_tile
                    # Dense scale TMA needs the aligned prefix; native paged
                    # coordinates begin at logical request column zero.
                    context_len = causal_base + q_valid
                    if cutlass.const_expr(not self.paged):
                        context_len = context_len + kv_start % 4

                # Materialize exact per-row bounds in the same launch.  The
                # prepared route can therefore refresh graph-stable metadata
                # without allocating an offsets tensor or launching PyTorch
                # pointwise kernels between the metadata and score kernels.
                for row in cutlass.range_constexpr(query_tile):
                    row_start = cutlass.Int32(0)
                    row_end = cutlass.Int32(0)
                    if row < q_valid:
                        row_end = causal_base + row + 1
                    tile_meta[tile_idx, 4 + row] = row_start
                    tile_meta[tile_idx, 4 + query_tile + row] = row_end

                context_lens[tile_idx] = context_len
                tile_meta[tile_idx, 0] = q_start
                # Dense uses the packed KV origin; paged ingress needs the
                # owning page-table row for this request-owned query tile.
                tile_meta[tile_idx, 1] = kv_start
                if cutlass.const_expr(self.paged):
                    tile_meta[tile_idx, 1] = owner
                tile_meta[tile_idx, 2] = q_valid
                tile_meta[tile_idx, 3] = causal_base
            tile_idx = tile_idx + 256
        cute.arch.sync_threads()

        # Warp 0 consumes the completed metadata and emits CTA boundaries.
        if warp_idx == 0:
            self._build_schedule(
                context_lens, schedule_meta, schedule_prefix, num_tiles
            )

    @cute.jit
    def _build_schedule(
        self,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        prefix_sum: cute.Tensor,
        num_tiles: cutlass.Int32,
    ):
        num_sms = cutlass.const_expr(self.num_sms)
        sm_chunks = cutlass.const_expr((num_sms + 32) // 32)
        lane_idx = cute.arch.lane_idx()

        # Inclusive scan one work item per exclusive query tile. Atomic modes may
        # divide it into fixed-size kv_chunk_size work items; candidate mode may
        # instead divide it among split_kv pair-aligned owners with disjoint
        # publisher-segment banks.
        carry = cutlass.Int32(0)
        chunk = cutlass.Int32(0)
        while chunk * 32 < num_tiles:
            tile_idx = cutlass.Int32(chunk * 32) + lane_idx
            context_len = cutlass.Int32(0)
            value = cutlass.Int32(0)
            if tile_idx < num_tiles:
                context_len = context_lens[tile_idx]
            if context_len > 0:
                if cutlass.const_expr(self.kv_chunk_size > 0):
                    value = cute.ceil_div(context_len, self.kv_chunk_size)
                elif cutlass.const_expr(self.split_kv > 1):
                    value = min(
                        cute.ceil_div(context_len, _KV_PAIR_TOKENS),
                        cutlass.Int32(self.split_kv),
                    )
                else:
                    value = cutlass.Int32(1)
            for step in cutlass.range_constexpr(5):
                offset = 1 << step
                shuffled = cute.arch.shuffle_sync_up(
                    value, offset, mask=0xFFFFFFFF, mask_and_clamp=0
                )
                if lane_idx >= offset:
                    value = value + shuffled
            value = value + carry
            prefix_sum[chunk * 32 + lane_idx] = value
            carry = cute.arch.shuffle_sync(value, 31)
            chunk = chunk + 1

        # Emit the exact PR4365 split-balanced boundaries for all persistent CTAs.
        total = carry
        quotient = total // num_sms
        remainder = total % num_sms
        for chunk in cutlass.range_constexpr(sm_chunks):
            sm_idx = cutlass.Int32(chunk * 32) + lane_idx
            if sm_idx <= num_sms:
                segment_start = sm_idx * quotient
                if sm_idx <= remainder:
                    segment_start = segment_start + sm_idx
                else:
                    segment_start = segment_start + remainder

                lo = cutlass.Int32(0)
                count = cutlass.Int32(num_tiles)
                while count > 0:
                    half = count // 2
                    mid = lo + half
                    take = cutlass.Boolean(False)
                    if count > 0:
                        if prefix_sum[mid] <= segment_start:
                            take = cutlass.Boolean(True)
                    if take:
                        lo = mid + 1
                        count = count - half - 1
                    else:
                        count = half
                q_idx = lo
                kv_split = segment_start
                if q_idx > 0:
                    kv_split = segment_start - prefix_sum[q_idx - 1]
                if cutlass.const_expr(self.split_kv > 1):
                    # Convert the split ordinal into its pair-aligned KV
                    # boundary. The scorer already consumes this coordinate
                    # as a KV-pair boundary, so its pipeline traversal and
                    # drain lifecycle remain unchanged.
                    if kv_split > 0:
                        num_pairs = cute.ceil_div(context_lens[q_idx], _KV_PAIR_TOKENS)
                        active_splits = min(num_pairs, cutlass.Int32(self.split_kv))
                        kv_split = cute.ceil_div(kv_split * num_pairs, active_splits)
                schedule_meta[sm_idx, 0] = q_idx
                schedule_meta[sm_idx, 1] = kv_split


class SelectiveLogitsExplicitMetadataScheduleKernel(
    SelectiveLogitsMetadataScheduleKernel
):
    """Build request-tile metadata from caller-owned exact local row bounds."""

    @cute.jit
    def __call__(
        self,
        cu_q: cute.Tensor,
        cu_kv: cute.Tensor,
        row_starts: cute.Tensor,
        row_ends: cute.Tensor,
        row_end_base: cutlass.Int32,
        context_lens: cute.Tensor,
        tile_meta: cute.Tensor,
        schedule_meta: cute.Tensor,
        schedule_prefix: cute.Tensor,
        num_batches: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel_explicit(
            cu_q,
            cu_kv,
            row_starts,
            row_ends,
            row_end_base,
            context_lens,
            tile_meta,
            schedule_meta,
            schedule_prefix,
            num_batches,
        ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel_explicit(
        self,
        cu_q: cute.Tensor,
        cu_kv: cute.Tensor,
        row_starts: cute.Tensor,
        row_ends: cute.Tensor,
        row_end_base: cutlass.Int32,
        context_lens: cute.Tensor,
        tile_meta: cute.Tensor,
        schedule_meta: cute.Tensor,
        schedule_prefix: cute.Tensor,
        num_batches: cutlass.Int32,
    ):
        query_tile = cutlass.const_expr(self.query_tile)
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        num_tiles = context_lens.shape[0]

        tile_idx = tidx
        while tile_idx < num_tiles:
            context_len = cutlass.Int32(0)
            if tile_idx < num_tiles:
                owner = cutlass.Int32(0)
                owner_tile_begin = cutlass.Int32(0)
                q_begin = cu_q[0]
                q_end = cu_q[1]
                q_len = q_end - q_begin
                owner_tile_end = cute.ceil_div(q_len, query_tile)
                while (owner + 1 < num_batches) & (tile_idx >= owner_tile_end):
                    owner_tile_begin = owner_tile_end
                    owner = owner + 1
                    q_begin = cu_q[owner]
                    q_end = cu_q[owner + 1]
                    q_len = q_end - q_begin
                    owner_tile_end = owner_tile_begin + cute.ceil_div(q_len, query_tile)

                active = tile_idx < owner_tile_end
                q_start = cutlass.Int32(0)
                kv_start = cutlass.Int32(0)
                q_valid = cutlass.Int32(0)
                if active:
                    local_tile = tile_idx - owner_tile_begin
                    q_start = q_begin + local_tile * query_tile
                    q_valid = min(q_end - q_start, query_tile)
                    kv_start = cu_kv[owner]
                    max_end = cutlass.Int32(0)
                    for row in cutlass.range_constexpr(query_tile):
                        start = cutlass.Int32(0)
                        end = cutlass.Int32(0)
                        if row < q_valid:
                            start = row_starts[q_start + row]
                            end = row_ends[q_start + row]
                            if end != 0:
                                end = end + row_end_base
                            max_end = max(max_end, end)
                        tile_meta[tile_idx, 4 + row] = start
                        tile_meta[tile_idx, 4 + query_tile + row] = end
                    context_len = max_end
                    if cutlass.const_expr(not self.paged):
                        # Contiguous scale TMA rounds the packed origin down to
                        # a four-float boundary; include that prefix in sizing.
                        context_len = context_len + kv_start % 4

                context_lens[tile_idx] = context_len
                tile_meta[tile_idx, 0] = q_start
                tile_meta[tile_idx, 1] = kv_start
                if cutlass.const_expr(self.paged):
                    tile_meta[tile_idx, 1] = owner
                tile_meta[tile_idx, 2] = q_valid
                tile_meta[tile_idx, 3] = cutlass.Int32(0)
            tile_idx = tile_idx + 256
        cute.arch.sync_threads()
        if warp_idx == 0:
            self._build_schedule(
                context_lens, schedule_meta, schedule_prefix, num_tiles
            )


class SelectiveLogitsCompactMetadataScheduleKernel(
    SelectiveLogitsMetadataScheduleKernel
):
    """Compact query tiles containing flagged rows into a scorer schedule."""

    @cute.jit
    def __call__(
        self,
        cu_q: cute.Tensor,
        row_starts: cute.Tensor,
        row_ends: cute.Tensor,
        row_flags: cute.Tensor,
        context_lens: cute.Tensor,
        tile_meta: cute.Tensor,
        schedule_meta: cute.Tensor,
        schedule_prefix: cute.Tensor,
        num_batches: cutlass.Int32,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel_compact(
            cu_q,
            row_starts,
            row_ends,
            row_flags,
            context_lens,
            tile_meta,
            schedule_meta,
            schedule_prefix,
            num_batches,
            rows,
        ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel_compact(
        self,
        cu_q: cute.Tensor,
        row_starts: cute.Tensor,
        row_ends: cute.Tensor,
        row_flags: cute.Tensor,
        context_lens: cute.Tensor,
        tile_meta: cute.Tensor,
        schedule_meta: cute.Tensor,
        schedule_prefix: cute.Tensor,
        num_batches: cutlass.Int32,
        rows: cutlass.Int32,
    ):
        query_tile = cutlass.const_expr(self.query_tile)
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        num_tiles = context_lens.shape[0]

        # Clear the fixed-capacity schedule first. Thread zero then emits the
        # active rows in global-Q order, which also makes request ownership a
        # monotonic scan instead of one search per repair row.
        tile_idx = tidx
        while tile_idx < num_tiles:
            context_lens[tile_idx] = cutlass.Int32(0)
            tile_idx = tile_idx + 256
        if tidx == 0:
            schedule_prefix[0] = cutlass.Int32(0)
        cute.arch.sync_threads()

        tile_idx = tidx
        while tile_idx < num_tiles:
            owner = cutlass.Int32(0)
            owner_tile_begin = cutlass.Int32(0)
            q_begin = cu_q[0]
            q_end = cu_q[1]
            owner_tile_end = cute.ceil_div(q_end - q_begin, query_tile)
            while (owner + 1 < num_batches) & (tile_idx >= owner_tile_end):
                owner_tile_begin = owner_tile_end
                owner = owner + 1
                q_begin = cu_q[owner]
                q_end = cu_q[owner + 1]
                owner_tile_end = owner_tile_begin + cute.ceil_div(
                    q_end - q_begin, query_tile
                )

            if tile_idx < owner_tile_end:
                q_start = q_begin + (tile_idx - owner_tile_begin) * query_tile
                q_valid = min(q_end - q_start, query_tile)
                context_len = cutlass.Int32(0)
                for tile_row in cutlass.range_constexpr(query_tile):
                    if (tile_row < q_valid) & (row_flags[q_start + tile_row] != 0):
                        context_len = max(context_len, row_ends[q_start + tile_row])
                if context_len != 0:
                    compact_idx = cute.arch.atomic_add(
                        schedule_prefix.iterator,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    context_lens[compact_idx] = context_len
                    tile_meta[(compact_idx, 0)] = q_start
                    tile_meta[(compact_idx, 1)] = owner
                    tile_meta[(compact_idx, 2)] = q_valid
                    tile_meta[(compact_idx, 3)] = cutlass.Int32(0)
                    for tile_row in cutlass.range_constexpr(query_tile):
                        start = cutlass.Int32(0)
                        end = cutlass.Int32(0)
                        if (tile_row < q_valid) & (row_flags[q_start + tile_row] != 0):
                            start = row_starts[q_start + tile_row]
                            end = row_ends[q_start + tile_row]
                        tile_meta[(compact_idx, 4 + tile_row)] = start
                        tile_meta[(compact_idx, 4 + query_tile + tile_row)] = end
            tile_idx = tile_idx + 256
        cute.arch.sync_threads()

        if warp_idx == 0:
            self._build_schedule(
                context_lens, schedule_meta, schedule_prefix, num_tiles
            )


class SelectiveRepairRowCompactKernel:
    """Append flagged rows and their first-repair policy to compact storage."""

    @cute.jit
    def __call__(
        self,
        flags: cute.Tensor,
        thresholds: cute.Tensor,
        row_ends: cute.Tensor,
        row_ids: cute.Tensor,
        packed_thresholds: cute.Tensor,
        packed_ends: cute.Tensor,
        active_count: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            flags,
            thresholds,
            row_ends,
            row_ids,
            packed_thresholds,
            packed_ends,
            active_count,
            rows,
        ).launch(
            grid=(cute.ceil_div(rows, 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        flags: cute.Tensor,
        thresholds: cute.Tensor,
        row_ends: cute.Tensor,
        row_ids: cute.Tensor,
        packed_thresholds: cute.Tensor,
        packed_ends: cute.Tensor,
        active_count: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * 256 + tidx
        if (row < rows) & (flags[row] != 0):
            slot = cute.arch.atomic_add(
                active_count.iterator,
                cutlass.Int32(1),
                sem="relaxed",
                scope="gpu",
            )
            if slot < rows:
                row_ids[slot] = row
                packed_thresholds[slot] = thresholds[row]
                packed_ends[slot] = row_ends[row]


class SelectiveRepairRowPackKernel:
    """Gather compact repair Q rows and weights without host synchronization."""

    def __init__(self, num_heads: int, head_dim: int, blocks: int = 148):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.blocks = blocks

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        weights: cute.Tensor,
        row_ids: cute.Tensor,
        active_count: cute.Tensor,
        packed_q: cute.Tensor,
        packed_weights: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q,
            weights,
            row_ids,
            active_count,
            packed_q,
            packed_weights,
            rows,
        ).launch(grid=(self.blocks, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        weights: cute.Tensor,
        row_ids: cute.Tensor,
        active_count: cute.Tensor,
        packed_q: cute.Tensor,
        packed_weights: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_size, _, _ = cute.arch.grid_dim()
        linear = bidx * 256 + tidx
        stride = grid_size * 256
        active = min(active_count[0], rows)
        q_elements = cutlass.const_expr(self.num_heads * self.head_dim)
        q_work = active * q_elements
        while linear < q_work:
            packed_row = linear // q_elements
            element = linear - packed_row * q_elements
            source_row = row_ids[packed_row]
            packed_q[linear] = q[source_row * q_elements + element]
            linear = linear + stride

        linear = bidx * 256 + tidx
        weight_work = active * self.num_heads
        while linear < weight_work:
            packed_row = linear // self.num_heads
            head = linear - packed_row * self.num_heads
            source_row = row_ids[packed_row]
            packed_weights[linear] = weights[source_row * self.num_heads + head]
            linear = linear + stride


class SelectiveCompactRepairResolveKernel:
    """Scatter successful K6 rows and compact rows requiring exact K7."""

    def __init__(self, top_k: int, capacity: int, threads: int = 256):
        self.top_k = top_k
        self.capacity = capacity
        self.threads = threads

    @cute.jit
    def __call__(
        self,
        packed_values: cute.Tensor,
        repair_lengths: cute.Tensor,
        positions: cute.Tensor,
        first_row_ids: cute.Tensor,
        first_count: cute.Tensor,
        repair_flags: cute.Tensor,
        global_row_ends: cute.Tensor,
        selected: cute.Tensor,
        second_row_ids: cute.Tensor,
        second_ends: cute.Tensor,
        second_ranks: cute.Tensor,
        second_flags: cute.Tensor,
        second_count: cute.Tensor,
        exact_error: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            packed_values,
            repair_lengths,
            positions,
            first_row_ids,
            first_count,
            repair_flags,
            global_row_ends,
            selected,
            second_row_ids,
            second_ends,
            second_ranks,
            second_flags,
            second_count,
            exact_error,
            rows,
        ).launch(grid=(rows, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        packed_values: cute.Tensor,
        repair_lengths: cute.Tensor,
        positions: cute.Tensor,
        first_row_ids: cute.Tensor,
        first_count: cute.Tensor,
        repair_flags: cute.Tensor,
        global_row_ends: cute.Tensor,
        selected: cute.Tensor,
        second_row_ids: cute.Tensor,
        second_ends: cute.Tensor,
        second_ranks: cute.Tensor,
        second_flags: cute.Tensor,
        second_count: cute.Tensor,
        exact_error: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        packed_row, _, _ = cute.arch.block_idx()
        active = packed_row < min(first_count[0], rows)
        source_row = cutlass.Int32(0)
        short_complete = cutlass.Boolean(False)
        if active:
            source_row = first_row_ids[packed_row]
            row_end = global_row_ends[source_row]
            # The finalizer flags every result shorter than TopK. For a short
            # causal row, K6 is already exact when it published every key, so
            # scatter that complete result instead of sending it through K7.
            short_complete = (row_end < self.top_k) & (
                repair_lengths[packed_row] == row_end
            )
        exact_repair = active & (repair_flags[packed_row] != 0) & (not short_complete)
        successful_repair = active & (not exact_repair)

        for item in cutlass.range_constexpr(cute.ceil_div(self.top_k, self.threads)):
            rank = item * self.threads + tidx
            if successful_repair & (rank < self.top_k):
                index = cutlass.Int32(0)
                if rank < repair_lengths[packed_row]:
                    position = positions[(packed_row, rank)]
                    if (position >= 0) & (position < self.capacity):
                        packed = cutlass.Uint64(packed_values[(packed_row, position)])
                        index = cutlass.Int32(packed & cutlass.Uint64(0xFFFFFFFF))
                selected[(source_row, rank)] = index

        if (tidx == 0) & exact_repair:
            slot = cute.arch.atomic_add(
                second_count.iterator,
                cutlass.Int32(1),
                sem="relaxed",
                scope="gpu",
            )
            if slot < rows:
                end = global_row_ends[source_row]
                second_row_ids[slot] = source_row
                second_ends[slot] = end
                second_ranks[slot] = min(end, cutlass.Int32(self.top_k))
                second_flags[source_row] = cutlass.Int32(1)
            else:
                cute.arch.atomic_add(
                    exact_error.iterator,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )


class SelectiveCompactTopKCombineExactKernel:
    """Combine compact exact K7 output into global-Q result rows."""

    def __init__(self, top_k: int):
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        greater_indices: cute.Tensor,
        greater_counts: cute.Tensor,
        equal_indices: cute.Tensor,
        equal_counts: cute.Tensor,
        ranks: cute.Tensor,
        row_ends: cute.Tensor,
        row_ids: cute.Tensor,
        active_count: cute.Tensor,
        output: cute.Tensor,
        error_count: cute.Tensor,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            greater_indices,
            greater_counts,
            equal_indices,
            equal_counts,
            ranks,
            row_ends,
            row_ids,
            active_count,
            output,
            error_count,
            rows,
        ).launch(grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        greater_indices: cute.Tensor,
        greater_counts: cute.Tensor,
        equal_indices: cute.Tensor,
        equal_counts: cute.Tensor,
        ranks: cute.Tensor,
        row_ends: cute.Tensor,
        row_ids: cute.Tensor,
        active_count: cute.Tensor,
        output: cute.Tensor,
        error_count: cute.Tensor,
        rows: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        packed_row, _, _ = cute.arch.block_idx()
        if packed_row < min(active_count[0], rows):
            output_row = row_ids[packed_row]
            greater = greater_counts[(packed_row, 0)]
            needed_equal = ranks[packed_row]
            required = min(row_ends[packed_row], cutlass.Int32(self.top_k))
            valid = (greater == required - needed_equal) & (
                equal_counts[(packed_row, 0)] >= needed_equal
            )
            for item_group in cutlass.range_constexpr((self.top_k + 255) // 256):
                column = tidx + item_group * 256
                if column < self.top_k:
                    if valid:
                        if column < required:
                            if column < greater:
                                output[(output_row, column)] = greater_indices[
                                    (packed_row, column)
                                ]
                            else:
                                output[(output_row, column)] = equal_indices[
                                    (packed_row, column - greater)
                                ]
                        else:
                            output[(output_row, column)] = cutlass.Int32(0)
                    else:
                        output[(output_row, column)] = cutlass.Int32(-1)
            if (tidx == 0) & (not valid):
                cute.arch.atomic_add(
                    error_count.iterator,
                    cutlass.Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
