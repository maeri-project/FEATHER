#!/usr/bin/env python3
"""Cycle estimation: estimate_cycles_for_gemm, model_instruction_fetch,
estimate_latency_from_config_stream."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .config import (
    FeatherPlusConfig, TraceStateTracker, CycleBreakdown, ceil_div,
    canonical_n_ivn, canonical_n_wvn, canonical_n_ovn,
    reduction_vn_sizes,
)


def _compute_tile_costs(
    Mt: int, Kt: int, Nt: int, cfg: FeatherPlusConfig,
) -> tuple:
    """Compute per-tile costs for uniform tiles (no edge effects).

    Returns (Lin, Lw, C, D, S, in_bytes_tile, w_bytes_tile, out_bytes_tile)
    where Lin/Lw are per-K-step load cycles, C is per-K-step compute cycles,
    and D/S are per-output-tile move/store cycles.
    """
    in_bytes_tile = int(Mt * Kt * cfg.in_bytes)
    Lin = int(math.ceil(in_bytes_tile / max(1, cfg.bw_load_in)))

    w_bytes_tile = int(Kt * Nt * cfg.w_bytes)
    Lw = int(math.ceil(w_bytes_tile / max(1, cfg.bw_load_w)))

    # Compute cycles per K-step
    n_col_types_raw = ceil_div(Nt, cfg.AH)
    n_col_types = min(n_col_types_raw, cfg.AW)
    n_replicas = max(1, cfg.AW // max(1, n_col_types))
    n_sub_passes = ceil_div(n_col_types_raw, cfg.AW)
    n_ivns = ceil_div(Mt, n_replicas)
    K_g = ceil_div(Kt, cfg.AH)

    # --- Sanity check: VN counts agree with canonical formulas ---
    _exp_ivn = canonical_n_ivn(Mt, Kt, cfg.AH)
    _exp_wvn = canonical_n_wvn(Kt, Nt, cfg.AH)
    _exp_ovn = canonical_n_ovn(Mt, Nt, cfg.AH)
    # n_ivns is the IVN streams *per ExecuteMapping* (per sub-pass),
    # not total IVNs.  Total IVNs = Mt × K_g.  Each EM streams
    # n_ivns = ceil(Mt / n_replicas) IVNs per column.
    assert _exp_ivn == Mt * K_g, (
        f"cycles: canonical IVN mismatch {_exp_ivn} != {Mt}*{K_g}")
    assert _exp_wvn == Nt * K_g, (
        f"cycles: canonical WVN mismatch {_exp_wvn} != {Nt}*{K_g}")
    assert _exp_ovn == Mt * ceil_div(Nt, cfg.AH), (
        f"cycles: canonical OVN mismatch {_exp_ovn} != {Mt}*{ceil_div(Nt, cfg.AH)}")
    # Per-K-group vn_size: all AH except possibly last
    vn_sizes = reduction_vn_sizes(Kt, cfg.AH)
    birrd = cfg.birrd_drain

    # C = wvn_load_0 + sum_{i=1}^{K_g-1} em_period_i + nest_time_{last} + birrd
    # Per EM i: vs=vn_sizes[i], nest=n_ivns*vs+vs, em_period=max(nest, vs²-vs)
    def _em_times(vs):
        streaming = n_ivns * vs
        nest = streaming + vs
        wvn_ld = vs * vs
        period = max(nest, wvn_ld - vs)
        return wvn_ld, nest, period

    vs0 = vn_sizes[0]
    wvn0, _, _ = _em_times(vs0)
    total = wvn0  # first EM: wvn_load
    for i in range(1, K_g):
        vs_i = vn_sizes[i] if i < len(vn_sizes) else cfg.AH
        _, _, period_i = _em_times(vs_i)
        total += period_i  # middle EMs: em_period
    vs_last = vn_sizes[-1]
    _, nest_last, _ = _em_times(vs_last)
    total += nest_last + birrd  # last EM: nest_time + birrd

    C_per_sub = total
    C = int(C_per_sub * n_sub_passes)

    out_bytes_tile = int(Mt * Nt * cfg.out_bytes)
    D = int(math.ceil(out_bytes_tile / max(1, cfg.bw_onchip_move)))
    S = int(math.ceil(out_bytes_tile / max(1, cfg.bw_store_out)))

    return Lin, Lw, C, D, S, in_bytes_tile, w_bytes_tile, out_bytes_tile


def estimate_cycles_for_gemm(M: int, K: int, N: int, cfg: FeatherPlusConfig,
                             Mt: int, Kt: int, Nt: int,
                             reuse_input_across_N: bool = False) -> CycleBreakdown:
    """Asynchronous MINISA / FEATHER+ cycle model with explicit latency hiding.

    On-chip data reuse: if the TOTAL input or weight data for ALL tiles fits
    in the corresponding on-chip buffer, each unique tile is loaded from
    off-chip only once. Subsequent accesses reuse on-chip data (zero load cost).

    Buffer turnaround: cfg.buf_turnaround cycles are added between the end
    of a DMA write and the earliest PE read from that buffer partition.

    For large workloads with uniform tiles (M%Mt==K%Kt==N%Nt==0), uses an
    O(k_tiles) fast path: simulates a few output tiles to capture steady-
    state throughput, then extrapolates analytically.
    """
    n_tiles = ceil_div(N, Nt)
    k_tiles = ceil_div(K, Kt)
    m_tiles = ceil_div(M, Mt)
    output_tiles = n_tiles * m_tiles

    # Fast path: when tile count is large, simulate a few output tiles
    # and extrapolate. Uses interior tile dimensions; edge tile error is
    # negligible for large workloads (proportional to 1/m_tiles + 1/n_tiles).
    FAST_THRESHOLD = 64

    if output_tiles > FAST_THRESHOLD:
        return _estimate_cycles_uniform(
            M, K, N, cfg, Mt, Kt, Nt,
            n_tiles, k_tiles, m_tiles, output_tiles,
            reuse_input_across_N)

    # --- Full simulation for small / non-uniform workloads ---
    return _estimate_cycles_full(
        M, K, N, cfg, Mt, Kt, Nt,
        n_tiles, k_tiles, m_tiles,
        reuse_input_across_N)


def _estimate_cycles_uniform(
    M: int, K: int, N: int, cfg: FeatherPlusConfig,
    Mt: int, Kt: int, Nt: int,
    n_tiles: int, k_tiles: int, m_tiles: int, output_tiles: int,
    reuse_input_across_N: bool,
) -> CycleBreakdown:
    """Fast O(k_tiles) cycle estimation for uniform tiles.

    Simulates a small number of output tiles (2 full M-tile periods)
    with actual per-tile dimensions (handling edge tiles correctly),
    then extrapolates for the remaining N-columns.
    """
    cb = CycleBreakdown()
    turnaround = getattr(cfg, 'buf_turnaround', 1)
    AH = cfg.AH

    # --- On-chip reuse analysis ---
    total_unique_in_bytes = 0
    for mt_idx in range(m_tiles):
        Mt_eff = min(Mt, M - mt_idx * Mt)
        for kt_idx in range(k_tiles):
            Kt_eff = min(Kt, K - kt_idx * Kt)
            total_unique_in_bytes += Mt_eff * Kt_eff * cfg.in_bytes

    total_unique_w_bytes = 0
    for kt_idx in range(k_tiles):
        Kt_eff = min(Kt, K - kt_idx * Kt)
        for nt_idx in range(n_tiles):
            Nt_eff = min(Nt, N - nt_idx * Nt)
            total_unique_w_bytes += Kt_eff * Nt_eff * cfg.w_bytes

    in_fits_onchip = (total_unique_in_bytes <= cfg.stationary_bytes)
    w_fits_onchip = (total_unique_w_bytes <= cfg.stream_bytes)

    # --- Precompute per-tile costs for each (mt, kt, nt) combination ---
    # Only edge tiles differ, so we cache per (mt_idx, kt_idx) and per nt_idx.
    def _tile_compute(Mt_e: int, Kt_e: int, Nt_e: int) -> int:
        groups = ceil_div(Nt_e, AH)
        n_ct = min(groups, cfg.AW)
        n_rep = max(1, cfg.AW // max(1, n_ct))
        n_sp = ceil_div(groups, cfg.AW)
        n_iv = ceil_div(Mt_e, n_rep)
        vn_szs = reduction_vn_sizes(Kt_e, AH)
        K_g_e = len(vn_szs)

        def _em(vs):
            nest = n_iv * vs + vs
            return vs * vs, nest, max(nest, vs * vs - vs)

        vs0 = vn_szs[0]
        wvn0, _, _ = _em(vs0)
        total = wvn0
        for i in range(1, K_g_e):
            vs_i = vn_szs[i] if i < len(vn_szs) else AH
            _, _, p_i = _em(vs_i)
            total += p_i
        _, nest_last, _ = _em(vn_szs[-1])
        total += nest_last + cfg.birrd_drain
        return int(total * n_sp)

    # --- Byte and cycle counters (analytical, O(1) per dimension) ---
    # Decompose each dimension into interior tiles + optional edge tile.
    m_full = M // Mt                     # number of full M-tiles
    m_edge = M % Mt                      # edge M-tile size (0 if exact)
    k_full = K // Kt
    k_edge = K % Kt
    n_full = N // Nt
    n_edge = N % Nt

    # Helper: sum a per-tile metric across all (mt, kt, nt) combinations.
    # Each dimension contributes (count_full, size_full) + optional (1, size_edge).
    def _sum_product(per_m: List[tuple], per_k: List[tuple], per_n: List[tuple],
                     fn) -> int:
        """Sum fn(Mt_e, Kt_e, Nt_e) * count_m * count_k * count_n."""
        total = 0
        for cm, mval in per_m:
            for ck, kval in per_k:
                for cn, nval in per_n:
                    total += cm * ck * cn * fn(mval, kval, nval)
        return total

    m_parts = [(m_full, Mt)]
    if m_edge: m_parts.append((1, m_edge))
    k_parts = [(k_full, Kt)]
    if k_edge: k_parts.append((1, k_edge))
    n_parts = [(n_full, Nt)]
    if n_edge: n_parts.append((1, n_edge))

    # Compute
    cb.compute = _sum_product(m_parts, k_parts, n_parts,
                              lambda me, ke, ne: _tile_compute(me, ke, ne))

    # Output move/store: per (mt, nt), summed over all m × n tiles
    for cm, mval in m_parts:
        for cn, nval in n_parts:
            out_b = int(mval * nval * cfg.out_bytes)
            count = cm * cn * n_tiles if cn == n_full else cm * cn  # wait...
    # Simpler: just iterate the decomposed parts
    cb.bytes_out_move = 0
    cb.bytes_out_store = 0
    cb.out_to_stream = 0
    cb.store_out = 0
    for cm, mval in m_parts:
        for cn, nval in n_parts:
            out_b = int(mval * nval * cfg.out_bytes)
            count = cm * cn
            cb.bytes_out_move += count * out_b
            cb.out_to_stream += count * int(math.ceil(out_b / max(1, cfg.bw_onchip_move)))
            cb.bytes_out_store += count * out_b
            cb.store_out += count * int(math.ceil(out_b / max(1, cfg.bw_store_out)))

    # Input loads: per (mt, kt), loaded n_tiles times (or once if on-chip/reuse)
    n_repeats_in = 1 if (in_fits_onchip or reuse_input_across_N) else n_tiles
    for cm, mval in m_parts:
        for ck, kval in k_parts:
            in_b = int(mval * kval * cfg.in_bytes)
            count = cm * ck * n_repeats_in
            cb.bytes_in += count * in_b
            cb.load_in += count * int(math.ceil(in_b / max(1, cfg.bw_load_in)))

    # Weight loads: per (kt, nt), loaded m_tiles times (or once if on-chip)
    n_repeats_w = 1 if w_fits_onchip else m_tiles
    for ck, kval in k_parts:
        for cn, nval in n_parts:
            w_b = int(kval * nval * cfg.w_bytes)
            count = ck * cn * n_repeats_w
            cb.bytes_w += count * w_b
            cb.load_w += count * int(math.ceil(w_b / max(1, cfg.bw_load_w)))

    # --- Wallclock estimation via steady-state simulation ---
    # Simulate 2 full M-tile periods with actual tile dimensions.
    SIM_TILES = min(2 * m_tiles, output_tiles)

    t_in = 0
    t_w = 0
    t_comp = 0
    t_move = 0
    t_store = 0
    stream_free = [0, 0]
    sta_free = [0, 0]
    out_free = [0, 0]

    in_loaded: set = set()
    w_loaded: set = set()

    wallclock_after: List[int] = []

    for sim_idx in range(SIM_TILES):
        nt = sim_idx // m_tiles
        mt = sim_idx % m_tiles
        Mt_eff = min(Mt, M - mt * Mt)
        Nt_eff = min(Nt, N - nt * Nt) if nt * Nt < N else Nt  # nt is 0 or 1

        out_pp = sim_idx & 1
        t_comp = max(t_comp, out_free[out_pp])
        tile_compute_done = t_comp

        for kt in range(k_tiles):
            Kt_eff = min(Kt, K - kt * Kt)

            in_tile_key = (mt, kt)
            need_in = True
            if reuse_input_across_N:
                need_in = (nt == 0)
            if in_fits_onchip and in_tile_key in in_loaded:
                need_in = False
            if need_in:
                in_loaded.add(in_tile_key)

            in_b = int(Mt_eff * Kt_eff * cfg.in_bytes)
            cur_Lin = int(math.ceil(in_b / max(1, cfg.bw_load_in))) if need_in else 0

            w_tile_key = (kt, nt)
            need_w = True
            if w_fits_onchip and w_tile_key in w_loaded:
                need_w = False
            if need_w:
                w_loaded.add(w_tile_key)

            w_b = int(Kt_eff * Nt_eff * cfg.w_bytes)
            cur_Lw = int(math.ceil(w_b / max(1, cfg.bw_load_w))) if need_w else 0

            cur_C = _tile_compute(Mt_eff, Kt_eff, Nt_eff)

            step_pp = (sim_idx * k_tiles + kt) & 1

            if cur_Lin > 0:
                start_in = max(t_in, stream_free[step_pp])
                finish_in = start_in + cur_Lin + turnaround
                t_in = finish_in
            else:
                finish_in = 0

            if cur_Lw > 0:
                start_w = max(t_w, sta_free[step_pp])
                finish_w = start_w + cur_Lw + turnaround
                t_w = finish_w
            else:
                finish_w = 0

            start_c = max(t_comp, finish_in, finish_w)
            finish_c = start_c + cur_C
            t_comp = finish_c
            tile_compute_done = finish_c

            stream_free[step_pp] = finish_c
            sta_free[step_pp] = finish_c

        out_b = int(Mt_eff * Nt_eff * cfg.out_bytes)
        cur_D = int(math.ceil(out_b / max(1, cfg.bw_onchip_move)))
        cur_S = int(math.ceil(out_b / max(1, cfg.bw_store_out)))

        start_move = max(t_move, tile_compute_done)
        finish_move = start_move + cur_D
        t_move = finish_move

        start_store = max(t_store, finish_move)
        finish_store = start_store + cur_S
        t_store = finish_store

        out_free[out_pp] = finish_move

        wallclock_after.append(max(t_in, t_w, t_comp, t_move, t_store))

    # Extrapolate using the steady-state rate per M-tile group
    if SIM_TILES >= 2 * m_tiles:
        # One full M-tile period: time from end of 1st period to end of 2nd
        period_wall = wallclock_after[2 * m_tiles - 1] - wallclock_after[m_tiles - 1]
        remaining_periods = n_tiles - 2  # already simulated 2 N-columns
        cb.total = int(wallclock_after[-1] + period_wall * remaining_periods)
    elif SIM_TILES >= 2:
        steady_period = wallclock_after[-1] - wallclock_after[-2]
        remaining = output_tiles - SIM_TILES
        cb.total = int(wallclock_after[-1] + steady_period * remaining)
    else:
        cb.total = int(wallclock_after[0] * output_tiles)

    return cb


def _estimate_cycles_full(
    M: int, K: int, N: int, cfg: FeatherPlusConfig,
    Mt: int, Kt: int, Nt: int,
    n_tiles: int, k_tiles: int, m_tiles: int,
    reuse_input_across_N: bool,
) -> CycleBreakdown:
    """Full tile-by-tile async simulation (original algorithm)."""
    cb = CycleBreakdown()
    turnaround = getattr(cfg, 'buf_turnaround', 1)

    # --- On-chip data reuse analysis ---
    total_unique_in_bytes = 0
    for mt_idx in range(m_tiles):
        Mt_eff = min(Mt, M - mt_idx * Mt)
        for kt_idx in range(k_tiles):
            Kt_eff = min(Kt, K - kt_idx * Kt)
            total_unique_in_bytes += Mt_eff * Kt_eff * cfg.in_bytes

    total_unique_w_bytes = 0
    for kt_idx in range(k_tiles):
        Kt_eff = min(Kt, K - kt_idx * Kt)
        for nt_idx in range(n_tiles):
            Nt_eff = min(Nt, N - nt_idx * Nt)
            total_unique_w_bytes += Kt_eff * Nt_eff * cfg.w_bytes

    in_fits_onchip = (total_unique_in_bytes <= cfg.stationary_bytes)
    w_fits_onchip = (total_unique_w_bytes <= cfg.stream_bytes)

    in_loaded: set = set()
    w_loaded: set = set()

    t_in = 0
    t_w = 0
    t_comp = 0
    t_move = 0
    t_store = 0

    stream_free = [0, 0]
    sta_free = [0, 0]
    out_free = [0, 0]

    step = 0
    out_tile_idx = 0

    for nt in range(n_tiles):
        n0 = nt * Nt
        Nt_eff = min(Nt, N - n0)

        for mt in range(m_tiles):
            m0 = mt * Mt
            Mt_eff = min(Mt, M - m0)

            out_pp = out_tile_idx & 1
            t_comp = max(t_comp, out_free[out_pp])
            tile_compute_done = t_comp

            for kt in range(k_tiles):
                k0 = kt * Kt
                Kt_eff = min(Kt, K - k0)

                # --- Input load ---
                in_elems = Mt_eff * Kt_eff
                in_bytes = int(in_elems * cfg.in_bytes)
                in_tile_key = (mt, kt)
                need_in = True
                if reuse_input_across_N:
                    need_in = (nt == 0)
                if in_fits_onchip and in_tile_key in in_loaded:
                    need_in = False
                if need_in:
                    in_loaded.add(in_tile_key)

                Lin = int(math.ceil(in_bytes / max(1, cfg.bw_load_in))) if need_in else 0
                if need_in:
                    cb.bytes_in += in_bytes
                    cb.load_in += Lin

                # --- Weight load ---
                w_elems = Kt_eff * Nt_eff
                w_bytes = int(w_elems * cfg.w_bytes)
                w_tile_key = (kt, nt)
                need_w = True
                if w_fits_onchip and w_tile_key in w_loaded:
                    need_w = False
                if need_w:
                    w_loaded.add(w_tile_key)

                Lw = int(math.ceil(w_bytes / max(1, cfg.bw_load_w))) if need_w else 0
                if need_w:
                    cb.bytes_w += w_bytes
                    cb.load_w += Lw

                n_col_types_raw = ceil_div(Nt_eff, cfg.AH)
                n_col_types = min(n_col_types_raw, cfg.AW)
                n_replicas = max(1, cfg.AW // max(1, n_col_types))
                n_sub_passes = ceil_div(n_col_types_raw, cfg.AW)

                n_ivns = ceil_div(Mt_eff, n_replicas)
                vn_szs = reduction_vn_sizes(Kt_eff, cfg.AH)
                K_g = len(vn_szs)
                birrd = cfg.birrd_drain

                def _em_t(vs):
                    nest = n_ivns * vs + vs
                    return vs * vs, nest, max(nest, vs * vs - vs)

                vs0 = vn_szs[0]
                wvn0, _, _ = _em_t(vs0)
                C_per_sub = wvn0
                for ii in range(1, K_g):
                    vs_i = vn_szs[ii] if ii < len(vn_szs) else cfg.AH
                    _, _, p_i = _em_t(vs_i)
                    C_per_sub += p_i
                _, nest_last, _ = _em_t(vn_szs[-1])
                C_per_sub += nest_last + birrd
                C = int(C_per_sub * n_sub_passes)
                cb.compute += C

                pp = step & 1

                if Lin > 0:
                    start_in = max(t_in, stream_free[pp])
                    finish_in = start_in + Lin + turnaround
                    t_in = finish_in
                else:
                    finish_in = 0

                if Lw > 0:
                    start_w = max(t_w, sta_free[pp])
                    finish_w = start_w + Lw + turnaround
                    t_w = finish_w
                else:
                    finish_w = 0

                start_c = max(t_comp, finish_in, finish_w)
                finish_c = start_c + C
                t_comp = finish_c
                tile_compute_done = finish_c

                stream_free[pp] = finish_c
                sta_free[pp] = finish_c

                step += 1

            out_elems = int(Mt_eff * Nt_eff)
            out_bytes = int(out_elems * cfg.out_bytes)
            D = int(math.ceil(out_bytes / max(1, cfg.bw_onchip_move)))
            cb.bytes_out_move += out_bytes
            cb.out_to_stream += D

            start_move = max(t_move, tile_compute_done)
            finish_move = start_move + D
            t_move = finish_move

            S = int(math.ceil(out_bytes / max(1, cfg.bw_store_out)))
            cb.bytes_out_store += out_bytes
            cb.store_out += S

            start_store = max(t_store, finish_move)
            finish_store = start_store + S
            t_store = finish_store

            out_free[out_pp] = finish_move

            out_tile_idx += 1

    cb.total = int(max(t_in, t_w, t_comp, t_move, t_store))
    return cb


def model_instruction_fetch(inst_bytes_total: int, base_cycles: int,
                            cfg: FeatherPlusConfig) -> Dict[str, int]:
    """Model instruction fetch with a finite on-chip instruction buffer."""
    inst_bytes_total = int(inst_bytes_total)
    base_cycles = int(base_cycles)
    bw = max(1, int(cfg.bw_inst_bytes))
    buf = max(0, int(cfg.inst_buf_bytes))

    prefetch_bytes = min(inst_bytes_total, buf)
    prefetch_cycles = int(math.ceil(prefetch_bytes / bw)) if prefetch_bytes > 0 else 0

    bytes_fetchable_during_run = int(base_cycles * bw)
    bytes_available = int(prefetch_bytes + bytes_fetchable_during_run)

    if bytes_available >= inst_bytes_total:
        stall_cycles = 0
    else:
        stall_bytes = int(inst_bytes_total - bytes_available)
        stall_cycles = int(math.ceil(stall_bytes / bw))

    required_buf_no_stall_bytes = int(max(0, inst_bytes_total - bytes_fetchable_during_run))
    total_extra_cycles = int(prefetch_cycles + stall_cycles)

    return {
        "inst_bytes_total": inst_bytes_total,
        "inst_buf_bytes": buf,
        "prefetch_bytes": prefetch_bytes,
        "prefetch_cycles": prefetch_cycles,
        "stall_cycles": stall_cycles,
        "required_buf_no_stall_bytes": required_buf_no_stall_bytes,
        "total_extra_cycles": total_extra_cycles,
    }


def estimate_latency_from_config_stream(
    trace: List[Dict[str, Any]],
    config_stream,  # to_config.ConfigStream or ConfigStreamSummary
    cfg: FeatherPlusConfig,
    M: int = 0, K: int = 0, N: int = 0,
    Mt: int = 0, Kt: int = 0, Nt: int = 0,
) -> CycleBreakdown:
    """Estimate workload latency using ConfigStream compute durations.

    Accepts either a full ConfigStream (with .cycles) or a lightweight
    ConfigStreamSummary (with .per_inst_durations).
    """
    cb = CycleBreakdown()

    compute_durations: List[int] = []
    dma_in_durations: List[int] = []
    dma_w_durations: List[int] = []
    dma_out_durations: List[int] = []
    layout_durations: List[int] = []

    # Pre-classify Load instructions as input vs weight using tracker
    load_is_input: List[bool] = []
    st_classify = TraceStateTracker(cfg.AH, cfg.AW, M, K, N, Mt, Kt, Nt)
    for inst in trace:
        st_classify.update(inst)
        if inst.get("op") == "Load":
            load_is_input.append(st_classify.last_layout == "I")
    load_cls_idx = 0

    if hasattr(config_stream, 'per_inst_durations'):
        # Lightweight path: durations already computed per instruction
        for i, inst in enumerate(trace):
            op = inst.get("op", "")
            dur = config_stream.per_inst_durations[i] if i < len(config_stream.per_inst_durations) else 0
            if op == "ExecuteMapping":
                compute_durations.append(dur)
            elif op == "Load":
                is_in = load_is_input[load_cls_idx] if load_cls_idx < len(load_is_input) else True
                load_cls_idx += 1
                if is_in:
                    dma_in_durations.append(dur)
                else:
                    dma_w_durations.append(dur)
            elif op == "Store":
                dma_out_durations.append(dur)
            elif op in ("SetWVNLayout", "SetIVNLayout", "SetOVNLayout", "ExecuteStreaming"):
                layout_durations.append(dur)

    else:
        # Full ConfigStream path: group cycles by source trace instruction.
        prev_idx = None
        cur_count = 0
        cur_trace_inst: Optional[Dict[str, Any]] = None

        def _flush():
            nonlocal load_cls_idx
            if cur_trace_inst is None:
                return
            cur_op_type = cur_trace_inst.get("op", "")
            if cur_op_type == "ExecuteMapping":
                compute_durations.append(cur_count)
            elif cur_op_type == "Load":
                is_in = load_is_input[load_cls_idx] if load_cls_idx < len(load_is_input) else True
                load_cls_idx += 1
                if is_in:
                    dma_in_durations.append(cur_count)
                else:
                    dma_w_durations.append(cur_count)
            elif cur_op_type == "Store":
                dma_out_durations.append(cur_count)
            elif cur_op_type in ("SetWVNLayout", "SetIVNLayout", "SetOVNLayout", "ExecuteStreaming"):
                layout_durations.append(cur_count)

        load_cls_idx = 0
        for cyc in config_stream.cycles:
            if cyc.source_idx != prev_idx:
                if prev_idx is not None:
                    _flush()
                prev_idx = cyc.source_idx
                cur_trace_inst = trace[cyc.source_idx] if cyc.source_idx < len(trace) else None
                cur_count = 1
            else:
                cur_count += 1
        if prev_idx is not None:
            _flush()

    t_in = 0
    t_w = 0
    t_comp = 0
    t_move = 0
    t_store = 0

    stream_free = [0, 0]
    sta_free = [0, 0]
    out_free = [0, 0]

    cur_output_tile: Optional[Dict[str, Any]] = None
    last_in_finish = 0
    last_w_finish = 0

    comp_idx = 0
    dma_in_idx = 0
    dma_w_idx = 0
    dma_out_idx = 0

    st = TraceStateTracker(cfg.AH, cfg.AW, M, K, N, Mt, Kt, Nt)
    load_sched_idx = 0

    for inst in trace:
        st.update(inst)
        op = inst.get("op", "")

        if op == "ExecuteMapping":
            # EM loads WVN into NEST; scheduling deferred to paired ES.
            continue

        if op == "SetOVNLayout":
            t_comp = max(t_comp, out_free[st.out_pp])
            cur_output_tile = {
                "Mt": st.Mt, "Nt": st.Nt,
                "out_pp": st.out_pp, "last_compute_done": t_comp,
            }

        elif op == "SetIVNLayout":
            pass

        elif op == "SetWVNLayout":
            pass

        elif op == "Load":
            target = int(inst.get("target", 1))
            target_free = sta_free if target == 0 else stream_free
            is_in = load_is_input[load_sched_idx] if load_sched_idx < len(load_is_input) else True
            load_sched_idx += 1
            pp = st.pp

            if is_in:
                data_bytes = st.Mt * st.Kt * cfg.in_bytes
                bw_time = ceil_div(data_bytes, max(1, cfg.bw_load_in))
                ctrl_time = 0
                if dma_in_idx < len(dma_in_durations):
                    ctrl_time = dma_in_durations[dma_in_idx]
                    dma_in_idx += 1
                Lin = max(bw_time, ctrl_time)
                cb.bytes_in += data_bytes
                cb.load_in += Lin
                start_in = max(t_in, target_free[pp])
                finish_in = start_in + Lin
                t_in = finish_in
                last_in_finish = finish_in
            else:
                data_bytes = st.Kt * st.Nt * cfg.w_bytes
                bw_time = ceil_div(data_bytes, max(1, cfg.bw_load_w))
                ctrl_time = 0
                if dma_w_idx < len(dma_w_durations):
                    ctrl_time = dma_w_durations[dma_w_idx]
                    dma_w_idx += 1
                Lw = max(bw_time, ctrl_time)
                cb.bytes_w += data_bytes
                cb.load_w += Lw
                start_w = max(t_w, target_free[pp])
                finish_w = start_w + Lw
                t_w = finish_w
                last_w_finish = finish_w

        elif op == "ExecuteStreaming":
            # ES triggers computation.  Use the compute duration from the
            # preceding EM (per_inst_durations is keyed by EM index).
            if comp_idx < len(compute_durations):
                C = compute_durations[comp_idx]
                comp_idx += 1
            else:
                vn_sz = st.vn_size
                Mt_sub = st.T
                wvn_load = cfg.wvn_load_cycles_for_vn(vn_sz)
                streaming_cycles = Mt_sub * vn_sz
                pipeline_fill = vn_sz
                birrd_pipe = cfg.birrd_drain + 1     # BIRRD drain + 1
                C = 1 + wvn_load + streaming_cycles + pipeline_fill + birrd_pipe

            cb.compute += C

            pp = st.pp
            start_c = max(t_comp, last_in_finish, last_w_finish)
            finish_c = start_c + C
            t_comp = finish_c

            stream_free[pp] = finish_c
            sta_free[pp] = finish_c

            if cur_output_tile is not None:
                cur_output_tile["last_compute_done"] = finish_c

        elif op == "Store":
            if cur_output_tile is None:
                continue

            out_bytes = st.Mt * st.Nt * cfg.out_bytes

            bw_move = ceil_div(out_bytes, max(1, cfg.bw_onchip_move))
            ctrl_time = 0
            if dma_out_idx < len(dma_out_durations):
                ctrl_time = dma_out_durations[dma_out_idx]
                dma_out_idx += 1

            D = max(bw_move, ctrl_time)
            cb.bytes_out_move += out_bytes
            cb.out_to_stream += D

            start_move = max(t_move, int(cur_output_tile["last_compute_done"]))
            finish_move = start_move + D
            t_move = finish_move

            S = ceil_div(out_bytes, max(1, cfg.bw_store_out))
            cb.bytes_out_store += out_bytes
            cb.store_out += S

            start_store = max(t_store, finish_move)
            finish_store = start_store + S
            t_store = finish_store

            out_free[st.out_pp] = finish_move

    cb.total = max(t_in, t_w, t_comp, t_move, t_store)

    inst_bytes_total = config_stream.total_config_bytes
    inst_model = model_instruction_fetch(inst_bytes_total, cb.total, cfg)
    cb.inst_prefetch = int(inst_model["prefetch_cycles"])
    cb.inst_stall = int(inst_model["stall_cycles"])
    cb.load_inst = int(inst_model["total_extra_cycles"])
    cb.total += cb.load_inst

    return cb
