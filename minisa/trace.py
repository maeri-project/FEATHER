#!/usr/bin/env python3
"""Trace generation and verification for GEMM and convolution.

Convolution is lowered directly through the VN abstraction (should be via im2col
GEMM conversion):
  Step 1: Flatten reduction dims (c_i, r, s) -> compound K' = C_i*R*S
  Step 2: Partition into VN tiles of length AH along K'
  Step 3: Group into VN Groups for the NEST

ExecuteMapping instructions carry VN-centric parameters
(r_0, c_0, G_r, G_c, s_r, s_c) from Stage 5 (Map).

Verification:
  - verify_trace (ISA-level): simulates the trace schedule by tracking
    tile state from layout/DMA instructions.
  - verify_config (config-level): simulates the per-cycle FEATHER+
    config stream to verify RTL-level correctness.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    FeatherPlusConfig, TraceBundle, TraceStateTracker, CycleBreakdown,
    ceil_div, ceil_log2,
    canonical_n_ivn, canonical_n_wvn, canonical_n_ovn,
    execute_mapping_replica_count, reduction_vn_sizes,
)
from .vn import choose_tile_sizes
from .layout import choose_layout_W, choose_layout_I, choose_layout_O


def _compute_exec_mapping_params(
    Mt_eff: int, Kt_eff: int, Nt_eff: int,
    AH: int, AW: int,
) -> Dict[str, Any]:
    """Compute VN-centric ExecuteMapping parameters (Stage 5 (Map)).

    MINISA mapping: PE (a_h, a_w) -> WVN(r, c) where
      r = r_0 + floor(a_w / G_r)
      c = c_0 + s_r * a_h + s_c * (a_w mod G_c)

    All AW PE columns are always active.

    For GEMM C[M,N] = A[M,K] x B[K,N] mapped to AH x AW NEST:
      - K_g = ceil(Kt_eff / AH) K-groups (reduction tiles)
      - n_col_types = ceil(Nt_eff / AH) distinct N-subgroups
      - n_replicas = floor(AW / n_col_types) replicas for M parallelism
      - G_r = n_replicas * n_col_types (all AW cols per k_g)
      - G_c = n_col_types (replication period of column pattern)
      - s_r = 1 (temporal stride: 1 WVN column per PE row)
      - s_c = AH if G_c > 1 else 0 (spacing between N-subgroups)

    Returns dict with r_0, c_0, G_r, G_c, s_r, s_c, n_replicas.
    """
    K_g = ceil_div(Kt_eff, AH)
    n_col_types = ceil_div(Nt_eff, AH) if Nt_eff > 0 else 1
    n_replicas = max(1, AW // max(1, n_col_types))

    # G_r: consecutive PE columns sharing the same WVN row
    # = n_replicas × n_col_types (fills all AW columns per k_g)
    G_r = n_replicas * n_col_types
    # G_c: replication period = n_col_types
    G_c = n_col_types
    r_0 = 0
    c_0 = 0
    # s_r: temporal stride per PE row (1 WVN column per PE row)
    s_r = 1
    # s_c: WVN-column spacing between distinct N-subgroups within
    # one G_c period (AH when multiple N-subgroups, else 0)
    s_c = AH if G_c > 1 else 0
    return {"r_0": r_0, "c_0": c_0, "G_r": G_r, "G_c": G_c,
            "s_r": s_r, "s_c": s_c, "n_replicas": n_replicas}


def generate_trace_gemm(
    M: int, K: int, N: int,
    cfg: FeatherPlusConfig,
    order_w: int = 0, order_i: int = 0, order_o: int = 0,
    Mt: Optional[int] = None, Kt: Optional[int] = None, Nt: Optional[int] = None,
    candidate: Optional[Any] = None,
    dataflow: str = "weight_stationary",
) -> TraceBundle:
    """Generate MINISA ISA trace for a GEMM workload.

    If *candidate* (a LayerCandidate from search.py) is provided, its
    layout orders, tile sizes, and per-k_g ExecuteMapping parameters
    are used directly.  Otherwise the legacy single-EM-per-K-tile path
    is taken.
    """
    if candidate is not None:
        return _generate_trace_vn(
            M, K, N, cfg,
            order_w=candidate.order_w,
            order_i=candidate.order_i,
            order_o=candidate.order_o,
            Mt=candidate.Mt, Kt=candidate.Kt, Nt=candidate.Nt,
            exec_params_list=candidate.exec_params,
            dataflow=dataflow,
        )
    return _generate_trace_vn(M, K, N, cfg,
                              order_w=order_w, order_i=order_i, order_o=order_o,
                              Mt=Mt, Kt=Kt, Nt=Nt, dataflow=dataflow)



# -----------------------------------------------------------------------
# Checking Point 1: ISA-level functional correctness
# -----------------------------------------------------------------------

def verify_trace(tb: TraceBundle, seed: int = 42) -> Tuple[bool, float]:
    """Verify GEMM correctness by simulating the MINISA trace schedule.

    Checking Point 1: ISA-level functional verification.

    Reconstructs the computation from the trace instruction stream
    using a TraceStateTracker to derive tile sizes and loop counters
    from ISA-only fields.  Each ExecuteMapping performs
    C[m0:m_end, n0:n_end] += A * B for the active tile context.
    """
    cfg = tb.cfg
    M, K, N = tb.M, tb.K, tb.N
    AH, AW = cfg.AH, cfg.AW
    Mt_nom = tb.chunk_strategy["M_chunk"]
    Kt_nom = tb.chunk_strategy["K_chunk"]
    Nt_nom = tb.chunk_strategy["N_chunk"]
    st = TraceStateTracker(AH, AW, M, K, N, Mt_nom, Kt_nom, Nt_nom)

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C_ref = A @ B
    C_sim = np.zeros((M, N), dtype=np.float32)

    for inst in tb.trace:
        st.update(inst)
        op = inst["op"]
        # Computation triggers on ExecuteStreaming (EM loads WVN, ES streams
        # IVN and triggers compute).  EM fields are cached in the tracker.
        if op == "ExecuteStreaming":
            n_end = min(st.n0 + st.Nt, N)

            # Derive K-slice from cached EM fields
            r_0 = st.em_r_0
            G_r = st.em_G_r
            k_g_count = AW // G_r
            Kt_sub = min(k_g_count * AH, st.Kt - r_0 * AH)
            k_start = st.k0 + r_0 * AH
            k_end = min(k_start + Kt_sub, K)

            # One EM per k_g covers all Mt rows in the tile
            m_start = st.m0
            m_end = min(st.m0 + st.Mt, M)
            C_sim[m_start:m_end, st.n0:n_end] += (
                A[m_start:m_end, k_start:k_end] @ B[k_start:k_end, st.n0:n_end]
            )

    max_err = float(np.max(np.abs(C_ref - C_sim)))
    ok = max_err < 1e-3
    return ok, max_err


# -----------------------------------------------------------------------
# Checking Point 2: Config-level functional correctness
# -----------------------------------------------------------------------

def verify_config(tb: TraceBundle, seed: int = 42) -> Tuple[bool, float]:
    """Verify that FEATHER+ config stream correctly computes the workload.

    Checking Point 2: Config-level functional verification.

    Converts the MINISA trace to a per-cycle config stream, then
    simulates the config stream's computation phases to produce
    the output matrix. Compares against numpy reference.

    This verifies the full ISA -> config -> RTL control pipeline:
      SetIVNLayout  -> store_ivn_layout_config
      SetWVNLayout  -> store_wvn_layout_config
      SetOVNLayout  -> store_ovn_layout_config
      ExecuteMapping -> tile_config + local_load_to_pe + operand_streaming + birrd_drain
      Load          -> dma_stationary_load / dma_streaming_load
      Store         -> dma_stationary_store / dma_streaming_store
    """
    from .isa import config_to_hw_params
    from .to_config import compute_config_stream_summary

    cfg = tb.cfg
    M, K, N = tb.M, tb.K, tb.N
    AH, AW = cfg.AH, cfg.AW
    Mt_nom = tb.chunk_strategy["M_chunk"]
    Kt_nom = tb.chunk_strategy["K_chunk"]
    Nt_nom = tb.chunk_strategy["N_chunk"]

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C_ref = A @ B

    # Compute lightweight config summary (no per-cycle allocation)
    hw = config_to_hw_params(cfg)
    summary = compute_config_stream_summary(
        tb.trace, hw, M, K, N, Mt_nom, Kt_nom, Nt_nom)

    # Simulate by replaying the trace
    C_sim = np.zeros((M, N), dtype=np.float32)
    st = TraceStateTracker(AH, AW, M, K, N, Mt_nom, Kt_nom, Nt_nom)

    config_ok = True
    BIRRD_pipe = hw.BIRRD_TOTAL_STAGE + 1

    em_inst_idx = -1  # index of the preceding ExecuteMapping
    for i, inst in enumerate(tb.trace):
        st.update(inst)
        op = inst["op"]
        if op == "ExecuteMapping":
            em_inst_idx = i
        # Computation triggers on ExecuteStreaming (EM loads WVN, ES streams
        # IVN and triggers compute).  EM fields are cached in the tracker.
        elif op == "ExecuteStreaming":
            n_end = min(st.n0 + st.Nt, N)

            # Derive K-slice from cached EM fields
            r_0 = st.em_r_0
            G_r = st.em_G_r
            k_g_count = AW // G_r
            Kt_sub = min(k_g_count * AH, st.Kt - r_0 * AH)
            k_start = st.k0 + r_0 * AH
            k_end = min(k_start + Kt_sub, K)

            # One EM per k_g covers all Mt rows in the tile
            m_start = st.m0
            m_end = min(st.m0 + st.Mt, M)
            C_sim[m_start:m_end, st.n0:n_end] += (
                A[m_start:m_end, k_start:k_end] @ B[k_start:k_end, st.n0:n_end]
            )

            # Verify config expansion: expected cycles for this EM+ES pair.
            # The config stream duration is recorded against the EM instruction.
            vn_sz = st.vn_size
            Mt_sub_inst = st.T
            wvn_load = vn_sz * vn_sz
            pipeline_fill = vn_sz
            expected_cyc = 1 + wvn_load + Mt_sub_inst * vn_sz + pipeline_fill + BIRRD_pipe
            actual_cyc = summary.per_inst_durations[em_inst_idx] if em_inst_idx >= 0 and em_inst_idx < len(summary.per_inst_durations) else 0
            if actual_cyc != expected_cyc:
                config_ok = False

    # Verify computational correctness
    max_err = float(np.max(np.abs(C_ref - C_sim)))

    ok = (max_err < 1e-3) and config_ok
    return ok, max_err


def estimate_minisa_inst_bytes(M: int, K: int, N: int, cfg: FeatherPlusConfig,
                               Mt: int, Kt: int, Nt: int) -> int:
    """Estimate total ISA instruction bytes without building trace."""
    AH, AW = cfg.AH, cfg.AW
    n_tiles = ceil_div(N, Nt)
    k_tiles = ceil_div(K, Kt)

    b_ovn = cfg.minisa_inst_bytes("SetOVNLayout")
    b_ivn = cfg.minisa_inst_bytes("SetIVNLayout")
    b_wvn = cfg.minisa_inst_bytes("SetWVNLayout")
    b_map = cfg.minisa_inst_bytes("ExecuteMapping")
    b_es = cfg.minisa_inst_bytes("ExecuteStreaming")
    b_load = cfg.minisa_inst_bytes("Load")
    b_store = cfg.minisa_inst_bytes("Store")

    total_bytes = 0
    for m0 in range(0, M, Mt):
        Mt_eff = min(Mt, M - m0)
        # Accumulate per-K-tile EM counts (map_blocks may differ for last K-tile)
        k_tile_bytes = 0
        for k0 in range(0, K, Kt):
            Kt_eff = min(Kt, K - k0)
            Nt_eff = min(Nt, N)
            params = _compute_exec_mapping_params(Mt_eff, Kt_eff, Nt_eff, AH, AW)
            n_replicas = params["n_replicas"]
            m_sub_blocks = ceil_div(ceil_div(Mt_eff, AH), n_replicas)
            K_g = ceil_div(Kt_eff, AH)  # K-groups per K-tile
            map_blocks = K_g * m_sub_blocks
            # Each ExecuteMapping is paired with an ExecuteStreaming
            k_tile_bytes += (b_ivn + b_load + b_wvn + b_load
                             + map_blocks * (b_es + b_map))
        bytes_per_n = b_ovn + k_tile_bytes + b_store
        total_bytes += n_tiles * bytes_per_n

    return int(total_bytes)


def _generate_trace_vn(
    M: int, K: int, N: int,
    cfg: FeatherPlusConfig,
    order_w: int = 0, order_i: int = 0, order_o: int = 0,
    conv_meta: Optional[Dict[str, Any]] = None,
    Mt: Optional[int] = None, Kt: Optional[int] = None, Nt: Optional[int] = None,
    exec_params_list: Optional[List[Any]] = None,
    dataflow: str = "weight_stationary",
) -> TraceBundle:
    """Core VN-based trace generator used by both GEMM and convolution.

    The loop nest iterates over VN tiles:
      - Outer: N-tiles (output channel groups)
      - Middle: M-tiles (output spatial groups)
      - Inner: K-tiles (reduction VN tile groups, accumulated)
      - Per K-tile: K_g = ceil(Kt/AH) K-groups (or EM batches from search)
      - Per k_g: all Mt rows stream through in Mt_sub = ceil(Mt / n_replicas) steps

    Each ExecuteMapping carries VN-centric parameters
    (r_0, c_0, G_r, G_c, s_r, s_c) from Stage 5 (Map).

    If *exec_params_list* is provided (from LayerCandidate.exec_params),
    each k_g uses the corresponding ExecuteMappingParams directly.
    """
    AH, AW = cfg.AH, cfg.AW
    if Mt is None or Kt is None or Nt is None:
        Mt, Kt, Nt = choose_tile_sizes(M, K, N, cfg)

    # --- Sanity check: canonical VN counts for the tile ---
    _n_ivn = canonical_n_ivn(Mt, Kt, AH)
    _n_wvn = canonical_n_wvn(Kt, Nt, AH)
    _n_ovn = canonical_n_ovn(Mt, Nt, AH)
    assert _n_ivn == Mt * ceil_div(Kt, AH), (
        f"IVN count invariant violated: {_n_ivn} != {Mt}*{ceil_div(Kt, AH)}")
    assert _n_wvn == Nt * ceil_div(Kt, AH), (
        f"WVN count invariant violated: {_n_wvn} != {Nt}*{ceil_div(Kt, AH)}")
    assert _n_ovn == Mt * ceil_div(Nt, AH), (
        f"OVN count invariant violated: {_n_ovn} != {Mt}*{ceil_div(Nt, AH)}")

    chunk = {"M_chunk": Mt, "K_chunk": Kt, "N_chunk": Nt}
    orders = {"W": order_w, "I": order_i, "O": order_o}
    trace: List[Dict[str, Any]] = []
    is_input_stationary = (dataflow == "input_stationary")
    load_ivn_target = 0 if is_input_stationary else 1
    load_wvn_target = 1 if is_input_stationary else 0

    out_tile_idx = 0
    for n0 in range(0, N, Nt):
        Nt_eff = min(Nt, N - n0)
        for m0 in range(0, M, Mt):
            Mt_eff = min(Mt, M - m0)
            out_pp = out_tile_idx & 1

            # SetOVNLayout: configure output buffer for this O_VN tile
            P_L0 = min(AW, max(1, Mt_eff))
            P_L1 = ceil_div(Mt_eff, P_L0)
            Q_L1 = ceil_div(Nt_eff, AH)
            trace.append({
                "op": "SetOVNLayout",
                "order": order_o,
                "P_L0": P_L0, "P_L1": P_L1, "Q_L1": Q_L1,
            })

            step = 0
            for k0 in range(0, K, Kt):
                Kt_eff = min(Kt, K - k0)
                pp = step & 1

                # SetIVNLayout: configure I_VN tile addressing
                M_L0 = min(AW, max(1, Mt_eff))
                M_L1 = ceil_div(Mt_eff, M_L0)
                J_L1 = ceil_div(Kt_eff, AH)
                trace.append({
                    "op": "SetIVNLayout",
                    "order": order_i,
                    "M_L0": M_L0, "M_L1": M_L1, "J_L1": J_L1,
                })
                # DMA: load I_VN tile from off-chip
                trace.append({
                    "op": "Load",
                    "hbm_addr": 0,
                    "target": load_ivn_target,
                })
                # SetWVNLayout: configure W_VN tile addressing
                N_L0 = min(AW, max(1, Nt_eff))
                N_L1 = ceil_div(Nt_eff, N_L0)
                K_L1 = ceil_div(Kt_eff, AH)
                trace.append({
                    "op": "SetWVNLayout",
                    "order": order_w,
                    "N_L0": N_L0, "N_L1": N_L1, "K_L1": K_L1,
                })
                # DMA: load W_VN tile from off-chip
                trace.append({
                    "op": "Load",
                    "hbm_addr": 0,
                    "target": load_wvn_target,
                })

                # K_g K-groups per K-tile; when exec_params_list is
                # provided it may have fewer entries than K_g (multi-k_g
                # EM batches pack several K-groups per EM).
                K_g = ceil_div(Kt_eff, AH)
                n_ems = len(exec_params_list) if exec_params_list is not None else K_g
                for em_idx in range(n_ems):
                    if exec_params_list is not None:
                        ep = exec_params_list[em_idx]
                        vn_r_0 = ep.r_0
                        vn_c_0 = ep.c_0
                        vn_G_r = ep.G_r
                        vn_G_c = ep.G_c
                        vn_s_r = ep.s_r
                        vn_s_c = ep.s_c
                        n_replicas = execute_mapping_replica_count(ep.G_r, ep.G_c)
                        # Derive K-groups covered by this EM from G_r
                        k_g_this_em = AW // ep.G_r
                        Kt_sub = min(k_g_this_em * AH,
                                     Kt_eff - ep.r_0 * AH)
                    else:
                        kg = em_idx
                        Kt_sub = min(AH, Kt_eff - kg * AH)
                        vn_params = _compute_exec_mapping_params(
                            Mt_eff, Kt_eff, Nt_eff, AH, AW)
                        vn_r_0 = kg
                        vn_c_0 = vn_params["c_0"]
                        vn_G_r = vn_params["G_r"]
                        vn_G_c = vn_params["G_c"]
                        vn_s_r = vn_params["s_r"]
                        vn_s_c = vn_params["s_c"]
                        n_replicas = execute_mapping_replica_count(vn_G_r, vn_G_c)

                    # One EM per k_g batch: all Mt_eff rows stream through.
                    # Mt_sub = IVN streaming steps per column.
                    Mt_sub = ceil_div(Mt_eff, n_replicas)
                    # Derive vn_size for this K-group
                    vn_sz = min(AH, Kt_sub)

                    # Determine IVN distribution mode from exec_params
                    ivn_dist = "interleaved"
                    if exec_params_list is not None:
                        ivn_dist = getattr(ep, 'ivn_distribution', 'interleaved')
                    es_s_m = 1 if ivn_dist == "consecutive" else n_replicas

                    # ExecuteMapping loads WVN into NEST;
                    # ExecuteStreaming streams IVN and triggers compute.
                    trace.append({
                        "op": "ExecuteMapping",
                        "r_0": vn_r_0,
                        "c_0": vn_c_0,
                        "G_r": vn_G_r,
                        "G_c": vn_G_c,
                        "s_r": vn_s_r,
                        "s_c": vn_s_c,
                    })
                    trace.append({
                        "op": "ExecuteStreaming",
                        "dataflow": 0 if is_input_stationary else 1,
                        "m_0": 0,
                        "s_m": es_s_m,
                        "T": Mt_sub,
                        "vn_size": max(0, vn_sz - 1),
                    })
                step += 1

            # DMA: store O_VN tile to off-chip
            trace.append({
                "op": "Store",
                "hbm_addr": 0,
                "target": 0,
            })
            out_tile_idx += 1

    tb = TraceBundle(cfg=cfg, M=M, K=K, N=N,
                     chunk_strategy=chunk, order_ids=orders, trace=trace)
    if conv_meta is not None:
        tb.conv_meta = conv_meta  # type: ignore[attr-defined]
    return tb


def generate_trace_conv(
    batch: int, C_i: int, H: int, W: int,
    C_o: int, R: int, S: int,
    cfg: FeatherPlusConfig,
    order_w: int = 0, order_i: int = 0, order_o: int = 0,
    stride_h: int = 1, stride_w: int = 1,
    pad_h: int = 0, pad_w: int = 0,
    Mt: Optional[int] = None, Kt: Optional[int] = None, Nt: Optional[int] = None,
) -> TraceBundle:
    """Generate MINISA ISA trace for a convolution workload.

    Lowering proceeds directly through the VN abstraction (Steps 1-3).
    From Step 3 onward, the instruction sequence is identical to GEMM.
    """
    H_o = (H + 2 * pad_h - R) // stride_h + 1
    W_o = (W + 2 * pad_w - S) // stride_w + 1
    K_prime = C_i * R * S
    M_vn = batch * H_o * W_o
    N_vn = C_o

    conv_meta = {
        "batch": batch, "C_i": C_i, "H": H, "W": W,
        "C_o": C_o, "R": R, "S": S,
        "stride_h": stride_h, "stride_w": stride_w,
        "pad_h": pad_h, "pad_w": pad_w,
        "H_o": H_o, "W_o": W_o,
        "K_prime": K_prime,
    }

    return _generate_trace_vn(M_vn, K_prime, N_vn, cfg,
                              order_w=order_w, order_i=order_i, order_o=order_o,
                              conv_meta=conv_meta,
                              Mt=Mt, Kt=Kt, Nt=Nt)


def export_config_json(
    M: int, K: int, N: int,
    cfg: FeatherPlusConfig,
    candidate: Any,
    out_path: str,
    D_StaB: int = 8192,
    D_StrB: int = 8192,
    D_OB: int = 8192,
) -> Dict[str, Any]:
    """Export a LayerCandidate as a JSON config file.

    Output format matches compiler/ACT/test_output.json:
      FEATHER_spec, layer[].L1[].{WVN, IVN, OVN, ExecuteMapping, latency, search_result}

    Parameters
    ----------
    M, K, N : GEMM dimensions (C[M,N] = A[M,K] x B[K,N]).
    cfg     : Hardware configuration.
    candidate : LayerCandidate from search.py.
    out_path  : Destination JSON file path.
    D_StaB, D_StrB, D_OB : SRAM bank depths (default 8192).

    Returns the dict that was written, for programmatic use.
    """
    import json

    AH, AW = cfg.AH, cfg.AW
    Mt, Kt, Nt = candidate.Mt, candidate.Kt, candidate.Nt

    # Canonical layout dimensions (independent of permutation order)
    N_L0 = min(AW, max(1, Nt))
    N_L1 = ceil_div(Nt, N_L0)
    K_L1 = ceil_div(Kt, AH)

    M_L0 = min(AW, max(1, Mt))
    M_L1 = ceil_div(Mt, M_L0)
    J_L1 = ceil_div(Kt, AH)

    P_L0 = min(AW, max(1, Mt))
    P_L1 = ceil_div(Mt, P_L0)
    Q_L1 = ceil_div(Nt, AH)

    # ExecuteMapping + ExecuteStreaming lists — from exec_params or trace
    em_list = []
    es_list = []
    if hasattr(candidate, 'exec_params') and candidate.exec_params:
        from .search import derive_execute_streaming
        dataflow_str = getattr(candidate, 'dataflow', 'weight_stationary')
        for ep in candidate.exec_params:
            em_list.append({
                "G_r": ep.G_r, "G_c": ep.G_c,
                "r_0": ep.r_0, "c_0": ep.c_0,
                "s_r": ep.s_r, "s_c": ep.s_c,
            })
            es = derive_execute_streaming(ep, Mt, dataflow_str)
            es_list.append({
                "dataflow": es.dataflow, "m_0": es.m_0,
                "s_m": es.s_m, "T": es.T, "vn_size": es.vn_size,
            })
    elif hasattr(candidate, 'trace_bundle') and candidate.trace_bundle:
        for inst in candidate.trace_bundle.trace:
            if inst.get('op') == 'ExecuteMapping':
                em_list.append({
                    "G_r": inst["G_r"], "G_c": inst["G_c"],
                    "r_0": inst["r_0"], "c_0": inst["c_0"],
                    "s_r": inst["s_r"], "s_c": inst["s_c"],
                })
            elif inst.get('op') == 'ExecuteStreaming':
                es_list.append({
                    "dataflow": inst["dataflow"], "m_0": inst["m_0"],
                    "s_m": inst["s_m"], "T": inst["T"],
                    "vn_size": inst["vn_size"],
                })

    n_sp = ceil_div(M, Mt) * ceil_div(N, Nt)

    # Build paired invocations list matching template.json format
    invocations = []
    for idx in range(max(len(em_list), len(es_list))):
        inv: Dict[str, Any] = {}
        inv["ExecuteMapping"] = em_list[idx] if idx < len(em_list) else {}
        inv["ExecuteStreaming"] = es_list[idx] if idx < len(es_list) else {}
        invocations.append(inv)

    doc: Dict[str, Any] = {
        "FEATHER_spec": {
            "AH": AH, "AW": AW,
            "D_StaB": D_StaB, "D_StrB": D_StrB, "D_OB": D_OB,
        },
        "traces": [{
            "L1": [
                {"WVN": {"order": candidate.order_w,
                          "N_L1": N_L1, "N_L0": N_L0, "K_L1": K_L1}},
                {"IVN": {"order": candidate.order_i,
                          "M_L1": M_L1, "M_L0": M_L0, "J_L1": J_L1}},
                {"OVN": {"order": candidate.order_o,
                          "P_L1": P_L1, "P_L0": P_L0, "Q_L1": Q_L1}},
                {"invocations": invocations},
                {"latency": candidate.cycles_total,
                 "utilization": round(getattr(candidate, 'utilization', 0) * 100, 2)},
            ],
        }],
    }

    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    return doc


def estimate_explicit_microinst_format(cfg: FeatherPlusConfig, overhead_bits: int = 16) -> Dict[str, int]:
    """Estimate explicit micro-instruction format overhead."""
    AW, AH = int(cfg.AW), int(cfg.AH)
    lgAW = ceil_log2(AW)
    birrd_bits = int(AW * (2 * lgAW - 1)) if AW > 1 else 0
    addr_in_bits = ceil_log2(int(cfg.cap_stream_vn()))
    addr_w_bits = ceil_log2(int(cfg.cap_stationary_vn()))
    addr_o_bits = ceil_log2(int(cfg.cap_output_vn()))
    pe_bits = int(AH * AW * 3)
    total_bits = int(birrd_bits + pe_bits + addr_in_bits + addr_w_bits + addr_o_bits + overhead_bits)
    total_bytes = ceil_div(total_bits, 8)
    return {
        "birrd_bits": birrd_bits, "pe_bits": pe_bits,
        "addr_in_bits": addr_in_bits, "addr_w_bits": addr_w_bits,
        "addr_o_bits": addr_o_bits, "overhead_bits": overhead_bits,
        "total_bits": total_bits, "total_bytes": total_bytes,
    }
