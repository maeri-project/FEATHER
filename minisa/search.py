#!/usr/bin/env python3
"""6-stage GEMM mapping/layout co-search for MINISA.

Pipeline stages:
  Stage 1 (Tile):    Enumerate legal (Mt, Kt, Nt) tiling choices
  Stage 2 (Lower):   Lower tile into IVN/WVN/OVN counts with per-VN sizes
  Stage 3 (Group):   Form VN groups VG(m_t, k_g, n_t)
  Stage 4 (Combine): Combine groups sharing WVNs into combined columns
  Stage 5 (Map):     Derive ExecuteMapping params with design choices
                       (broadcast, interleaved, contiguous patterns)
  Stage 6 (Layout):  Search bank-conflict-free buffer layouts

Public API:
  co_search_layout_mapping   -- end-to-end search (backward compat)
  co_search_gemm             -- new-style search returning SearchCandidate
  brute_force_layer_search   -- per-layer search
  layout_constrained_search  -- fixed layout orders, vary tile/mapping
  multi_layer_search         -- inter-layer conflict resolution
"""

from __future__ import annotations

import itertools
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import (
    FeatherPlusConfig, TraceBundle, CycleBreakdown, ceil_div,
    canonical_n_ivn, canonical_n_wvn, canonical_n_ovn,
    canonical_n_vndp, canonical_n_vn_groups,
    reduction_vn_sizes,
)
from .layout import (
    choose_layout_W, choose_layout_I, choose_layout_O,
    choose_tile_sizes, LayoutSpec,
    check_ob_port_conflict, layouts_match, derive_ovn_from_ivn,
    check_inter_layer_layout, OVN_TO_IVN_ORDER,
)
from .trace import generate_trace_gemm, estimate_minisa_inst_bytes
from .cycles import estimate_cycles_for_gemm, model_instruction_fetch


# ===================================================================
# Constants
# ===================================================================

# Sequential layout search priority (Stage 6 fast-path)
ORDER_SEARCH_PRIORITY = (0, 1, 2, 3, 4, 5)

# Dataflow pruning heuristics
DATAFLOW_PRUNE_WORK_RATIO = 8.0
DATAFLOW_PRUNE_CYCLE_RATIO = 1.002

# Dataflow name normalization
DATAFLOW_ALIASES: Dict[str, str] = {
    "WO-S": "weight_stationary",
    "WO_S": "weight_stationary",
    "wo-s": "weight_stationary",
    "wo_s": "weight_stationary",
    "weight_stationary": "weight_stationary",
    "IO-S": "input_stationary",
    "IO_S": "input_stationary",
    "io-s": "input_stationary",
    "io_s": "input_stationary",
    "input_stationary": "input_stationary",
    "auto": "auto",
}


def normalize_dataflow(dataflow: str) -> str:
    """Normalize dataflow string to canonical form."""
    key = dataflow.strip()
    normalized = DATAFLOW_ALIASES.get(key)
    if normalized is None:
        normalized = DATAFLOW_ALIASES.get(key.lower())
    if normalized is None:
        raise ValueError(f"Unsupported dataflow: {dataflow!r}")
    return normalized


# ===================================================================
# Stage 1: Tiling — data structures and enumeration
# ===================================================================

@dataclass(frozen=True)
class TilingChoice:
    """A candidate (Mt, Kt, Nt) tile size."""
    Mt: int
    Kt: int
    Nt: int


def _power_of_2_candidates(start: int, maximum: int) -> List[int]:
    """Generate power-of-2 values from *start* up to *maximum*.

    Always includes *maximum* itself (even if not a power of 2) to
    ensure the full-dimension tiling is considered.
    """
    vals = set()
    v = max(1, int(start))
    while v <= maximum:
        vals.add(v)
        v *= 2
    vals.add(maximum)
    return sorted(vals)


def enumerate_tiling_choices(
    M: int, K: int, N: int, cfg: FeatherPlusConfig,
) -> List[TilingChoice]:
    """Stage 1: Enumerate all legal (Mt, Kt, Nt) tiling strategies.

    A tiling is legal if the resulting buffer occupancy fits on-chip
    and the N-subgroups fit within AW columns.
    """
    AH, AW = cfg.AH, cfg.AW
    cap_str = cfg.cap_stream_vn()
    cap_sta = cfg.cap_stationary_vn()
    cap_out = cfg.cap_output_vn()

    Mt_candidates = _power_of_2_candidates(AH, M)
    Kt_candidates = _power_of_2_candidates(AH, K)
    Nt_candidates = _power_of_2_candidates(1, N)
    if N not in Nt_candidates:
        Nt_candidates.append(N)
        Nt_candidates.sort()

    valid: List[TilingChoice] = []
    for Mt in Mt_candidates:
        for Kt in Kt_candidates:
            for Nt in Nt_candidates:
                lw = choose_layout_W(cfg, Kt, Nt)
                li = choose_layout_I(cfg, Mt, Kt)
                lo = choose_layout_O(cfg, Mt, Nt)
                if (lw.vn_count() <= cap_sta
                        and li.vn_count() <= cap_str
                        and lo.vn_count() <= cap_out):
                    valid.append(TilingChoice(Mt, Kt, Nt))

    valid.sort(key=lambda t: -(t.Mt * t.Kt * t.Nt))
    MAX_STRATEGIES = 512
    if len(valid) > MAX_STRATEGIES:
        valid = valid[:MAX_STRATEGIES]
    return valid


# ===================================================================
# Stage 2: Lowering — derive VN structure from tile dimensions
# ===================================================================

@dataclass(frozen=True)
class LoweredTile:
    """Result of lowering a tile into VN structure.

    vn_sizes: per-K-group VN sizes.  For K divisible by AH, all are AH.
              For K not divisible by AH, the last group has fewer elements.
              Example: Kt=25, AH=16 → vn_sizes=(16, 9)
    """
    Mt: int
    Kt: int
    Nt: int
    K_g: int                        # number of K-groups = len(vn_sizes)
    vn_sizes: Tuple[int, ...]      # per-K-group VN size (1..AH)
    n_col_types: int                # ceil(Nt / AH) — number of N-subgroups
    n_ivn: int
    n_wvn: int
    n_ovn: int
    n_vn_groups: int
    n_vndp: int


def lower_tile(tile: TilingChoice, cfg: FeatherPlusConfig) -> LoweredTile:
    """Stage 2: Lower a tiling choice into VN counts and per-K-group sizes."""
    vn_sizes = reduction_vn_sizes(tile.Kt, cfg.AH)
    K_g = len(vn_sizes)
    n_col_types = ceil_div(tile.Nt, cfg.AH)
    return LoweredTile(
        Mt=tile.Mt,
        Kt=tile.Kt,
        Nt=tile.Nt,
        K_g=K_g,
        vn_sizes=vn_sizes,
        n_col_types=n_col_types,
        n_ivn=canonical_n_ivn(tile.Mt, tile.Kt, cfg.AH),
        n_wvn=canonical_n_wvn(tile.Kt, tile.Nt, cfg.AH),
        n_ovn=canonical_n_ovn(tile.Mt, tile.Nt, cfg.AH),
        n_vn_groups=canonical_n_vn_groups(tile.Mt, tile.Kt, tile.Nt, cfg.AH),
        n_vndp=canonical_n_vndp(tile.Mt, tile.Kt, tile.Nt, cfg.AH),
    )


# ===================================================================
# Stage 3: VN Group formation
# ===================================================================

@dataclass
class _SearchVNGroup:
    """A VN Group for search: VNDPs sharing the same IVN and fitting in one column.

    Each PE column has AH PEs, each holding one full WVN (AH weight elements).
    Therefore a VN Group can contain **at most AH WVNs** (one per PE).

    When Nt_eff > AH, the N output channels are split into
    ceil(Nt_eff / AH) sub-groups, each mapping to one PE column.
    """
    m: int
    k_g: int    # K-group index (0..K_g-1)
    wvn_row: int
    wvn_cols: List[int]
    vn_subgroup: int = 0


def _form_vn_groups(
    Mt_eff: int, Kt_eff: int, Nt_eff: int, AH: int, AW: int,
) -> List[_SearchVNGroup]:
    """Stage 3: Form VN Groups from tile dimensions.

    For a tile of size (Mt_eff, Kt_eff, Nt_eff) on AH x AW NEST:
      - K_g = ceil(Kt_eff / AH) K-groups (reduction tiles)
      - Each VN Group = 1 IVN (one output row) + ≤AH WVNs
      - With Nt_eff output channels: ceil(Nt_eff / AH) sub-groups per (m_t, k_g)
      - Each group indexed by (m_t, k_g, subgroup)
    """
    K_g = ceil_div(Kt_eff, AH)
    n_col_types = ceil_div(Nt_eff, AH)

    groups = []
    for kg in range(K_g):
        for g in range(n_col_types):
            col_start = g * AH
            col_end = min(col_start + AH, Nt_eff)
            wvn_cols = list(range(col_start, col_end))
            for m in range(Mt_eff):
                groups.append(_SearchVNGroup(
                    m=m, k_g=kg,
                    wvn_row=kg,
                    wvn_cols=wvn_cols,
                    vn_subgroup=g,
                ))

    # --- Sanity checks ---
    expected_n_groups = canonical_n_vn_groups(Mt_eff, Kt_eff, Nt_eff, AH)
    assert len(groups) == expected_n_groups, (
        f"VN group count mismatch: got {len(groups)}, "
        f"expected {expected_n_groups} = {Mt_eff}×{K_g}×{n_col_types}")

    for g in groups:
        assert 1 <= len(g.wvn_cols) <= AH

    total_vndps = sum(len(g.wvn_cols) for g in groups)
    expected_vndps = canonical_n_vndp(Mt_eff, Kt_eff, Nt_eff, AH)
    assert total_vndps == expected_vndps

    return groups


# ===================================================================
# Stage 4: Combining — merge VN Groups into combined columns
# ===================================================================

@dataclass
class CombinedColumn:
    """A PE column after combining VN Groups (Stage 4).

    Multiple VN Groups that share the same WVN set can be packed into
    the same PE column by streaming multiple IVNs sequentially.
    """
    groups: List[_SearchVNGroup]
    wvn_row: int
    wvn_cols: List[int]
    ivn_sequence: List[Tuple[int, int]]


def _combine_vn_groups_general(
    groups: List[_SearchVNGroup],
    Nt_eff: int,
    AH: int,
    AW: int,
    dup_factor: int = 0,
) -> Tuple[List[CombinedColumn], int]:
    """Stage 4: Combine VN Groups with explicit weight duplication.

    Design choices (controlled by dup_factor):
      dup_factor = 0 (auto) : maximize duplication — broadcast pattern
          All AW columns share the same WVN set.  Maximum M-parallelism.
      dup_factor = 1        : no duplication — contiguous/interleaved pattern
          Pack max k_g per EM.  Fewest EMs, fewest streaming passes.
      intermediate values   : balanced trade-off

    All variants use **interleaved stride distribution** for IVN assignment:
      replica `rep` takes rows [rep, rep+n_replicas, rep+2*n_replicas, ...]
    to ensure bank-conflict-free streaming buffer access.
    """
    if not groups:
        return [], 0

    K_g = max(g.wvn_row for g in groups) + 1
    n_col_types = max(1, ceil_div(Nt_eff, AH))

    max_dup = max(1, AW // n_col_types)
    if dup_factor <= 0 or dup_factor > max_dup:
        dup_factor = max_dup

    cols_per_k_g = n_col_types * dup_factor
    k_g_per_em = max(1, AW // max(1, cols_per_k_g))

    by_type: Dict[Tuple[int, int], List[_SearchVNGroup]] = {}
    for g in groups:
        key = (g.wvn_row, g.vn_subgroup)
        by_type.setdefault(key, []).append(g)
    for key in by_type:
        by_type[key].sort(key=lambda g: g.m)

    all_combined: List[CombinedColumn] = []

    for kg_start in range(0, K_g, k_g_per_em):
        kg_end = min(kg_start + k_g_per_em, K_g)
        actual_k_g = kg_end - kg_start
        actual_dup = max(1, AW // max(1, actual_k_g * n_col_types))

        for kg in range(kg_start, kg_end):
            for rep in range(actual_dup):
                for sg in range(n_col_types):
                    key = (kg, sg)
                    all_m_groups = by_type.get(key, [])
                    rep_groups = all_m_groups[rep::actual_dup]

                    if not rep_groups:
                        continue

                    ivn_seq = [(g.m, g.k_g) for g in rep_groups]
                    all_combined.append(CombinedColumn(
                        groups=rep_groups,
                        wvn_row=kg,
                        wvn_cols=rep_groups[0].wvn_cols,
                        ivn_sequence=ivn_seq,
                    ))

    total_cols = len(all_combined)
    n_exec_mappings = ceil_div(total_cols, AW)
    return all_combined, n_exec_mappings


def _enumerate_dup_factors(Nt_eff: int, AH: int, AW: int) -> List[int]:
    """Enumerate valid weight duplication factors (1..max)."""
    n_col_types = max(1, ceil_div(Nt_eff, AH))
    max_dup = max(1, AW // n_col_types)
    return list(range(1, max_dup + 1))


def _dup_factor_pattern(dup_factor: int, n_col_types: int, AW: int) -> str:
    """Classify the design choice pattern for a duplication factor.

    broadcast:    all AW columns share same WVN row (dup = max, 1 k_g/EM)
    interleaved:  multiple k_g packed per EM (dup = 1, max k_g/EM)
    contiguous:   balanced trade-off (1 < dup < max)
    """
    max_dup = max(1, AW // max(1, n_col_types))
    if dup_factor >= max_dup:
        return "broadcast"
    elif dup_factor == 1:
        return "interleaved"
    else:
        return "contiguous"


# ===================================================================
# Stage 5: Mapping — derive ExecuteMapping parameters
# ===================================================================

@dataclass
class ExecuteMappingParams:
    """Parameters for one ExecuteMapping instruction.

    MINISA mapping: PE (a_h, a_w) -> WVN(r, c) where
      r = r_0 + floor(a_w / G_r)
      c = c_0 + s_r * a_h + s_c * (a_w mod G_c)

    All AW PE columns are always active. Out-of-bounds WVN accesses
    are zero-padded in hardware.

    vn_size: active VN height for this EM (1..AH).  For K divisible by AH,
             always equals AH.  For K not divisible by AH, the last EM in
             a tile may have vn_size < AH.
    pattern: design choice label ("broadcast", "interleaved", "contiguous")
    """
    r_0: int
    c_0: int
    G_r: int
    G_c: int
    s_r: int
    s_c: int
    n_ivn_per_col: int
    n_replicas: int = 1
    vn_size: int = 0        # 0 = full AH (set by caller)
    pattern: str = ""
    ivn_distribution: str = "interleaved"  # "interleaved" or "consecutive"


@dataclass
class ExecuteStreamingParams:
    """Parameters for one ExecuteStreaming instruction (paired with ExecuteMapping).

    ExecuteStreaming exposes the IVN streaming pattern as explicit ISA fields:
      dataflow: 0=IO-S, 1=WO-S
      m_0:      starting IVN row offset per column
      s_m:      stride of m_t increment per cycle (= n_replicas for interleaved)
      T:        number of m_t values streamed per column = ceil(Mt_eff / n_replicas)
      vn_size:  active VN height, encoded as (actual_vn_size - 1)
    """
    dataflow: int   # 0=IO-S, 1=WO-S
    m_0: int        # starting IVN row offset per column
    s_m: int        # stride of m_t increment per cycle
    T: int          # number of m_t values streamed per column
    vn_size: int    # encoded as actual_vn_size - 1


def derive_execute_streaming(
    em: ExecuteMappingParams,
    Mt_eff: int,
    dataflow: str = "weight_stationary",
) -> ExecuteStreamingParams:
    """Derive ExecuteStreaming parameters from an ExecuteMapping and tile context.

    Two IVN distribution modes:
      interleaved (default):
        col rep reads m_t = rep, rep+n_rep, rep+2*n_rep, ...
        m_0 = 0, s_m = n_replicas, T = ceil(Mt_eff / n_replicas)
      consecutive:
        col rep reads m_t = rep*T, rep*T+1, ..., rep*T+T-1
        m_0 = 0, s_m = 1, T = ceil(Mt_eff / n_replicas)
    """
    n_replicas = max(1, em.n_replicas)
    vn_sz = em.vn_size if em.vn_size > 0 else 1
    T = ceil_div(Mt_eff, n_replicas)
    ivn_dist = getattr(em, 'ivn_distribution', 'interleaved')

    if ivn_dist == "consecutive":
        s_m = 1
    else:
        s_m = n_replicas

    return ExecuteStreamingParams(
        dataflow=0 if dataflow == "input_stationary" else 1,
        m_0=0,
        s_m=s_m,
        T=T,
        vn_size=max(0, vn_sz - 1),
    )


def _analytical_em_params(
    Mt_eff: int, Kt_eff: int, Nt_eff: int,
    AH: int, AW: int, dup_factor: int,
    vn_sizes: Tuple[int, ...] = (),
) -> List[ExecuteMappingParams]:
    """Stage 5 (analytical fast-path): Compute EM params from tile counts.

    Produces the same result as the chain:
      _form_vn_groups → _combine_vn_groups_general → _derive_execute_mapping_params
    but in O(n_EMs) time and O(1) memory.

    Mixed G_r: within a tile, different EMs may have different G_r values
    when K-groups are packed into batches of varying size (e.g., the last
    batch may have fewer k_g).
    """
    K_g = ceil_div(Kt_eff, AH)
    n_col_types = max(1, ceil_div(Nt_eff, AH))

    max_dup = max(1, AW // n_col_types)
    if dup_factor <= 0 or dup_factor > max_dup:
        dup_factor = max_dup

    cols_per_k_g = n_col_types * dup_factor
    k_g_per_em = max(1, AW // max(1, cols_per_k_g))

    # Derive per-K-group VN sizes if not provided
    if not vn_sizes:
        vn_sizes = reduction_vn_sizes(Kt_eff, AH)

    pattern = _dup_factor_pattern(dup_factor, n_col_types, AW)

    params_list: List[ExecuteMappingParams] = []
    for kg_start in range(0, K_g, k_g_per_em):
        kg_end = min(kg_start + k_g_per_em, K_g)
        actual_k_g = kg_end - kg_start
        actual_dup = max(1, AW // max(1, actual_k_g * n_col_types))
        effective_reps = min(actual_dup, Mt_eff)

        if effective_reps == 0:
            continue

        r_0 = kg_start
        G_c = n_col_types
        s_c = AH if G_c > 1 else 0
        s_r = 1
        G_r = max(1, AW // actual_k_g) if actual_k_g > 0 else AW
        n_ivn_per_col = ceil_div(Mt_eff, effective_reps)
        n_replicas = max(1, G_r // max(1, G_c))

        # Per-VN_size: use the VN size of the first K-group in this batch.
        # All K-groups in one EM batch should have the same VN size
        # (ensured by the batch splitting logic below when sizes differ).
        vn_sz = int(vn_sizes[kg_start]) if kg_start < len(vn_sizes) else AH

        params_list.append(ExecuteMappingParams(
            r_0=r_0, c_0=0, G_r=G_r, G_c=G_c,
            s_r=s_r, s_c=s_c,
            n_ivn_per_col=n_ivn_per_col,
            n_replicas=n_replicas,
            vn_size=vn_sz,
            pattern=pattern,
        ))

    return params_list


def _analytical_em_params_vn_safe(
    Mt_eff: int, Kt_eff: int, Nt_eff: int,
    AH: int, AW: int, dup_factor: int,
    vn_sizes: Tuple[int, ...] = (),
    wvn_col_stride: str = "block",
    ivn_distribution: str = "interleaved",
) -> List[ExecuteMappingParams]:
    """Like _analytical_em_params but ensures K-groups with different VN sizes
    are never packed into the same EM batch.

    When K is not divisible by AH, the last K-group has vn_size < AH.
    This function splits EM batches at VN-size boundaries to ensure each
    EM has a uniform vn_size.

    Design choices
    --------------
    wvn_col_stride : "block" (default) or "strided"
        "block"  : consecutive WVN columns within each N-subgroup → s_r=1, s_c=AH
        "strided": WVN columns interleaved across N-subgroups → s_r=n_col_types, s_c=1
    ivn_distribution : "interleaved" (default) or "consecutive"
        Controls IVN row assignment across replicated columns.
        Stored in each EM and used by bank conflict checks and ES derivation.
    """
    K_g = ceil_div(Kt_eff, AH)
    n_col_types = max(1, ceil_div(Nt_eff, AH))

    max_dup = max(1, AW // n_col_types)
    if dup_factor <= 0 or dup_factor > max_dup:
        dup_factor = max_dup

    cols_per_k_g = n_col_types * dup_factor
    k_g_per_em = max(1, AW // max(1, cols_per_k_g))

    if not vn_sizes:
        vn_sizes = reduction_vn_sizes(Kt_eff, AH)

    pattern = _dup_factor_pattern(dup_factor, n_col_types, AW)

    params_list: List[ExecuteMappingParams] = []
    kg_start = 0
    while kg_start < K_g:
        # Find how many consecutive K-groups share the same VN size
        vn_sz = int(vn_sizes[kg_start]) if kg_start < len(vn_sizes) else AH
        kg_end = kg_start + 1
        while kg_end < min(kg_start + k_g_per_em, K_g):
            next_sz = int(vn_sizes[kg_end]) if kg_end < len(vn_sizes) else AH
            if next_sz != vn_sz:
                break
            kg_end += 1

        actual_k_g = kg_end - kg_start
        actual_dup = max(1, AW // max(1, actual_k_g * n_col_types))
        effective_reps = min(actual_dup, Mt_eff)

        if effective_reps > 0:
            G_c = n_col_types
            if wvn_col_stride == "strided" and G_c > 1:
                s_r = n_col_types
                s_c = 1
            else:
                s_r = 1
                s_c = AH if G_c > 1 else 0
            G_r = max(1, AW // actual_k_g) if actual_k_g > 0 else AW
            n_ivn_per_col = ceil_div(Mt_eff, effective_reps)
            n_replicas = max(1, G_r // max(1, G_c))

            params_list.append(ExecuteMappingParams(
                r_0=kg_start, c_0=0, G_r=G_r, G_c=G_c,
                s_r=s_r, s_c=s_c,
                n_ivn_per_col=n_ivn_per_col,
                n_replicas=n_replicas,
                vn_size=vn_sz,
                pattern=pattern,
                ivn_distribution=ivn_distribution,
            ))

        kg_start = kg_end

    return params_list


def _derive_execute_mapping_params(
    combined_columns: List[CombinedColumn],
    AH: int,
    AW: int,
    Nt_eff: int = 0,
) -> List[ExecuteMappingParams]:
    """Stage 5 (materialized path): Derive EM params from combined columns."""
    if not combined_columns:
        return []

    n_col_types = ceil_div(Nt_eff, AH) if Nt_eff > 0 else 1
    params_list: List[ExecuteMappingParams] = []

    for start in range(0, len(combined_columns), AW):
        end = min(start + AW, len(combined_columns))
        cols_in_exec = combined_columns[start:end]

        wvn_rows_used = sorted(set(c.wvn_row for c in cols_in_exec))
        n_distinct_rows = len(wvn_rows_used)

        r_0 = wvn_rows_used[0] if wvn_rows_used else 0
        G_c = n_col_types
        s_c = AH if G_c > 1 else 0
        s_r = 1

        if n_distinct_rows > 0:
            G_r = max(1, AW // n_distinct_rows)
        else:
            G_r = AW

        n_ivn_per_col = max(len(c.ivn_sequence) for c in cols_in_exec) if cols_in_exec else 1
        n_replicas = max(1, G_r // max(1, G_c))

        params_list.append(ExecuteMappingParams(
            r_0=r_0, c_0=0, G_r=G_r, G_c=G_c,
            s_r=s_r, s_c=s_c, n_ivn_per_col=n_ivn_per_col,
            n_replicas=n_replicas,
        ))

    return params_list


# ===================================================================
# IVN row index helper (used by bank conflict checks)
# ===================================================================

def _ivn_row_at_step(
    ivn_dist: str, rep: int, step: int,
    effective_reps: int, max_steps: int,
) -> int:
    """Compute the IVN m_t row index for a given replica and streaming step.

    interleaved : col rep reads rows [rep, rep+n_rep, rep+2*n_rep, ...]
    consecutive : col rep reads rows [rep*T, rep*T+1, ..., rep*T+T-1]
    """
    if ivn_dist == "consecutive":
        return rep * max_steps + step
    else:  # interleaved
        return rep + step * effective_reps


# ===================================================================
# Stage 6: Layout search — bank conflict checks
# ===================================================================

def _analytical_batch_info(
    Kt_eff: int, Nt_eff: int, Mt_eff: int,
    AH: int, AW: int, dup_factor: int,
) -> List[Tuple[int, int, int, int]]:
    """Per-EM-batch metadata for analytical conflict checks.

    Returns [(kg_start, actual_k_g, actual_dup, n_col_types), ...].
    """
    K_g = ceil_div(Kt_eff, AH)
    n_col_types = max(1, ceil_div(Nt_eff, AH))

    max_dup = max(1, AW // n_col_types)
    if dup_factor <= 0 or dup_factor > max_dup:
        dup_factor = max_dup

    cols_per_k_g = n_col_types * dup_factor
    k_g_per_em = max(1, AW // max(1, cols_per_k_g))

    batches = []
    for kg_start in range(0, K_g, k_g_per_em):
        kg_end = min(kg_start + k_g_per_em, K_g)
        actual_k_g = kg_end - kg_start
        actual_dup = max(1, AW // max(1, actual_k_g * n_col_types))
        batches.append((kg_start, actual_k_g, actual_dup, n_col_types))
    return batches


def _analytical_bank_conflict_check(
    params_list: List[ExecuteMappingParams],
    Mt_eff: int, Kt_eff: int, Nt_eff: int,
    AH: int, AW: int, dup_factor: int,
    lw, li,
) -> bool:
    """Analytical WVN + IVN bank conflict check.

    Returns True if conflict-free, False if conflict exists.
    """
    # --- WVN check (stationary buffer) ---
    N_L0 = lw.a0
    N_L1 = lw.a1
    K_L1 = lw.a2
    max_c_w = N_L0 * N_L1

    checked_groups: set = set()
    for em in params_list:
        group_key = (em.G_r, em.G_c, em.s_r, em.s_c, em.c_0)
        if em.G_r >= AW and group_key in checked_groups:
            continue
        checked_groups.add(group_key)

        # Use per-VN_size: only check active rows
        active_rows = em.vn_size if em.vn_size > 0 else AH
        for a_h in range(active_rows):
            bank_to_addr: dict = {}
            for a_w in range(AW):
                r = em.r_0 + a_w // max(1, em.G_r)
                c = em.c_0 + em.s_r * a_h + em.s_c * (a_w % max(1, em.G_c))
                if r < 0 or r >= K_L1 or c < 0 or c >= max_c_w:
                    continue
                nL0 = c % N_L0
                nL1 = c // N_L0
                kL1 = r
                addr = lw.linear_index({"kL1": kL1, "nL0": nL0, "nL1": nL1})
                bank = addr % AW
                if bank in bank_to_addr:
                    if bank_to_addr[bank] != addr:
                        return False
                else:
                    bank_to_addr[bank] = addr

    # --- IVN check (streaming buffer) ---
    M_L0 = li.a0
    M_L1 = li.a1
    J_L1 = li.a2

    # Read IVN distribution mode from exec_params
    ivn_dist = params_list[0].ivn_distribution if params_list else "interleaved"

    batches = _analytical_batch_info(Kt_eff, Nt_eff, Mt_eff, AH, AW, dup_factor)
    for _batch_idx, (kg_start, actual_k_g, actual_dup, n_col_types) in enumerate(batches):
        effective_reps = min(actual_dup, Mt_eff)
        if actual_k_g * effective_reps <= 1:
            continue

        max_steps = ceil_div(Mt_eff, effective_reps)
        for step in range(max_steps):
            bank_to_addr = {}
            for kg_offset in range(actual_k_g):
                kg = kg_start + kg_offset
                for rep in range(effective_reps):
                    m_row = _ivn_row_at_step(
                        ivn_dist, rep, step, effective_reps, max_steps)
                    if m_row >= Mt_eff:
                        continue
                    mL0 = m_row % M_L0 if M_L0 > 0 else 0
                    mL1 = m_row // M_L0 if M_L0 > 0 else m_row
                    jL1 = kg
                    if mL0 >= M_L0 or mL1 >= M_L1 or jL1 >= J_L1:
                        continue
                    addr = li.linear_index({"jL1": jL1, "mL0": mL0, "mL1": mL1})
                    bank = addr % AW
                    if bank in bank_to_addr:
                        if bank_to_addr[bank] != addr:
                            return False
                    else:
                        bank_to_addr[bank] = addr

    return True


def _analytical_ob_conflict_check(
    layout_o,
    params_list: List[ExecuteMappingParams],
    Mt_eff: int, Kt_eff: int, Nt_eff: int,
    AH: int, AW: int, dup_factor: int,
) -> bool:
    """Analytical OB port conflict check.

    Returns True if there IS a conflict, False if conflict-free.
    """
    P_L0 = layout_o.a0
    P_L1 = layout_o.a1
    Q_L1 = layout_o.a2

    # Read IVN distribution mode from exec_params
    ivn_dist = params_list[0].ivn_distribution if params_list else "interleaved"

    batches = _analytical_batch_info(Kt_eff, Nt_eff, Mt_eff, AH, AW, dup_factor)
    checked_ob: set = set()
    for _batch_idx, (kg_start, actual_k_g, actual_dup, n_col_types) in enumerate(batches):
        effective_reps = min(actual_dup, Mt_eff)

        ob_key = (effective_reps, n_col_types)
        if ob_key in checked_ob:
            continue
        checked_ob.add(ob_key)

        max_steps = ceil_div(Mt_eff, effective_reps)
        for step in range(max_steps):
            bank_to_row: dict = {}
            for rep in range(effective_reps):
                m_row = _ivn_row_at_step(
                    ivn_dist, rep, step, effective_reps, max_steps)
                if m_row >= Mt_eff:
                    continue
                for sg in range(n_col_types):
                    pL0 = m_row % P_L0 if P_L0 > 0 else 0
                    pL1 = m_row // P_L0 if P_L0 > 0 else m_row
                    qL1 = sg
                    if pL0 >= P_L0 or pL1 >= P_L1 or qL1 >= Q_L1:
                        continue
                    L = layout_o.linear_index(
                        {"pL0": pL0, "pL1": pL1, "qL1": qL1})
                    addr_col = L % AW
                    addr_row = L // AW
                    if addr_col in bank_to_row:
                        if bank_to_row[addr_col] != addr_row:
                            return True
                    else:
                        bank_to_row[addr_col] = addr_row

    return False


def _find_all_layout_selections(
    exec_params: List[ExecuteMappingParams],
    Mt_eff: int, Kt_eff: int, Nt_eff: int,
    AH: int, AW: int, dup_factor: int,
    cfg: FeatherPlusConfig,
) -> List[Tuple[int, int, int]]:
    """Stage 6 (exhaustive): Search all 216 layout order combos.

    Returns list of (order_w, order_i, order_o) tuples that are
    bank-conflict-free and fit in buffers.
    """
    cap_str = cfg.cap_stream_vn()
    cap_sta = cfg.cap_stationary_vn()
    cap_out = cfg.cap_output_vn()

    # Pre-compute valid order_o
    valid_oo: List[int] = []
    for oo in range(6):
        lo = choose_layout_O(cfg, Mt_eff, Nt_eff, oo)
        if lo.vn_count() > cap_out or lo.a0 > AW:
            continue
        if _analytical_ob_conflict_check(lo, exec_params,
                                          Mt_eff, Kt_eff, Nt_eff,
                                          AH, AW, dup_factor):
            continue
        valid_oo.append(oo)

    if not valid_oo:
        return []

    # Pre-compute valid (order_w, order_i) pairs
    valid_wi: List[Tuple[int, int]] = []
    for ow in range(6):
        lw = choose_layout_W(cfg, Kt_eff, Nt_eff, ow)
        if lw.vn_count() > cap_sta or lw.a0 > AW:
            continue
        for oi in range(6):
            li = choose_layout_I(cfg, Mt_eff, Kt_eff, oi)
            if li.vn_count() > cap_str or li.a0 > AW:
                continue
            if not _analytical_bank_conflict_check(
                    exec_params, Mt_eff, Kt_eff, Nt_eff,
                    AH, AW, dup_factor, lw, li):
                continue
            valid_wi.append((ow, oi))

    results = []
    for ow, oi in valid_wi:
        for oo in valid_oo:
            results.append((ow, oi, oo))

    return results


def _find_first_layout_selection(
    exec_params: List[ExecuteMappingParams],
    Mt_eff: int, Kt_eff: int, Nt_eff: int,
    AH: int, AW: int, dup_factor: int,
    cfg: FeatherPlusConfig,
) -> Optional[Tuple[int, int, int]]:
    """Stage 6 (sequential fast-path): Find first legal layout per operand.

    Searches I, then W, then O layouts in ORDER_SEARCH_PRIORITY order.
    Returns (order_w, order_i, order_o) or None.
    """
    cap_str = cfg.cap_stream_vn()
    cap_sta = cfg.cap_stationary_vn()
    cap_out = cfg.cap_output_vn()

    # Find first valid I layout
    best_oi = None
    best_li = None
    for oi in ORDER_SEARCH_PRIORITY:
        li = choose_layout_I(cfg, Mt_eff, Kt_eff, oi)
        if li.vn_count() > cap_str or li.a0 > AW:
            continue
        best_oi = oi
        best_li = li
        break

    if best_oi is None:
        return None

    # Find first valid W layout compatible with chosen I
    best_ow = None
    for ow in ORDER_SEARCH_PRIORITY:
        lw = choose_layout_W(cfg, Kt_eff, Nt_eff, ow)
        if lw.vn_count() > cap_sta or lw.a0 > AW:
            continue
        if _analytical_bank_conflict_check(
                exec_params, Mt_eff, Kt_eff, Nt_eff,
                AH, AW, dup_factor, lw, best_li):
            best_ow = ow
            break

    if best_ow is None:
        return None

    # Find first valid O layout
    best_oo = None
    for oo in ORDER_SEARCH_PRIORITY:
        lo = choose_layout_O(cfg, Mt_eff, Nt_eff, oo)
        if lo.vn_count() > cap_out or lo.a0 > AW:
            continue
        if not _analytical_ob_conflict_check(lo, exec_params,
                                              Mt_eff, Kt_eff, Nt_eff,
                                              AH, AW, dup_factor):
            best_oo = oo
            break

    if best_oo is None:
        return None

    return (best_ow, best_oi, best_oo)


# ===================================================================
# Per-tile search: run all 6 stages for one tile
# ===================================================================

# Fallback design choices: (wvn_col_stride, ivn_distribution)
# Default first, then alternatives. Strided WVN only useful when n_col_types > 1.
_DESIGN_CHOICES = [
    ("block", "interleaved"),       # default: block WVN columns, interleaved IVN
    ("block", "consecutive"),       # fallback 1: block WVN columns, consecutive IVN
    ("strided", "interleaved"),     # fallback 2: strided WVN columns, interleaved IVN
    ("strided", "consecutive"),     # fallback 3: strided WVN columns, consecutive IVN
]


def _search_for_tile(
    M: int, K: int, N: int,
    Mt: int, Kt: int, Nt: int,
    cfg: FeatherPlusConfig,
    exhaustive_layout: bool = True,
) -> List[Tuple[int, int, int, List[ExecuteMappingParams]]]:
    """Run 6-stage pipeline for a single (Mt, Kt, Nt) tile.

    Stage 1: tile is given
    Stage 2: lower_tile
    Stage 3-4: analytical combining (implicit in _analytical_em_params)
    Stage 5: derive EM params for each dup_factor
    Stage 6: search layouts (exhaustive or sequential)

    When Stage 6 fails for all dup_factors with default design choices,
    falls back to alternative choices:
      - Strided WVN column grouping (changes s_r, s_c in EM)
      - Consecutive IVN distribution (changes m_0, s_m in ES)

    Returns list of (order_w, order_i, order_o, exec_params) tuples.
    """
    AH, AW = cfg.AH, cfg.AW
    Mt_eff = min(Mt, M)
    Kt_eff = min(Kt, K)
    Nt_eff = min(Nt, N)

    # Stage 2: lower
    lowered = lower_tile(TilingChoice(Mt_eff, Kt_eff, Nt_eff), cfg)
    dup_factors = _enumerate_dup_factors(Nt_eff, AH, AW)
    n_col_types = max(1, ceil_div(Nt_eff, AH))

    for wvn_cs, ivn_dist in _DESIGN_CHOICES:
        # Skip strided WVN when n_col_types == 1 (no effect on s_r/s_c)
        if wvn_cs == "strided" and n_col_types <= 1:
            continue

        # Stage 3-5: enumerate dup factors → analytical EM params
        combining_results: List[Tuple[List[ExecuteMappingParams], int]] = []
        seen_params: set = set()

        for dup in dup_factors:
            exec_params = _analytical_em_params_vn_safe(
                Mt_eff, Kt_eff, Nt_eff, AH, AW, dup,
                vn_sizes=lowered.vn_sizes,
                wvn_col_stride=wvn_cs, ivn_distribution=ivn_dist)
            sig = tuple((p.G_r, p.G_c, p.r_0, p.c_0, p.s_r, p.s_c,
                         p.vn_size, p.ivn_distribution)
                        for p in exec_params)
            if sig in seen_params:
                continue
            seen_params.add(sig)
            combining_results.append((exec_params, dup))

        # Stage 6: Layout search
        if exhaustive_layout:
            best_for_layout: Dict[Tuple[int, int, int], List[ExecuteMappingParams]] = {}
            for exec_params, dup in combining_results:
                layouts = _find_all_layout_selections(
                    exec_params, Mt_eff, Kt_eff, Nt_eff,
                    AH, AW, dup, cfg)
                for ow, oi, oo in layouts:
                    key = (ow, oi, oo)
                    if key not in best_for_layout or len(exec_params) < len(best_for_layout[key]):
                        best_for_layout[key] = exec_params

            if best_for_layout:
                return [(ow, oi, oo, ep) for (ow, oi, oo), ep in best_for_layout.items()]

        else:
            # Sequential fast-path: first valid layout per dup_factor
            for exec_params, dup in combining_results:
                result = _find_first_layout_selection(
                    exec_params, Mt_eff, Kt_eff, Nt_eff,
                    AH, AW, dup, cfg)
                if result is not None:
                    ow, oi, oo = result
                    return [(ow, oi, oo, exec_params)]

    return []  # all design choices exhausted


# ===================================================================
# Verification helpers
# ===================================================================

def _verify_combined_columns(
    combined: List[CombinedColumn],
    groups: List[_SearchVNGroup],
    Mt_eff: int, Kt_eff: int, Nt_eff: int,
    AH: int, AW: int,
    label: str = "",
) -> None:
    """Verify invariants on combined columns."""
    pfx = f"[{label}] " if label else ""
    K_g = ceil_div(Kt_eff, AH)
    n_col_types = ceil_div(Nt_eff, AH)

    total_vndps = sum(len(c.ivn_sequence) * len(c.wvn_cols) for c in combined)
    expected_vndps = canonical_n_vndp(Mt_eff, Kt_eff, Nt_eff, AH)
    assert total_vndps == expected_vndps, (
        f"{pfx}VNDP conservation: got {total_vndps}, expected {expected_vndps}")

    total_groups = sum(len(c.groups) for c in combined)
    expected_groups = canonical_n_vn_groups(Mt_eff, Kt_eff, Nt_eff, AH)
    assert total_groups == expected_groups

    for i, c in enumerate(combined):
        for g in c.groups:
            assert g.wvn_row == c.wvn_row
            assert g.wvn_cols == c.wvn_cols

    seen: Dict[Tuple[int, int, int], int] = {}
    for i, c in enumerate(combined):
        sg = c.groups[0].vn_subgroup if c.groups else 0
        for m_row, j in c.ivn_sequence:
            key = (m_row, j, sg)
            assert key not in seen, (
                f"{pfx}IVN(m={m_row},j={j},sg={sg}) in columns {seen[key]} and {i}")
            seen[key] = i


# ===================================================================
# Public API: backward-compatible data structures
# ===================================================================

@dataclass
class SearchResult:
    """Result of a layout+mapping search.

    Attributes
    ----------
    dataflow : "weight_stationary" or "input_stationary"
    search_M, search_K, search_N : int
        The (possibly transposed) dimensions used by the search.
    """
    order_w: int
    order_i: int
    order_o: int
    Mt: int
    Kt: int
    Nt: int
    cycles_total: int
    inst_bytes: int
    dataflow: str = "weight_stationary"
    search_M: int = 0
    search_K: int = 0
    search_N: int = 0
    trace_bundle: Optional[TraceBundle] = None


@dataclass
class LayerCandidate:
    """One legal (SetIVNLayout, SetWVNLayout, ExecuteMapping) candidate."""
    order_w: int
    order_i: int
    order_o: int
    Mt: int
    Kt: int
    Nt: int
    cycles_total: int
    inst_bytes: int
    utilization: float
    layout_w: LayoutSpec
    layout_i: LayoutSpec
    layout_o: LayoutSpec
    exec_params: Optional[List[ExecuteMappingParams]] = None


@dataclass
class LayerSpec:
    """Workload specification for one layer."""
    M: int
    K: int
    N: int
    name: str = ""


@dataclass
class MultiLayerResult:
    """Result of multi-layer search with inter-layer resolution."""
    layers: List[LayerCandidate]
    total_cycles: int
    total_inst_bytes: int
    inter_layer_matches: List[bool]


# ===================================================================
# Public API: brute_force_layer_search
# ===================================================================

def brute_force_layer_search(
    M: int, K: int, N: int,
    cfg: FeatherPlusConfig,
    first_valid_only: bool = False,
) -> List[LayerCandidate]:
    """6-stage brute-force search over VN Group strategies and layouts.

    For each (Mt, Kt, Nt) tile:
      Stage 1: Enumerate tiling choices
      Stage 2: Lower tile into VN structure
      Stage 3: Form VN groups (analytical)
      Stage 4: Combine with all dup_factors
      Stage 5: Derive ExecuteMapping params
      Stage 6: Search all 216 layout combos

    Returns all valid candidates sorted by utilization (highest first).
    """
    candidates: List[LayerCandidate] = []
    total_macs = M * K * N
    peak = cfg.effective_peak_macs_per_cycle()

    strategies = enumerate_tiling_choices(M, K, N, cfg)

    for tile in strategies:
        Mt, Kt, Nt = tile.Mt, tile.Kt, tile.Nt
        valid_layouts = _search_for_tile(M, K, N, Mt, Kt, Nt, cfg,
                                          exhaustive_layout=True)
        if not valid_layouts:
            continue

        cb = estimate_cycles_for_gemm(M, K, N, cfg, Mt, Kt, Nt,
                                      reuse_input_across_N=False)
        inst_bytes = estimate_minisa_inst_bytes(M, K, N, cfg, Mt, Kt, Nt)
        inst_model = model_instruction_fetch(inst_bytes, cb.total, cfg)
        total = cb.total + int(inst_model["total_extra_cycles"])

        compute_cycles = cb.compute + cb.out_to_stream
        utilization = (total_macs / max(1, compute_cycles * peak)
                       if compute_cycles > 0 else 0.0)

        for ow, oi, oo, exec_params in valid_layouts:
            lw = choose_layout_W(cfg, Kt, Nt, ow)
            li = choose_layout_I(cfg, Mt, Kt, oi)
            lo = choose_layout_O(cfg, Mt, Nt, oo)

            candidates.append(LayerCandidate(
                order_w=ow, order_i=oi, order_o=oo,
                Mt=Mt, Kt=Kt, Nt=Nt,
                cycles_total=total, inst_bytes=inst_bytes,
                utilization=utilization,
                layout_w=lw, layout_i=li, layout_o=lo,
                exec_params=exec_params,
            ))

        if first_valid_only and candidates:
            break

    candidates.sort(key=lambda c: (-c.utilization, c.cycles_total))
    return candidates


# ===================================================================
# Public API: layout_constrained_search
# ===================================================================

def layout_constrained_search(
    M: int, K: int, N: int,
    cfg: FeatherPlusConfig,
    order_w: int, order_i: int, order_o: int,
) -> List[LayerCandidate]:
    """Layout-constrained search: fix layout orders, vary tiling/grouping/combining.

    Exhaustively searches over:
      (1) Tiling       — all legal (Mt, Kt, Nt)
      (2) Combining    — all dup_factors
      (3) Mapping      — analytical EM params
    with fixed layout orders, checking feasibility at Stage 6.
    """
    candidates: List[LayerCandidate] = []
    total_macs = M * K * N
    peak = cfg.effective_peak_macs_per_cycle()
    AH, AW = cfg.AH, cfg.AW

    strategies = enumerate_tiling_choices(M, K, N, cfg)

    for tile in strategies:
        Mt, Kt, Nt = tile.Mt, tile.Kt, tile.Nt
        Mt_eff = min(Mt, M)
        Kt_eff = min(Kt, K)
        Nt_eff = min(Nt, N)

        lowered = lower_tile(TilingChoice(Mt_eff, Kt_eff, Nt_eff), cfg)
        dup_factors = _enumerate_dup_factors(Nt_eff, AH, AW)

        lw = choose_layout_W(cfg, Kt_eff, Nt_eff, order_w)
        li = choose_layout_I(cfg, Mt_eff, Kt_eff, order_i)

        n_col_types_t = max(1, ceil_div(Nt_eff, AH))
        best_exec: Optional[List[ExecuteMappingParams]] = None

        for wvn_cs, ivn_dist in _DESIGN_CHOICES:
            if wvn_cs == "strided" and n_col_types_t <= 1:
                continue

            seen_params: set = set()
            for dup in dup_factors:
                exec_params = _analytical_em_params_vn_safe(
                    Mt_eff, Kt_eff, Nt_eff, AH, AW, dup,
                    vn_sizes=lowered.vn_sizes,
                    wvn_col_stride=wvn_cs, ivn_distribution=ivn_dist)

                sig = tuple((p.G_r, p.G_c, p.r_0, p.c_0, p.s_r, p.s_c,
                             p.vn_size, p.ivn_distribution)
                            for p in exec_params)
                if sig in seen_params:
                    continue
                seen_params.add(sig)

                lo = choose_layout_O(cfg, Mt_eff, Nt_eff, order_o)
                cap_str_v = cfg.cap_stream_vn()
                cap_sta_v = cfg.cap_stationary_vn()
                cap_out_v = cfg.cap_output_vn()
                if lw.vn_count() > cap_sta_v or lw.a0 > AW:
                    continue
                if li.vn_count() > cap_str_v or li.a0 > AW:
                    continue
                if lo.vn_count() > cap_out_v or lo.a0 > AW:
                    continue
                if _analytical_ob_conflict_check(lo, exec_params,
                                                  Mt_eff, Kt_eff, Nt_eff,
                                                  AH, AW, dup):
                    continue
                if not _analytical_bank_conflict_check(
                        exec_params, Mt_eff, Kt_eff, Nt_eff,
                        AH, AW, dup, lw, li):
                    continue

                if best_exec is None or len(exec_params) < len(best_exec):
                    best_exec = exec_params

            if best_exec is not None:
                break  # found a valid config, no need to try more design choices

        if best_exec is None:
            continue

        cb = estimate_cycles_for_gemm(M, K, N, cfg, Mt, Kt, Nt,
                                      reuse_input_across_N=False)
        inst_bytes = estimate_minisa_inst_bytes(M, K, N, cfg, Mt, Kt, Nt)
        inst_model = model_instruction_fetch(inst_bytes, cb.total, cfg)
        total = cb.total + int(inst_model["total_extra_cycles"])

        compute_cycles = cb.compute + cb.out_to_stream
        utilization = (total_macs / max(1, compute_cycles * peak)
                       if compute_cycles > 0 else 0.0)

        lw = choose_layout_W(cfg, Kt, Nt, order_w)
        li = choose_layout_I(cfg, Mt, Kt, order_i)
        lo = choose_layout_O(cfg, Mt, Nt, order_o)

        candidates.append(LayerCandidate(
            order_w=order_w, order_i=order_i, order_o=order_o,
            Mt=Mt, Kt=Kt, Nt=Nt,
            cycles_total=total, inst_bytes=inst_bytes,
            utilization=utilization,
            layout_w=lw, layout_i=li, layout_o=lo,
            exec_params=best_exec,
        ))

    candidates.sort(key=lambda c: (-c.utilization, c.cycles_total))
    return candidates


# ===================================================================
# Public API: co_search_layout_mapping (backward compat)
# ===================================================================

def co_search_layout_mapping(
    M: int, K: int, N: int,
    cfg: FeatherPlusConfig,
    generate_trace: bool = True,
    dataflow: str = "auto",
) -> SearchResult:
    """End-to-end 6-stage search with dataflow selection.

    Searches both weight-stationary and input-stationary dataflows,
    selecting the one with lower total latency.

    Dataflows
    ---------
    weight_stationary (WO-S): WVNs stored in PE registers, IVNs stream.
    input_stationary (IO-S):  IVNs stored in PE registers, WVNs stream.
        Internally maps the workload as {M'=N, K, N'=M}.
    """
    dataflow = normalize_dataflow(dataflow)
    try_ws = dataflow in ("auto", "weight_stationary")
    try_is = dataflow in ("auto", "input_stationary")

    best_ws: Optional[LayerCandidate] = None
    best_is: Optional[LayerCandidate] = None
    ws_total: int = 0
    is_total: int = 0

    if try_ws:
        ws_cands = brute_force_layer_search(M, K, N, cfg, first_valid_only=True)
        if ws_cands:
            best_ws = ws_cands[0]
            ws_total = best_ws.cycles_total

    if try_is and not (M == N):
        is_cands = brute_force_layer_search(N, K, M, cfg, first_valid_only=True)
        if is_cands:
            best_is = is_cands[0]
            is_total = best_is.cycles_total

    if best_ws is None and best_is is None:
        raise RuntimeError(
            f"No valid mapping found for M={M}, K={K}, N={N}, "
            f"AH={cfg.AH}, AW={cfg.AW}")

    if best_ws is not None and best_is is not None:
        use_is = is_total < ws_total
    elif best_is is not None:
        use_is = True
    else:
        use_is = False

    if use_is:
        best = best_is
        chosen_dataflow = "input_stationary"
        search_M, search_K, search_N = N, K, M
    else:
        best = best_ws
        chosen_dataflow = "weight_stationary"
        search_M, search_K, search_N = M, K, N

    tb = None
    if generate_trace:
        tb = generate_trace_gemm(search_M, search_K, search_N, cfg,
                                  candidate=best,
                                  dataflow=chosen_dataflow)

    return SearchResult(
        order_w=best.order_w, order_i=best.order_i, order_o=best.order_o,
        Mt=best.Mt, Kt=best.Kt, Nt=best.Nt,
        cycles_total=best.cycles_total, inst_bytes=best.inst_bytes,
        dataflow=chosen_dataflow,
        search_M=search_M, search_K=search_K, search_N=search_N,
        trace_bundle=tb,
    )


# ===================================================================
# Public API: co_search_gemm (new-style)
# ===================================================================

@dataclass
class SearchCandidate:
    """New-style search result with full pipeline details."""
    dataflow: str
    AH: int
    AW: int
    search_M: int
    search_K: int
    search_N: int
    Mt: int
    Kt: int
    Nt: int
    lowered_tile: LoweredTile
    exec_params: List[ExecuteMappingParams]
    order_w: int
    order_i: int
    order_o: int
    cycles_total: int
    inst_bytes: int
    utilization: float

    def streaming_params(self) -> List[ExecuteStreamingParams]:
        """Derive ExecuteStreaming params for each paired ExecuteMapping."""
        Mt_eff = min(self.Mt, self.search_M)
        return [derive_execute_streaming(em, Mt_eff, self.dataflow)
                for em in self.exec_params]

    def to_dict(self) -> Dict[str, Any]:
        """Export in template.json format."""
        cfg_dummy = None  # layout dims from exec_params
        lw = choose_layout_W(
            type('_', (), {'AH': self.AH, 'AW': self.AW,
                           'cap_stationary_vn': lambda: 999999,
                           'stationary_bytes': 999999, 'w_bytes': 1})(),
            self.Kt, self.Nt, self.order_w) if False else None

        Mt_eff = min(self.Mt, self.search_M)
        Kt_eff = min(self.Kt, self.search_K)
        Nt_eff = min(self.Nt, self.search_N)
        AH, AW = self.AH, self.AW

        N_L0 = min(AW, max(1, Nt_eff))
        N_L1 = ceil_div(Nt_eff, N_L0)
        K_L1 = ceil_div(Kt_eff, AH)
        M_L0 = min(AW, max(1, Mt_eff))
        M_L1 = ceil_div(Mt_eff, M_L0)
        J_L1 = ceil_div(Kt_eff, AH)
        P_L0 = min(AW, max(1, Mt_eff))
        P_L1 = ceil_div(Mt_eff, P_L0)
        Q_L1 = ceil_div(Nt_eff, AH)

        es_list = self.streaming_params()
        invocations = []
        for em, es in zip(self.exec_params, es_list):
            invocations.append({
                "ExecuteMapping": {
                    "r_0": em.r_0, "c_0": em.c_0,
                    "G_r": em.G_r, "G_c": em.G_c,
                    "s_r": em.s_r, "s_c": em.s_c,
                },
                "ExecuteStreaming": {
                    "dataflow": es.dataflow, "m_0": es.m_0,
                    "s_m": es.s_m, "T": es.T, "vn_size": es.vn_size,
                },
            })

        return {
            "dataflow": self.dataflow,
            "tiling": {"Mt": self.Mt, "Kt": self.Kt, "Nt": self.Nt},
            "L1": [
                {"WVN": {"order": self.order_w, "N_L1": N_L1, "N_L0": N_L0, "K_L1": K_L1}},
                {"IVN": {"order": self.order_i, "M_L1": M_L1, "M_L0": M_L0, "J_L1": J_L1}},
                {"OVN": {"order": self.order_o, "P_L1": P_L1, "P_L0": P_L0, "Q_L1": Q_L1}},
                {"invocations": invocations},
                {"latency": self.cycles_total, "utilization": round(self.utilization, 6)},
            ],
        }


@dataclass
class GemmSearchResult:
    """Result of co_search_gemm."""
    best: SearchCandidate
    candidates: List[SearchCandidate]


def _search_tile_candidates_new(
    M: int, K: int, N: int,
    tile: TilingChoice,
    cfg: FeatherPlusConfig,
    dataflow: str,
    search_dims: Tuple[int, int, int],
    exhaustive_layout: bool = True,
) -> List[SearchCandidate]:
    """Run full 6-stage pipeline for one tile, returning SearchCandidates.

    Tries default design choices first; if Stage 6 fails for all dup_factors,
    falls back to alternative WVN column stride and IVN distribution choices.
    """
    sM, sK, sN = search_dims
    Mt, Kt, Nt = tile.Mt, tile.Kt, tile.Nt
    Mt_eff = min(Mt, sM)
    Kt_eff = min(Kt, sK)
    Nt_eff = min(Nt, sN)
    AH, AW = cfg.AH, cfg.AW

    lowered = lower_tile(TilingChoice(Mt_eff, Kt_eff, Nt_eff), cfg)

    dup_factors = _enumerate_dup_factors(Nt_eff, AH, AW)
    total_macs = sM * sK * sN
    peak = cfg.effective_peak_macs_per_cycle()
    n_col_types = max(1, ceil_div(Nt_eff, AH))

    for wvn_cs, ivn_dist in _DESIGN_CHOICES:
        # Skip strided WVN when n_col_types == 1 (no effect on s_r/s_c)
        if wvn_cs == "strided" and n_col_types <= 1:
            continue

        candidates: List[SearchCandidate] = []
        seen_params: set = set()

        for dup in dup_factors:
            exec_params = _analytical_em_params_vn_safe(
                Mt_eff, Kt_eff, Nt_eff, AH, AW, dup,
                vn_sizes=lowered.vn_sizes,
                wvn_col_stride=wvn_cs, ivn_distribution=ivn_dist)
            sig = tuple((p.G_r, p.G_c, p.r_0, p.c_0, p.s_r, p.s_c,
                         p.vn_size, p.ivn_distribution)
                        for p in exec_params)
            if sig in seen_params:
                continue
            seen_params.add(sig)

            if exhaustive_layout:
                layouts = _find_all_layout_selections(
                    exec_params, Mt_eff, Kt_eff, Nt_eff, AH, AW, dup, cfg)
            else:
                result = _find_first_layout_selection(
                    exec_params, Mt_eff, Kt_eff, Nt_eff, AH, AW, dup, cfg)
                layouts = [result] if result else []

            if not layouts:
                continue

            # Cycle estimation (same for all layouts of same tile)
            cb = estimate_cycles_for_gemm(sM, sK, sN, cfg, Mt, Kt, Nt,
                                          reuse_input_across_N=False)
            inst_bytes = estimate_minisa_inst_bytes(sM, sK, sN, cfg, Mt, Kt, Nt)
            inst_model = model_instruction_fetch(inst_bytes, cb.total, cfg)
            total = cb.total + int(inst_model["total_extra_cycles"])
            compute_cycles = cb.compute + cb.out_to_stream
            utilization = (total_macs / max(1, compute_cycles * peak)
                           if compute_cycles > 0 else 0.0)

            for ow, oi, oo in layouts:
                candidates.append(SearchCandidate(
                    dataflow=dataflow,
                    AH=AH, AW=AW,
                    search_M=sM, search_K=sK, search_N=sN,
                    Mt=Mt, Kt=Kt, Nt=Nt,
                    lowered_tile=lowered,
                    exec_params=exec_params,
                    order_w=ow, order_i=oi, order_o=oo,
                    cycles_total=total, inst_bytes=inst_bytes,
                    utilization=utilization,
                ))

        if candidates:
            return candidates

    return []  # all design choices exhausted


def co_search_gemm(
    M: int, K: int, N: int,
    cfg: FeatherPlusConfig,
    *,
    dataflow: str = "auto",
    jobs: int = 1,
    exhaustive_layout: bool = True,
) -> GemmSearchResult:
    """New-style co-search returning SearchCandidate with full details.

    Parameters
    ----------
    M, K, N : workload dimensions
    cfg : FeatherPlusConfig
    dataflow : "auto", "weight_stationary"/"WO-S", "input_stationary"/"IO-S"
    jobs : number of parallel tile search threads
    exhaustive_layout : True for 216-combo, False for sequential fast-path
    """
    dataflow = normalize_dataflow(dataflow)

    all_candidates: List[SearchCandidate] = []

    dataflows_to_try = []
    if dataflow in ("auto", "weight_stationary"):
        dataflows_to_try.append(("weight_stationary", M, K, N))
    if dataflow in ("auto", "input_stationary") and M != N:
        dataflows_to_try.append(("input_stationary", N, K, M))

    for df, sM, sK, sN in dataflows_to_try:
        tiles = enumerate_tiling_choices(sM, sK, sN, cfg)

        if jobs <= 1 or len(tiles) <= 1:
            for tile in tiles:
                all_candidates.extend(_search_tile_candidates_new(
                    M, K, N, tile, cfg, df, (sM, sK, sN),
                    exhaustive_layout))
        else:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futures = [
                    ex.submit(_search_tile_candidates_new,
                              M, K, N, tile, cfg, df, (sM, sK, sN),
                              exhaustive_layout)
                    for tile in tiles
                ]
                for fut in as_completed(futures):
                    all_candidates.extend(fut.result())

    if not all_candidates:
        raise RuntimeError(
            f"No legal candidate found for GEMM M={M}, K={K}, N={N}, "
            f"AH={cfg.AH}, AW={cfg.AW}")

    all_candidates.sort(key=lambda c: (c.cycles_total, -c.utilization, c.inst_bytes))
    best = all_candidates[0]

    return GemmSearchResult(best=best, candidates=all_candidates)


# ===================================================================
# Public API: multi_layer_search
# ===================================================================

def multi_layer_search(
    layers: List[LayerSpec],
    cfg: FeatherPlusConfig,
    generate_trace: bool = True,
) -> MultiLayerResult:
    """Select mappings across layers with inter-layer conflict resolution.

    For consecutive layers, tries to match:
        SetOVNLayout^(i) == SetIVNLayout^(i+1)
    """
    n_layers = len(layers)
    if n_layers == 0:
        return MultiLayerResult([], 0, 0, [])

    all_candidates: List[List[LayerCandidate]] = []
    for layer in layers:
        cands = brute_force_layer_search(layer.M, layer.K, layer.N, cfg)
        if not cands:
            raise RuntimeError(
                f"No valid candidates for layer '{layer.name}' "
                f"(M={layer.M}, K={layer.K}, N={layer.N})")
        all_candidates.append(cands)

    if n_layers == 1:
        best = all_candidates[0][0]
        return MultiLayerResult(
            layers=[best],
            total_cycles=best.cycles_total,
            total_inst_bytes=best.inst_bytes,
            inter_layer_matches=[],
        )

    selected: List[LayerCandidate] = [all_candidates[0][0]]
    inter_matches: List[bool] = []

    for i in range(1, n_layers):
        prev = selected[-1]
        best_next = None
        matched = False

        for cand in all_candidates[i]:
            if layouts_match(prev.layout_o, cand.layout_i):
                best_next = cand
                matched = True
                break

        if best_next is None:
            for cand in all_candidates[i]:
                derived_o = derive_ovn_from_ivn(
                    cand.layout_i, cfg, prev.Mt, prev.Nt)
                if derived_o is not None:
                    prev_updated = LayerCandidate(
                        order_w=prev.order_w, order_i=prev.order_i,
                        order_o=derived_o.order_id,
                        Mt=prev.Mt, Kt=prev.Kt, Nt=prev.Nt,
                        cycles_total=prev.cycles_total,
                        inst_bytes=prev.inst_bytes,
                        utilization=prev.utilization,
                        layout_w=prev.layout_w, layout_i=prev.layout_i,
                        layout_o=derived_o,
                        exec_params=prev.exec_params,
                    )
                    selected[-1] = prev_updated
                    best_next = cand
                    matched = True
                    break

        if best_next is None:
            best_next = all_candidates[i][0]
            matched = False

        selected.append(best_next)
        inter_matches.append(matched)

    # Validate inter-layer layout continuity and warn on mismatches
    for i in range(1, len(selected)):
        prev_layer = selected[i - 1]
        next_layer = selected[i]
        prev_name = layers[i - 1].name if hasattr(layers[i - 1], "name") else str(i - 1)
        next_name = layers[i].name if hasattr(layers[i], "name") else str(i)
        errs = check_inter_layer_layout(
            prev_layer.layout_o, next_layer.layout_i,
            layer_i_name=prev_name, layer_next_name=next_name)
        for err in errs:
            warnings.warn(f"Inter-layer layout mismatch (layer {i-1}→{i}): {err}",
                          stacklevel=2)

    total_cycles = sum(c.cycles_total for c in selected)
    total_inst_bytes = sum(c.inst_bytes for c in selected)

    return MultiLayerResult(
        layers=selected,
        total_cycles=total_cycles,
        total_inst_bytes=total_inst_bytes,
        inter_layer_matches=inter_matches,
    )
