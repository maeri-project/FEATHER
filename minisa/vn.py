#!/usr/bin/env python3
"""Virtual Neuron (VN) abstraction for FEATHER+ MINISA compilation.

Implements the 6-step lowering procedure:
  Step 1: Convert workloads (GEMM / Conv) into logical VNs
  Step 2: Divide logical VNs into hardware-sized VN tiles (length AH)
  Step 3: Assemble VN tiles into VN Groups for the NEST

After Step 3, convolution and matrix multiplication are indistinguishable
to the backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    FeatherPlusConfig, ceil_div,
    canonical_n_ivn, canonical_n_wvn, canonical_n_ovn,
    canonical_n_vndp, canonical_n_vn_groups,
)
from .layout import choose_layout_W, choose_layout_I, choose_layout_O


# ---------------------------------------------------------------------------
# Workload descriptors
# ---------------------------------------------------------------------------

@dataclass
class GEMMWorkload:
    """Matrix multiplication: C[M,N] = A[M,K] * B[K,N]."""
    M: int
    K: int
    N: int

    @property
    def output_elements(self) -> int:
        return self.M * self.N

    @property
    def reduction_length(self) -> int:
        return self.K

    @property
    def total_macs(self) -> int:
        return self.M * self.K * self.N


@dataclass
class ConvWorkload:
    """Convolution: Y[n,c_o,h_o,w_o] = sum_{c_i,r,s} X[n,c_i,h_o+r,w_o+s] * W[c_o,c_i,r,s].

    Parameters:
        batch:  N  (batch size)
        C_i:    input channels
        H, W:   input spatial dimensions
        C_o:    output channels
        R, S:   filter spatial dimensions
        stride_h, stride_w: convolution stride
        pad_h, pad_w: padding
    """
    batch: int
    C_i: int
    H: int
    W: int
    C_o: int
    R: int
    S: int
    stride_h: int = 1
    stride_w: int = 1
    pad_h: int = 0
    pad_w: int = 0

    @property
    def H_o(self) -> int:
        return (self.H + 2 * self.pad_h - self.R) // self.stride_h + 1

    @property
    def W_o(self) -> int:
        return (self.W + 2 * self.pad_w - self.S) // self.stride_w + 1

    @property
    def K_prime(self) -> int:
        """Flattened reduction dimension: C_i * R * S."""
        return self.C_i * self.R * self.S

    @property
    def output_elements(self) -> int:
        return self.batch * self.C_o * self.H_o * self.W_o

    @property
    def total_macs(self) -> int:
        return self.output_elements * self.K_prime

    def to_gemm(self) -> GEMMWorkload:
        """Convert to equivalent GEMM via im2col.

        M = batch * H_o * W_o   (output spatial positions)
        K = C_i * R * S         (flattened reduction)
        N = C_o                 (output channels)

        From this point onward, convolution and matrix multiplication
        are indistinguishable to the backend.
        """
        M = self.batch * self.H_o * self.W_o
        K = self.K_prime
        N = self.C_o
        return GEMMWorkload(M=M, K=K, N=N)


# ---------------------------------------------------------------------------
# Step 1: Logical VN (one output element = one complete dot product)
# ---------------------------------------------------------------------------

@dataclass
class LogicalVN:
    """One output element represented as a maximal dot product.

    For GEMM C[m,n] = sum_k A[m,k]*B[k,n]:
        output_coords = (m, n)
        reduction_length = K

    For Conv Y[n,c_o,h_o,w_o] = sum_{c_i,r,s} X*W:
        output_coords = (n, c_o, h_o, w_o)
        reduction_length = K' = C_i * R * S
    """
    output_coords: Tuple[int, ...]
    reduction_length: int


def workload_to_logical_vns(workload) -> List[LogicalVN]:
    """Step 1: Convert workload into logical VNs.

    Each output element becomes one logical VN with the full reduction
    length. This representation is used conceptually; in practice we
    operate on aggregate dimensions (M, K, N) rather than enumerating
    all M*N individual VNs.

    Returns a summary list (one per output row for GEMM, since VNs in
    the same row share the reduction dimension).
    """
    if isinstance(workload, ConvWorkload):
        gemm = workload.to_gemm()
    elif isinstance(workload, GEMMWorkload):
        gemm = workload
    else:
        raise TypeError(f"Unknown workload type: {type(workload)}")

    # Each row m has N logical VNs, all with reduction length K.
    # We return one representative per row for efficiency.
    vns = []
    for m in range(gemm.M):
        vns.append(LogicalVN(
            output_coords=(m,),
            reduction_length=gemm.K,
        ))
    return vns


# ---------------------------------------------------------------------------
# Step 2: VN Tiles (hardware-sized chunks of length AH)
# ---------------------------------------------------------------------------

@dataclass
class VNTile:
    """One hardware-sized VN tile: AH-element dot product.

    Produced by partitioning a logical VN of reduction length K into
    ceil(K/AH) tiles. Each tile contributes one partial sum.

    Attributes:
        m: row index in the output matrix
        r: reduction tile index (0 .. ceil(K/AH)-1)
        tile_length: actual length (AH, or remainder for last tile)
    """
    m: int
    r: int
    tile_length: int


def tile_logical_vns(workload, cfg: FeatherPlusConfig) -> Dict[str, Any]:
    """Step 2: Partition logical VNs into hardware-sized VN tiles.

    For GEMM with reduction length K and array height AH:
        Each logical VN is split into ceil(K/AH) VN tiles.
        Total VN tiles = M * ceil(K/AH) * N

    For convolution, first converts to equivalent GEMM.

    Returns a summary dict (not individual tiles, for efficiency).
    """
    if isinstance(workload, ConvWorkload):
        gemm = workload.to_gemm()
    elif isinstance(workload, GEMMWorkload):
        gemm = workload
    else:
        raise TypeError(f"Unknown workload type: {type(workload)}")

    AH = cfg.AH
    K = gemm.K
    n_reduction_tiles = ceil_div(K, AH)

    # --- Sanity check: VN tile counts match canonical formulas ---
    # At the full-workload level (before outer tiling), the canonical
    # counts describe the total distinct VN addresses.
    M_, K_, N_ = gemm.M, gemm.K, gemm.N
    assert M_ * n_reduction_tiles == canonical_n_ivn(M_, K_, AH), (
        f"tile_logical_vns: IVN count mismatch for M={M_}, K={K_}, AH={AH}")
    assert N_ * n_reduction_tiles == canonical_n_wvn(K_, N_, AH), (
        f"tile_logical_vns: WVN count mismatch for K={K_}, N={N_}, AH={AH}")
    assert M_ * ceil_div(N_, AH) == canonical_n_ovn(M_, N_, AH), (
        f"tile_logical_vns: OVN count mismatch for M={M_}, N={N_}, AH={AH}")

    # VNDP count = M × N × ceil(K/AH): each (m, n, r) triple is one VNDP
    total_vndp = M_ * N_ * n_reduction_tiles
    assert total_vndp == canonical_n_vndp(M_, K_, N_, AH), (
        f"tile_logical_vns: VNDP count mismatch: {total_vndp} != "
        f"canonical {canonical_n_vndp(M_, K_, N_, AH)} for M={M_}, K={K_}, N={N_}, AH={AH}")

    return {
        "M": M_,
        "K": K_,
        "N": N_,
        "AH": AH,
        "n_reduction_tiles": n_reduction_tiles,
        "total_vn_tiles": M_ * n_reduction_tiles * N_,
        "tile_length": AH,
        "last_tile_length": K - (n_reduction_tiles - 1) * AH,
        "ivn_shape": f"I_VN(m, r)[0..{AH - 1}] = A[m, r*{AH} + ell]",
        "wvn_shape": f"W_VN(r, n)[0..{AH - 1}] = B[r*{AH} + ell, n]",
        "pvn_shape": f"P_VN(m, r, n) = sum_ell I_VN(m,r)[ell] * W_VN(r,n)[ell]",
        "ovn_shape": f"O_VN(m, n) = sum_r P_VN(m, r, n)",
    }


# ---------------------------------------------------------------------------
# Step 3: VN Groups (AH VN tiles → one NEST column)
# ---------------------------------------------------------------------------

@dataclass
class VNGroup:
    """A group of AH VN tiles assigned to one NEST column.

    All tiles in a group share the same streamed input operand list
    (I_VN), while each tile has its own weight operand (W_VN) loaded
    into the PE's local registers.

    Attributes:
        column_id: which of the AW columns this group maps to
        m_start: starting M index
        r: reduction tile index
        n_indices: which N outputs this group contributes to
    """
    column_id: int
    m_start: int
    r: int
    n_indices: List[int]


def assemble_vn_groups(
    M: int, K: int, N: int, cfg: FeatherPlusConfig,
) -> Dict[str, Any]:
    """Step 3: Assemble VN tiles into VN Groups for the NEST.

    Groups VN tiles by the shared-input constraint required by FEATHER+'s
    execution semantics: inputs stream from top to bottom of a PE column
    in pipelined order, so only VN tiles sharing the same input operand
    list can be packed into one VN Group (one NEST column).

    For GEMM C[M,N] = A[M,K] × B[K,N]:
      VN tiles with the same (m, r) share input A[m, r*AH..(r+1)*AH].
      Total VN Groups = M × ceil(K/AH) × N.

    One VN Group maps to exactly one column of the AH × AW NEST:
    - AH PEs in the column share one streamed input (I_VN)
    - Each PE holds one W_VN in its local registers
    - The column produces AH partial sums that flow into BIRRD

    One NEST run processes AW VN Groups in parallel.

    This step is tile-size-independent. Step 4 then decides how to
    schedule these VN Groups into NEST runs by choosing tile sizes.

    Returns a summary of the VN grouping.
    """
    AH, AW = cfg.AH, cfg.AW

    n_reduction_tiles = ceil_div(K, AH)
    total_vn_groups = M * n_reduction_tiles * N

    # --- Sanity check: total IVNs = M × n_reduction_tiles ---
    assert canonical_n_ivn(M, K, AH) == M * n_reduction_tiles, (
        f"assemble_vn_groups: IVN count mismatch")
    assert canonical_n_wvn(K, N, AH) == N * n_reduction_tiles, (
        f"assemble_vn_groups: WVN count mismatch")

    # --- VNDP count = M × N × ceil(K/AH) ---
    assert total_vn_groups == canonical_n_vndp(M, K, N, AH), (
        f"assemble_vn_groups: VNDP count mismatch: {total_vn_groups} != "
        f"canonical {canonical_n_vndp(M, K, N, AH)}")

    # --- VN Group count = M × ceil(K/AH) × ceil(N/AH) ---
    expected_vn_groups = canonical_n_vn_groups(M, K, N, AH)
    assert expected_vn_groups == M * n_reduction_tiles * ceil_div(N, AH), (
        f"assemble_vn_groups: VN Group count mismatch: {expected_vn_groups} != "
        f"M×K_g×n_col_types = {M}×{n_reduction_tiles}×{ceil_div(N, AH)}")

    return {
        "AH": AH,
        "AW": AW,
        "M": M,
        "K": K,
        "N": N,
        "n_reduction_tiles": n_reduction_tiles,
        "total_vn_groups": total_vn_groups,
        "vn_groups_per_nest_run": AW,
        "min_nest_runs": ceil_div(total_vn_groups, AW),
        "semantics": (
            f"Each NEST run: {AW} columns process {AW} VN Groups in parallel. "
            f"Each column: {AH} PEs share one I_VN stream, "
            f"each PE holds one W_VN in {AH} registers. "
            f"BIRRD reduces {AH} partial sums per column."
        ),
    }


# ---------------------------------------------------------------------------
# Unified lowering entry point
# ---------------------------------------------------------------------------

def lower_workload_to_vn_summary(
    workload,
    cfg: FeatherPlusConfig,
    Mt: Optional[int] = None,
    Kt: Optional[int] = None,
    Nt: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Steps 1-3 and return a unified summary.

    If tile sizes are not provided, uses default tiling from layout.py.
    """
    if isinstance(workload, ConvWorkload):
        gemm = workload.to_gemm()
        workload_type = "conv"
        conv_info = {
            "original": {
                "batch": workload.batch, "C_i": workload.C_i,
                "H": workload.H, "W": workload.W,
                "C_o": workload.C_o, "R": workload.R, "S": workload.S,
                "stride": (workload.stride_h, workload.stride_w),
                "padding": (workload.pad_h, workload.pad_w),
            },
            "H_o": workload.H_o, "W_o": workload.W_o,
            "K_prime": workload.K_prime,
        }
    elif isinstance(workload, GEMMWorkload):
        gemm = workload
        workload_type = "gemm"
        conv_info = None
    else:
        raise TypeError(f"Unknown workload type: {type(workload)}")

    M, K, N = gemm.M, gemm.K, gemm.N

    if Mt is None or Kt is None or Nt is None:
        Mt, Kt, Nt = choose_tile_sizes(M, K, N, cfg)

    AH = cfg.AH
    step2 = tile_logical_vns(gemm, cfg)
    step3 = assemble_vn_groups(M, K, N, cfg)

    # Step 4 scheduling info (depends on tile sizes chosen by search)
    step4 = {
        "Mt": Mt, "Kt": Kt, "Nt": Nt,
        "n_m_tiles": ceil_div(M, Mt),
        "n_k_tiles": ceil_div(K, Kt),
        "n_n_tiles": ceil_div(N, Nt),
        "m_subtiles_per_m_tile": ceil_div(Mt, AH),
        "total_nest_runs": (ceil_div(M, Mt)
                           * ceil_div(N, Nt)
                           * ceil_div(K, Kt)
                           * ceil_div(Mt, AH)),
    }

    return {
        "workload_type": workload_type,
        "gemm": {"M": M, "K": K, "N": N},
        "conv_info": conv_info,
        "total_macs": gemm.total_macs,
        "step2_vn_tiling": step2,
        "step3_vn_grouping": step3,
        "step4_scheduling": step4,
        "tile_sizes": {"Mt": Mt, "Kt": Kt, "Nt": Nt},
    }


# ---------------------------------------------------------------------------
# Tiling: partition workload into on-chip tiles
# ---------------------------------------------------------------------------

def _divisors_descending(n: int) -> List[int]:
    """Return all divisors of *n* in descending order."""
    divs: List[int] = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.sort(reverse=True)
    return divs


def choose_tile_sizes(M: int, K: int, N: int, cfg: FeatherPlusConfig) -> Tuple[int, int, int]:
    """Compute largest tile sizes that fit on-chip buffers.

    Tile sizes are determined by workload-level constraints:
      - Streaming buffer:   Mt × ⌈Kt/AH⌉ ≤ cap_stream (elements)
      - Stationary buffer:  Nt × ⌈Kt/AH⌉ ≤ cap_stationary (elements)
      - Output buffer:      Mt × Nt × 4 ≤ cap_output (bytes)

    After finding the largest tile that fits, this function also checks
    spatial efficiency: whether ceil(Nt/AH) N-subgroups divide evenly
    into the AW PE columns.  If not, Nt is reduced to d*AH where d is
    the largest divisor of AW such that d*AH ≤ Nt, ensuring all AW
    columns are active (no idle columns from uneven replication).
    """
    AH, AW = cfg.AH, cfg.AW
    cap_str = cfg.cap_stream_vn()
    cap_sta = cfg.cap_stationary_vn()
    cap_out = cfg.cap_output_vn()

    Mt = M
    Kt = K
    # Nt cannot exceed AW*AH: each NEST column handles AH N-outputs,
    # so at most AW columns × AH outputs = AW*AH distinct N per tile.
    Nt = min(N, AW * AH)

    def fits(Mt_: int, Kt_: int, Nt_: int) -> bool:
        lw = choose_layout_W(cfg, Kt_, Nt_)
        li = choose_layout_I(cfg, Mt_, Kt_)
        lo = choose_layout_O(cfg, Mt_, Nt_)
        return lw.vn_count() <= cap_sta and li.vn_count() <= cap_str and lo.vn_count() <= cap_out

    if not fits(Mt, Kt, Nt):
        # Reduce dimensions based on which buffer overflows:
        #   Streaming (IVN):    depends on Mt × ⌈Kt/AH⌉  → reduce Mt or Kt
        #   Stationary (WVN):   depends on Nt × ⌈Kt/AH⌉  → reduce Nt or Kt
        #   Output (OVN):       depends on Mt × Nt         → reduce Mt or Nt
        # Strategy: first try reducing Mt alone (preserves Kt=K to avoid
        # multi-K-tile overhead), then fall back to reducing Kt, then Nt.
        Kt2, Mt2, Nt2 = Kt, Mt, Nt
        # Phase 1: try reducing Mt while keeping Kt=K
        while Mt2 > AH and not fits(Mt2, Kt2, Nt2):
            Mt2 = max(AH, Mt2 // 2)
        # Phase 2: if Mt alone wasn't enough, reduce Kt
        while Kt2 > AH and not fits(Mt2, Kt2, Nt2):
            Kt2 = max(AH, Kt2 // 2)
        # Phase 3: if still doesn't fit, reduce Mt further
        while Mt2 > AH and not fits(Mt2, Kt2, Nt2):
            Mt2 = max(AH, Mt2 // 2)
        # Phase 4: last resort, reduce Nt
        while Nt2 > 1 and not fits(Mt2, Kt2, Nt2):
            Nt2 = max(1, Nt2 // 2)
        if not fits(Mt2, Kt2, Nt2):
            raise RuntimeError(
                f"No tiling fits buffers: M={M}, K={K}, N={N}, AH={cfg.AH}, AW={cfg.AW}, "
                f"caps=({cap_str},{cap_sta},{cap_out})")
        Mt, Kt, Nt = Mt2, Kt2, Nt2

    # --- Spatial efficiency: ensure N-subgroups fill AW columns ---
    # n_col_types = ceil(Nt/AH) is the number of distinct PE-column
    # patterns.  If AW % n_col_types != 0, some PE columns are idle.
    # Fix: reduce Nt to d*AH where d is the largest divisor of AW
    # that still fits on-chip, giving perfect column utilization.
    n_col_types = ceil_div(Nt, AH)
    if n_col_types < AW and AW % n_col_types != 0:
        best_nt = None
        for d in _divisors_descending(AW):
            candidate = d * AH
            if candidate > Nt:
                continue
            if candidate < AH:
                continue
            if fits(Mt, Kt, candidate):
                best_nt = candidate
                break
        if best_nt is not None:
            Nt = best_nt

    return int(Mt), int(Kt), int(Nt)
