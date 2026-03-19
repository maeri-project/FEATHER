#!/usr/bin/env python3
"""FeatherPlusConfig, TraceBundle, CycleBreakdown, and make_cfg."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def ceil_log2(x: int) -> int:
    if x <= 1:
        return 0
    return int(math.ceil(math.log2(x)))


def execute_mapping_replica_count(G_r: int, G_c: int) -> int:
    """Derive the duplication count encoded by ExecuteMapping.

    The horizontal mapping pattern repeats every ``G_c`` PE columns, while
    ``G_r`` columns share the same WVN row before the row index increments.
    Their ratio therefore gives the number of repeated column-pattern
    replicas that one ExecuteMapping covers in parallel.
    """
    return max(1, int(G_r) // max(1, int(G_c)))


def reduction_vn_sizes(Kt: int, AH: int) -> Tuple[int, ...]:
    """Split a reduction tile into full AH VNs plus one residue VN if needed.

    For K not divisible by AH, the last VN has fewer than AH elements.
    Example: Kt=25, AH=16 → (16, 9)
    """
    if Kt <= 0 or AH <= 0:
        return tuple()
    full_tiles, residue = divmod(int(Kt), int(AH))
    sizes = (int(AH),) * int(full_tiles)
    if residue:
        sizes += (int(residue),)
    return sizes if sizes else (int(min(Kt, AH)),)


# ---------------------------------------------------------------------------
# Canonical VN counts — ground truth derived solely from tile sizes and AH.
#
# Every module that computes or consumes VN counts MUST agree with these
# formulas.  Sanity checks throughout the toolchain assert against them.
#
# Each VN is a vector of AH elements.  The dimension that maps to
# PE rows (K for IVN/WVN, N for OVN) is divided by AH:
#
#   IVN(m, j): AH elements from K for output row m, K-sub-tile j
#              → count = Mt × ceil(Kt / AH)
#
#   WVN(r, c): AH elements from K for K-sub-tile r, output column c
#              → count = Nt × ceil(Kt / AH)
#
#   OVN(p, q): AH output elements for M-index p, N-subgroup q
#              → count = Mt × ceil(Nt / AH)
# ---------------------------------------------------------------------------

def canonical_n_ivn(Mt: int, Kt: int, AH: int) -> int:
    """Canonical IVN count: Mt × ceil(Kt / AH)."""
    return Mt * ceil_div(Kt, AH)


def canonical_n_wvn(Kt: int, Nt: int, AH: int) -> int:
    """Canonical WVN count: Nt × ceil(Kt / AH)."""
    return Nt * ceil_div(Kt, AH)


def canonical_n_ovn(Mt: int, Nt: int, AH: int) -> int:
    """Canonical OVN count: Mt × ceil(Nt / AH)."""
    return Mt * ceil_div(Nt, AH)


def canonical_n_vndp(Mt: int, Kt: int, Nt: int, AH: int) -> int:
    """Canonical VNDP count: Mt × Nt × ceil(Kt / AH).

    Each VNDP is one AH-element partial dot product between one IVN
    and one WVN.  There are Mt output rows × Nt output columns × K_g
    K-groups = Mt × Nt × ceil(Kt/AH) VNDPs per tile.
    """
    return Mt * Nt * ceil_div(Kt, AH)


def canonical_n_vn_groups(Mt: int, Kt: int, Nt: int, AH: int) -> int:
    """Canonical VN Group count: Mt × ceil(Kt / AH) × ceil(Nt / AH).

    Each VN Group = 1 IVN + ≤AH WVNs.  There is one VN Group per
    (output_row, k_group, n_subgroup) = Mt × K_g × n_col_types.
    """
    return Mt * ceil_div(Kt, AH) * ceil_div(Nt, AH)


class TraceStateTracker:
    """Derive tile sizes and loop counters from ISA-only trace instructions.

    Consumers create one instance per trace pass.  Call ``update(inst)``
    for every instruction in order; the tracker maintains:

    * Effective tile sizes (Mt, Kt, Nt) from nominal sizes + loop counters.
    * Loop counters (m0, n0, k0) advanced on SetOVNLayout / SetIVNLayout.
    * Ping-pong indices (pp, out_pp) for double-buffering.
    * ExecuteMapping fields (r_0, G_r, G_c, c_0, s_r, s_c) from last EM.
    * ExecuteStreaming state (vn_size, T = Mt_sub).
    * last_layout flag ("I" or "W") to classify Load instructions.

    Trace order is ExecuteMapping → ExecuteStreaming.  EM loads WVNs into
    the NEST; ES streams IVNs and triggers computation.  Consumers should
    act on ``op == "ExecuteStreaming"`` when the (EM, ES) pair is complete.
    """

    def __init__(self, AH: int, AW: int,
                 M: int, K: int, N: int,
                 Mt_nom: int, Kt_nom: int, Nt_nom: int):
        self.AH, self.AW = AH, AW
        self.M, self.K, self.N = M, K, N
        self.Mt_nom, self.Kt_nom, self.Nt_nom = Mt_nom, Kt_nom, Nt_nom

        # Effective tile sizes (set by update)
        self.Mt: int = 0
        self.Nt: int = 0
        self.Kt: int = 0

        # From ExecuteMapping (cached until paired ES arrives)
        self.em_r_0: int = 0
        self.em_c_0: int = 0
        self.em_G_r: int = AW
        self.em_G_c: int = 1
        self.em_s_r: int = 1
        self.em_s_c: int = 0

        # From ExecuteStreaming
        self.vn_size: int = AH
        self.T: int = 0          # streaming steps = Mt_sub

        # Loop counters
        self.m0: int = 0
        self.n0: int = 0
        self.k0: int = 0

        # Internal state
        self._k_step: int = 0
        self._out_tile_idx: int = 0
        self._first_out: bool = True
        self.last_layout: str = ""   # "I" or "W"

    @property
    def pp(self) -> int:
        """Ping-pong index for the current K-step."""
        return max(0, self._k_step - 1) & 1

    @property
    def out_pp(self) -> int:
        """Output ping-pong index for the current output tile."""
        return self._out_tile_idx & 1

    def update(self, inst) -> None:
        """Process one trace instruction and advance state."""
        op = inst.get("op", "")

        if op == "SetOVNLayout":
            if not self._first_out:
                self._out_tile_idx += 1
                self.m0 += self.Mt_nom
                if self.m0 >= self.M:
                    self.m0 = 0
                    self.n0 += self.Nt_nom
            else:
                self._first_out = False
            self.k0 = 0
            self._k_step = 0
            self.Mt = min(self.Mt_nom, self.M - self.m0)
            self.Nt = min(self.Nt_nom, self.N - self.n0)

        elif op == "SetIVNLayout":
            if self._k_step > 0:
                self.k0 += self.Kt_nom
            self._k_step += 1
            self.Kt = min(self.Kt_nom, self.K - self.k0)
            self.last_layout = "I"

        elif op == "SetWVNLayout":
            self.last_layout = "W"

        elif op == "ExecuteMapping":
            self.em_r_0 = int(inst.get("r_0", 0))
            self.em_c_0 = int(inst.get("c_0", 0))
            self.em_G_r = int(inst.get("G_r", self.AW))
            self.em_G_c = int(inst.get("G_c", 1))
            self.em_s_r = int(inst.get("s_r", 1))
            self.em_s_c = int(inst.get("s_c", 0))

        elif op == "ExecuteStreaming":
            self.vn_size = int(inst.get("vn_size", self.AH - 1)) + 1
            self.T = int(inst.get("T", 0))


@dataclass
class FeatherPlusConfig:
    """Parametric FEATHER+ configuration used by the model.

    Conventions:
      * AH = array height (PE column depth), AW = array width (number of columns).
        AH and AW may differ (rectangular PE array).
      * Input/weight elements are 1 byte each; output elements are 4 bytes each.
      * bw_load_in / bw_load_w / bw_store_out / bw_onchip_move are bytes per cycle.
      * Instructions are fetched from off-chip into an on-chip instruction buffer
        of size inst_buf_mb.
    """
    ah: int
    aw: int
    total_sram_mb: float

    inst_buf_mb: float = 0.0

    frac_stream: float = 0.4
    frac_stationary: float = 0.4
    frac_output: float = 0.2

    in_bytes: int = 1
    w_bytes: int = 1
    out_bytes: int = 4

    bw_load_in: int = 256
    bw_load_w: int = 256
    bw_store_out: int = 1024
    bw_onchip_move: int = 1024

    inst_bytes_fallback: int = 16
    inst_bits_store_out: int = 64
    bw_inst_bytes: int = 64

    nest_fill: int = 0
    nest_drain: int = 0
    birrd_drain: int = 0
    pipeline_fill: int = 0

    xbar_latency: int = 0
    autopick_latency: int = 1
    is_birrd_plus: bool = True

    k_distribution: bool = True
    pe_flush_latency: int = 3
    weights_depth: int = 0  # 0 = auto-set to AH in __post_init__

    buf_turnaround: int = 1   # cycles between buffer write and read

    macs_per_pe_per_cycle: int = 1
    freq_ghz: float = 1.0

    @property
    def AH(self) -> int:
        return self.ah

    @property
    def AW(self) -> int:
        return self.aw

    def __post_init__(self):
        s = self.frac_stream + self.frac_stationary + self.frac_output
        if abs(s - 1.0) > 1e-6:
            raise ValueError("frac_stream + frac_stationary + frac_output must sum to 1.0")
        if self.ah <= 0:
            raise ValueError("ah must be positive")
        if self.aw <= 0:
            raise ValueError("aw must be positive")
        if self.bw_load_in <= 0 or self.bw_load_w <= 0 or self.bw_store_out <= 0 or self.bw_onchip_move <= 0:
            raise ValueError("bandwidths must be positive")

        if self.nest_fill == 0:
            self.nest_fill = self.AH - 1
        if self.nest_drain == 0:
            self.nest_drain = self.pe_flush_latency if self.k_distribution else self.AH
        if self.birrd_drain == 0 and self.AW >= 2:
            level = ceil_log2(self.AW)
            self.birrd_drain = (2 * level - 1) if self.AW == 4 else (2 * level)
        if self.xbar_latency == 0:
            self.xbar_latency = 2 if ceil_log2(self.AW) <= 4 else 3
        if self.weights_depth == 0:
            self.weights_depth = self.AH

    def wvn_load_cycles_for_vn(self, vn_size: int) -> int:
        """Cycles to load one VN-sized WVN into active PE rows.

        For a VN of size `vn_size`, only vn_size PEs are active,
        each needing vn_size elements. Total: vn_size² cycles.
        """
        active_rows = max(1, min(int(vn_size), self.AH))
        return active_rows * active_rows

    @property
    def wvn_load_cycles(self) -> int:
        """Cycles to load a full-height WVN into PE registers (AH²)."""
        return self.wvn_load_cycles_for_vn(self.AH)

    @property
    def total_sram_bytes(self) -> int:
        return int(round(self.total_sram_mb * 1024 * 1024))

    @property
    def inst_buf_bytes(self) -> int:
        return int(round(self.inst_buf_mb * 1024 * 1024))

    @property
    def stream_bytes(self) -> int:
        return int(self.total_sram_bytes * self.frac_stream)

    @property
    def stationary_bytes(self) -> int:
        return int(self.total_sram_bytes * self.frac_stationary)

    @property
    def output_bytes(self) -> int:
        return int(self.total_sram_bytes * self.frac_output)

    def vn_bytes_in(self) -> int:
        return int(self.AH * self.in_bytes)

    def vn_bytes_w(self) -> int:
        return int(self.AH * self.w_bytes)

    def vn_bytes_out(self) -> int:
        return int(self.AH * self.out_bytes)

    def cap_stream_vn(self) -> int:
        return max(0, self.stream_bytes // max(1, self.vn_bytes_in()))

    def cap_stationary_vn(self) -> int:
        return max(0, self.stationary_bytes // max(1, self.vn_bytes_w()))

    def cap_output_vn(self) -> int:
        return max(0, self.output_bytes // max(1, self.vn_bytes_out()))

    def peak_macs_per_cycle(self) -> int:
        return int(self.AH * self.AW * self.macs_per_pe_per_cycle)

    def effective_peak_macs_per_cycle(self) -> int:
        return self.peak_macs_per_cycle()

    def minisa_inst_bits(self, op: str) -> int:
        """ISA 2.0 instruction bit widths.

        Buffer assignment:
          SetWVNLayout → stationary buffer
          SetIVNLayout → streaming buffer
          ExecuteMapping → stationary buffer for addressing
          ExecuteStreaming → streaming buffer for addressing
        """
        from .isa import config_to_isa_params, minisa_bitwidths as _bw
        isa_p = config_to_isa_params(self)
        field_bits = _bw(isa_p).get(op)
        if field_bits is None:
            return int(self.inst_bytes_fallback * 8)
        return int(sum(int(b) for b in field_bits.values()))

    def minisa_inst_bytes(self, op: str) -> int:
        bits = int(self.minisa_inst_bits(op))
        return int((bits + 7) // 8)

    def minisa_trace_inst_bytes(self, trace: Sequence[Dict[str, Any]]) -> int:
        total = 0
        for inst in trace:
            op = str(inst.get("op", ""))
            total += int(self.minisa_inst_bytes(op))
        return int(total)


@dataclass
class TraceBundle:
    cfg: FeatherPlusConfig
    M: int
    K: int
    N: int
    chunk_strategy: Dict[str, int]
    order_ids: Dict[str, int]
    trace: List[Dict[str, Any]]


@dataclass
class CycleBreakdown:
    total: int = 0
    load_in: int = 0
    load_w: int = 0
    load_inst: int = 0

    inst_prefetch: int = 0
    inst_stall: int = 0

    compute: int = 0
    out_to_stream: int = 0
    store_out: int = 0

    bytes_in: int = 0
    bytes_w: int = 0
    bytes_out_store: int = 0
    bytes_out_move: int = 0


def make_cfg(ah: int, aw: int,
             sram_mb_map: Dict[int, float],
             instbuf_mb_map: Dict[int, float],
             alloc: Tuple[float, float, float],
             freq_ghz: float) -> FeatherPlusConfig:
    """Construct a scaled FEATHER+ config.

    Args:
        ah: Array height (PE column depth).
        aw: Array width (number of PE columns).
        sram_mb_map: Dict mapping AH values to total SRAM in MB.
        instbuf_mb_map: Dict mapping AH values to instruction buffer in MB.
        alloc: (frac_stream, frac_stationary, frac_output) SRAM allocation.
        freq_ghz: Clock frequency in GHz.
    """
    ah = int(ah)
    aw = int(aw)
    if ah not in sram_mb_map:
        raise KeyError(f"No SRAM entry for AH={ah}. Provide --sram-map with {ah}:<MB>.")
    if ah not in instbuf_mb_map:
        raise KeyError(f"No inst-buffer entry for AH={ah}. Provide --instbuf-map with {ah}:<MB>.")

    # SRAM and inst-buffer are keyed by AH only (fixed per AH group).
    # AW=4,16,64 all share the same 4 MB when AH=4, etc.
    base_sram_mb = float(sram_mb_map[ah])
    base_inst_mb = float(instbuf_mb_map[ah])
    sram_mb = base_sram_mb
    inst_mb = base_inst_mb
    fs, fsta, fout = alloc

    bw_load_in = aw
    bw_load_w = aw
    bw_store_out = 4 * aw
    bw_onchip_move = 4 * aw

    return FeatherPlusConfig(
        ah=ah,
        aw=aw,
        total_sram_mb=sram_mb,
        inst_buf_mb=inst_mb,
        frac_stream=fs,
        frac_stationary=fsta,
        frac_output=fout,
        bw_load_in=bw_load_in,
        bw_load_w=bw_load_w,
        bw_store_out=bw_store_out,
        bw_onchip_move=bw_onchip_move,
        freq_ghz=freq_ghz,
        is_birrd_plus=True,
        k_distribution=True,
        pe_flush_latency=3,
        weights_depth=ah,  # weight register depth = AH
    )
