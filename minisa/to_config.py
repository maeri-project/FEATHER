#!/usr/bin/env python3
"""ConfigStream, convert_trace_to_config, compute_memory_comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

from .config import TraceStateTracker
from .isa import FeatherPlusParams, MinisaIsaParams, clog2, minisa_bitwidths


@dataclass
class ConfigCycle:
    phase: str
    source_op: str
    source_idx: int
    signals: Dict[str, int]
    total_bits: int


@dataclass
class ConfigStream:
    cycles: List[ConfigCycle]
    total_config_bits: int
    total_config_bytes: int


def convert_trace_to_config(
    trace: List[Dict[str, Any]],
    p: FeatherPlusParams,
    M: int = 0, K: int = 0, N: int = 0,
    Mt: int = 0, Kt: int = 0, Nt: int = 0,
) -> ConfigStream:
    """Convert MINISA ISA trace to per-cycle FEATHER+ configuration stream."""
    cycles: List[ConfigCycle] = []
    BIRRD_pipe = p.BIRRD_TOTAL_STAGE + 1
    st = TraceStateTracker(p.AH, p.AW, M, K, N, Mt, Kt, Nt)

    layout_w_signals = {
        "wvn_order_reg": 3, "wvn_N_L0_reg": clog2(p.AW),
        "wvn_K_L1_reg": clog2(p.D_sta // max(1, p.AH)),
        "wvn_N_L1_reg": clog2(p.D_sta // max(1, p.AH)),
    }
    layout_i_signals = {
        "ivn_order_reg": 3, "ivn_M_L0_reg": clog2(p.AW),
        "ivn_M_L1_reg": clog2(p.D_str // max(1, p.AH)),
        "ivn_J_L1_reg": clog2(p.D_str // max(1, p.AH)),
    }
    layout_o_signals = {
        "ovn_order_reg": 3, "ovn_P_L0_reg": clog2(p.AW),
        "ovn_P_L1_reg": clog2(p.D_str // max(1, p.AH)),
        "ovn_Q_L1_reg": clog2(p.D_str // max(1, p.AH)),
    }
    tile_cfg_signals = {
        "i_iacts_xbar_en": 1, "i_iacts_xbar_cmd": p.XBAR_TOTAL_CMD,
        "i_weights_xbar_en": 1, "i_weights_xbar_cmd": p.XBAR_TOTAL_CMD,
        "pe_output_mask": p.AH * p.AW,
        "pe_weights_to_use": p.LOG2_WEIGHTS_DEPTH * p.AW,
        "i_birrd_cmd": p.BIRRD_TOTAL_CMD,
        "weights_pp_sel": 1, "iacts_pp_sel": 1,
    }
    local_load_signals = {
        "weights_addr": p.STRB_ADDR_WIDTH, "pe_weights_valid": p.AW,
        "pe_weights_pp_sel": p.AW, "pe_sel": p.PE_SEL_WIDTH * p.AW,
    }
    stream_signals = {"iacts_addr": p.STAB_ADDR_WIDTH, "pe_iacts_valid": p.AW}
    dma_stream_load_signals = {"weights_wr_en": 1, "weights_addr": p.STRB_ADDR_WIDTH}
    dma_stationary_load_signals = {"iacts_wr_en": 1, "iacts_addr": p.STAB_ADDR_WIDTH}
    dma_stream_store_signals = {"weights_rd_en": 1, "weights_addr": p.STRB_ADDR_WIDTH}
    dma_stationary_store_signals = {"iacts_rd_en": 1, "iacts_addr": p.STAB_ADDR_WIDTH}
    es_signals = {
        "dataflow_reg": 1,
        "m_0_reg": clog2(p.D_str // max(1, p.AH)),
        "s_m_reg": clog2(p.D_str // max(1, p.AH)),
        "T_reg": clog2(p.D_str // max(1, p.AH)),
        "vn_size_reg": clog2(p.AH),
    }

    def _make_cycle(phase: str, source_op: str, source_idx: int,
                    signals: Dict[str, int]) -> ConfigCycle:
        return ConfigCycle(
            phase=phase, source_op=source_op, source_idx=source_idx,
            signals=dict(signals), total_bits=sum(signals.values()),
        )

    em_inst_idx = -1  # index of the preceding ExecuteMapping
    for inst_idx, inst in enumerate(trace):
        st.update(inst)
        op = inst.get("op", "")

        if op == "SetWVNLayout":
            cycles.append(_make_cycle("store_wvn_layout_config", op, inst_idx, layout_w_signals))

        elif op == "SetIVNLayout":
            cycles.append(_make_cycle("store_ivn_layout_config", op, inst_idx, layout_i_signals))

        elif op == "SetOVNLayout":
            cycles.append(_make_cycle("store_ovn_layout_config", op, inst_idx, layout_o_signals))

        elif op == "ExecuteMapping":
            # EM configures the PE-to-WVN mapping; config expansion is
            # deferred until the paired ES arrives with vn_size and T.
            em_inst_idx = inst_idx

        elif op == "ExecuteStreaming":
            # ES triggers computation.  Emit the full EM+ES config expansion:
            # 1 cycle ES config + 1 cycle tile_config + vn_size² WVN load +
            # T*vn_size streaming + vn_size pipeline fill + BIRRD drain.
            # ES config cycle
            cycles.append(_make_cycle("configure_streaming", op, inst_idx, es_signals))
            # EM config expansion (attributed to the EM instruction)
            vn_size = st.vn_size
            Mt_sub = st.T
            cycles.append(_make_cycle("tile_config", "ExecuteMapping", em_inst_idx, tile_cfg_signals))
            # Weight load to PE: vn_size PEs × vn_size elements = vn_size² cycles
            wvn_load = vn_size * vn_size
            for _ in range(wvn_load):
                cycles.append(_make_cycle("local_load_to_pe", "ExecuteMapping", em_inst_idx, local_load_signals))
            # IVN streaming: Mt_sub IVNs × vn_size cycles each
            streaming_cycles = Mt_sub * vn_size
            for _ in range(streaming_cycles):
                cycles.append(_make_cycle("operand_streaming", "ExecuteMapping", em_inst_idx, stream_signals))
            # Pipeline fill: last IVN propagates through vn_size active PE rows
            for _ in range(vn_size):
                cycles.append(_make_cycle("pipeline_fill", "ExecuteMapping", em_inst_idx, {}))
            for _ in range(BIRRD_pipe):
                cycles.append(_make_cycle("birrd_drain", "ExecuteMapping", em_inst_idx, {}))

        elif op == "Load":
            target = int(inst.get("target", 1))
            if st.last_layout == "I":
                dma_cyc = math.ceil(st.Mt * st.Kt / max(1, p.AH * p.AW)) * p.AH
            else:
                dma_cyc = math.ceil(st.Kt * st.Nt / max(1, p.AH * p.AW)) * p.AH
            dma_cyc = max(1, dma_cyc)
            phase = "dma_stationary_load" if target == 0 else "dma_streaming_load"
            signals = dma_stationary_load_signals if target == 0 else dma_stream_load_signals
            for _ in range(dma_cyc):
                cycles.append(_make_cycle(phase, op, inst_idx, signals))

        elif op == "Store":
            dma_cyc = math.ceil(st.Mt * st.Nt / max(1, p.AH * p.AW)) * p.AH
            dma_cyc = max(1, dma_cyc)
            target = int(inst.get("target", 0))
            phase = "dma_stationary_store" if target == 0 else "dma_streaming_store"
            signals = dma_stationary_store_signals if target == 0 else dma_stream_store_signals
            for _ in range(dma_cyc):
                cycles.append(_make_cycle(phase, op, inst_idx, signals))

    total_bits = sum(c.total_bits for c in cycles)
    return ConfigStream(
        cycles=cycles,
        total_config_bits=total_bits,
        total_config_bytes=math.ceil(total_bits / 8),
    )


@dataclass
class ConfigStreamSummary:
    """Lightweight summary: totals + per-instruction durations (no per-cycle allocation)."""
    total_config_bits: int
    total_config_bytes: int
    total_cycles: int
    per_inst_durations: List[int]   # one entry per trace instruction


def compute_config_stream_summary(
    trace: List[Dict[str, Any]],
    p: FeatherPlusParams,
    M: int = 0, K: int = 0, N: int = 0,
    Mt: int = 0, Kt: int = 0, Nt: int = 0,
) -> ConfigStreamSummary:
    """Compute config stream metrics without materializing per-cycle objects.

    This replaces the heavyweight convert_trace_to_config for evaluation
    purposes, reducing memory from O(total_cycles) to O(len(trace)).
    """
    BIRRD_pipe = p.BIRRD_TOTAL_STAGE + 1
    st = TraceStateTracker(p.AH, p.AW, M, K, N, Mt, Kt, Nt)

    layout_w_bits = sum({
        "wvn_order_reg": 3, "wvn_N_L0_reg": clog2(p.AW),
        "wvn_K_L1_reg": clog2(p.D_sta // max(1, p.AH)),
        "wvn_N_L1_reg": clog2(p.D_sta // max(1, p.AH)),
    }.values())
    layout_i_bits = sum({
        "ivn_order_reg": 3, "ivn_M_L0_reg": clog2(p.AW),
        "ivn_M_L1_reg": clog2(p.D_str // max(1, p.AH)),
        "ivn_J_L1_reg": clog2(p.D_str // max(1, p.AH)),
    }.values())
    layout_o_bits = sum({
        "ovn_order_reg": 3, "ovn_P_L0_reg": clog2(p.AW),
        "ovn_P_L1_reg": clog2(p.D_str // max(1, p.AH)),
        "ovn_Q_L1_reg": clog2(p.D_str // max(1, p.AH)),
    }.values())
    tile_cfg_bits = sum({
        "i_iacts_xbar_en": 1, "i_iacts_xbar_cmd": p.XBAR_TOTAL_CMD,
        "i_weights_xbar_en": 1, "i_weights_xbar_cmd": p.XBAR_TOTAL_CMD,
        "pe_output_mask": p.AH * p.AW,
        "pe_weights_to_use": p.LOG2_WEIGHTS_DEPTH * p.AW,
        "i_birrd_cmd": p.BIRRD_TOTAL_CMD,
        "weights_pp_sel": 1, "iacts_pp_sel": 1,
    }.values())
    local_load_bits = sum({
        "weights_addr": p.STRB_ADDR_WIDTH, "pe_weights_valid": p.AW,
        "pe_weights_pp_sel": p.AW, "pe_sel": p.PE_SEL_WIDTH * p.AW,
    }.values())
    stream_bits = sum({
        "iacts_addr": p.STAB_ADDR_WIDTH, "pe_iacts_valid": p.AW,
    }.values())
    dma_stream_load_bits = sum({"weights_wr_en": 1, "weights_addr": p.STRB_ADDR_WIDTH}.values())
    dma_stationary_load_bits = sum({"iacts_wr_en": 1, "iacts_addr": p.STAB_ADDR_WIDTH}.values())
    dma_stream_store_bits = sum({"weights_rd_en": 1, "weights_addr": p.STRB_ADDR_WIDTH}.values())
    dma_stationary_store_bits = sum({"iacts_rd_en": 1, "iacts_addr": p.STAB_ADDR_WIDTH}.values())
    es_bits = sum({
        "dataflow_reg": 1,
        "m_0_reg": clog2(p.D_str // max(1, p.AH)),
        "s_m_reg": clog2(p.D_str // max(1, p.AH)),
        "T_reg": clog2(p.D_str // max(1, p.AH)),
        "vn_size_reg": clog2(p.AH),
    }.values())

    total_bits = 0
    total_cycles = 0
    per_inst_durations: List[int] = []

    for inst in trace:
        st.update(inst)
        op = inst.get("op", "")

        if op == "SetWVNLayout":
            total_bits += layout_w_bits
            total_cycles += 1
            per_inst_durations.append(1)

        elif op == "SetIVNLayout":
            total_bits += layout_i_bits
            total_cycles += 1
            per_inst_durations.append(1)

        elif op == "SetOVNLayout":
            total_bits += layout_o_bits
            total_cycles += 1
            per_inst_durations.append(1)

        elif op == "ExecuteMapping":
            # EM configures the PE-to-WVN mapping.  Config expansion is
            # deferred until the paired ES provides vn_size and T.
            # Record EM's duration as the full EM+ES compute duration;
            # the paired ES will record 1 cycle for its own config.
            per_inst_durations.append(0)  # placeholder, updated below

        elif op == "ExecuteStreaming":
            # ES triggers computation.  Compute the full EM+ES pair duration.
            vn_size = st.vn_size
            Mt_sub = st.T
            wvn_load = vn_size * vn_size  # vn_size² weight load to PE
            streaming_cycles = Mt_sub * vn_size
            pipeline_fill = vn_size  # last IVN through vn_size active PE rows
            n_cyc = 1 + wvn_load + streaming_cycles + pipeline_fill + BIRRD_pipe
            bits = (es_bits + tile_cfg_bits + wvn_load * local_load_bits
                    + streaming_cycles * stream_bits)
            # pipeline_fill and birrd_drain cycles have 0 bits
            total_bits += bits + es_bits
            total_cycles += n_cyc + 1  # +1 for ES configure_streaming cycle
            # Store compute duration on the EM entry (for verify_config)
            # and 1 on the ES entry itself (its configure_streaming cycle).
            em_idx = len(per_inst_durations) - 1  # preceding EM
            per_inst_durations[em_idx] = n_cyc
            per_inst_durations.append(1)

        elif op == "Load":
            target = int(inst.get("target", 1))
            if st.last_layout == "I":
                dma_cyc = max(1, math.ceil(st.Mt * st.Kt / max(1, p.AH * p.AW)) * p.AH)
            else:
                dma_cyc = max(1, math.ceil(st.Kt * st.Nt / max(1, p.AH * p.AW)) * p.AH)
            bits = dma_stationary_load_bits if target == 0 else dma_stream_load_bits
            total_bits += dma_cyc * bits
            total_cycles += dma_cyc
            per_inst_durations.append(dma_cyc)

        elif op == "Store":
            dma_cyc = max(1, math.ceil(st.Mt * st.Nt / max(1, p.AH * p.AW)) * p.AH)
            target = int(inst.get("target", 0))
            bits = dma_stationary_store_bits if target == 0 else dma_stream_store_bits
            total_bits += dma_cyc * bits
            total_cycles += dma_cyc
            per_inst_durations.append(dma_cyc)

        else:
            per_inst_durations.append(0)

    return ConfigStreamSummary(
        total_config_bits=total_bits,
        total_config_bytes=math.ceil(total_bits / 8),
        total_cycles=total_cycles,
        per_inst_durations=per_inst_durations,
    )


def compute_memory_comparison(
    trace: List[Dict[str, Any]],
    config_stream,
    p: FeatherPlusParams,
) -> Dict[str, Any]:
    """Compare MINISA ISA program memory vs FEATHER+ config stream memory.

    config_stream can be ConfigStream or ConfigStreamSummary.
    """
    isa_p = MinisaIsaParams(
        AH=p.AH, AW=p.AW,
        D_str=p.D_str, D_sta=p.D_sta,
        D_ob=p.OB_DEPTH,
    )
    isa_bw = minisa_bitwidths(isa_p)
    isa_bits = 0
    for inst in trace:
        op = inst.get("op", "")
        if op in isa_bw:
            isa_bits += sum(isa_bw[op].values())

    num_config_cycles = (len(config_stream.cycles) if hasattr(config_stream, 'cycles')
                         else config_stream.total_cycles)
    return {
        "minisa_bits": isa_bits,
        "minisa_bytes": math.ceil(isa_bits / 8),
        "config_bits": config_stream.total_config_bits,
        "config_bytes": config_stream.total_config_bytes,
        "compression_ratio": config_stream.total_config_bits / max(isa_bits, 1),
        "num_isa_instructions": len(trace),
        "num_config_cycles": num_config_cycles,
    }
