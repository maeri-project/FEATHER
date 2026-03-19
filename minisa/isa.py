#!/usr/bin/env python3
"""MINISA ISA 2.0: FeatherPlusParams, MinisaIsaParams, bitwidths, config widths, phase mappers.

MINISA ISA 2.0 (8 instructions, 3-bit opcode):
  000: SetWVNLayout   — configure stationary buffer layout
  001: SetIVNLayout   — configure streaming buffer layout
  010: SetOVNLayout   — configure output buffer layout
  011: ExecuteStreaming — configure operand streaming parameters
  100: Store          — DMA store to off-chip
  101: Load           — DMA load from off-chip
  110: Activation     — activation function (reserved)
  111: ExecuteMapping — configure PE-to-WVN mapping

Buffer assignment:
  SetWVNLayout → Stationary Buffer (weights are stationary)
  SetIVNLayout → Streaming Buffer  (inputs stream through)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

from .config import FeatherPlusConfig


def clog2(n: int) -> int:
    """ceil(log2(n)), minimum 1."""
    return max(1, math.ceil(math.log2(max(int(n), 2))))


# ---- RTL-level hardware parameters (mirrors feather_plus_top.v) ----

def config_to_hw_params(cfg: FeatherPlusConfig) -> FeatherPlusParams:
    """Derive RTL hardware parameters from a FeatherPlusConfig."""
    return FeatherPlusParams(
        AH=cfg.AH,
        AW=cfg.AW,
        WEIGHTS_DEPTH=cfg.weights_depth,
        D_str=cfg.stream_bytes // max(1, cfg.AW * cfg.in_bytes),
        D_sta=cfg.stationary_bytes // max(1, cfg.AW * cfg.w_bytes),
        OB_DEPTH=cfg.output_bytes // max(1, cfg.AW * cfg.out_bytes),
    )


@dataclass
class FeatherPlusParams:
    """Hardware parameters of feather_plus_top (mirrors RTL parameters)."""
    AH:             int = 4
    AW:             int = 4
    IACTS_WIDTH:    int = 8
    WEIGHTS_WIDTH:  int = 8
    PE_OUTPUT_WIDTH: int = 32
    WEIGHTS_DEPTH:  int = 4
    D_str:          int = 64
    D_sta:          int = 64
    OB_DEPTH:       int = 64
    HBM_ADDR_BITS:  int = 29

    @property
    def LOG2_WEIGHTS_DEPTH(self):  return clog2(self.WEIGHTS_DEPTH)
    @property
    def PE_SEL_WIDTH(self):        return clog2(self.AH * self.AW)
    @property
    def STRB_ADDR_WIDTH(self):     return clog2(self.D_str)
    @property
    def STAB_ADDR_WIDTH(self):     return clog2(self.D_sta)
    @property
    def OB_ADDR_WIDTH(self):       return clog2(self.OB_DEPTH)
    @property
    def XBAR_TAG_WIDTH(self):      return clog2(self.AW)
    @property
    def XBAR_TOTAL_CMD(self):      return self.XBAR_TAG_WIDTH * self.AW
    @property
    def XBAR_LATENCY(self):        return 2 if self.XBAR_TAG_WIDTH <= 4 else 3
    @property
    def BIRRD_LEVEL(self):         return clog2(self.AW)
    @property
    def BIRRD_NUM_SWITCH(self):    return self.AW >> 1
    @property
    def BIRRD_TOTAL_STAGE(self):
        if self.AW == 4:
            return 2 * self.BIRRD_LEVEL - 1
        return 2 * self.BIRRD_LEVEL
    @property
    def BIRRD_CMD_WIDTH_PER_ROW(self): return 2 * self.BIRRD_TOTAL_STAGE
    @property
    def BIRRD_IN_CMD_WIDTH(self):
        return (self.BIRRD_CMD_WIDTH_PER_ROW * self.AW) >> 1
    @property
    def BIRRD_TOTAL_CMD(self):
        return self.BIRRD_NUM_SWITCH * self.BIRRD_IN_CMD_WIDTH
    @property
    def VN_per_buf(self):
        return (self.D_sta * self.AW) // self.AH


# ---- ISA-level parameters (for instruction encoding) ----

@dataclass(frozen=True)
class MinisaIsaParams:
    """ISA-level parameters for MINISA instruction encoding.

    D values are per-bank scalar depths:
      D_str = stream_bytes // (AW * in_bytes)    — streaming buffer per-bank depth
      D_sta = stationary_bytes // (AW * w_bytes)  — stationary buffer per-bank depth
      D_ob  = output_bytes // (AW * out_bytes)    — output buffer per-bank depth

    VN row counts:  D // AH  (number of VN rows per bank)
    Total VN count: (D // AH) * AW  (total VNs in the buffer)
    """
    AH: int
    AW: int
    D_str: int
    D_sta: int
    D_ob: int
    HBM_ADDR_BITS: int = 29

    @property
    def str_vn_rows(self) -> int:
        return max(1, self.D_str // max(1, self.AH))

    @property
    def sta_vn_rows(self) -> int:
        return max(1, self.D_sta // max(1, self.AH))

    @property
    def ob_vn_rows(self) -> int:
        return max(1, self.D_ob // max(1, self.AH))

    @property
    def str_vn_total(self) -> int:
        return int(self.str_vn_rows * self.AW)

    @property
    def sta_vn_total(self) -> int:
        return int(self.sta_vn_rows * self.AW)

    @property
    def ob_vn_total(self) -> int:
        return int(self.ob_vn_rows * self.AW)


def config_to_isa_params(cfg: FeatherPlusConfig) -> MinisaIsaParams:
    """Convert FeatherPlusConfig to per-bank buffer depths used by ISA sizing.

    D = total_buffer_bytes // (AW * element_bytes)
    This is the per-bank scalar depth. No AH in the denominator.
    """
    return MinisaIsaParams(
        AH=cfg.AH,
        AW=cfg.AW,
        D_str=cfg.stream_bytes // max(1, cfg.AW * cfg.in_bytes),
        D_sta=cfg.stationary_bytes // max(1, cfg.AW * cfg.w_bytes),
        D_ob=cfg.output_bytes // max(1, cfg.AW * cfg.out_bytes),
    )


def minisa_opcode_values() -> Dict[str, str]:
    """Return opcode binary strings for all 8 MINISA ISA 2.0 instructions."""
    return {
        "SetWVNLayout": "000",
        "SetIVNLayout": "001",
        "SetOVNLayout": "010",
        "ExecuteStreaming": "011",
        "Store": "100",
        "Load": "101",
        "Activation": "110",
        "ExecuteMapping": "111",
    }


def minisa_bitwidths(p: MinisaIsaParams) -> Dict[str, Dict[str, int]]:
    """Return ISA field widths for MINISA ISA 2.0.

    Buffer assignment (ISA 2.0):
      SetWVNLayout → stationary buffer (sta)
      SetIVNLayout → streaming buffer (str)
      SetOVNLayout → output buffer (ob)
      ExecuteMapping → stationary buffer for r_0/c_0/s_r addressing
      ExecuteStreaming → streaming buffer for m_0/s_m/T addressing
    """
    op = 3
    b_aw = clog2(p.AW)
    b_str_rows = clog2(p.str_vn_rows)
    b_sta_rows = clog2(p.sta_vn_rows)
    b_ob_rows = clog2(p.ob_vn_rows)
    b_sta_total = clog2(p.sta_vn_total)
    b_vn_size = clog2(p.AH)

    return {
        "SetWVNLayout": {
            "opcode": op, "order": 3,
            "N_L0": b_aw, "N_L1": b_sta_rows, "K_L1": b_sta_rows,
        },
        "SetIVNLayout": {
            "opcode": op, "order": 3,
            "M_L0": b_aw, "M_L1": b_str_rows, "J_L1": b_str_rows,
        },
        "SetOVNLayout": {
            "opcode": op, "order": 3,
            "P_L0": b_aw, "P_L1": b_str_rows, "Q_L1": b_str_rows,
        },
        "ExecuteMapping": {
            "opcode": op,
            "G_r": b_aw, "G_c": b_aw,
            "r_0": b_sta_total, "c_0": b_sta_total,
            "s_r": b_sta_total, "s_c": b_sta_rows,
        },
        "ExecuteStreaming": {
            "opcode": op,
            "dataflow": 1,
            "m_0": b_str_rows, "s_m": b_str_rows,
            "T": b_str_rows, "vn_size": b_vn_size,
        },
        "Load": {"opcode": op, "target": 1, "hbm_addr": p.HBM_ADDR_BITS},
        "Store": {"opcode": op, "target": 1, "hbm_addr": p.HBM_ADDR_BITS},
        "Activation": {"opcode": op, "tbd": 8},
    }


def minisa_value_ranges(p: MinisaIsaParams) -> Dict[str, Dict[str, Any]]:
    """Return human-readable ISA value ranges for documentation."""
    return {
        "SetWVNLayout": {
            "opcode": "000", "order": [0, 5],
            "N_L0": [1, p.AW], "N_L1": [1, p.sta_vn_rows],
            "K_L1": [1, p.sta_vn_rows],
        },
        "SetIVNLayout": {
            "opcode": "001", "order": [0, 5],
            "M_L0": [1, p.AW], "M_L1": [1, p.str_vn_rows],
            "J_L1": [1, p.str_vn_rows],
        },
        "SetOVNLayout": {
            "opcode": "010", "order": [0, 5],
            "P_L0": [1, p.AW], "P_L1": [1, p.str_vn_rows],
            "Q_L1": [1, p.str_vn_rows],
        },
        "ExecuteStreaming": {
            "opcode": "011",
            "dataflow": {"IO-S": 0, "WO-S": 1},
            "m_0": [0, p.str_vn_rows - 1],
            "s_m": [0, p.str_vn_rows - 1],
            "T": [0, p.str_vn_rows],
            "vn_size": [0, p.AH - 1],
            "actual_vn_size": [1, p.AH],
        },
        "ExecuteMapping": {
            "opcode": "111",
            "G_r": [1, p.AW], "G_c": [1, p.AW],
            "r_0": [0, p.sta_vn_total], "c_0": [0, p.sta_vn_total],
            "s_r": [0, p.sta_vn_total], "s_c": [0, p.sta_vn_rows],
        },
        "Load": {
            "opcode": "101", "target": {"stationary_buffer": 0, "streaming_buffer": 1},
            "hbm_addr": f"[0, 2^{p.HBM_ADDR_BITS} - 1]",
        },
        "Store": {
            "opcode": "100", "target": {"stationary_buffer": 0, "streaming_buffer": 1},
            "hbm_addr": f"[0, 2^{p.HBM_ADDR_BITS} - 1]",
        },
        "Activation": {"opcode": "110", "tbd": 8},
    }


# ---- Legacy bitwidths API (accepts FeatherPlusParams) ----

def minisa_bitwidths_legacy(p: FeatherPlusParams) -> Dict[str, Dict[str, int]]:
    """Legacy bitwidths from FeatherPlusParams (for backward compatibility)."""
    isa_p = MinisaIsaParams(
        AH=p.AH, AW=p.AW,
        D_str=p.D_str, D_sta=p.D_sta, D_ob=p.OB_DEPTH,
        HBM_ADDR_BITS=p.HBM_ADDR_BITS,
    )
    return minisa_bitwidths(isa_p)


def hw_config_widths(p: FeatherPlusParams) -> Dict[str, Dict[str, int]]:
    """Return {group: {signal: bits}} for every feather_plus_top port."""
    return {
        "streaming_buffer": {
            "weights_wr_en": 1, "weights_addr": p.STRB_ADDR_WIDTH, "weights_pp_sel": 1,
        },
        "stationary_buffer": {
            "iacts_wr_en": 1, "iacts_addr": p.STAB_ADDR_WIDTH, "iacts_pp_sel": 1,
        },
        "crossbar_iacts": {"iacts_xbar_en": 1, "iacts_xbar_cmd": p.XBAR_TOTAL_CMD},
        "crossbar_weights": {"weights_xbar_en": 1, "weights_xbar_cmd": p.XBAR_TOTAL_CMD},
        "pe_control": {
            "pe_iacts_valid": p.AW, "pe_weights_valid": p.AW, "pe_weights_pp_sel": p.AW,
            "pe_sel": p.PE_SEL_WIDTH * p.AW,
            "pe_weights_to_use": p.LOG2_WEIGHTS_DEPTH * p.AW,
            "pe_output_mask": p.AH * p.AW,
        },
        "birrd": {"birrd_cmd": p.BIRRD_TOTAL_CMD},
        "output_buffer": {
            "ob_wr_en": p.AW, "ob_addr_wr": p.OB_ADDR_WIDTH * p.AW,
            "ob_addr_rd": p.OB_ADDR_WIDTH * p.AW, "ob_rd_en": p.AW,
        },
        "zero_point": {
            "iacts_zp": p.IACTS_WIDTH, "iacts_zp_valid": 1,
            "weights_zp": p.WEIGHTS_WIDTH, "weights_zp_valid": 1,
        },
        "quantization": {"o_iacts_scale": p.PE_OUTPUT_WIDTH, "o_iacts_zp": p.IACTS_WIDTH},
    }


# ---- Phase mappers (ISA instruction -> config phases) ----

@dataclass
class ConfigPhase:
    name: str
    cycles: str
    signals: Dict[str, int]
    once: bool = False

    @property
    def bits_per_cycle(self):
        return sum(self.signals.values())

    def total_bits_sym(self):
        return f"{self.bits_per_cycle} * {self.cycles}"


def map_setwvnlayout(p: FeatherPlusParams) -> List[ConfigPhase]:
    return [ConfigPhase(
        name="store_wvn_layout_config", cycles="1",
        signals={
            "wvn_order_reg": 3, "wvn_N_L0_reg": clog2(p.AW),
            "wvn_K_L1_reg": clog2(p.D_sta // max(1, p.AH)),
            "wvn_N_L1_reg": clog2(p.D_sta // max(1, p.AH)),
        }, once=True,
    )]


def map_setivnlayout(p: FeatherPlusParams) -> List[ConfigPhase]:
    return [ConfigPhase(
        name="store_ivn_layout_config", cycles="1",
        signals={
            "ivn_order_reg": 3, "ivn_M_L0_reg": clog2(p.AW),
            "ivn_M_L1_reg": clog2(p.D_str // max(1, p.AH)),
            "ivn_J_L1_reg": clog2(p.D_str // max(1, p.AH)),
        }, once=True,
    )]


def map_setovnlayout(p: FeatherPlusParams) -> List[ConfigPhase]:
    return [ConfigPhase(
        name="store_ovn_layout_config", cycles="1",
        signals={
            "ovn_order_reg": 3, "ovn_P_L0_reg": clog2(p.AW),
            "ovn_P_L1_reg": clog2(p.D_str // max(1, p.AH)),
            "ovn_Q_L1_reg": clog2(p.D_str // max(1, p.AH)),
        }, once=True,
    )]


def map_executemapping(p: FeatherPlusParams) -> List[ConfigPhase]:
    return [
        ConfigPhase(
            name="tile_config", cycles="1",
            signals={
                "i_iacts_xbar_en": 1, "i_iacts_xbar_cmd": p.XBAR_TOTAL_CMD,
                "i_weights_xbar_en": 1, "i_weights_xbar_cmd": p.XBAR_TOTAL_CMD,
                "pe_output_mask": p.AH * p.AW,
                "pe_weights_to_use": p.LOG2_WEIGHTS_DEPTH * p.AW,
                "i_birrd_cmd": p.BIRRD_TOTAL_CMD,
                "weights_pp_sel": 1, "iacts_pp_sel": 1,
            }, once=True,
        ),
        ConfigPhase(
            name="weight_load_to_pe", cycles=f"WEIGHTS_DEPTH={p.WEIGHTS_DEPTH}",
            signals={
                "weights_addr": p.STRB_ADDR_WIDTH, "pe_weights_valid": p.AW,
                "pe_weights_pp_sel": p.AW, "pe_sel": p.PE_SEL_WIDTH * p.AW,
            },
        ),
        ConfigPhase(
            name="iacts_streaming", cycles="M * WEIGHTS_DEPTH",
            signals={"iacts_addr": p.STAB_ADDR_WIDTH, "pe_iacts_valid": p.AW},
        ),
        ConfigPhase(
            name="birrd_drain", cycles=f"BIRRD_pipe={p.BIRRD_TOTAL_STAGE + 1}",
            signals={},
        ),
    ]


def map_executestreaming(p: FeatherPlusParams) -> List[ConfigPhase]:
    """Phase mapper for ExecuteStreaming (ISA 2.0)."""
    return [ConfigPhase(
        name="configure_streaming", cycles="1",
        signals={
            "dataflow_reg": 1,
            "m_0_reg": clog2(p.D_str // max(1, p.AH)),
            "s_m_reg": clog2(p.D_str // max(1, p.AH)),
            "T_reg": clog2(p.D_str // max(1, p.AH)),
            "vn_size_reg": clog2(p.AH),
        }, once=True,
    )]


def map_store(p: FeatherPlusParams) -> List[ConfigPhase]:
    return [ConfigPhase(
        name="dma_store", cycles="N_VN * AH",
        signals={"target": 1, "hbm_addr": p.HBM_ADDR_BITS},
    )]


def map_load(p: FeatherPlusParams) -> List[ConfigPhase]:
    return [ConfigPhase(
        name="dma_load", cycles="N_VN * AH",
        signals={"target": 1, "hbm_addr": p.HBM_ADDR_BITS},
    )]


def map_swap(p: FeatherPlusParams) -> List[ConfigPhase]:
    return [ConfigPhase(
        name="set_dataflow", cycles="1",
        signals={"dataflow": 1}, once=True,
    )]


def map_activation(p: FeatherPlusParams) -> List[ConfigPhase]:
    return [ConfigPhase(
        name="activation", cycles="TBD",
        signals={},
    )]
