#!/usr/bin/env python3
"""
MINISA GUI - Unified Interactive Visualization for FEATHER+ Accelerator

This module consolidates:
- minisa_gui.py: Main GUI with tkinter
- minisa_enhancements.py: Verification and comparison tools
- minisa_animation.py: Animation engine

Key Features:
1) Manually configuring ISAs and their execution order
2) Visualizing NEST (PE Array) with accurate per-cycle pipeline animation
3) Visualizing BIRRD (Reduction Network) configurations
4) Displaying performance metrics under different configurations
5) Verification tests and table comparisons

NEST PE Model (Accurate RTL Implementation):
- Each PE contains AH local weight registers W[0..AH-1] and accumulator PSUM
- Weight-load phase: W[0], W[1], ..., W[AH-1] filled from top→bottom pipeline
- Compute/streaming phase: inputs stream top→bottom, MAC with W[k] in index order
- Output to BIRRD when AH inputs consumed (dot-product complete)
- Top row completes first, round-robin row output to BIRRD

Based on RTL:
- /work/RTL/feather_4x4/feather_pe.v
- /work/RTL/feather_4x4/birrd_simple_cmd_flow_seq.v
- MINISA paper: "MINISA: Minimal Instruction Set Architecture for Next-gen
  Reconfigurable Inference Accelerator"
"""

import sys
import os
import json
import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Sequence
from enum import Enum
import threading
import time
import copy

# Try to import matplotlib for visualizations
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, FancyBboxPatch, Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.animation as animation
    import matplotlib.colors as mcolors
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Using basic visualization.")

# Import from the minisa package
try:
    from .config import FeatherPlusConfig, ceil_div, ceil_log2
    from .cycles import estimate_cycles_for_gemm
    HAS_MINISA = True
    HAS_ACCURATE_MODELS = True
except ImportError:
    HAS_MINISA = False
    HAS_ACCURATE_MODELS = False
    print("Warning: minisa package not fully available. Using built-in definitions.")

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    def ceil_log2(x: int) -> int:
        if x <= 1:
            return 0
        return int(math.ceil(math.log2(x)))


# ============================================================================
# ISA Definitions (from MINISA paper Table II/III)
# ============================================================================

class ISAType(Enum):
    SetIVNLayout = "SetIVNLayout"
    SetWVNLayout = "SetWVNLayout"
    SetOVNLayout = "SetOVNLayout"
    ExecuteMapping = "ExecuteMapping"
    ExecuteStreaming = "ExecuteStreaming"

# Order permutations from Table III in the paper
ORDER_PERMUTATIONS = {
    0: {"W": "kL1 -> nL0 -> nL1", "I": "jL1 -> mL0 -> mL1", "O": "pL1 -> pL0 -> qL1"},
    1: {"W": "kL1 -> nL1 -> nL0", "I": "jL1 -> mL1 -> mL0", "O": "pL1 -> qL1 -> pL0"},
    2: {"W": "nL0 -> kL1 -> nL1", "I": "mL0 -> jL1 -> mL1", "O": "pL0 -> pL1 -> qL1"},
    3: {"W": "nL0 -> nL1 -> kL1", "I": "mL0 -> mL1 -> jL1", "O": "pL0 -> qL1 -> pL1"},
    4: {"W": "nL1 -> kL1 -> nL0", "I": "mL1 -> jL1 -> mL0", "O": "qL1 -> pL1 -> pL0"},
    5: {"W": "nL1 -> nL0 -> kL1", "I": "mL1 -> mL0 -> jL1", "O": "qL1 -> pL0 -> pL1"},
}

@dataclass
class ISAInstruction:
    """Represents a single MINISA instruction with all reconfigurable knobs.

    Based on MINISA paper Table II and specifications:
    - SetIVNLayout: Order (0-5), M_L0 [1,AW], M_L1, J_L1
    - SetWVNLayout: Order (0-5), N_L0 [1,AW], N_L1, K_L1
    - SetOVNLayout: Order (0-5), P_L0 [1,AW], P_L1, Q_L1
    - ExecuteMapping: r0, c0, Gr [1,AH], Gc [1,AW], sr, sc
    """
    isa_type: ISAType
    params: Dict[str, Any] = field(default_factory=dict)

    def get_description(self) -> str:
        if self.isa_type == ISAType.SetIVNLayout:
            order = self.params.get('order', 0)
            m_l0 = self.params.get('M_L0', 1)
            m_l1 = self.params.get('M_L1', 1)
            j_l1 = self.params.get('J_L1', 1)
            return f"SetIVNLayout(Order={order}, M_L0={m_l0}, M_L1={m_l1}, J_L1={j_l1})"
        elif self.isa_type == ISAType.SetWVNLayout:
            order = self.params.get('order', 0)
            n_l0 = self.params.get('N_L0', 1)
            n_l1 = self.params.get('N_L1', 1)
            k_l1 = self.params.get('K_L1', 1)
            return f"SetWVNLayout(Order={order}, N_L0={n_l0}, N_L1={n_l1}, K_L1={k_l1})"
        elif self.isa_type == ISAType.SetOVNLayout:
            order = self.params.get('order', 0)
            p_l0 = self.params.get('P_L0', 1)
            p_l1 = self.params.get('P_L1', 1)
            q_l1 = self.params.get('Q_L1', 1)
            return f"SetOVNLayout(Order={order}, P_L0={p_l0}, P_L1={p_l1}, Q_L1={q_l1})"
        elif self.isa_type == ISAType.ExecuteMapping:
            r0 = self.params.get('r0', 0)
            c0 = self.params.get('c0', 0)
            Gr = self.params.get('Gr', 1)
            Gc = self.params.get('Gc', 1)
            sr = self.params.get('sr', 0)
            sc = self.params.get('sc', 0)
            return f"ExecuteMapping(r0={r0}, c0={c0}, Gr={Gr}, Gc={Gc}, sr={sr}, sc={sc})"
        elif self.isa_type == ISAType.ExecuteStreaming:
            dataflow = self.params.get('dataflow', 1)
            m_0 = self.params.get('m_0', 0)
            s_m = self.params.get('s_m', 1)
            T = self.params.get('T', 1)
            vn_size = self.params.get('vn_size', 0)
            df_str = "IO-S" if dataflow == 0 else "WO-S"
            return f"ExecuteStreaming(dataflow={df_str}, m_0={m_0}, s_m={s_m}, T={T}, vn_size={vn_size})"
        return str(self.isa_type.value)

    def get_short_description(self) -> str:
        """Get a shorter description for listbox display"""
        if self.isa_type == ISAType.SetIVNLayout:
            order = self.params.get('order', 0)
            return f"IVN(O={order}, {ORDER_PERMUTATIONS[order]['I']})"
        elif self.isa_type == ISAType.SetWVNLayout:
            order = self.params.get('order', 0)
            return f"WVN(O={order}, {ORDER_PERMUTATIONS[order]['W']})"
        elif self.isa_type == ISAType.SetOVNLayout:
            order = self.params.get('order', 0)
            return f"OVN(O={order}, {ORDER_PERMUTATIONS[order]['O']})"
        elif self.isa_type == ISAType.ExecuteMapping:
            r0 = self.params.get('r0', 0)
            c0 = self.params.get('c0', 0)
            Gr = self.params.get('Gr', 1)
            Gc = self.params.get('Gc', 1)
            return f"Map(r0={r0}, c0={c0}, Gr={Gr}, Gc={Gc})"
        elif self.isa_type == ISAType.ExecuteStreaming:
            T = self.params.get('T', 1)
            vn_size = self.params.get('vn_size', 0)
            dataflow = self.params.get('dataflow', 1)
            df_str = "IO-S" if dataflow == 0 else "WO-S"
            return f"ES(T={T}, vn={vn_size}, df={df_str})"
        return str(self.isa_type.value)

class ISAConfigValidator:
    """Validates ISA instruction configurations based on MINISA paper constraints.

    Constraint Summary (from MINISA paper):
    - SetIVNLayout: Order (0-5), M_L0 in [1, AW], M_L1*J_L1 <= D_str/M_L0
    - SetWVNLayout: Order (0-5), N_L0 in [1, AW], N_L1*K_L1 <= D_sta/N_L0
    - SetOVNLayout: Order (0-5), P_L0 in [1, AW], P_L1*Q_L1 <= D_str/P_L0
    - ExecuteMapping: r0 < K/AH, c0 < N, Gr in [1, AH], Gc in [1, AW], sr <= K/2, sc <= N/2
    """

    def __init__(self, hw_config: 'HardwareConfig', workload_config: 'WorkloadConfig'):
        self.hw = hw_config
        self.wl = workload_config
        # Calculate buffer depths from overall capacity using the user-specified
        # SRAM size and allocation fractions (matching config.py logic).
        total_sram_bytes = int(hw_config.sram_mb * 1024 * 1024)
        in_bytes_per_vn = hw_config.AH * 1   # 1 byte per input element
        w_bytes_per_vn  = hw_config.AH * 1   # 1 byte per weight element
        out_bytes_per_vn = hw_config.AH * 4   # 4 bytes per output element
        self.D_str = max(1, int(total_sram_bytes * hw_config.frac_stream) // max(1, in_bytes_per_vn))
        self.D_sta = max(1, int(total_sram_bytes * hw_config.frac_stationary) // max(1, w_bytes_per_vn))
        self.D_out = max(1, int(total_sram_bytes * hw_config.frac_output) // max(1, out_bytes_per_vn))

    def validate_instruction(self, instr: ISAInstruction) -> Tuple[bool, List[str]]:
        """Validate an ISA instruction. Returns (is_valid, list_of_errors)."""
        errors = []

        if instr.isa_type == ISAType.SetIVNLayout:
            errors.extend(self._validate_ivn_layout(instr.params))
        elif instr.isa_type == ISAType.SetWVNLayout:
            errors.extend(self._validate_wvn_layout(instr.params))
        elif instr.isa_type == ISAType.SetOVNLayout:
            errors.extend(self._validate_ovn_layout(instr.params))
        elif instr.isa_type == ISAType.ExecuteMapping:
            errors.extend(self._validate_mapping(instr.params))
        elif instr.isa_type == ISAType.ExecuteStreaming:
            errors.extend(self._validate_streaming(instr.params))

        return (len(errors) == 0, errors)

    def _validate_ivn_layout(self, params: Dict[str, Any]) -> List[str]:
        """Validate SetIVNLayout parameters"""
        errors = []
        order = params.get('order', 0)
        m_l0 = params.get('M_L0', 1)
        m_l1 = params.get('M_L1', 1)
        j_l1 = params.get('J_L1', 1)

        # Order must be 0-5 (3-bit, 6 legal permutations)
        if order < 0 or order > 5:
            errors.append(f"SetIVNLayout: Order must be 0-5 (got {order})")

        # M_L0 constrained by array width [1, AW]
        if m_l0 < 1 or m_l0 > self.hw.AW:
            errors.append(f"SetIVNLayout: M_L0 must be in [1, {self.hw.AW}] (got {m_l0})")

        # M_L1 and J_L1 must fit in streaming buffer
        if m_l0 > 0 and m_l1 * j_l1 > self.D_str // m_l0:
            max_product = self.D_str // m_l0
            errors.append(f"SetIVNLayout: M_L1*J_L1 ({m_l1*j_l1}) exceeds buffer capacity ({max_product})")

        # M_L1 range check: up to total rows / M_L0
        max_m_l1 = ceil_div(self.wl.M, m_l0) if m_l0 > 0 else 1
        if m_l1 > max_m_l1:
            errors.append(f"SetIVNLayout: M_L1 ({m_l1}) exceeds max ({max_m_l1})")

        # J_L1 range check: up to total reduction size / AH
        max_j_l1 = ceil_div(self.wl.K, self.hw.AH)
        if j_l1 > max_j_l1:
            errors.append(f"SetIVNLayout: J_L1 ({j_l1}) exceeds max ({max_j_l1})")

        return errors

    def _validate_wvn_layout(self, params: Dict[str, Any]) -> List[str]:
        """Validate SetWVNLayout parameters"""
        errors = []
        order = params.get('order', 0)
        n_l0 = params.get('N_L0', 1)
        n_l1 = params.get('N_L1', 1)
        k_l1 = params.get('K_L1', 1)

        # Order must be 0-5
        if order < 0 or order > 5:
            errors.append(f"SetWVNLayout: Order must be 0-5 (got {order})")

        # N_L0 constrained by array width [1, AW]
        if n_l0 < 1 or n_l0 > self.hw.AW:
            errors.append(f"SetWVNLayout: N_L0 must be in [1, {self.hw.AW}] (got {n_l0})")

        # N_L1 and K_L1 must fit in stationary buffer
        if n_l0 > 0 and n_l1 * k_l1 > self.D_sta // n_l0:
            max_product = self.D_sta // n_l0
            errors.append(f"SetWVNLayout: N_L1*K_L1 ({n_l1*k_l1}) exceeds buffer capacity ({max_product})")

        # N_L1 range check: up to total weight columns / N_L0
        max_n_l1 = ceil_div(self.wl.N, n_l0) if n_l0 > 0 else 1
        if n_l1 > max_n_l1:
            errors.append(f"SetWVNLayout: N_L1 ({n_l1}) exceeds max ({max_n_l1})")

        # K_L1 range check: up to total weight rows / AH
        max_k_l1 = ceil_div(self.wl.K, self.hw.AH)
        if k_l1 > max_k_l1:
            errors.append(f"SetWVNLayout: K_L1 ({k_l1}) exceeds max ({max_k_l1})")

        return errors

    def _validate_ovn_layout(self, params: Dict[str, Any]) -> List[str]:
        """Validate SetOVNLayout parameters"""
        errors = []
        order = params.get('order', 0)
        p_l0 = params.get('P_L0', 1)
        p_l1 = params.get('P_L1', 1)
        q_l1 = params.get('Q_L1', 1)

        # Order must be 0-5
        if order < 0 or order > 5:
            errors.append(f"SetOVNLayout: Order must be 0-5 (got {order})")

        # P_L0 constrained by array width [1, AW]
        if p_l0 < 1 or p_l0 > self.hw.AW:
            errors.append(f"SetOVNLayout: P_L0 must be in [1, {self.hw.AW}] (got {p_l0})")

        # P_L1 and Q_L1 must fit in output buffer
        if p_l0 > 0 and p_l1 * q_l1 > self.D_out // p_l0:
            max_product = self.D_out // p_l0
            errors.append(f"SetOVNLayout: P_L1*Q_L1 ({p_l1*q_l1}) exceeds buffer capacity ({max_product})")

        # P_L1 range check: up to total output rows / P_L0
        max_p_l1 = ceil_div(self.wl.M, p_l0) if p_l0 > 0 else 1
        if p_l1 > max_p_l1:
            errors.append(f"SetOVNLayout: P_L1 ({p_l1}) exceeds max ({max_p_l1})")

        # Q_L1 range check: up to total output columns / AH
        max_q_l1 = ceil_div(self.wl.N, self.hw.AH)
        if q_l1 > max_q_l1:
            errors.append(f"SetOVNLayout: Q_L1 ({q_l1}) exceeds max ({max_q_l1})")

        return errors

    def _validate_mapping(self, params: Dict[str, Any]) -> List[str]:
        """Validate ExecuteMapping parameters"""
        errors = []
        r0 = params.get('r0', 0)
        c0 = params.get('c0', 0)
        Gr = params.get('Gr', 1)
        Gc = params.get('Gc', 1)
        sr = params.get('sr', 0)
        sc = params.get('sc', 0)

        # r0: starting row index, range [0, K/AH)
        max_r0 = ceil_div(self.wl.K, self.hw.AH)
        if r0 < 0 or r0 >= max_r0:
            errors.append(f"ExecuteMapping: r0 must be in [0, {max_r0-1}] (got {r0})")

        # c0: starting column index, range [0, N)
        if c0 < 0 or c0 >= self.wl.N:
            errors.append(f"ExecuteMapping: c0 must be in [0, {self.wl.N-1}] (got {c0})")

        # Gr: multicast replication rows, range [1, AH]
        if Gr < 1 or Gr > self.hw.AH:
            errors.append(f"ExecuteMapping: Gr must be in [1, {self.hw.AH}] (got {Gr})")

        # Gc: group size for column replication, range [1, AW]
        if Gc < 1 or Gc > self.hw.AW:
            errors.append(f"ExecuteMapping: Gc must be in [1, {self.hw.AW}] (got {Gc})")

        # sr: stride for rows, reasonable range
        max_sr = ceil_div(self.wl.K, self.hw.AH) // 2 if self.wl.K > 0 else 0
        if sr < 0 or sr > max(max_sr, self.hw.AH):
            errors.append(f"ExecuteMapping: sr ({sr}) may cause out-of-bounds access")

        # sc: stride for columns, reasonable range
        max_sc = self.wl.N // 2 if self.wl.N > 0 else 0
        if sc < 0 or sc > max(max_sc, self.hw.AW):
            errors.append(f"ExecuteMapping: sc ({sc}) may cause out-of-bounds access")

        return errors

    def _validate_streaming(self, params: Dict[str, Any]) -> List[str]:
        """Validate ExecuteStreaming parameters"""
        errors = []
        dataflow = params.get('dataflow', 1)
        m_0 = params.get('m_0', 0)
        s_m = params.get('s_m', 1)
        T = params.get('T', 1)
        vn_size = params.get('vn_size', 0)

        # dataflow: 0 = IO-S (input-output stationary), 1 = WO-S (weight-output stationary)
        if dataflow not in (0, 1):
            errors.append(f"ExecuteStreaming: dataflow must be 0 (IO-S) or 1 (WO-S) (got {dataflow})")

        # m_0: base streaming row index, must be non-negative
        if m_0 < 0:
            errors.append(f"ExecuteStreaming: m_0 must be >= 0 (got {m_0})")

        # s_m: streaming row stride, must be >= 1
        if s_m < 1:
            errors.append(f"ExecuteStreaming: s_m must be >= 1 (got {s_m})")

        # T: streaming steps, must be >= 1
        if T < 1:
            errors.append(f"ExecuteStreaming: T must be >= 1 (got {T})")

        # vn_size: active VN height, encoded as AH-1 (range [0, AH-1])
        if vn_size < 0 or vn_size >= self.hw.AH:
            errors.append(f"ExecuteStreaming: vn_size must be in [0, {self.hw.AH - 1}] (got {vn_size})")

        return errors

    def validate_trace_sequence(self, instructions: List[ISAInstruction]) -> Tuple[bool, List[str]]:
        """Validate a sequence of ISA instructions for semantic correctness.

        The correct trace order is ExecuteMapping (loads WVN into NEST) followed
        by ExecuteStreaming (streams IVN and triggers compute).  ES must appear
        after its paired EM.
        """
        errors = []

        # Check that layout instructions come before mapping/streaming
        has_ivn = False
        has_wvn = False
        has_ovn = False
        mapping_count = 0
        streaming_count = 0
        last_was_mapping = False  # Track EM->ES pairing

        for i, instr in enumerate(instructions):
            # Validate individual instruction
            is_valid, instr_errors = self.validate_instruction(instr)
            for err in instr_errors:
                errors.append(f"Instruction {i+1}: {err}")

            if instr.isa_type == ISAType.SetIVNLayout:
                has_ivn = True
                last_was_mapping = False
            elif instr.isa_type == ISAType.SetWVNLayout:
                has_wvn = True
                last_was_mapping = False
            elif instr.isa_type == ISAType.SetOVNLayout:
                has_ovn = True
                last_was_mapping = False
            elif instr.isa_type == ISAType.ExecuteMapping:
                mapping_count += 1
                if not (has_ivn and has_wvn and has_ovn):
                    missing = []
                    if not has_ivn:
                        missing.append("SetIVNLayout")
                    if not has_wvn:
                        missing.append("SetWVNLayout")
                    if not has_ovn:
                        missing.append("SetOVNLayout")
                    errors.append(f"Instruction {i+1}: ExecuteMapping issued before: {', '.join(missing)}")
                last_was_mapping = True
            elif instr.isa_type == ISAType.ExecuteStreaming:
                streaming_count += 1
                if not last_was_mapping:
                    errors.append(f"Instruction {i+1}: ExecuteStreaming must follow an ExecuteMapping (EM loads WVN, ES streams IVN)")
                last_was_mapping = False

        if streaming_count == 0 and len(instructions) > 0:
            errors.append("Warning: No ExecuteStreaming instruction in trace (no computation will be triggered)")

        if mapping_count > 0 and streaming_count == 0:
            errors.append("Warning: ExecuteMapping without paired ExecuteStreaming (ES triggers compute, not EM)")

        if mapping_count != streaming_count and mapping_count > 0 and streaming_count > 0:
            errors.append(f"Warning: Mismatched EM/ES count ({mapping_count} EM vs {streaming_count} ES); each EM needs a paired ES")

        return (len(errors) == 0, errors)


class ISAConfigDialog(tk.Toplevel):
    """Dialog for configuring ISA instruction parameters.

    Shows different configuration panels based on ISA type with proper
    constraints and range validation based on hardware and workload config.
    """

    def __init__(self, parent, isa_type: ISAType, hw_config: 'HardwareConfig',
                 workload_config: 'WorkloadConfig', existing_params: Dict[str, Any] = None,
                 title: str = "Configure ISA Instruction"):
        super().__init__(parent)
        self.title(title)
        self.isa_type = isa_type
        self.hw_config = hw_config
        self.workload_config = workload_config
        self.existing_params = existing_params or {}
        self.result = None  # Will be set to params dict if OK is pressed

        # Calculate constraints from overall capacity using user-specified
        # SRAM size and allocation fractions (matching config.py logic).
        total_sram_bytes = int(hw_config.sram_mb * 1024 * 1024)
        in_bytes_per_vn = hw_config.AH * 1
        w_bytes_per_vn  = hw_config.AH * 1
        out_bytes_per_vn = hw_config.AH * 4
        self.D_str = max(1, int(total_sram_bytes * hw_config.frac_stream) // max(1, in_bytes_per_vn))
        self.D_sta = max(1, int(total_sram_bytes * hw_config.frac_stationary) // max(1, w_bytes_per_vn))
        self.D_out = max(1, int(total_sram_bytes * hw_config.frac_output) // max(1, out_bytes_per_vn))

        self.geometry("500x450")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self._create_widgets()
        self._center_window()

    def _center_window(self):
        """Center the dialog on the parent window"""
        self.update_idletasks()
        parent = self.master
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def _create_widgets(self):
        """Create dialog widgets based on ISA type"""
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_text = f"Configure {self.isa_type.value}"
        ttk.Label(main_frame, text=title_text, font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 10))

        # Description based on ISA type
        desc_frame = ttk.LabelFrame(main_frame, text="Description", padding=5)
        desc_frame.pack(fill=tk.X, pady=5)

        desc_text = self._get_isa_description()
        ttk.Label(desc_frame, text=desc_text, wraplength=450, justify=tk.LEFT).pack()

        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.param_vars = {}

        if self.isa_type == ISAType.SetIVNLayout:
            self._create_ivn_layout_params(params_frame)
        elif self.isa_type == ISAType.SetWVNLayout:
            self._create_wvn_layout_params(params_frame)
        elif self.isa_type == ISAType.SetOVNLayout:
            self._create_ovn_layout_params(params_frame)
        elif self.isa_type == ISAType.ExecuteMapping:
            self._create_mapping_params(params_frame)
        elif self.isa_type == ISAType.ExecuteStreaming:
            self._create_streaming_params(params_frame)

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(btn_frame, text="OK", command=self._on_ok, width=10).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel, width=10).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Validate", command=self._on_validate, width=10).pack(side=tk.LEFT, padx=5)

    def _get_isa_description(self) -> str:
        """Get description text for ISA type"""
        if self.isa_type == ISAType.SetIVNLayout:
            return ("Configures how Input Virtual Neurons (IVN) are organized in the streaming buffer.\n"
                    f"Order: 3-bit (0-5) nested-loop permutation\n"
                    f"M_L0: inner non-reduction block size [1, {self.hw_config.AW}]\n"
                    f"M_L1: outer loop extent for rows (up to M/{self.hw_config.AW})\n"
                    f"J_L1: outer loop extent for reduction dimension")
        elif self.isa_type == ISAType.SetWVNLayout:
            return ("Configures the stationary buffer layout for Weight Virtual Neurons (WVN).\n"
                    f"Order: 3-bit (0-5) nested-loop permutation\n"
                    f"N_L0: inner block width [1, {self.hw_config.AW}]\n"
                    f"N_L1: outer loop for weight columns\n"
                    f"K_L1: outer loop for weight rows (up to K/{self.hw_config.AH})")
        elif self.isa_type == ISAType.SetOVNLayout:
            return ("Manages the layout of Output Virtual Neurons (OVN) in the output buffer.\n"
                    f"Order: 3-bit (0-5) nested-loop permutation\n"
                    f"P_L0: inner tile height [1, {self.hw_config.AW}]\n"
                    f"P_L1: outer loop for output rows\n"
                    f"Q_L1: outer loop for output columns")
        elif self.isa_type == ISAType.ExecuteMapping:
            return ("Execution trigger that maps a compute tile to processing elements.\n"
                    f"r0, c0: starting row/column indices for weight VNs\n"
                    f"Gr: multicast replication rows [1, {self.hw_config.AH}]\n"
                    f"Gc: group size for column replication [1, {self.hw_config.AW}]\n"
                    f"sr, sc: stride steps for traversing the weight matrix")
        elif self.isa_type == ISAType.ExecuteStreaming:
            return ("Stream inputs through NEST and trigger compute (paired with ExecuteMapping).\n"
                    f"dataflow: 0=IO-S (input-output stationary), 1=WO-S (weight-output stationary)\n"
                    f"m_0: base streaming row index\n"
                    f"s_m: streaming row stride (interleaved distribution)\n"
                    f"T: number of streaming steps\n"
                    f"vn_size: active VN height [0, {self.hw_config.AH - 1}] (encoded as AH-1)")
        return ""

    def _add_param_row(self, parent, row: int, label: str, key: str,
                       default: int, min_val: int, max_val: int, tooltip: str = ""):
        """Add a parameter row with label, spinbox, and range info"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)

        var = tk.IntVar(value=self.existing_params.get(key, default))
        self.param_vars[key] = var

        spinbox = ttk.Spinbox(parent, from_=min_val, to=max_val, textvariable=var, width=10)
        spinbox.grid(row=row, column=1, padx=10, pady=3)

        range_text = f"[{min_val}, {max_val}]"
        ttk.Label(parent, text=range_text, foreground='gray').grid(row=row, column=2, sticky=tk.W, pady=3)

        if tooltip:
            ttk.Label(parent, text=tooltip, foreground='blue', font=('TkDefaultFont', 8)).grid(
                row=row, column=3, sticky=tk.W, padx=5, pady=3)

    def _create_ivn_layout_params(self, parent):
        """Create SetIVNLayout parameter widgets"""
        AW = self.hw_config.AW
        AH = self.hw_config.AH
        M = self.workload_config.M
        K = self.workload_config.K

        # Order parameter with permutation display
        order_frame = ttk.Frame(parent)
        order_frame.pack(fill=tk.X, pady=5)

        ttk.Label(order_frame, text="Order:").pack(side=tk.LEFT)
        self.param_vars['order'] = tk.IntVar(value=self.existing_params.get('order', 0))
        order_combo = ttk.Combobox(order_frame, textvariable=self.param_vars['order'],
                                    values=list(range(6)), state="readonly", width=5)
        order_combo.pack(side=tk.LEFT, padx=10)

        self.order_label = ttk.Label(order_frame, text=ORDER_PERMUTATIONS[0]['I'], foreground='blue')
        self.order_label.pack(side=tk.LEFT, padx=10)
        order_combo.bind("<<ComboboxSelected>>", lambda e: self._update_order_label('I'))

        # Other parameters
        params_grid = ttk.Frame(parent)
        params_grid.pack(fill=tk.X, pady=5)

        self._add_param_row(params_grid, 0, "M_L0:", 'M_L0', 1, 1, AW, "Inner block size")
        self._add_param_row(params_grid, 1, "M_L1:", 'M_L1', 1, 1, max(1, ceil_div(M, AW)), "Rows / M_L0")
        self._add_param_row(params_grid, 2, "J_L1:", 'J_L1', 1, 1, max(1, ceil_div(K, AH)), "Reduction / AH")

    def _create_wvn_layout_params(self, parent):
        """Create SetWVNLayout parameter widgets"""
        AW = self.hw_config.AW
        AH = self.hw_config.AH
        K = self.workload_config.K
        N = self.workload_config.N

        # Order parameter
        order_frame = ttk.Frame(parent)
        order_frame.pack(fill=tk.X, pady=5)

        ttk.Label(order_frame, text="Order:").pack(side=tk.LEFT)
        self.param_vars['order'] = tk.IntVar(value=self.existing_params.get('order', 0))
        order_combo = ttk.Combobox(order_frame, textvariable=self.param_vars['order'],
                                    values=list(range(6)), state="readonly", width=5)
        order_combo.pack(side=tk.LEFT, padx=10)

        self.order_label = ttk.Label(order_frame, text=ORDER_PERMUTATIONS[0]['W'], foreground='blue')
        self.order_label.pack(side=tk.LEFT, padx=10)
        order_combo.bind("<<ComboboxSelected>>", lambda e: self._update_order_label('W'))

        params_grid = ttk.Frame(parent)
        params_grid.pack(fill=tk.X, pady=5)

        self._add_param_row(params_grid, 0, "N_L0:", 'N_L0', 1, 1, AW, "Inner block width")
        self._add_param_row(params_grid, 1, "N_L1:", 'N_L1', 1, 1, max(1, ceil_div(N, AW)), "Cols / N_L0")
        self._add_param_row(params_grid, 2, "K_L1:", 'K_L1', 1, 1, max(1, ceil_div(K, AH)), "Rows / AH")

    def _create_ovn_layout_params(self, parent):
        """Create SetOVNLayout parameter widgets"""
        AW = self.hw_config.AW
        AH = self.hw_config.AH
        M = self.workload_config.M
        N = self.workload_config.N

        # Order parameter
        order_frame = ttk.Frame(parent)
        order_frame.pack(fill=tk.X, pady=5)

        ttk.Label(order_frame, text="Order:").pack(side=tk.LEFT)
        self.param_vars['order'] = tk.IntVar(value=self.existing_params.get('order', 0))
        order_combo = ttk.Combobox(order_frame, textvariable=self.param_vars['order'],
                                    values=list(range(6)), state="readonly", width=5)
        order_combo.pack(side=tk.LEFT, padx=10)

        self.order_label = ttk.Label(order_frame, text=ORDER_PERMUTATIONS[0]['O'], foreground='blue')
        self.order_label.pack(side=tk.LEFT, padx=10)
        order_combo.bind("<<ComboboxSelected>>", lambda e: self._update_order_label('O'))

        params_grid = ttk.Frame(parent)
        params_grid.pack(fill=tk.X, pady=5)

        self._add_param_row(params_grid, 0, "P_L0:", 'P_L0', 1, 1, AW, "Inner tile height")
        self._add_param_row(params_grid, 1, "P_L1:", 'P_L1', 1, 1, max(1, ceil_div(M, AW)), "Rows / P_L0")
        self._add_param_row(params_grid, 2, "Q_L1:", 'Q_L1', 1, 1, max(1, ceil_div(N, AH)), "Cols / AH")

    def _create_mapping_params(self, parent):
        """Create ExecuteMapping parameter widgets"""
        AW = self.hw_config.AW
        AH = self.hw_config.AH
        K = self.workload_config.K
        N = self.workload_config.N

        params_grid = ttk.Frame(parent)
        params_grid.pack(fill=tk.X, pady=5)

        max_r0 = max(1, ceil_div(K, AH))
        max_c0 = max(1, N)

        self._add_param_row(params_grid, 0, "r0:", 'r0', 0, 0, max_r0 - 1, "Starting WVN row")
        self._add_param_row(params_grid, 1, "c0:", 'c0', 0, 0, max_c0 - 1, "Starting WVN col")
        self._add_param_row(params_grid, 2, "Gr:", 'Gr', 1, 1, AH, "Row multicast")
        self._add_param_row(params_grid, 3, "Gc:", 'Gc', 1, 1, AW, "Col group size")
        self._add_param_row(params_grid, 4, "sr:", 'sr', 0, 0, max(AH, max_r0 // 2), "Row stride")
        self._add_param_row(params_grid, 5, "sc:", 'sc', 0, 0, max(AW, max_c0 // 2), "Col stride")

    def _create_streaming_params(self, parent):
        """Create ExecuteStreaming parameter widgets"""
        AH = self.hw_config.AH
        M = self.workload_config.M

        params_grid = ttk.Frame(parent)
        params_grid.pack(fill=tk.X, pady=5)

        self._add_param_row(params_grid, 0, "dataflow:", 'dataflow', 1, 0, 1, "0=IO-S, 1=WO-S")
        self._add_param_row(params_grid, 1, "m_0:", 'm_0', 0, 0, max(0, M - 1), "Base streaming row")
        self._add_param_row(params_grid, 2, "s_m:", 's_m', 1, 1, max(1, M), "Row stride")
        self._add_param_row(params_grid, 3, "T:", 'T', AH, 1, max(1, M), "Streaming steps")
        self._add_param_row(params_grid, 4, "vn_size:", 'vn_size', AH - 1, 0, AH - 1, f"Active VN height (AH-1={AH-1})")

    def _update_order_label(self, operand: str):
        """Update the order permutation label"""
        order = self.param_vars['order'].get()
        if order in ORDER_PERMUTATIONS:
            self.order_label.config(text=ORDER_PERMUTATIONS[order][operand])

    def _on_validate(self):
        """Validate current configuration"""
        params = {key: var.get() for key, var in self.param_vars.items()}
        instr = ISAInstruction(self.isa_type, params)
        validator = ISAConfigValidator(self.hw_config, self.workload_config)
        is_valid, errors = validator.validate_instruction(instr)

        if is_valid:
            messagebox.showinfo("Validation", "Configuration is valid!", parent=self)
        else:
            error_text = "Configuration errors:\n\n" + "\n".join(f"• {e}" for e in errors)
            messagebox.showerror("Validation Failed", error_text, parent=self)

    def _on_ok(self):
        """OK button handler"""
        params = {key: var.get() for key, var in self.param_vars.items()}
        instr = ISAInstruction(self.isa_type, params)
        validator = ISAConfigValidator(self.hw_config, self.workload_config)
        is_valid, errors = validator.validate_instruction(instr)

        if not is_valid:
            error_text = "Configuration has errors:\n\n" + "\n".join(f"• {e}" for e in errors)
            error_text += "\n\nDo you want to save anyway?"
            if not messagebox.askyesno("Validation Warning", error_text, parent=self):
                return

        self.result = params
        self.destroy()

    def _on_cancel(self):
        """Cancel button handler"""
        self.result = None
        self.destroy()


@dataclass
class WorkloadConfig:
    """GEMM workload configuration"""
    M: int = 16
    K: int = 16
    N: int = 16

@dataclass
class HardwareConfig:
    """FEATHER+ hardware configuration"""
    AH: int = 4  # NEST height (also number of weight registers per PE)
    AW: int = 4  # NEST width
    sram_mb: float = 4.0
    frac_stream: float = 0.4
    frac_stationary: float = 0.4
    frac_output: float = 0.2


# ============================================================================
# Accurate PE State Machine (from RTL: feather_pe.v)
# ============================================================================

class PEPhase(Enum):
    """Phase of a single PE during execution"""
    IDLE = "idle"
    LOADING_WEIGHTS = "loading_weights"  # Receiving weights into W[k]
    WEIGHTS_READY = "weights_ready"      # All W[0..AH-1] loaded
    COMPUTING = "computing"              # Performing MAC operations
    DOT_PRODUCT_DONE = "dp_done"         # Finished AH MACs, waiting for bus
    OUTPUTTING = "outputting"            # Driving output bus to BIRRD

@dataclass
class PEState:
    """Accurate state of a single Processing Element.

    Each PE contains:
    - W[0..AH-1]: Local weight registers
    - PSUM: Accumulator for partial sum
    - mac_count: Number of MACs completed in current dot-product
    - weight_idx: Next weight register to be written during loading
    """
    row: int
    col: int
    AH: int  # Number of weight registers

    # Weight registers W[0..AH-1]
    weights: List[float] = field(default_factory=list)
    weight_load_idx: int = 0  # Which W[k] to write next
    weights_ready: bool = False  # True when all W[0..AH-1] loaded

    # Accumulator and MAC state
    psum: float = 0.0
    mac_count: int = 0  # How many MACs done for current dot-product
    current_input: Optional[float] = None

    # Phase tracking
    phase: PEPhase = PEPhase.IDLE
    dot_products_completed: int = 0  # How many full dot-products have been output

    def __post_init__(self):
        if not self.weights:
            self.weights = [0.0] * self.AH

    def reset_for_new_tile(self):
        """Reset PE for a new weight tile"""
        self.weights = [0.0] * self.AH
        self.weight_load_idx = 0
        self.weights_ready = False
        self.psum = 0.0
        self.mac_count = 0
        self.current_input = None
        self.phase = PEPhase.IDLE
        self.dot_products_completed = 0

    def load_weight(self, value: float) -> bool:
        """Load a weight into W[weight_load_idx]. Returns True when all weights loaded."""
        if self.weight_load_idx < self.AH:
            self.weights[self.weight_load_idx] = value
            self.weight_load_idx += 1
            self.phase = PEPhase.LOADING_WEIGHTS

            if self.weight_load_idx >= self.AH:
                self.weights_ready = True
                self.phase = PEPhase.WEIGHTS_READY
                return True
        return False

    def receive_input_and_mac(self, input_val: float) -> bool:
        """Receive input, perform MAC with W[mac_count]. Returns True when dot-product done."""
        if not self.weights_ready:
            return False

        self.current_input = input_val
        self.phase = PEPhase.COMPUTING

        # MAC: PSUM += input * W[mac_count]
        w_idx = self.mac_count % self.AH
        self.psum += input_val * self.weights[w_idx]
        self.mac_count += 1

        # Check if dot-product is complete (AH MACs done)
        if self.mac_count % self.AH == 0:
            self.phase = PEPhase.DOT_PRODUCT_DONE
            return True

        return False

    def output_psum(self) -> float:
        """Output PSUM to column bus (for BIRRD). Resets PSUM for next dot-product."""
        self.phase = PEPhase.OUTPUTTING
        result = self.psum
        self.psum = 0.0  # Reset for next dot-product
        self.dot_products_completed += 1
        return result

    def finish_output(self):
        """Called after output is sent, PE returns to computing or waiting state."""
        if self.weights_ready:
            self.phase = PEPhase.WEIGHTS_READY
        else:
            self.phase = PEPhase.IDLE


# ============================================================================
# NEST Execution Model (Accurate Pipeline)
# ============================================================================

class NESTPhase(Enum):
    """Global NEST execution phase"""
    IDLE = "idle"
    WEIGHT_LOAD = "weight_load"      # Loading weights from top→bottom
    COMPUTE_STREAM = "compute"       # Streaming inputs, computing MACs
    OUTPUT_DRAIN = "output_drain"    # Draining outputs row by row to BIRRD

@dataclass
class NESTState:
    """State of the entire NEST PE array.

    Accurate pipeline model:
    1. Weight-load phase: Weights flow top→bottom, each PE fills W[0], W[1], ..., W[AH-1]
    2. Compute phase: Inputs stream top→bottom, each row holds different input elements
    3. Output phase: Round-robin row output (row 0→1→...→AH-1→repeat)
    """
    AH: int
    AW: int

    # Per-PE state
    pe_states: Dict[Tuple[int, int], PEState] = field(default_factory=dict)

    # Global phase
    phase: NESTPhase = NESTPhase.IDLE
    global_cycle: int = 0

    # Weight loading tracking
    weight_load_row: int = 0  # Which row is receiving weights
    weight_load_idx: int = 0  # Which W[k] is being written
    weights_loaded_count: int = 0  # Total weight values loaded

    # Input streaming tracking
    input_stream_cycle: int = 0  # Cycle within compute phase
    inputs_in_flight: List[Optional[float]] = field(default_factory=list)  # Pipeline registers

    # Output tracking (round-robin)
    output_row: int = 0  # Current row outputting to BIRRD
    outputs_emitted: int = 0

    def __post_init__(self):
        # Initialize PE states
        self.pe_states = {}
        for row in range(self.AH):
            for col in range(self.AW):
                self.pe_states[(row, col)] = PEState(row=row, col=col, AH=self.AH)

        # Initialize pipeline registers (one per row for input streaming)
        self.inputs_in_flight = [None] * self.AH

    def reset(self):
        """Reset NEST for new execution"""
        for pe in self.pe_states.values():
            pe.reset_for_new_tile()
        self.phase = NESTPhase.IDLE
        self.global_cycle = 0
        self.weight_load_row = 0
        self.weight_load_idx = 0
        self.weights_loaded_count = 0
        self.input_stream_cycle = 0
        self.inputs_in_flight = [None] * self.AH
        self.output_row = 0
        self.outputs_emitted = 0

    def step_weight_load(self, weight_value: float) -> Dict[str, Any]:
        """Execute one cycle of weight loading.

        Weights propagate top→bottom through NEST pipeline.
        Each PE fills W[0], then W[1], ..., up to W[AH-1].
        """
        self.phase = NESTPhase.WEIGHT_LOAD
        result = {
            'phase': 'weight_load',
            'cycle': self.global_cycle,
            'active_row': self.weight_load_row,
            'weight_idx': self.weight_load_idx,
            'events': []
        }

        # Load weight into all PEs in current row
        for col in range(self.AW):
            pe = self.pe_states[(self.weight_load_row, col)]
            all_loaded = pe.load_weight(weight_value)
            result['events'].append({
                'type': 'weight_load',
                'pe': (self.weight_load_row, col),
                'w_idx': self.weight_load_idx,
                'value': weight_value
            })

        self.weights_loaded_count += self.AW

        # Advance pipeline: move to next row
        self.weight_load_row += 1

        # If all rows received current W[k], advance to W[k+1]
        if self.weight_load_row >= self.AH:
            self.weight_load_row = 0
            self.weight_load_idx += 1

            # Check if all weights loaded
            if self.weight_load_idx >= self.AH:
                result['all_weights_loaded'] = True
                self.phase = NESTPhase.COMPUTE_STREAM

        self.global_cycle += 1
        return result

    def step_compute(self, new_input: Optional[float] = None) -> Dict[str, Any]:
        """Execute one cycle of compute/streaming phase.

        Inputs stream top→bottom through the pipeline.
        At cycle c, row r holds input that was injected at cycle c-r.
        Each PE performs MAC with W[mac_count % AH].
        """
        self.phase = NESTPhase.COMPUTE_STREAM
        result = {
            'phase': 'compute',
            'cycle': self.global_cycle,
            'input_injected': new_input,
            'row_inputs': {},
            'mac_events': [],
            'dp_complete': []
        }

        # Shift pipeline: propagate inputs top→bottom
        # Row AH-1 (bottom) receives from row AH-2
        # Row 0 (top) receives new input
        prev_inputs = self.inputs_in_flight.copy()

        for row in range(self.AH - 1, 0, -1):
            self.inputs_in_flight[row] = prev_inputs[row - 1]
        self.inputs_in_flight[0] = new_input

        # Each PE performs MAC with its current input
        for row in range(self.AH):
            input_val = self.inputs_in_flight[row]
            result['row_inputs'][row] = input_val

            if input_val is not None:
                for col in range(self.AW):
                    pe = self.pe_states[(row, col)]
                    dp_done = pe.receive_input_and_mac(input_val)

                    mac_event = {
                        'pe': (row, col),
                        'input': input_val,
                        'w_idx': (pe.mac_count - 1) % self.AH,
                        'psum': pe.psum
                    }
                    result['mac_events'].append(mac_event)

                    if dp_done:
                        result['dp_complete'].append((row, col))

        self.input_stream_cycle += 1
        self.global_cycle += 1
        return result

    def step_output(self) -> Dict[str, Any]:
        """Execute one cycle of output drain phase.

        Round-robin output: row 0 outputs first, then row 1, ..., row AH-1.
        Only one row drives the column-wise bus per cycle.
        """
        self.phase = NESTPhase.OUTPUT_DRAIN
        result = {
            'phase': 'output',
            'cycle': self.global_cycle,
            'output_row': self.output_row,
            'psum_values': [],
            'events': []
        }

        # Current row outputs its PSUM values to BIRRD
        for col in range(self.AW):
            pe = self.pe_states[(self.output_row, col)]
            psum = pe.output_psum()
            result['psum_values'].append(psum)
            result['events'].append({
                'type': 'output',
                'pe': (self.output_row, col),
                'psum': psum,
                'col_bus': col
            })

        self.outputs_emitted += self.AW

        # Advance to next row (round-robin)
        self.output_row = (self.output_row + 1) % self.AH

        self.global_cycle += 1
        return result

    def get_pe_state(self, row: int, col: int) -> PEState:
        """Get state of specific PE"""
        return self.pe_states.get((row, col))


# ============================================================================
# Data Flow Animation Structures
# ============================================================================

class DataFlowPhase(Enum):
    """Phases of data flow in a compute cycle"""
    IDLE = "idle"
    LOAD_INPUT = "load_input"
    LOAD_WEIGHT = "load_weight"
    DISTRIBUTE_INPUT = "distribute_input"
    DISTRIBUTE_WEIGHT = "distribute_weight"
    COMPUTE = "compute"
    REDUCE_PE = "reduce_pe"
    REDUCE_BIRRD = "reduce_birrd"
    ACCUMULATE = "accumulate"
    STORE_OUTPUT = "store_output"

@dataclass
class VirtualNeuron:
    """Represents a Virtual Neuron (VN) - AH-element dot product atom"""
    vn_type: str  # "I", "W", "O", "P" (partial sum)
    row: int
    col: int
    data: Optional[Any] = None
    linear_index: int = 0

    def __str__(self):
        return f"{self.vn_type}VN({self.row},{self.col})"

@dataclass
class AnimationFrame:
    """A single frame of animation state with cycle-accurate pipeline modeling.

    Cycle-Accurate Pipeline Semantics:
    1. Weight INJECTION phase (AH cycles):
       - Weights injected into top row from Stationary Buffer
       - Weight-stream ends when last weight injected into top row (cycle AH-1)

    2. OVERLAP phase (starts immediately after weight injection):
       - Input streaming begins (Streaming Buffer → top row)
       - Weight propagation continues to lower rows
       - Rows start computing as they receive weights + inputs

    3. Compute + Output overlap:
       - PEs compute as inputs arrive
       - Row outputs to BIRRD via round-robin when dot-product complete
       - No stalling of input stream

    WVN Semantics:
    - PE at (row, col) loads from WVN(row, col) in Stationary Buffer
    - Each WVN has AH elements: WVN(r,c)[0..AH-1]
    - Column-aligned: PE column j reads from buffer column j
    """
    cycle: int
    phase: str  # Description of current phase
    nest_phase: NESTPhase = NESTPhase.IDLE

    # Per-PE state for visualization
    pe_phases: Dict[Tuple[int, int], PEPhase] = field(default_factory=dict)
    pe_weight_idx: Dict[Tuple[int, int], int] = field(default_factory=dict)  # How many W[k] loaded (0 to AH)
    pe_mac_count: Dict[Tuple[int, int], int] = field(default_factory=dict)
    pe_psum: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # WVN annotations per PE: stores the current loading progress as "WVN(r,c)[0:k]"
    pe_wvn_annotation: Dict[Tuple[int, int], str] = field(default_factory=dict)

    # Currently loading element per PE (which WVN element is being received this cycle)
    pe_loading_element: Dict[Tuple[int, int], int] = field(default_factory=dict)  # -1 if not loading

    # Row-level state
    row_inputs: Dict[int, Optional[float]] = field(default_factory=dict)  # Input at each row
    active_output_row: int = -1  # Which row is outputting (-1 = none)

    # Which rows are actively receiving weights this cycle (propagation)
    active_weight_rows: List[int] = field(default_factory=list)

    # Which rows are receiving inputs this cycle
    active_input_rows: List[int] = field(default_factory=list)

    # Overlap indicators
    is_weight_injection: bool = False      # True if injecting weights into top row
    is_weight_propagation: bool = False    # True if weights still propagating to lower rows
    is_input_streaming: bool = False       # True if inputs being streamed
    is_compute_active: bool = False        # True if any PE is computing
    is_output_active: bool = False         # True if any PE outputting to BIRRD

    # Output eligibility per row (True if row has completed dot-product)
    row_output_eligible: Dict[int, bool] = field(default_factory=dict)
    row_dot_products_done: Dict[int, int] = field(default_factory=dict)

    # Data flow annotations
    data_flows: List[Dict] = field(default_factory=list)

    # BIRRD state
    birrd_active_stage: int = -1
    birrd_inputs: List[float] = field(default_factory=list)
    birrd_egg_configs: Dict[Tuple[int, int], int] = field(default_factory=dict)  # (stage, sw) -> mode


# ============================================================================
# Animation Generator with Cycle-Accurate NEST Pipeline
# ============================================================================

class AccurateAnimationGenerator:
    """Generates animation frames with cycle-accurate NEST pipeline behavior.

    Cycle-Accurate Pipeline Model:
    =============================

    1. WEIGHT INJECTION Phase (AH cycles):
       - Cycles 0 to AH-1: Inject W[0], W[1], ..., W[AH-1] into top row
       - "Weight-stream end" = cycle AH-1 (last weight injected into top row)
       - Column-aligned: PE column j reads from Stationary Buffer column j

    2. OVERLAP Phase (starts at cycle AH):
       - Input streaming begins IMMEDIATELY after last weight injection
       - Weight propagation continues to lower rows (weights still moving down)
       - This creates explicit overlap: inputs enter while weights still propagating
       - Row r can start computing when it has received all AH weights AND an input

    3. COMPUTE + OUTPUT Overlap:
       - PEs compute as inputs arrive (if weights are ready)
       - When a PE completes AH MACs (dot-product), it becomes output-eligible
       - Round-robin row arbitration for column-wise bus to BIRRD
       - Input streaming is NOT stalled by output

    4. NEXT ITERATION:
       - After all inputs injected and outputs emitted, loop back to Phase 1
    """

    def __init__(self, AH: int = 4, AW: int = 4):
        self.AH = AH
        self.AW = AW
        self.nest_state = NESTState(AH=AH, AW=AW)
        self.frames: List[AnimationFrame] = []

        # BIRRD parameters
        if AW == 4:
            self.birrd_stages = 3
        else:
            self.birrd_stages = 2 * ceil_log2(AW)

    def reset(self):
        """Reset generator state"""
        self.nest_state.reset()
        self.frames = []

    def generate_full_iteration(self,
                                 num_weight_vns: int = 1,
                                 num_input_vns: int = 4,
                                 num_output_rows: int = 4,
                                 vn_size: int = None,
                                 s_m: int = 1,
                                 m_0: int = 0,
                                 wvn_preloaded_rows: int = 0) -> List[AnimationFrame]:
        """Generate frames for a complete iteration with cycle-accurate overlap.

        Args:
            num_weight_vns: Number of weight VNs to load
            num_input_vns: Number of input streaming steps (T from ExecuteStreaming)
            num_output_rows: Number of output rows
            vn_size: Active VN height (default AH). When vn_size < AH, only the
                     top vn_size rows are active during streaming/compute phases.
            s_m: Streaming row stride from ExecuteStreaming (for IVN index annotation)
            m_0: Base streaming row index from ExecuteStreaming (for IVN index annotation)
            wvn_preloaded_rows: Number of PE rows whose weights were pre-loaded
                during the previous EM's streaming phase (inter-EM pipelining).
                These rows skip the weight loading phase entirely.

        Timeline:
        - Cycles 0 to AH-1: Weight injection into top row (skipped for pre-loaded rows)
        - Cycle AH onwards: Input streaming begins (overlap with weight propagation)
        - Output happens via round-robin as dot-products complete
        """
        if vn_size is None:
            vn_size = self.AH
        self.reset()
        frames = []
        num_inputs = max(num_input_vns, self.AH)  # Need at least AH inputs

        # Generate all frames with unified cycle-accurate model
        frames = self._generate_cycle_accurate_frames(num_inputs, vn_size=vn_size,
                                                       s_m=s_m, m_0=m_0,
                                                       wvn_preloaded_rows=wvn_preloaded_rows)

        self.frames = frames
        return frames

    def _generate_birrd_configs(self, output_row: int) -> Dict[Tuple[int, int], int]:
        """Generate BIRRD EGG configurations for a reduction pattern.

        Creates sample configurations that demonstrate different EGG modes
        based on which row is outputting. This simulates the BIRRD being
        configured for different reduction patterns.

        Args:
            output_row: The PE row that is outputting to BIRRD

        Returns:
            Dict mapping (stage, sw) -> 2-bit mode
        """
        AW = self.AW
        LEVEL = ceil_log2(AW)
        TOTAL_STAGES = 2 * LEVEL
        NUM_SWITCHES = AW // 2

        configs = {}

        # Generate different patterns based on output_row for visual variety
        # Mode: 0b00=Pass, 0b11=Swap, 0b01=Add-L, 0b10=Add-R
        for stage in range(TOTAL_STAGES):
            for sw in range(NUM_SWITCHES):
                # Create a pattern that varies with output_row and stage
                # This is for visualization - actual configs would come from MINISA instructions
                if stage < LEVEL:
                    # First half: mix of pass and add based on row
                    if (output_row + sw) % 2 == 0:
                        mode = 0b01  # Add-L
                    else:
                        mode = 0b00  # Pass
                elif stage == LEVEL:
                    # Middle stage: swap pattern
                    if sw % 2 == output_row % 2:
                        mode = 0b11  # Swap
                    else:
                        mode = 0b00  # Pass
                else:
                    # Second half: reduction pattern
                    if (stage + sw + output_row) % 3 == 0:
                        mode = 0b10  # Add-R
                    elif (stage + sw + output_row) % 3 == 1:
                        mode = 0b01  # Add-L
                    else:
                        mode = 0b00  # Pass

                configs[(stage, sw)] = mode

        return configs

    def _generate_cycle_accurate_frames(self, num_inputs: int,
                                          vn_size: int = None,
                                          s_m: int = 1,
                                          m_0: int = 0,
                                          wvn_preloaded_rows: int = 0) -> List[AnimationFrame]:
        """Generate frames with cycle-accurate weight/input/output overlap.

        CORRECTED Weight Loading Model (with input sharing constraint):
        ================================================================
        Weights are loaded ROW-BY-ROW (not pipelined across all rows simultaneously):
        - Row 0: cycles 0 to AH-1 (loads W[0], W[1], ..., W[AH-1])
        - Row 1: cycles AH to 2*AH-1
        - Row r: cycles r*AH to (r+1)*AH-1
        - Total weight loading: AH * AH cycles

        Each PE in a row loads from its corresponding column in the Stationary Buffer:
        - PE(row, col) loads WVN(row, col)[0..AH-1]

        INPUT SHARING CONSTRAINT:
        =========================
        Because all PEs in the same column receive the same input data and the
        vertical pipeline latency from the top row to the bottom row is AH cycles,
        Row 0 cannot start immediately when its weights are done.

        Row 0 may only begin loading input data during the final AH cycles of
        the weight-loading phase - precisely when Row (AH-1) starts loading weights.

        Input start cycle = AH * (AH - 1) = AH² - AH
        (Precisely when the last row starts loading its weights)

        Pipeline latency: Input at Row 0 at cycle X reaches Row r at cycle X + r

        Key timing for 4x4 NEST:
        - Cycles 0-3:   Row 0 loads W[0..3]
        - Cycles 4-7:   Row 1 loads W[0..3]
        - Cycles 8-11:  Row 2 loads W[0..3]
        - Cycle 12:     Input streaming starts (Row 0 receives I[0])
                        Row 3 starts loading W[0]
        - Cycles 12-15: Row 3 loads W[0..3], Rows 0-2 compute
        - Cycle 15:     I[0] reaches Row 3, Row 3 finishes loading
                        Row 3 computes with I[0] (perfect alignment)
        - Cycle 16+:    All rows compute, outputs drain

        Inter-EM Pipelining:
        ====================
        When wvn_preloaded_rows > 0, the first `wvn_preloaded_rows` PE rows
        already have their weights loaded (pre-fetched during the previous EM's
        IVN streaming phase). Only the remaining rows need loading, reducing
        the weight loading phase and allowing earlier input streaming start.

        Args:
            wvn_preloaded_rows: Number of PE rows (0..AH) whose weights are
                already loaded from the previous EM's overlap period.
        """
        if vn_size is None:
            vn_size = self.AH
        # Clamp vn_size to valid range
        vn_size = max(1, min(vn_size, self.AH))
        wvn_preloaded_rows = max(0, min(wvn_preloaded_rows, self.AH))

        frames = []

        # Persistent state tracking
        pe_weight_count = {(r, c): 0 for r in range(self.AH) for c in range(self.AW)}
        pe_mac_count = {(r, c): 0 for r in range(self.AH) for c in range(self.AW)}
        pe_psum = {(r, c): 0.0 for r in range(self.AH) for c in range(self.AW)}
        row_dp_complete = {r: 0 for r in range(self.AH)}  # Dot-products completed per row
        row_dp_output = {r: 0 for r in range(self.AH)}    # Dot-products output per row

        # Pre-initialize weights for rows that were pre-loaded during previous EM
        for r in range(wvn_preloaded_rows):
            for c in range(self.AW):
                pe_weight_count[(r, c)] = self.AH  # fully loaded

        # Output arbitration state
        next_output_row = 0  # Round-robin pointer
        pending_outputs = []  # Rows waiting to output

        # INPUT BUFFERING: Each row maintains a queue of pending inputs
        # This handles the case where input arrives before weights are ready
        # (e.g., Row 3 receives I[0] at cycle 14 but weights aren't ready until cycle 15)
        row_input_queue = {r: [] for r in range(self.AH)}  # Queue of (input_idx, arrival_cycle)
        row_next_input_to_process = {r: 0 for r in range(self.AH)}  # Next input to compute

        # Weight loading: only remaining rows need loading (after pre-loaded ones)
        # Row wvn_preloaded_rows loads during cycles [0, AH-1]
        # Row wvn_preloaded_rows+1 loads during cycles [AH, 2*AH-1]
        # etc.
        remaining_wvn_rows = self.AH - wvn_preloaded_rows
        total_weight_cycles = remaining_wvn_rows * self.AH

        # INPUT SHARING CONSTRAINT (adjusted for pre-loaded rows):
        # Input streaming starts when the last row (AH-1) begins loading.
        # If all rows are pre-loaded, streaming starts immediately (cycle 0).
        # Otherwise, the last row starts at cycle (AH-1-wvn_preloaded_rows)*AH.
        if remaining_wvn_rows <= 0:
            input_start_cycle = 0
        else:
            input_start_cycle = (remaining_wvn_rows - 1) * self.AH

        # Pipeline latency: 1 cycle per row transition
        # Input i reaches Row r at cycle (input_start_cycle + i + r)

        total_cycles = total_weight_cycles + num_inputs + self.AH + self.AH

        for cycle in range(total_cycles):
            # === Determine which row is loading weights this cycle ===
            # Only remaining (non-preloaded) rows need loading.
            # Row index within remaining = cycle // AH; actual row = that + wvn_preloaded_rows
            remaining_row_idx = cycle // self.AH if cycle < total_weight_cycles else -1
            loading_row = (wvn_preloaded_rows + remaining_row_idx) if remaining_row_idx >= 0 else -1
            weight_element = cycle % self.AH if cycle < total_weight_cycles else -1

            is_weight_loading = (loading_row >= 0 and loading_row < self.AH)
            active_weight_rows = [loading_row] if is_weight_loading else []
            row_weight_element = {loading_row: weight_element} if is_weight_loading else {}

            # Is this weight injection into top row? (cycles 0 to AH-1)
            is_weight_injection = (cycle < self.AH)

            # Input streaming (starts at input_start_cycle)
            # Input i is injected into top row at cycle: input_start_cycle + i
            is_input_streaming = (cycle >= input_start_cycle) and (cycle < input_start_cycle + num_inputs)
            input_inject_idx = cycle - input_start_cycle if is_input_streaming else -1

            # Input pipeline with buffering:
            # - Input i arrives at Row r at cycle (input_start_cycle + i + r)
            # - If row has weights ready, compute immediately
            # - If row doesn't have weights ready, buffer the input until ready
            #
            # For 4x4 with input_start=12:
            #   - I[0] arrives at Row 0: cycle 12 (weights ready since cycle 3) → compute
            #   - I[0] arrives at Row 1: cycle 13 (weights ready since cycle 7) → compute
            #   - I[0] arrives at Row 2: cycle 14 (weights ready since cycle 11) → compute
            #   - I[0] arrives at Row 3: cycle 15 (weights ready at cycle 15) → compute
            #
            # Perfect alignment: I[0] reaches Row 3 exactly when Row 3 finishes loading
            #
            # Step 1: Add newly arriving inputs to each row's queue
            arriving_inputs = {}  # row -> input_idx that just arrived
            for row in range(self.AH):
                # Input i arrives at row r at cycle: input_start_cycle + i + row
                # Solving for i: i = cycle - input_start_cycle - row
                input_idx = cycle - input_start_cycle - row
                if 0 <= input_idx < num_inputs:
                    # Check if this input already in queue (avoid duplicates)
                    if input_idx not in [q[0] for q in row_input_queue[row]]:
                        row_input_queue[row].append((input_idx, cycle))
                        arriving_inputs[row] = input_idx

            # Step 2: Determine which rows have pending inputs (arrived or buffered)
            active_input_rows = []
            row_input_idx = {}
            for row in range(self.AH):
                if row_input_queue[row]:
                    # Row has at least one pending input
                    active_input_rows.append(row)
                    # Show the oldest pending input (front of queue)
                    row_input_idx[row] = row_input_queue[row][0][0]

            # === Build phase description ===
            phase_parts = []

            if is_weight_loading:
                phase_parts.append(f"W[{weight_element}]→R{loading_row}")

            if is_input_streaming:
                ivn_idx = m_0 + input_inject_idx * s_m
                phase_parts.append(f"IVN[{ivn_idx}]→R0")

            # Step 3: Find which rows can compute (have all weights + have pending input)
            computing_rows = []
            for r in active_input_rows:
                row_fully_loaded = (pe_weight_count.get((r, 0), 0) >= self.AH)
                if row_fully_loaded and row_input_queue[r]:
                    computing_rows.append(r)

            if computing_rows:
                phase_parts.append(f"MAC@R{computing_rows}")

            if vn_size < self.AH and (is_input_streaming or computing_rows):
                phase_parts.append(f"vn_size={vn_size}")

            if phase_parts:
                phase_desc = " | ".join(phase_parts)
            else:
                phase_desc = "Pipeline idle"

            # Determine NEST phase
            if is_weight_loading and not is_input_streaming:
                nest_phase = NESTPhase.WEIGHT_LOAD
            elif is_input_streaming or computing_rows:
                nest_phase = NESTPhase.COMPUTE_STREAM
            else:
                nest_phase = NESTPhase.OUTPUT_DRAIN

            # === Create frame ===
            frame = AnimationFrame(
                cycle=cycle,
                phase=phase_desc,
                nest_phase=nest_phase,
                is_weight_injection=is_weight_injection,
                is_weight_propagation=is_weight_loading and not is_weight_injection,
                is_input_streaming=is_input_streaming,
                active_weight_rows=active_weight_rows.copy(),
                active_input_rows=active_input_rows.copy(),
            )

            # === Update per-PE state ===
            for r in range(self.AH):
                for c in range(self.AW):
                    # Update weight count if this row is receiving weight this cycle
                    if r == loading_row and is_weight_loading:
                        pe_weight_count[(r, c)] = weight_element + 1

                    weights_loaded = pe_weight_count[(r, c)]
                    frame.pe_weight_idx[(r, c)] = weights_loaded

                    # WVN annotation
                    if weights_loaded == 0:
                        frame.pe_wvn_annotation[(r, c)] = f"WVN({r},{c})[-]"
                    elif weights_loaded >= self.AH:
                        frame.pe_wvn_annotation[(r, c)] = f"WVN({r},{c})[0:{self.AH-1}]✓"
                    else:
                        frame.pe_wvn_annotation[(r, c)] = f"WVN({r},{c})[0:{weights_loaded-1}]"

                    # Loading element indicator
                    if r == loading_row and is_weight_loading:
                        frame.pe_loading_element[(r, c)] = weight_element
                    else:
                        frame.pe_loading_element[(r, c)] = -1

                    # Compute if row has all weights AND has pending input in queue
                    has_all_weights = (weights_loaded >= self.AH)
                    has_pending_input = bool(row_input_queue[r])

                    if has_all_weights and has_pending_input:
                        # Perform MAC with the oldest pending input
                        pe_mac_count[(r, c)] += 1
                        w_idx = (pe_mac_count[(r, c)] - 1) % self.AH
                        pe_psum[(r, c)] += 1.0  # Symbolic

                        frame.pe_mac_count[(r, c)] = pe_mac_count[(r, c)]
                        frame.pe_psum[(r, c)] = pe_psum[(r, c)]
                        frame.is_compute_active = True

                        # Check dot-product completion (after AH MACs)
                        if pe_mac_count[(r, c)] % self.AH == 0:
                            frame.pe_phases[(r, c)] = PEPhase.DOT_PRODUCT_DONE
                            if c == 0:  # Track per-row (just check col 0)
                                row_dp_complete[r] += 1
                                frame.row_output_eligible[r] = True
                        else:
                            frame.pe_phases[(r, c)] = PEPhase.COMPUTING
                    elif r == loading_row and is_weight_loading:
                        frame.pe_phases[(r, c)] = PEPhase.LOADING_WEIGHTS
                    elif has_all_weights:
                        frame.pe_phases[(r, c)] = PEPhase.WEIGHTS_READY
                    elif weights_loaded > 0:
                        # Partially loaded but not currently loading
                        frame.pe_phases[(r, c)] = PEPhase.LOADING_WEIGHTS
                    else:
                        frame.pe_phases[(r, c)] = PEPhase.IDLE

                    frame.row_dot_products_done[r] = row_dp_complete[r]

            # === Consume inputs from queue after dot-product completion ===
            # This happens once per row (not per PE) when a dot-product completes
            for r in computing_rows:
                # Check if this row just completed a dot-product (MAC count divisible by AH)
                if pe_mac_count[(r, 0)] > 0 and pe_mac_count[(r, 0)] % self.AH == 0:
                    # Pop the processed input from queue
                    if row_input_queue[r]:
                        row_input_queue[r].pop(0)

            # === Output arbitration (round-robin) ===
            # Check which rows have completed dot-products that haven't been output
            for r in range(self.AH):
                if row_dp_complete[r] > row_dp_output[r]:
                    if r not in pending_outputs:
                        pending_outputs.append(r)

            # Grant output to next row in round-robin
            if pending_outputs:
                # Find next eligible row in round-robin order
                output_row = None
                for _ in range(self.AH):
                    if next_output_row in pending_outputs:
                        output_row = next_output_row
                        break
                    next_output_row = (next_output_row + 1) % self.AH

                if output_row is not None:
                    frame.active_output_row = output_row
                    frame.is_output_active = True
                    row_dp_output[output_row] += 1
                    pending_outputs.remove(output_row)
                    next_output_row = (output_row + 1) % self.AH

                    # Mark outputting PEs
                    for c in range(self.AW):
                        frame.pe_phases[(output_row, c)] = PEPhase.OUTPUTTING

                    frame.data_flows.append({
                        'type': 'row_output',
                        'row': output_row,
                        'to': 'birrd',
                        'info': f'Row {output_row} PSUM → BIRRD (round-robin)'
                    })
                    frame.birrd_active_stage = 0

                    # Generate sample EGG configurations for BIRRD visualization
                    # This creates a reduction pattern based on which row is outputting
                    frame.birrd_egg_configs = self._generate_birrd_configs(output_row)

            # === Data flow annotations ===
            if is_weight_loading:
                frame.data_flows.append({
                    'type': 'weight_load',
                    'row': loading_row,
                    'element': weight_element,
                    'info': f'Stationary Buffer col * → W[{weight_element}] into Row {loading_row}'
                })

            if is_input_streaming:
                ivn_idx = m_0 + input_inject_idx * s_m
                frame.data_flows.append({
                    'type': 'input_inject',
                    'input_idx': input_inject_idx,
                    'info': f'Streaming Buffer → IVN[{ivn_idx}] into Row 0'
                })

            for row in active_input_rows:
                inp = row_input_idx[row]
                ivn_idx = m_0 + inp * s_m
                frame.row_inputs[row] = f"IVN[{ivn_idx}]"
                if pe_weight_count.get((row, 0), 0) >= self.AH:
                    frame.data_flows.append({
                        'type': 'compute',
                        'row': row,
                        'input_idx': inp,
                        'info': f'Row {row}: IVN[{ivn_idx}] x W[{(pe_mac_count[(row, 0)] - 1) % self.AH}]'
                    })

            frames.append(frame)

            # Early termination check
            all_weights_done = (cycle >= total_weight_cycles)
            all_inputs_done = (cycle >= input_start_cycle + num_inputs + self.AH - 1)
            all_outputs_done = all(row_dp_complete[r] == row_dp_output[r] for r in range(self.AH))
            if all_weights_done and all_inputs_done and all_outputs_done and not pending_outputs:
                break

        return frames


# ============================================================================
# NEST Visualizer with Accurate Pipeline Animation
# ============================================================================

class PERowState(Enum):
    """High-level state of a PE row for visualization"""
    IDLE = "idle"
    LOADING_WEIGHTS = "load_wt"
    WEIGHTS_READY = "wt_ready"
    RECEIVING_INPUT = "recv_in"
    COMPUTING = "compute"
    OUTPUTTING = "output"

class NESTVisualizer:
    """Visualizes the NEST PE Array with accurate per-cycle pipeline states.

    Key visual elements:
    - PE grid with per-PE state colors
    - Weight loading: top→bottom flow, showing W[k] index
    - Input streaming: top→bottom flow, showing which input at each row
    - Output: highlighting active output row
    - Weight flow from TOP (not right side)
    """

    PE_COLORS = {
        PERowState.IDLE: '#E8EAED',
        PERowState.LOADING_WEIGHTS: '#C8E6C9',   # Green (WVN loading)
        PERowState.WEIGHTS_READY: '#A5D6A7',      # Darker green (WVN ready)
        PERowState.RECEIVING_INPUT: '#BBDEFB',     # Blue  (IVN streaming)
        PERowState.COMPUTING: '#FFE082',
        PERowState.OUTPUTTING: '#FFCDD2',          # Red   (OVN output)
    }

    def __init__(self, canvas_frame: tk.Frame, hw_config: HardwareConfig):
        self.hw_config = hw_config
        self.canvas_frame = canvas_frame
        self.fig = None
        self.ax = None
        self.canvas = None
        self.pe_patches = {}
        self.pe_labels = {}
        self.pe_weight_labels = {}
        self.pe_mac_labels = {}
        self.row_state_labels = {}
        self.input_arrows = []
        self.weight_arrows = []
        self.output_arrows = []

        self._setup_canvas()

    def _setup_canvas(self):
        """Set up matplotlib canvas"""
        if not HAS_MATPLOTLIB:
            label = tk.Label(self.canvas_frame, text="Matplotlib required for visualization")
            label.pack(expand=True)
            return

        self.fig = Figure(figsize=(12, 10), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._draw_nest_structure()

    def _draw_nest_structure(self):
        """Draw NEST structure with corrected data flow arrows.

        IMPORTANT: Weights flow from TOP (same as inputs), NOT from right side.
        """
        if self.ax is None:
            return

        self.ax.clear()
        AH, AW = self.hw_config.AH, self.hw_config.AW

        # Layout parameters
        pe_size = 1.0
        pe_spacing = 1.4
        margin_left = 2.0
        margin_top = 1.5

        self.pe_patches = {}
        self.pe_labels = {}
        self.pe_weight_labels = {}
        self.pe_mac_labels = {}
        self.row_state_labels = {}

        # Draw PE grid
        for row in range(AH):
            y = margin_top + (AH - 1 - row) * pe_spacing

            # Row state label
            state_label = self.ax.text(
                margin_left - 1.5, y + pe_size/2,
                f'Row {row}\nIdle',
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8EAED',
                         edgecolor='#666', alpha=0.9)
            )
            self.row_state_labels[row] = state_label

            for col in range(AW):
                x = margin_left + col * pe_spacing

                # PE rectangle
                pe_rect = FancyBboxPatch(
                    (x, y), pe_size, pe_size,
                    boxstyle="round,pad=0.03,rounding_size=0.08",
                    facecolor=self.PE_COLORS[PERowState.IDLE],
                    edgecolor='#1565C0', linewidth=1.5
                )
                self.ax.add_patch(pe_rect)
                self.pe_patches[(row, col)] = pe_rect

                # PE ID label
                pe_label = self.ax.text(
                    x + pe_size/2, y + pe_size*0.75,
                    f'PE({row},{col})',
                    ha='center', va='center', fontsize=9, fontweight='bold', color='#333'
                )
                self.pe_labels[(row, col)] = pe_label

                # WVN annotation label: WVN(row,col)[0:k]
                w_label = self.ax.text(
                    x + pe_size/2, y + pe_size*0.45,
                    f'WVN({row},{col})[-]',
                    ha='center', va='center', fontsize=7, color='#444'
                )
                self.pe_weight_labels[(row, col)] = w_label

                # MAC count label
                mac_label = self.ax.text(
                    x + pe_size/2, y + pe_size*0.18,
                    'MAC:0',
                    ha='center', va='center', fontsize=8, color='#666'
                )
                self.pe_mac_labels[(row, col)] = mac_label

        # Draw data flow arrows and buffer boxes
        # Top of NEST array (where row 0 is)
        nest_top_y = margin_top + (AH - 1) * pe_spacing + pe_size
        arrow_y_bottom = margin_top - 0.8

        # Buffer box positions (above NEST array)
        buffer_box_y = nest_top_y + 1.8  # Y position for buffer boxes
        arrow_start_y = buffer_box_y - 0.6  # Arrow starts below buffer box
        arrow_end_y = nest_top_y + 0.15  # Arrow ends just above NEST (not touching)

        # Calculate center positions for left and right halves
        nest_center_x = margin_left + (AW - 1) * pe_spacing / 2 + pe_size / 2
        input_buffer_x = margin_left + (AW / 4) * pe_spacing  # Left quarter
        weight_buffer_x = margin_left + (3 * AW / 4) * pe_spacing  # Right quarter

        # STREAMING BUFFER (Inputs) - upper left area
        self.ax.text(
            input_buffer_x, buffer_box_y,
            'Streaming Buffer\n(Inputs)',
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#BBDEFB',
                     edgecolor='#1565C0', linewidth=2, alpha=0.95)
        )

        # Arrow from Streaming Buffer pointing down toward NEST (not touching)
        self.ax.annotate(
            '', xy=(input_buffer_x, arrow_end_y),
            xytext=(input_buffer_x, arrow_start_y),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2.5,
                           connectionstyle='arc3,rad=0')
        )

        # STATIONARY BUFFER (WVN) - upper right area
        self.ax.text(
            weight_buffer_x, buffer_box_y,
            'Stationary Buffer\n(WVN)',
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#C8E6C9',
                     edgecolor='#2E7D32', linewidth=2, alpha=0.95)
        )

        # Arrow from Stationary Buffer pointing down toward NEST (not touching)
        self.ax.annotate(
            '', xy=(weight_buffer_x, arrow_end_y),
            xytext=(weight_buffer_x, arrow_start_y),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2.5,
                           connectionstyle='arc3,rad=0')
        )

        # Show pipeline flow with arrows between rows (inside NEST)
        for col in range(AW):
            x = margin_left + col * pe_spacing + pe_size/2
            for row in range(AH - 1):
                y_from = margin_top + (AH - 1 - row) * pe_spacing - 0.05
                y_to = margin_top + (AH - 2 - row) * pe_spacing + pe_size + 0.05
                self.ax.annotate(
                    '', xy=(x, y_to), xytext=(x, y_from),
                    arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=1, alpha=0.6)
                )

        # OUTPUT ARROWS (to BIRRD, from bottom)
        for col in range(AW):
            x = margin_left + col * pe_spacing + pe_size/2
            self.ax.annotate(
                '', xy=(x, arrow_y_bottom + 0.5),
                xytext=(x, margin_top - 0.1),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=2)
            )

        self.ax.text(
            nest_center_x,
            arrow_y_bottom + 0.1,
            'OUTPUT to BIRRD\n(round-robin row)',
            ha='center', va='top', fontsize=11, color='#C62828', fontweight='bold'
        )

        # Legend
        legend_y = margin_top - 2.0
        self.ax.text(margin_left - 0.5, legend_y, 'PE States:', fontsize=11, fontweight='bold')
        legend_items = [
            (PERowState.IDLE, 'Idle'),
            (PERowState.LOADING_WEIGHTS, 'Loading W'),
            (PERowState.WEIGHTS_READY, 'W Ready'),
            (PERowState.COMPUTING, 'Computing'),
            (PERowState.OUTPUTTING, 'Outputting'),
        ]
        for idx, (state, name) in enumerate(legend_items):
            x_leg = margin_left + idx * 2.0
            rect = Rectangle(
                (x_leg, legend_y - 0.5), 0.4, 0.3,
                facecolor=self.PE_COLORS[state], edgecolor='#333', linewidth=1
            )
            self.ax.add_patch(rect)
            self.ax.text(x_leg + 0.5, legend_y - 0.35, name, fontsize=9, va='center')

        # Title
        self.ax.set_title(
            f'NEST PE Array ({AH}×{AW})\n'
            f'Each PE loads WVN(row,col)[0..{AH-1}] from Stationary Buffer',
            fontsize=12, fontweight='bold'
        )

        # Axis settings
        self.ax.set_xlim(-1, margin_left + AW * pe_spacing + 2)
        self.ax.set_ylim(legend_y - 1.5, buffer_box_y + 1.5)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        self.fig.tight_layout()
        self.canvas.draw()

    def update_from_frame(self, frame: AnimationFrame):
        """Update visualization from animation frame with overlap awareness"""
        AH, AW = self.hw_config.AH, self.hw_config.AW

        # Get overlap state indicators
        is_weight_prop = getattr(frame, 'is_weight_propagation', False)
        is_input_stream = getattr(frame, 'is_input_streaming', False)
        active_weight_rows = getattr(frame, 'active_weight_rows', [])
        active_input_rows = getattr(frame, 'active_input_rows', [])

        for row in range(AH):
            # Get PE state for this row
            pe_phase = frame.pe_phases.get((row, 0), PEPhase.IDLE)
            loading_elem = frame.pe_loading_element.get((row, 0), -1)
            w_idx = frame.pe_weight_idx.get((row, 0), 0)
            has_all_weights = (w_idx >= AH)

            # Determine row state with overlap awareness
            if frame.active_output_row == row:
                row_state = PERowState.OUTPUTTING
                row_label = f'Row {row}\nOUT→BIRRD'
            elif pe_phase == PEPhase.COMPUTING or pe_phase == PEPhase.DOT_PRODUCT_DONE:
                # Row is computing (has weights + input)
                inp = frame.row_inputs.get(row)
                row_state = PERowState.COMPUTING
                row_label = f'Row {row}\n{inp}' if inp else f'Row {row}\nMAC'
            elif row in active_weight_rows and loading_elem >= 0:
                # Row is receiving a weight (possibly during overlap)
                row_state = PERowState.LOADING_WEIGHTS
                if is_input_stream:
                    # Overlap: weights still coming while inputs started
                    row_label = f'Row {row}\n←W[{loading_elem}]⚡'
                else:
                    row_label = f'Row {row}\n←W[{loading_elem}]'
            elif row in active_input_rows and not has_all_weights:
                # Row has input but not all weights yet (waiting)
                row_state = PERowState.LOADING_WEIGHTS
                row_label = f'Row {row}\n[0:{w_idx-1}]⏳' if w_idx > 0 else f'Row {row}\nWait⏳'
            elif has_all_weights:
                # Row has all weights, waiting for input or idle
                if row in active_input_rows:
                    row_state = PERowState.COMPUTING
                    inp = frame.row_inputs.get(row)
                    row_label = f'Row {row}\n{inp}' if inp else f'Row {row}\nReady'
                else:
                    row_state = PERowState.WEIGHTS_READY
                    row_label = f'Row {row}\nWVN✓'
            elif w_idx > 0:
                # Partially loaded
                row_state = PERowState.LOADING_WEIGHTS
                row_label = f'Row {row}\n[0:{w_idx-1}]'
            else:
                row_state = PERowState.IDLE
                row_label = f'Row {row}\nIdle'

            # Update row label
            if row in self.row_state_labels:
                self.row_state_labels[row].set_text(row_label)
                self.row_state_labels[row].set_bbox(
                    dict(boxstyle='round,pad=0.2',
                         facecolor=self.PE_COLORS[row_state],
                         edgecolor='#666', alpha=0.9)
                )

            # Update PE patches
            for col in range(AW):
                if (row, col) in self.pe_patches:
                    pe_phase = frame.pe_phases.get((row, col), PEPhase.IDLE)

                    if pe_phase == PEPhase.OUTPUTTING:
                        color = self.PE_COLORS[PERowState.OUTPUTTING]
                    elif pe_phase == PEPhase.COMPUTING or pe_phase == PEPhase.DOT_PRODUCT_DONE:
                        color = self.PE_COLORS[PERowState.COMPUTING]
                    elif pe_phase == PEPhase.WEIGHTS_READY:
                        color = self.PE_COLORS[PERowState.WEIGHTS_READY]
                    elif pe_phase == PEPhase.LOADING_WEIGHTS:
                        color = self.PE_COLORS[PERowState.LOADING_WEIGHTS]
                    else:
                        color = self.PE_COLORS[PERowState.IDLE]

                    self.pe_patches[(row, col)].set_facecolor(color)

                # Update WVN annotation label
                if (row, col) in self.pe_weight_labels:
                    # Use WVN annotation from frame if available
                    wvn_ann = frame.pe_wvn_annotation.get((row, col), None)
                    if wvn_ann:
                        self.pe_weight_labels[(row, col)].set_text(wvn_ann)
                    else:
                        # Fallback: generate from weight index
                        w_idx = frame.pe_weight_idx.get((row, col), 0)
                        if w_idx >= self.hw_config.AH:
                            self.pe_weight_labels[(row, col)].set_text(
                                f'WVN({row},{col})[0:{self.hw_config.AH-1}]✓')
                        elif w_idx > 0:
                            self.pe_weight_labels[(row, col)].set_text(
                                f'WVN({row},{col})[0:{w_idx-1}]')
                        else:
                            self.pe_weight_labels[(row, col)].set_text(
                                f'WVN({row},{col})[-]')

                # Update MAC label
                if (row, col) in self.pe_mac_labels:
                    mac = frame.pe_mac_count.get((row, col), 0)
                    self.pe_mac_labels[(row, col)].set_text(f'MAC:{mac}')

        # Update title with cycle info
        self.ax.set_title(
            f'NEST PE Array ({AH}×{AW}) - Cycle {frame.cycle}\n'
            f'Phase: {frame.phase}',
            fontsize=11, fontweight='bold'
        )

        self.canvas.draw()

    def update_config(self, hw_config: HardwareConfig):
        """Update hardware configuration and redraw"""
        self.hw_config = hw_config
        self._draw_nest_structure()

    def reset_display(self):
        """Reset display to initial state"""
        self._draw_nest_structure()


# ============================================================================
# BIRRD Visualizer
# ============================================================================

class BIRRDVisualizer:
    """Visualizes the BIRRD Reduction Network with VERTICAL orientation.

    Topology based on birrd_simple_cmd_flow_seq.v:
    - LEVEL = log2(AW), TOTAL_STAGES = 2*LEVEL
    - Each stage has AW/2 EGG switches
    - Connection patterns:
      * Stage 0: Adjacent pairs
      * Stages 1 to LEVEL-1: Inverse shuffle (loop left shift)
      * Stage LEVEL: Bit-reverse
      * Stages LEVEL+1 to TOTAL_STAGES-1: Shuffle (loop right shift)

    Data flows from TOP to BOTTOM (vertical orientation).
    """

    EGG_MODES = {
        0b00: ('Pass', '#E8F5E9', '#2E7D32'),      # Through/Pass
        0b11: ('Swap', '#FFF3E0', '#E65100'),      # Switch/Swap
        0b01: ('Add-L', '#E3F2FD', '#1565C0'),     # Add to Left output
        0b10: ('Add-R', '#F3E5F5', '#7B1FA2'),     # Add to Right output
    }

    def __init__(self, canvas_frame: tk.Frame, hw_config: HardwareConfig):
        self.hw_config = hw_config
        self.canvas_frame = canvas_frame
        self.fig = None
        self.ax = None
        self.canvas = None
        self.switch_patches = {}
        self.switch_labels = {}
        self.switch_mode_labels = {}
        self.egg_configs = {}  # Track EGG configurations: (stage, sw) -> mode

        self._setup_canvas()

    def _setup_canvas(self):
        """Set up matplotlib canvas"""
        if not HAS_MATPLOTLIB:
            label = tk.Label(self.canvas_frame, text="Matplotlib required")
            label.pack(expand=True)
            return

        self.fig = Figure(figsize=(10, 9), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._draw_birrd_structure()

    def _get_total_stages(self) -> int:
        """Get correct number of stages based on RTL.

        N=4 (LEVEL=2): 3 stages (2*LEVEL - 1), matching RTL birrd_plus_cmd_flow_seq.v.
        N>4: 2*LEVEL stages.
        """
        AW = self.hw_config.AW
        LEVEL = ceil_log2(AW)
        if AW == 4:
            return 2 * LEVEL - 1  # 3 stages for 4-input BIRRD/BIRRD+
        return 2 * LEVEL

    # ========== RTL-accurate connection functions ==========

    def _loop_left_shift(self, val: int, width: int) -> int:
        """Loop left shift (inverse shuffle) - from RTL first_half_stages"""
        if width <= 1:
            return val
        msb = (val >> (width - 1)) & 1
        shifted = ((val << 1) | msb) & ((1 << width) - 1)
        return shifted

    def _loop_right_shift(self, val: int, width: int) -> int:
        """Loop right shift (shuffle) - from RTL second_half_stages"""
        if width <= 1:
            return val
        lsb = val & 1
        shifted = (val >> 1) | (lsb << (width - 1))
        return shifted & ((1 << width) - 1)

    def _bit_reverse(self, val: int, width: int) -> int:
        """Bit reverse - from RTL middle_stage"""
        if width <= 0:
            return val
        result = 0
        for i in range(width):
            if val & (1 << i):
                result |= 1 << (width - 1 - i)
        return result

    def _classify_stage(self, stage: int) -> str:
        """Classify a physical stage into its RTL role.

        For N=4 (3 stages): stage 0=first, stage 1=middle, stage 2=last.
        For N>4 (2*LEVEL stages): stage 0=first, 1..LEVEL-1=first_half,
            LEVEL=middle, LEVEL+1..TOTAL-2=second_half, TOTAL-1=last.
        """
        AW = self.hw_config.AW
        LEVEL = ceil_log2(AW)
        TOTAL_STAGES = self._get_total_stages()
        SKIP = 1 if AW == 4 else 0

        if stage == 0:
            return "first"

        # Map physical stage to logical stage accounting for skipped first-half
        logical = stage + SKIP * (LEVEL - 1)

        if logical < LEVEL:
            return "first_half"
        elif logical == LEVEL:
            return "middle"
        elif logical < TOTAL_STAGES + SKIP * (LEVEL - 1) - 1:
            return "second_half"
        else:
            return "last"

    def _get_stage_input_indices(self, stage: int, sw: int) -> tuple:
        """Get the input port indices for a switch at given stage.

        Returns (low_input_idx, high_input_idx) based on RTL connection patterns.
        Correctly handles N=4 3-stage topology (SKIP_FIRST_HALF=1).
        """
        AW = self.hw_config.AW
        LEVEL = ceil_log2(AW)
        NUM_SWITCHES = AW // 2
        TOTAL_STAGES = self._get_total_stages()
        SKIP = 1 if AW == 4 else 0
        role = self._classify_stage(stage)

        if role == "first":
            # First stage: adjacent pairs
            return (2 * sw, 2 * sw + 1)

        elif role == "first_half":
            # First half stages (1 to LEVEL-1): inverse shuffle (skipped for N=4)
            s = stage  # physical stage = logical stage here (SKIP=0 for this branch)
            num_groups = 1 << (s - 1)
            switches_per_group = NUM_SWITCHES >> (s - 1)
            group = sw // switches_per_group
            sw_in_group = sw % switches_per_group
            group_offset = group * (AW >> (s - 1))
            width = LEVEL - s + 1

            l_idx = (sw_in_group << 1)
            l_shifted = self._loop_left_shift(l_idx, width)
            h_idx = (sw_in_group << 1) + 1
            h_shifted = self._loop_left_shift(h_idx, width)
            return (l_shifted + group_offset, h_shifted + group_offset)

        elif role == "middle":
            # Middle stage: bit-reverse
            out_low = sw << 1
            out_high = (sw << 1) + 1
            in_low = self._bit_reverse(out_low, LEVEL)
            in_high = self._bit_reverse(out_high, LEVEL)
            return (in_low, in_high)

        else:
            # Second half stages and last stage: shuffle (loop right shift)
            # Compute logical second-half index
            logical = stage + SKIP * (LEVEL - 1)
            num_groups_from_end = (2 * LEVEL - 1) - logical
            num_groups = 1 << num_groups_from_end if num_groups_from_end >= 0 else 1
            switches_per_group = NUM_SWITCHES >> num_groups_from_end if num_groups_from_end >= 0 else NUM_SWITCHES
            group = sw // switches_per_group if switches_per_group > 0 else 0
            sw_in_group = sw % switches_per_group if switches_per_group > 0 else sw
            group_offset = group * (AW >> num_groups_from_end) if num_groups_from_end >= 0 else 0
            width = LEVEL - num_groups_from_end if num_groups_from_end >= 0 else LEVEL

            l_idx = (sw_in_group << 1)
            l_shifted = self._loop_right_shift(l_idx, width)
            h_idx = (sw_in_group << 1) + 1
            h_shifted = self._loop_right_shift(h_idx, width)
            return (l_shifted + group_offset, h_shifted + group_offset)

    def _get_interstage_permutation(self, stage: int) -> List[int]:
        """Get the permutation applied to outputs of stage before they become inputs to stage+1.

        Derives the wiring from the next stage's input indices (consistent with
        _get_stage_input_indices). This correctly handles N=4's 3-stage topology.

        Returns: List where result[i] = j means output port i connects to input port j of next stage.
        """
        AW = self.hw_config.AW
        TOTAL_STAGES = self._get_total_stages()
        NUM_SWITCHES = AW // 2

        if stage >= TOTAL_STAGES - 1:
            # Last stage: identity to output
            return list(range(AW))

        # Build permutation from next stage's switch input indices.
        # Each switch at stage+1 reads from two input ports. These input ports
        # correspond to output ports from the current stage.
        perm = [0] * AW
        for sw in range(NUM_SWITCHES):
            lo_in, hi_in = self._get_stage_input_indices(stage + 1, sw)
            out_lo = 2 * sw
            out_hi = 2 * sw + 1
            perm[lo_in] = out_lo
            perm[hi_in] = out_hi

        return perm

    def _draw_birrd_structure(self):
        """Draw BIRRD butterfly network with VERTICAL orientation (top to bottom).

        Shows inter-stage connections with crossing wires based on RTL patterns:
        - Inverse shuffle (first half stages)
        - Bit-reverse (before middle stage)
        - Shuffle (second half stages)
        """
        if self.ax is None:
            return

        self.ax.clear()
        AW = self.hw_config.AW
        LEVEL = ceil_log2(AW)
        TOTAL_STAGES = self._get_total_stages()
        NUM_SWITCHES = AW // 2

        # Layout parameters (vertical orientation)
        stage_spacing = 1.8   # Vertical spacing between stages (increased for crossing wires)
        port_spacing = 0.9    # Horizontal spacing between ports
        egg_width = 0.7
        egg_height = 0.5
        crossing_zone = 0.5   # Vertical space for crossing wires

        self.switch_patches = {}
        self.switch_labels = {}
        self.switch_mode_labels = {}

        # Initialize EGG configs to default (Pass mode)
        for stage in range(TOTAL_STAGES):
            for sw in range(NUM_SWITCHES):
                self.egg_configs[(stage, sw)] = 0b00

        # Calculate positions
        def port_x(port):
            return port * port_spacing

        def stage_y(stage):
            return (TOTAL_STAGES - stage) * stage_spacing

        # Draw input labels (top)
        y_input = stage_y(-0.3)
        for port in range(AW):
            x = port_x(port)
            self.ax.text(x, y_input + 0.25, f'In{port}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color='#1565C0')
            circle = Circle((x, y_input), 0.1, facecolor='#BBDEFB',
                           edgecolor='#1565C0', linewidth=1.5, zorder=5)
            self.ax.add_patch(circle)

        # Draw output labels (bottom)
        y_output = stage_y(TOTAL_STAGES + 0.3)
        for port in range(AW):
            x = port_x(port)
            self.ax.text(x, y_output - 0.25, f'Out{port}', ha='center', va='top',
                        fontsize=10, fontweight='bold', color='#C62828')
            circle = Circle((x, y_output), 0.1, facecolor='#FFCDD2',
                           edgecolor='#C62828', linewidth=1.5, zorder=5)
            self.ax.add_patch(circle)

        # Draw each stage and inter-stage connections
        for stage in range(TOTAL_STAGES):
            y_stage = stage_y(stage)

            # Get stage type description for inter-stage connection AFTER this stage
            next_role = self._classify_stage(stage + 1) if stage + 1 < TOTAL_STAGES else "output"
            conn_type_map = {
                "first_half": "InvShuf",
                "middle": "BitRev",
                "second_half": "Shuffle",
                "last": "Shuffle",
                "output": "Output",
            }
            conn_type = conn_type_map.get(next_role, "Output")

            # Stage label on the left
            self.ax.text(-0.8, y_stage, f'S{stage}',
                        ha='right', va='center', fontsize=10, fontweight='bold', color='#333')

            # Draw EGGs for this stage
            # Each EGG sw has outputs at ports 2*sw and 2*sw+1
            for sw in range(NUM_SWITCHES):
                out_low = 2 * sw
                out_high = 2 * sw + 1

                x_low = port_x(out_low)
                x_high = port_x(out_high)
                x_center = (x_low + x_high) / 2

                # Draw EGG box
                egg_rect = FancyBboxPatch(
                    (x_center - egg_width/2, y_stage - egg_height/2),
                    egg_width, egg_height,
                    boxstyle="round,pad=0.02",
                    facecolor='#FFF9C4', edgecolor='#FF8F00',
                    linewidth=1.5, zorder=3
                )
                self.ax.add_patch(egg_rect)
                self.switch_patches[(stage, sw)] = egg_rect

                # EGG label
                label = self.ax.text(x_center, y_stage, f'E{sw}',
                                    ha='center', va='center',
                                    fontsize=9, fontweight='bold', color='#333', zorder=4)
                self.switch_labels[(stage, sw)] = label

                # Draw input lines from above to EGG
                if stage == 0:
                    # First stage: direct from input ports
                    y_above = y_input
                    self.ax.plot([x_low, x_low], [y_above - 0.1, y_stage + egg_height/2],
                               color='#1565C0', linewidth=1.0, zorder=1)
                    self.ax.plot([x_high, x_high], [y_above - 0.1, y_stage + egg_height/2],
                               color='#1565C0', linewidth=1.0, zorder=1)

                # Draw output lines from EGG downward
                self.ax.plot([x_low, x_low], [y_stage - egg_height/2, y_stage - egg_height/2 - 0.15],
                           color='#546E7A', linewidth=1.0, zorder=1)
                self.ax.plot([x_high, x_high], [y_stage - egg_height/2, y_stage - egg_height/2 - 0.15],
                           color='#546E7A', linewidth=1.0, zorder=1)

            # Draw inter-stage connections (crossing wires)
            if stage < TOTAL_STAGES - 1:
                y_cross_start = y_stage - egg_height/2 - 0.15
                y_cross_end = stage_y(stage + 1) + egg_height/2
                y_cross_mid = (y_cross_start + y_cross_end) / 2

                perm = self._get_interstage_permutation(stage)

                # Connection type label
                self.ax.text((AW - 1) * port_spacing + 0.6, y_cross_mid, conn_type,
                            ha='left', va='center', fontsize=9, color='#666', style='italic')

                # Draw crossing connections
                for src_port in range(AW):
                    dst_port = perm[src_port]
                    x_src = port_x(src_port)
                    x_dst = port_x(dst_port)

                    is_crossing = (src_port != dst_port)
                    if is_crossing:
                        # Draw crossing wire with bezier-like path
                        color = '#E65100'  # Orange for crossing
                        lw = 0.8
                        # Use intermediate point for smoother crossing
                        self.ax.plot([x_src, x_src], [y_cross_start, y_cross_mid + 0.1],
                                   color=color, linewidth=lw, zorder=1)
                        self.ax.plot([x_src, x_dst], [y_cross_mid + 0.1, y_cross_mid - 0.1],
                                   color=color, linewidth=lw, zorder=1)
                        self.ax.plot([x_dst, x_dst], [y_cross_mid - 0.1, y_cross_end],
                                   color=color, linewidth=lw, zorder=1)
                    else:
                        # Straight connection
                        color = '#1565C0'  # Blue for straight
                        self.ax.plot([x_src, x_dst], [y_cross_start, y_cross_end],
                                   color=color, linewidth=0.8, zorder=1)

            else:
                # Last stage: connect to outputs
                y_to_output = y_stage - egg_height/2 - 0.15
                for port in range(AW):
                    x = port_x(port)
                    self.ax.plot([x, x], [y_to_output, y_output + 0.1],
                               color='#C62828', linewidth=1.0, zorder=1)

        # Draw legend on the right
        legend_x = (AW - 1) * port_spacing + 2.0
        legend_y = stage_y(TOTAL_STAGES / 2)
        self.ax.text(legend_x, legend_y + 1.2, 'EGG Modes:', fontsize=11, fontweight='bold')
        for idx, (mode, (name, bg_color, edge_color)) in enumerate(self.EGG_MODES.items()):
            y_leg = legend_y + 0.8 - idx * 0.5
            rect = Rectangle((legend_x, y_leg - 0.15), 0.4, 0.3,
                            facecolor=bg_color, edgecolor=edge_color, linewidth=1.5)
            self.ax.add_patch(rect)
            self.ax.text(legend_x + 0.55, y_leg, f'{name} ({mode:02b})',
                        fontsize=10, va='center')

        # Title
        config_bits = TOTAL_STAGES * NUM_SWITCHES * 2
        stage_formula = f"2x{LEVEL}-1" if AW == 4 else f"2x{LEVEL}"
        self.ax.set_title(
            f'BIRRD Reduction Network (AW={AW}, Vertical)\n'
            f'{TOTAL_STAGES} stages ({stage_formula}) | {NUM_SWITCHES} EGGs/stage | {config_bits} config bits',
            fontsize=11, fontweight='bold', pad=10
        )

        # Set axis limits
        margin = 1.5
        self.ax.set_xlim(-margin, (AW - 1) * port_spacing + 4.0)
        self.ax.set_ylim(y_output - 1.0, y_input + 1.0)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        self.fig.tight_layout()
        self.canvas.draw()

    def update_config(self, hw_config: HardwareConfig):
        """Update hardware configuration and redraw"""
        self.hw_config = hw_config
        self._draw_birrd_structure()

    def set_egg_config(self, stage: int, sw: int, mode: int):
        """Set configuration for a specific EGG switch.

        Args:
            stage: Stage number (0 to TOTAL_STAGES-1)
            sw: Switch number within stage (0 to NUM_SWITCHES-1)
            mode: 2-bit mode (0b00=Pass, 0b11=Swap, 0b01=Add-L, 0b10=Add-R)
        """
        self.egg_configs[(stage, sw)] = mode & 0b11
        self._update_egg_display(stage, sw)

    def set_all_egg_configs(self, configs: dict):
        """Set configurations for all EGG switches.

        Args:
            configs: Dict mapping (stage, sw) to mode
        """
        for (stage, sw), mode in configs.items():
            self.egg_configs[(stage, sw)] = mode & 0b11
        self._update_all_egg_displays()

    def _update_egg_display(self, stage: int, sw: int):
        """Update display for a single EGG"""
        if (stage, sw) not in self.switch_patches:
            return

        mode = self.egg_configs.get((stage, sw), 0b00)
        name, bg_color, edge_color = self.EGG_MODES.get(mode, ('?', '#FFF', '#000'))

        patch = self.switch_patches[(stage, sw)]
        patch.set_facecolor(bg_color)
        patch.set_edgecolor(edge_color)

        if (stage, sw) in self.switch_mode_labels:
            self.switch_mode_labels[(stage, sw)].set_text(name)

    def _update_all_egg_displays(self):
        """Update display for all EGGs"""
        AW = self.hw_config.AW
        TOTAL_STAGES = self._get_total_stages()
        NUM_SWITCHES = AW // 2

        for stage in range(TOTAL_STAGES):
            for sw in range(NUM_SWITCHES):
                self._update_egg_display(stage, sw)
        self.canvas.draw()

    def highlight_active_stage(self, stage: int):
        """Highlight active stage and update EGG configurations"""
        AW = self.hw_config.AW
        TOTAL_STAGES = self._get_total_stages()
        NUM_SWITCHES = AW // 2

        for s in range(TOTAL_STAGES):
            for sw in range(NUM_SWITCHES):
                patch = self.switch_patches.get((s, sw))
                if patch is None:
                    continue

                if s == stage:
                    # Active stage: highlight with yellow
                    patch.set_facecolor('#FFE082')
                    patch.set_edgecolor('#F57C00')
                    patch.set_linewidth(2.5)
                else:
                    # Inactive: show configured color
                    mode = self.egg_configs.get((s, sw), 0b00)
                    _, bg_color, edge_color = self.EGG_MODES.get(mode, ('?', '#FFF9C4', '#FF8F00'))
                    patch.set_facecolor(bg_color)
                    patch.set_edgecolor(edge_color)
                    patch.set_linewidth(1.5)

        self.canvas.draw()

    def animate_data_flow(self, input_values: List[float], configs: dict = None):
        """Animate data flowing through BIRRD with given configurations.

        Args:
            input_values: List of AW input values
            configs: Optional dict of (stage, sw) -> mode configurations
        """
        if configs:
            self.set_all_egg_configs(configs)
        else:
            self._update_all_egg_displays()

    def reset_display(self):
        """Reset all EGGs to default state"""
        AW = self.hw_config.AW
        TOTAL_STAGES = self._get_total_stages()
        NUM_SWITCHES = AW // 2

        for stage in range(TOTAL_STAGES):
            for sw in range(NUM_SWITCHES):
                self.egg_configs[(stage, sw)] = 0b00

        self._update_all_egg_displays()


# ============================================================================
# VN Buffer Layout Visualization
# ============================================================================

class VNBufferVisualizer:
    """Visualizes VN data layouts in buffers.

    The visualization reflects the actual layout specified by Set*VNLayout
    instructions.  Each buffer is drawn as a grid of ``AW`` columns (matching
    buffer banks) with enough rows to hold all VNs.  The mapping from
    multi-dimensional VN indices to linear buffer addresses follows the
    ``LayoutSpec.linear_index`` logic from ``layout.py``: the three nested-loop
    dimensions are traversed in the order selected by the ``order`` field and
    each VN is placed at the resulting linear address.

    Buffer capacity (total number of VN slots per buffer) is derived from the
    external ``HardwareConfig`` (sram_mb, allocation fractions, AH) so that
    it always matches the hardware model.
    """

    BUFFER_COLORS = {
        'streaming': ('#BBDEFB', '#1565C0'),   # Blue  (IVN)
        'stationary': ('#C8E6C9', '#2E7D32'),  # Green (WVN)
        'output': ('#FFCDD2', '#C62828'),       # Red   (OVN)
    }

    # Permutation tables for the three operands (outer → inner)
    _PERM_TABLE = {
        0: {"W": ("kL1", "nL0", "nL1"), "I": ("jL1", "mL0", "mL1"), "O": ("pL1", "pL0", "qL1")},
        1: {"W": ("kL1", "nL1", "nL0"), "I": ("jL1", "mL1", "mL0"), "O": ("pL1", "qL1", "pL0")},
        2: {"W": ("nL0", "kL1", "nL1"), "I": ("mL0", "jL1", "mL1"), "O": ("pL0", "pL1", "qL1")},
        3: {"W": ("nL0", "nL1", "kL1"), "I": ("mL0", "mL1", "jL1"), "O": ("pL0", "qL1", "pL1")},
        4: {"W": ("nL1", "kL1", "nL0"), "I": ("mL1", "jL1", "mL0"), "O": ("qL1", "pL1", "pL0")},
        5: {"W": ("nL1", "nL0", "kL1"), "I": ("mL1", "mL0", "jL1"), "O": ("qL1", "pL0", "pL1")},
    }

    def __init__(self, canvas_frame: tk.Frame, hw_config: HardwareConfig):
        self.hw_config = hw_config
        self.canvas_frame = canvas_frame
        self.fig = None
        self.axes = {}
        self.canvas = None
        self.cell_patches = {}
        self.cell_labels = {}
        # Layout parameters from Set*VNLayout instructions
        self.layout_params = {
            'streaming':  {'order': 0, 'a0': 1, 'a1': 1, 'a2': 1},  # M_L0, M_L1, J_L1
            'stationary': {'order': 0, 'a0': 1, 'a1': 1, 'a2': 1},  # N_L0, N_L1, K_L1
            'output':     {'order': 0, 'a0': 1, 'a1': 1, 'a2': 1},  # P_L0, P_L1, Q_L1
        }

        self._setup_canvas()

    def _buffer_vn_capacity(self, buffer_type: str) -> int:
        """Compute VN capacity for a buffer from overall SRAM and allocation."""
        total_sram_bytes = int(self.hw_config.sram_mb * 1024 * 1024)
        AH = self.hw_config.AH
        if buffer_type == 'streaming':
            buf_bytes = int(total_sram_bytes * self.hw_config.frac_stream)
            vn_size = AH * 1  # 1 byte per input element
        elif buffer_type == 'stationary':
            buf_bytes = int(total_sram_bytes * self.hw_config.frac_stationary)
            vn_size = AH * 1  # 1 byte per weight element
        else:  # output
            buf_bytes = int(total_sram_bytes * self.hw_config.frac_output)
            vn_size = AH * 4  # 4 bytes per output element
        return max(1, buf_bytes // max(1, vn_size))

    def _setup_canvas(self):
        """Set up matplotlib canvas"""
        if not HAS_MATPLOTLIB:
            label = tk.Label(self.canvas_frame, text="Matplotlib required")
            label.pack(expand=True)
            return

        self.fig = Figure(figsize=(14, 8), dpi=100)
        self.axes['streaming'] = self.fig.add_subplot(131)
        self.axes['stationary'] = self.fig.add_subplot(132)
        self.axes['output'] = self.fig.add_subplot(133)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._draw_buffer_layouts()

    def _draw_buffer_layouts(self):
        """Draw all buffer layouts"""
        self._draw_single_buffer('streaming', 'Input (IVN)', 'I')
        self._draw_single_buffer('stationary', 'Weight (WVN)', 'W')
        self._draw_single_buffer('output', 'Output (OVN)', 'O')

        self.fig.tight_layout()
        self.canvas.draw()

    @staticmethod
    def _linear_index(order_id: int, operand: str, dims: Dict[str, int],
                      idxs: Dict[str, int]) -> int:
        """Compute linear buffer address for a VN using the Table II permutation.

        This mirrors ``LayoutSpec.linear_index`` from ``layout.py``.
        ``dims`` maps dimension names to their sizes; ``idxs`` maps names to
        the current index values.
        """
        perm = VNBufferVisualizer._PERM_TABLE[order_id][operand]
        i0, i1, i2 = idxs[perm[0]], idxs[perm[1]], idxs[perm[2]]
        D1, D2 = dims[perm[1]], dims[perm[2]]
        return i0 * (D1 * D2) + i1 * D2 + i2

    def _draw_single_buffer(self, buffer_type: str, title: str, operand: str):
        """Draw a single buffer, placing VNs according to the current layout params."""
        ax = self.axes[buffer_type]
        ax.clear()

        AH, AW = self.hw_config.AH, self.hw_config.AW
        lp = self.layout_params[buffer_type]
        order_id = lp['order']
        a0, a1, a2 = lp['a0'], lp['a1'], lp['a2']
        fill_color, edge_color = self.BUFFER_COLORS[buffer_type]
        vn_prefix = {'I': 'IVN', 'W': 'WVN', 'O': 'OVN'}.get(operand, 'VN')

        # Build dimension name map and VN labels keyed by linear address
        if operand == 'W':
            dim_names = {'nL0': a0, 'nL1': a1, 'kL1': a2}
            iter_names = ('kL1', 'nL0', 'nL1')  # iteration order for labelling
            def vn_label(idxs):
                return f'{vn_prefix}\nk={idxs["kL1"]},n={idxs["nL0"]*a1+idxs["nL1"]}'
        elif operand == 'I':
            dim_names = {'mL0': a0, 'mL1': a1, 'jL1': a2}
            iter_names = ('jL1', 'mL0', 'mL1')
            def vn_label(idxs):
                return f'{vn_prefix}\nm={idxs["mL0"]*a1+idxs["mL1"]},j={idxs["jL1"]}'
        else:  # 'O'
            dim_names = {'pL0': a0, 'pL1': a1, 'qL1': a2}
            iter_names = ('pL0', 'pL1', 'qL1')
            def vn_label(idxs):
                return f'{vn_prefix}\np={idxs["pL0"]*a1+idxs["pL1"]},q={idxs["qL1"]}'

        total_vns = a0 * a1 * a2  # actual VN count from layout dimensions
        cap = self._buffer_vn_capacity(buffer_type)

        # Grid sizing: AW columns (= buffer banks), enough rows for all VNs
        num_cols = max(1, min(AW, total_vns))
        num_rows = min(16, max(1, ceil_div(total_vns, num_cols)))

        # Build linear-address → label mapping via the Table II permutation
        addr_to_label: Dict[int, str] = {}
        for d0 in range(dim_names[iter_names[0]]):
            for d1 in range(dim_names[iter_names[1]]):
                for d2 in range(dim_names[iter_names[2]]):
                    idxs = {iter_names[0]: d0, iter_names[1]: d1, iter_names[2]: d2}
                    L = self._linear_index(order_id, operand, dim_names, idxs)
                    addr_to_label[L] = vn_label(idxs)

        cell_w, cell_h = 1.2, 0.85

        self.cell_patches[buffer_type] = {}
        self.cell_labels[buffer_type] = {}

        for row in range(num_rows):
            for col in range(num_cols):
                x = col * (cell_w + 0.08)
                y = (num_rows - 1 - row) * (cell_h + 0.08)
                L = row * num_cols + col

                is_valid = L < total_vns
                is_over_cap = L >= cap
                if is_over_cap and is_valid:
                    fc = '#FFCDD2'  # red tint for over-capacity
                elif is_valid:
                    fc = fill_color
                else:
                    fc = '#F5F5F5'

                rect = FancyBboxPatch(
                    (x, y), cell_w, cell_h,
                    boxstyle="round,pad=0.02",
                    facecolor=fc,
                    edgecolor=edge_color if is_valid else '#CCC',
                    linewidth=1 if is_valid else 0.5
                )
                ax.add_patch(rect)
                self.cell_patches[buffer_type][(row, col)] = rect

                if is_valid:
                    label_text = addr_to_label.get(L, f'addr {L}')
                    fontsize = 7 if num_cols > 8 else 8
                    label = ax.text(x + cell_w/2, y + cell_h/2, label_text,
                                   ha='center', va='center', fontsize=fontsize,
                                   fontweight='bold', color='#333')
                    self.cell_labels[buffer_type][(row, col)] = label

        # Column / row index labels
        for col in range(num_cols):
            ax.text(col * (cell_w + 0.08) + cell_w/2,
                   num_rows * (cell_h + 0.08) + 0.15,
                   f'{col}', ha='center', va='bottom', fontsize=10, color='#666')
        for row in range(num_rows):
            ax.text(-0.35, (num_rows - 1 - row) * (cell_h + 0.08) + cell_h/2,
                   f'{row}', ha='right', va='center', fontsize=10, color='#666')

        perm_str = ORDER_PERMUTATIONS.get(order_id, {}).get(operand, '?')
        cap_info = f'  (VNs={total_vns}, cap={cap})'
        ax.set_title(f'{title} Buffer\nOrder {order_id}: {perm_str}{cap_info}',
                    fontsize=10, fontweight='bold', color=edge_color)

        ax.set_xlim(-1.4, num_cols * (cell_w + 0.08) + 0.5)
        ax.set_ylim(-1.1, num_rows * (cell_h + 0.08) + 1.0)
        ax.set_aspect('equal')
        ax.axis('off')

    def update_layout_from_instructions(self, instructions: list):
        """Extract layout parameters from ISA instructions and redraw.

        Scans the instruction list for Set*VNLayout instructions and uses
        the latest parameters for each buffer type to update the
        visualization.
        """
        for instr in instructions:
            if instr.isa_type == ISAType.SetIVNLayout:
                self.layout_params['streaming'] = {
                    'order': instr.params.get('order', 0),
                    'a0': max(1, instr.params.get('M_L0', 1)),
                    'a1': max(1, instr.params.get('M_L1', 1)),
                    'a2': max(1, instr.params.get('J_L1', 1)),
                }
            elif instr.isa_type == ISAType.SetWVNLayout:
                self.layout_params['stationary'] = {
                    'order': instr.params.get('order', 0),
                    'a0': max(1, instr.params.get('N_L0', 1)),
                    'a1': max(1, instr.params.get('N_L1', 1)),
                    'a2': max(1, instr.params.get('K_L1', 1)),
                }
            elif instr.isa_type == ISAType.SetOVNLayout:
                self.layout_params['output'] = {
                    'order': instr.params.get('order', 0),
                    'a0': max(1, instr.params.get('P_L0', 1)),
                    'a1': max(1, instr.params.get('P_L1', 1)),
                    'a2': max(1, instr.params.get('Q_L1', 1)),
                }
        self._draw_buffer_layouts()

    def update_config(self, hw_config: HardwareConfig):
        """Update hardware configuration and redraw"""
        self.hw_config = hw_config
        self._draw_buffer_layouts()


# ============================================================================
# Enhancement Tools (from minisa_enhancements.py)
# ============================================================================

# Paper vs Code comparison tables
PAPER_TABLE_III = {
    0: {"W": ("kL1", "nL0", "nL1"), "I": ("mL1", "mL0", "jL1"), "O": ("pL1", "pL0", "qL1")},
    1: {"W": ("kL1", "nL1", "nL0"), "I": ("mL1", "jL1", "mL0"), "O": ("pL1", "qL1", "pL0")},
    2: {"W": ("nL0", "kL1", "nL1"), "I": ("mL0", "mL1", "jL1"), "O": ("pL0", "pL1", "qL1")},
    3: {"W": ("nL0", "nL1", "kL1"), "I": ("mL0", "jL1", "mL1"), "O": ("pL0", "qL1", "pL1")},
    4: {"W": ("nL1", "kL1", "nL0"), "I": ("jL1", "mL1", "mL0"), "O": ("qL1", "pL1", "pL0")},
    5: {"W": ("nL1", "nL0", "kL1"), "I": ("jL1", "mL0", "mL1"), "O": ("qL1", "pL0", "pL1")},
}

CODE_TABLE_II = {
    0: {"W": ("kL1", "nL0", "nL1"), "I": ("jL1", "mL0", "mL1"), "O": ("pL1", "pL0", "qL1")},
    1: {"W": ("kL1", "nL1", "nL0"), "I": ("jL1", "mL1", "mL0"), "O": ("pL1", "qL1", "pL0")},
    2: {"W": ("nL0", "kL1", "nL1"), "I": ("mL0", "jL1", "mL1"), "O": ("pL0", "pL1", "qL1")},
    3: {"W": ("nL0", "nL1", "kL1"), "I": ("mL0", "mL1", "jL1"), "O": ("pL0", "qL1", "pL1")},
    4: {"W": ("nL1", "kL1", "nL0"), "I": ("mL1", "jL1", "mL0"), "O": ("qL1", "pL1", "pL0")},
    5: {"W": ("nL1", "nL0", "kL1"), "I": ("mL1", "mL0", "jL1"), "O": ("qL1", "pL0", "pL1")},
}

@dataclass
class ISABitwidth:
    """Calculate ISA instruction bitwidths"""
    AH: int
    AW: int
    Dsta: int
    Dstr: int

    def __post_init__(self):
        self.log2_AH = ceil_log2(self.AH)
        self.log2_AW = ceil_log2(self.AW)
        self.log2_Dsta = ceil_log2(self.Dsta)
        self.log2_Dstr = ceil_log2(self.Dstr)

    def summary(self) -> Dict[str, Any]:
        return {
            "config": {"AH": self.AH, "AW": self.AW, "Dsta": self.Dsta, "Dstr": self.Dstr},
            "SetWVNLayout": {"bits": 3 + 3 + self.log2_AW + 2*self.log2_Dsta},
            "SetIVNLayout": {"bits": 3 + 3 + self.log2_AW + 2*self.log2_Dstr},
            "SetOVNLayout": {"bits": 3 + 3 + self.log2_AW + 2*self.log2_Dstr},
            "ExecuteMapping": {"bits": 3 + self.log2_AH + self.log2_AW + 2*self.log2_Dsta},
            "ExecuteStreaming": {"bits": 3 + 1 + 2*self.log2_Dstr + self.log2_Dstr + self.log2_AH},
        }

class PerformanceComparator:
    """Compare performance under different configurations"""

    def __init__(self, M: int, K: int, N: int):
        self.M = M
        self.K = K
        self.N = N
        self.results = []

    def evaluate_config(self, AH: int, AW: int, sram_mb: float, label: str = "") -> Dict[str, Any]:
        """Evaluate a configuration"""
        if not HAS_MINISA:
            return {"error": "MINISA system not available", "label": label}

        try:
            cfg = FeatherPlusConfig(nest=AH, total_sram_mb=sram_mb)
            tb = generate_trace_gemm(self.M, self.K, self.N, cfg)
            Mt = tb.chunk_strategy['M_chunk']
            Kt = tb.chunk_strategy['K_chunk']
            Nt = tb.chunk_strategy['N_chunk']
            cyc = estimate_cycles_for_gemm(self.M, self.K, self.N, cfg, Mt, Kt, Nt)

            macs = self.M * self.K * self.N
            peak = cfg.peak_macs_per_cycle()
            utilization = macs / (cyc.total * peak) if cyc.total > 0 else 0

            result = {
                "label": label or f"{AH}x{AW}",
                "cycles": cyc.total,
                "utilization": utilization,
                "trace_len": len(tb.trace),
            }
            self.results.append(result)
            return result
        except Exception as e:
            return {"error": str(e), "label": label}

    def print_summary(self):
        """Print summary"""
        print(f"\nPerformance Summary for GEMM({self.M}x{self.K})@({self.K}x{self.N})")
        print("-" * 60)
        for r in sorted(self.results, key=lambda x: -x.get('utilization', 0)):
            if 'error' not in r:
                print(f"{r['label']}: {r['cycles']:,} cycles, {r['utilization']*100:.2f}% util")

def run_all_tests():
    """Run verification tests"""
    print("\n" + "=" * 60)
    print("MINISA Verification Tests")
    print("=" * 60)

    # Test BIRRD stage counts
    print("\n[Test 1] BIRRD Stage Counts:")
    for aw in [4, 8, 16]:
        if aw == 4:
            stages = 3
        else:
            stages = 2 * ceil_log2(aw)
        print(f"  AW={aw}: {stages} stages, {aw//2} EGGs/stage")

    # Test ISA bitwidth
    print("\n[Test 2] ISA Bitwidth:")
    bw = ISABitwidth(AH=8, AW=8, Dsta=256, Dstr=256)
    summary = bw.summary()
    for isa, info in summary.items():
        if isa != "config" and isinstance(info, dict):
            print(f"  {isa}: {info.get('bits', '?')} bits")

    print("\nAll tests completed!")


# ============================================================================
# Main GUI Application
# ============================================================================

class MINISAGui:
    """Main GUI application"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MINISA - FEATHER+ Accelerator Visualizer (Unified)")

        # Set initial window size and minimum size
        self.root.geometry("1920x1080")
        self.root.minsize(900, 600)  # Allow small screens

        # Try to set a reasonable size based on screen
        try:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            width = max(900, min(int(screen_width * 0.95), 2560))
            height = max(600, min(int(screen_height * 0.95), 1440))
            self.root.geometry(f"{width}x{height}")
        except Exception:
            pass  # Fallback to default geometry

        self.hw_config = HardwareConfig()
        self.workload_config = WorkloadConfig()
        self.instructions: List[ISAInstruction] = []
        self.current_trace = None
        self.animation_running = False
        self.animation_generator = None
        self.animation_frames = []
        self.current_frame_idx = 0

        self._create_menu()
        self._create_main_layout()
        self._create_default_instructions()

    def _create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Trace", command=self._load_trace)
        file_menu.add_command(label="Save Trace", command=self._save_trace)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset View", command=self._reset_view)

        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Run Tests", command=self._run_tests)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_main_layout(self):
        """Create main layout"""
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: scrollable configuration area
        left_outer = ttk.Frame(main_paned, width=520)
        main_paned.add(left_outer, weight=1)

        # Canvas + scrollbar for scrollable left panel
        self._left_canvas = tk.Canvas(left_outer, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_outer, orient=tk.VERTICAL,
                                       command=self._left_canvas.yview)
        self._left_inner = ttk.Frame(self._left_canvas)

        self._left_inner.bind(
            "<Configure>",
            lambda e: self._left_canvas.configure(
                scrollregion=self._left_canvas.bbox("all")))

        self._left_canvas_window = self._left_canvas.create_window(
            (0, 0), window=self._left_inner, anchor="nw")

        # Make inner frame match canvas width on resize
        def _on_canvas_configure(event):
            self._left_canvas.itemconfig(self._left_canvas_window,
                                         width=event.width)
        self._left_canvas.bind("<Configure>", _on_canvas_configure)

        self._left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind mousewheel scrolling on the left panel
        def _on_mousewheel(event):
            self._left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_linux(event):
            if event.num == 4:
                self._left_canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                self._left_canvas.yview_scroll(3, "units")

        def _bind_mousewheel(event):
            if sys.platform == 'darwin':
                self._left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            elif sys.platform.startswith('linux'):
                self._left_canvas.bind_all("<Button-4>", _on_mousewheel_linux)
                self._left_canvas.bind_all("<Button-5>", _on_mousewheel_linux)
            else:
                self._left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            if sys.platform == 'darwin':
                self._left_canvas.unbind_all("<MouseWheel>")
            elif sys.platform.startswith('linux'):
                self._left_canvas.unbind_all("<Button-4>")
                self._left_canvas.unbind_all("<Button-5>")
            else:
                self._left_canvas.unbind_all("<MouseWheel>")

        self._left_canvas.bind("<Enter>", _bind_mousewheel)
        self._left_canvas.bind("<Leave>", _unbind_mousewheel)

        # Right panel for visualizations
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)

        self._create_config_panel(self._left_inner)
        self._create_visualization_panel(right_frame)

    def _create_config_panel(self, parent: ttk.Frame):
        """Create configuration panel"""
        # Hardware config
        hw_frame = ttk.LabelFrame(parent, text="Hardware Configuration", padding=10)
        hw_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(hw_frame, text="NEST Size:").grid(row=0, column=0, sticky=tk.W)
        self.nest_size_var = tk.StringVar(value="4x4")
        nest_combo = ttk.Combobox(hw_frame, textvariable=self.nest_size_var,
                                  values=["4x4", "8x8", "16x16"], state="readonly", width=15)
        nest_combo.grid(row=0, column=1, padx=5, pady=2)
        nest_combo.bind("<<ComboboxSelected>>", self._on_nest_size_change)

        ttk.Label(hw_frame, text="SRAM (MB):").grid(row=1, column=0, sticky=tk.W)
        self.sram_var = tk.DoubleVar(value=4.0)
        ttk.Spinbox(hw_frame, from_=1, to=4096, textvariable=self.sram_var, width=15).grid(row=1, column=1, padx=5, pady=2)
        self.sram_var.trace_add('write', self._on_sram_change)

        # Workload config (derived from Set*VNLayout instructions)
        wl_frame = ttk.LabelFrame(parent, text="Workload (GEMM) — derived from ISA trace", padding=10)
        wl_frame.pack(fill=tk.X, padx=5, pady=5)

        for i, (label, default) in enumerate([("M:", 16), ("K:", 16), ("N:", 16)]):
            ttk.Label(wl_frame, text=label).grid(row=0, column=i*2, sticky=tk.W, padx=2)
            var = tk.IntVar(value=default)
            setattr(self, f'{label[0].lower()}_var', var)
            lbl = ttk.Label(wl_frame, textvariable=var, width=8,
                            relief='sunken', anchor='center')
            lbl.grid(row=0, column=i*2+1, padx=2, pady=2)

        # ISA list
        isa_frame = ttk.LabelFrame(parent, text="ISA Trace Sequence", padding=10)
        isa_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        list_frame = ttk.Frame(isa_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.isa_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, selectmode=tk.SINGLE, height=8)
        self.isa_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.isa_listbox.yview)
        self.isa_listbox.bind('<Double-1>', lambda e: self._edit_instruction())
        self.isa_listbox.bind('<<ListboxSelect>>', self._on_instruction_select)

        # Buttons for ISA trace manipulation
        btn_frame = ttk.Frame(isa_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        # Row 1: Add instruction with ISA type dropdown
        add_frame = ttk.Frame(btn_frame)
        add_frame.pack(fill=tk.X, pady=3)

        ttk.Button(add_frame, text="Add", command=self._add_instruction_from_dropdown,
                   width=8).pack(side=tk.LEFT, padx=2)
        self.isa_type_var = tk.StringVar(value="SetWVNLayout")
        isa_dropdown = ttk.Combobox(add_frame, textvariable=self.isa_type_var,
                                     values=["SetIVNLayout", "SetWVNLayout", "SetOVNLayout", "ExecuteMapping", "ExecuteStreaming"],
                                     state="readonly", width=18)
        isa_dropdown.pack(side=tk.LEFT, padx=5)
        ttk.Label(add_frame, text="(Select ISA type)", foreground='gray').pack(side=tk.LEFT, padx=5)

        # Row 2: Edit and Delete buttons
        edit_frame = ttk.Frame(btn_frame)
        edit_frame.pack(fill=tk.X, pady=3)

        ttk.Button(edit_frame, text="Edit", command=self._edit_instruction, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(edit_frame, text="Delete", command=self._delete_instruction, width=10).pack(side=tk.LEFT, padx=2)

        # Row 3: Move Up and Move Down buttons
        move_frame = ttk.Frame(btn_frame)
        move_frame.pack(fill=tk.X, pady=3)

        ttk.Button(move_frame, text="Move Up", command=self._move_instruction_up, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(move_frame, text="Move Down", command=self._move_instruction_down, width=10).pack(side=tk.LEFT, padx=2)

        # Row 4: Validate and Clear buttons
        action_frame = ttk.Frame(btn_frame)
        action_frame.pack(fill=tk.X, pady=3)

        ttk.Button(action_frame, text="Validate Trace", command=self._validate_trace, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Clear All", command=self._clear_all_instructions, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Reset Default", command=self._reset_default_instructions, width=12).pack(side=tk.LEFT, padx=2)

        # Instruction Details / Metrics
        detail_frame = ttk.LabelFrame(parent, text="Instruction Details", padding=10)
        detail_frame.pack(fill=tk.X, padx=5, pady=5)

        self.metrics_text = tk.Text(detail_frame, height=6, width=45, state=tk.DISABLED, wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.X)

        # Configure text tags for colored display
        self.metrics_text.tag_configure('header', font=('TkDefaultFont', 10, 'bold'))
        self.metrics_text.tag_configure('param', foreground='#0066CC')
        self.metrics_text.tag_configure('value', foreground='#006600')
        self.metrics_text.tag_configure('error', foreground='#CC0000')

        # Animation controls (in left panel for visibility on all screen sizes)
        anim_frame = ttk.LabelFrame(parent, text="Animation Controls", padding=5)
        anim_frame.pack(fill=tk.X, padx=5, pady=5)

        # Generate button and trace info
        ttk.Button(anim_frame, text="Generate Animation from ISA Trace",
                   command=self._generate_animation).pack(fill=tk.X, padx=5, pady=2)

        self.anim_info_var = tk.StringVar(value="No animation generated")
        ttk.Label(anim_frame, textvariable=self.anim_info_var,
                  foreground='gray').pack(fill=tk.X, padx=5, pady=2)

        # Playback controls
        ctrl_frame = ttk.Frame(anim_frame)
        ctrl_frame.pack(fill=tk.X, pady=5)

        ttk.Button(ctrl_frame, text="<<", width=3, command=lambda: self._step_animation(-10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="<", width=3, command=lambda: self._step_animation(-1)).pack(side=tk.LEFT, padx=2)
        self.play_btn = ttk.Button(ctrl_frame, text="Play", width=6, command=self._toggle_animation)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text=">", width=3, command=lambda: self._step_animation(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text=">>", width=3, command=lambda: self._step_animation(10)).pack(side=tk.LEFT, padx=2)

        # Speed and frame info row
        info_frame = ttk.Frame(anim_frame)
        info_frame.pack(fill=tk.X, pady=2)

        ttk.Label(info_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(info_frame, from_=0.1, to=5.0, variable=self.speed_var,
                  orient=tk.HORIZONTAL, length=80).pack(side=tk.LEFT)

        ttk.Label(info_frame, text="  Frame:").pack(side=tk.LEFT, padx=5)
        self.frame_var = tk.IntVar(value=0)
        self.frame_label = ttk.Label(info_frame, textvariable=self.frame_var, width=5)
        self.frame_label.pack(side=tk.LEFT)

        self.total_frames_var = tk.StringVar(value="/ 0")
        ttk.Label(info_frame, textvariable=self.total_frames_var).pack(side=tk.LEFT)

    def _create_visualization_panel(self, parent: ttk.Frame):
        """Create visualization panel"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # NEST tab
        nest_frame = ttk.Frame(notebook)
        notebook.add(nest_frame, text="NEST PE Array")
        self.nest_visualizer = NESTVisualizer(nest_frame, self.hw_config)

        # BIRRD tab
        birrd_frame = ttk.Frame(notebook)
        notebook.add(birrd_frame, text="BIRRD Network")
        self.birrd_visualizer = BIRRDVisualizer(birrd_frame, self.hw_config)

        # Buffer tab
        buffer_frame = ttk.Frame(notebook)
        notebook.add(buffer_frame, text="VN Buffers")
        self.buffer_visualizer = VNBufferVisualizer(buffer_frame, self.hw_config)


    def _create_default_instructions(self):
        """Create default ISA sequence from co_search result for GEMM {M,K,N}={16,12,8} on AH=AW=4.

        This is a real MINISA trace produced by co_search_layout_mapping().
        It demonstrates mixed G_r (two EM/ES pairs with different reduction-group widths):
          EM[0]: G_r=2 packs K-rows 0,1 (8 of 12 K-elements), T=16 streaming steps
          EM[1]: G_r=4 handles K-row 2 (last 4 K-elements) with full duplication, T=8
        Load/Store instructions are omitted (not visualized by the GUI).
        """
        self.instructions = [
            # 1. SetOVNLayout: output buffer layout (Mt=16, Nt=8, AH=4)
            ISAInstruction(ISAType.SetOVNLayout, {
                'order': 0,  # pL1 -> pL0 -> qL1
                'P_L0': 4,   # min(AW, Mt_eff) = 4
                'P_L1': 4,   # ceil(Mt_eff / P_L0) = 16/4 = 4
                'Q_L1': 2    # ceil(Nt_eff / AH) = 8/4 = 2
            }),
            # 2. SetIVNLayout: input buffer layout (Mt=16, Kt=12, AH=4)
            ISAInstruction(ISAType.SetIVNLayout, {
                'order': 5,  # mL1 -> mL0 -> jL1
                'M_L0': 4,   # min(AW, Mt_eff) = 4
                'M_L1': 4,   # ceil(Mt_eff / M_L0) = 16/4 = 4
                'J_L1': 3    # ceil(Kt / AH) = 12/4 = 3 K-groups
            }),
            # 3. SetWVNLayout: weight buffer layout (Nt=8, Kt=12, AH=4)
            ISAInstruction(ISAType.SetWVNLayout, {
                'order': 2,  # nL0 -> kL1 -> nL1
                'N_L0': 4,   # min(AW, Nt_eff) = 4
                'N_L1': 2,   # ceil(Nt_eff / N_L0) = 8/4 = 2
                'K_L1': 3    # ceil(Kt / AH) = 12/4 = 3 K-groups
            }),
            # 4. EM[0]: K-rows 0,1 packed (mixed G_r), all 16 M-rows streamed
            ISAInstruction(ISAType.ExecuteMapping, {
                'r0': 0,   # Starting WVN row index (K-group 0)
                'c0': 0,   # Starting WVN column index
                'Gr': 2,   # 2 PE columns per K-group (packs 2 K-rows)
                'Gc': 2,   # Replication period = ceil(Nt/AH) = 2
                'sr': 1,   # Temporal stride per PE row
                'sc': 4    # WVN-column spacing between N-subgroups (= AH)
            }),
            # 5. ES[0]: Stream all 16 M-rows, stride 1 (no duplication for this batch)
            ISAInstruction(ISAType.ExecuteStreaming, {
                'dataflow': 1,      # WO-S (weight-output stationary)
                'm_0': 0,           # Base streaming row index
                's_m': 1,           # Streaming row stride
                'T': 16,            # 16 streaming steps (all M-rows)
                'vn_size': 3        # Active VN height (AH-1 = 3)
            }),
            # 6. EM[1]: K-row 2 with full duplication (G_r=4 = AW)
            ISAInstruction(ISAType.ExecuteMapping, {
                'r0': 2,   # Starting WVN row index (K-group 2)
                'c0': 0,   # Starting WVN column index
                'Gr': 4,   # All 4 PE columns share same K-group (full dup)
                'Gc': 2,   # Replication period = 2
                'sr': 1,   # Temporal stride per PE row
                'sc': 4    # WVN-column spacing (= AH)
            }),
            # 7. ES[1]: Stream 8 steps with stride 2 (2 replicas, interleaved)
            ISAInstruction(ISAType.ExecuteStreaming, {
                'dataflow': 1,      # WO-S
                'm_0': 0,           # Base streaming row index
                's_m': 2,           # Stride 2 (interleaved across 2 replicas)
                'T': 8,             # ceil(16/2) = 8 streaming steps
                'vn_size': 3        # AH-1 = 3
            }),
        ]
        self._update_isa_listbox()

    def _update_isa_listbox(self):
        """Update ISA listbox with detailed instruction descriptions"""
        self.isa_listbox.delete(0, tk.END)
        for i, instr in enumerate(self.instructions):
            # Use short description for listbox, full description for tooltips
            self.isa_listbox.insert(tk.END, f"{i+1}. {instr.get_short_description()}")

            # Color code by instruction type
            if instr.isa_type == ISAType.SetIVNLayout:
                self.isa_listbox.itemconfig(i, {'bg': '#E3F2FD'})  # Light blue (IVN = streaming)
            elif instr.isa_type == ISAType.SetWVNLayout:
                self.isa_listbox.itemconfig(i, {'bg': '#E8F5E9'})  # Light green (WVN = stationary)
            elif instr.isa_type == ISAType.SetOVNLayout:
                self.isa_listbox.itemconfig(i, {'bg': '#FFEBEE'})  # Light red (OVN = output)
            elif instr.isa_type == ISAType.ExecuteMapping:
                self.isa_listbox.itemconfig(i, {'bg': '#FFF8E1'})  # Light amber for mapping
            elif instr.isa_type == ISAType.ExecuteStreaming:
                self.isa_listbox.itemconfig(i, {'bg': '#FFF8E1'})  # Light amber for streaming

        # Update buffer layout visualization from the current ISA trace
        self.buffer_visualizer.update_layout_from_instructions(self.instructions)

        # Derive M, K, N from the Set*VNLayout instructions
        self._derive_mkn_from_instructions()

    def _derive_mkn_from_instructions(self, up_to_idx: int = -1):
        """Derive workload M, K, N from the latest Set*VNLayout instructions.

        Scans instructions (up to up_to_idx, or all if -1) and extracts
        tile dimensions from the most recent SetIVNLayout, SetWVNLayout,
        and SetOVNLayout:
          - M = M_L0 * M_L1   (from SetIVNLayout) or P_L0 * P_L1 (from SetOVNLayout)
          - K = J_L1 * AH      (from SetIVNLayout) or K_L1 * AH   (from SetWVNLayout)
          - N = N_L0 * N_L1   (from SetWVNLayout) or Q_L1 * AH   (from SetOVNLayout)
        """
        AH = self.hw_config.AH
        instrs = self.instructions
        if up_to_idx >= 0:
            instrs = instrs[:up_to_idx + 1]

        M, K, N = None, None, None

        # Scan in order so the last Set*VNLayout wins
        for instr in instrs:
            if instr.isa_type == ISAType.SetIVNLayout:
                m_l0 = instr.params.get('M_L0', 1)
                m_l1 = instr.params.get('M_L1', 1)
                j_l1 = instr.params.get('J_L1', 1)
                M = m_l0 * m_l1
                K = j_l1 * AH
            elif instr.isa_type == ISAType.SetWVNLayout:
                n_l0 = instr.params.get('N_L0', 1)
                n_l1 = instr.params.get('N_L1', 1)
                k_l1 = instr.params.get('K_L1', 1)
                N = n_l0 * n_l1
                K = k_l1 * AH  # K from WVN overwrites IVN's K if seen later
            elif instr.isa_type == ISAType.SetOVNLayout:
                p_l0 = instr.params.get('P_L0', 1)
                p_l1 = instr.params.get('P_L1', 1)
                q_l1 = instr.params.get('Q_L1', 1)
                if M is None:
                    M = p_l0 * p_l1
                if N is None:
                    N = q_l1 * AH

        if M is not None:
            self.m_var.set(M)
        if K is not None:
            self.k_var.set(K)
        if N is not None:
            self.n_var.set(N)

        # Sync to workload_config
        self.workload_config.M = self.m_var.get()
        self.workload_config.K = self.k_var.get()
        self.workload_config.N = self.n_var.get()

    def _on_nest_size_change(self, event=None):
        """Handle NEST size change"""
        size_str = self.nest_size_var.get()
        size = int(size_str.split('x')[0])
        self.hw_config.AH = size
        self.hw_config.AW = size
        self.nest_visualizer.update_config(self.hw_config)
        self.birrd_visualizer.update_config(self.hw_config)
        self.buffer_visualizer.update_config(self.hw_config)

    def _on_sram_change(self, *args):
        """Handle SRAM size change — update buffer capacity visualization"""
        try:
            self.hw_config.sram_mb = self.sram_var.get()
            if hasattr(self, 'buffer_visualizer'):
                self.buffer_visualizer.update_config(self.hw_config)
        except (tk.TclError, ValueError):
            pass  # ignore transient parse errors while typing

    def _generate_trace(self):
        """Generate trace"""
        if not HAS_MINISA:
            messagebox.showwarning("Warning", "MINISA system not available")
            return

        try:
            M = self.m_var.get()
            K = self.k_var.get()
            N = self.n_var.get()

            cfg = FeatherPlusConfig(nest=self.hw_config.AH, total_sram_mb=self.sram_var.get())
            self.current_trace = generate_trace_gemm(M, K, N, cfg)

            self._update_metrics()
            messagebox.showinfo("Success", f"Generated {len(self.current_trace.trace)} instructions")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _generate_animation(self):
        """Generate animation frames based on current ISA trace.

        Extracts parameters from the ISA trace to determine:
        - Number of weight VNs from SetWVNLayout (N_L1 * K_L1)
        - Number of input VNs from SetIVNLayout (M_L1 * J_L1)
        - Number of output rows from SetOVNLayout (P_L1)
        - Mapping parameters from ExecuteMapping instructions
        """
        if not self.instructions:
            messagebox.showwarning("No Trace",
                                   "No ISA instructions in trace.\n"
                                   "Please add instructions or use 'Reset Default' first.")
            return

        # Validate trace first
        self.workload_config.M = self.m_var.get()
        self.workload_config.K = self.k_var.get()
        self.workload_config.N = self.n_var.get()

        validator = ISAConfigValidator(self.hw_config, self.workload_config)
        is_valid, errors = validator.validate_trace_sequence(self.instructions)

        if not is_valid:
            # Show warning but allow generation
            error_summary = "\n".join(f"• {e}" for e in errors[:5])
            if len(errors) > 5:
                error_summary += f"\n... and {len(errors) - 5} more errors"
            if not messagebox.askyesno("Validation Warning",
                                       f"ISA trace has errors:\n\n{error_summary}\n\n"
                                       f"Generate animation anyway?"):
                return

        # Extract parameters from ISA trace
        num_weight_vns = 1
        num_input_vns = self.hw_config.AH
        num_output_rows = self.hw_config.AH
        num_mappings = 0

        # ExecuteStreaming parameters (per EM/ES pair)
        es_params_list = []  # List of ES param dicts, one per EM/ES pair
        current_es_params = None

        for instr in self.instructions:
            if instr.isa_type == ISAType.SetWVNLayout:
                # Weight VNs = N_L1 * K_L1
                n_l1 = instr.params.get('N_L1', 1)
                k_l1 = instr.params.get('K_L1', 1)
                num_weight_vns = max(1, n_l1 * k_l1)

            elif instr.isa_type == ISAType.SetIVNLayout:
                # Input VNs = M_L1 * J_L1 (determines streaming cycles)
                m_l1 = instr.params.get('M_L1', 1)
                j_l1 = instr.params.get('J_L1', 1)
                num_input_vns = max(self.hw_config.AH, m_l1 * j_l1)

            elif instr.isa_type == ISAType.SetOVNLayout:
                # Output rows = P_L1
                p_l1 = instr.params.get('P_L1', 1)
                num_output_rows = max(1, p_l1)

            elif instr.isa_type == ISAType.ExecuteMapping:
                num_mappings += 1

            elif instr.isa_type == ISAType.ExecuteStreaming:
                # Extract ES parameters for the preceding EM
                current_es_params = {
                    'T': instr.params.get('T', self.hw_config.AH),
                    'vn_size': instr.params.get('vn_size', self.hw_config.AH - 1),
                    's_m': instr.params.get('s_m', 1),
                    'm_0': instr.params.get('m_0', 0),
                    'dataflow': instr.params.get('dataflow', 1),
                }
                es_params_list.append(current_es_params)

        # Generate animation
        self.animation_generator = AccurateAnimationGenerator(
            AH=self.hw_config.AH, AW=self.hw_config.AW
        )

        # Generate frames for each mapping instruction (or at least one iteration)
        # Inter-EM pipelining: WVN loading for the next EM overlaps with the
        # current EM's IVN streaming phase. This reduces the total cycle count.
        AH = self.hw_config.AH
        all_frames = []
        iterations = max(1, num_mappings)
        prev_em_start = 0
        prev_wvn_preloaded = 0

        for iteration in range(iterations):
            # Use ES parameters if available for this iteration
            es_p = es_params_list[iteration] if iteration < len(es_params_list) else None
            iter_num_inputs = num_input_vns
            iter_vn_size = self.hw_config.AH
            iter_s_m = 1
            iter_m_0 = 0

            if es_p is not None:
                # ES T field overrides num_input_vns for streaming steps
                iter_num_inputs = max(self.hw_config.AH, es_p['T'])
                # vn_size is encoded as AH-1; active height = vn_size + 1
                iter_vn_size = es_p['vn_size'] + 1
                iter_s_m = es_p['s_m']
                iter_m_0 = es_p['m_0']

            # Calculate how many WVN rows were pre-loaded during previous EM
            wvn_preloaded = 0
            if iteration > 0 and all_frames:
                # Previous EM's streaming phase duration (frames after input_start)
                # provides time to pre-load WVN rows for this EM.
                prev_remaining_rows = AH - prev_wvn_preloaded
                prev_input_start = max(0, (prev_remaining_rows - 1) * AH)
                prev_streaming_duration = all_frames[-1].cycle - (prev_em_start + prev_input_start) + 1
                # Each row takes AH cycles to load; how many can fit in the overlap?
                wvn_preloaded = min(AH, max(0, prev_streaming_duration // AH))

            frames = self.animation_generator.generate_full_iteration(
                num_weight_vns=num_weight_vns,
                num_input_vns=iter_num_inputs,
                num_output_rows=num_output_rows,
                vn_size=iter_vn_size,
                s_m=iter_s_m,
                m_0=iter_m_0,
                wvn_preloaded_rows=wvn_preloaded,
            )

            # Adjust cycle numbers for multi-iteration
            if iteration > 0:
                offset = all_frames[-1].cycle + 1 if all_frames else 0
                for frame in frames:
                    frame.cycle += offset

            # Annotate the previous EM's streaming frames with WVN prefetch info
            if iteration > 0 and wvn_preloaded > 0 and all_frames:
                prev_remaining_rows = AH - prev_wvn_preloaded
                prev_input_start_cycle = prev_em_start + max(0, (prev_remaining_rows - 1) * AH)
                for f in all_frames:
                    if f.cycle >= prev_input_start_cycle:
                        # Which WVN row is being pre-loaded at this cycle?
                        cycles_into_prefetch = f.cycle - prev_input_start_cycle
                        prefetch_row = cycles_into_prefetch // AH
                        if prefetch_row < wvn_preloaded:
                            prefetch_elem = cycles_into_prefetch % AH
                            f.phase += f" | WVN(next)→R{prefetch_row}[{prefetch_elem}]"

            # Track EM start info for next iteration's overlap calculation
            prev_em_start = frames[0].cycle if frames else 0
            prev_wvn_preloaded = wvn_preloaded

            all_frames.extend(frames)

        self.animation_frames = all_frames
        self.current_frame_idx = 0
        self._update_animation_display()

        # Show summary
        vn_size_info = ""
        if es_params_list:
            vn_sizes = [p['vn_size'] + 1 for p in es_params_list]
            if any(v < self.hw_config.AH for v in vn_sizes):
                vn_size_info = f"\n  • Active VN heights: {vn_sizes}"
        pipeline_info = ""
        if iterations > 1:
            pipeline_info = f"\n  • Inter-EM pipelining: WVN prefetch during IVN streaming"
        info_text = (f"Generated {len(self.animation_frames)} animation frames\n\n"
                     f"From ISA Trace:\n"
                     f"  • Weight VNs: {num_weight_vns}\n"
                     f"  • Input VNs: {num_input_vns}\n"
                     f"  • Output rows: {num_output_rows}\n"
                     f"  • Mapping iterations: {iterations}"
                     f"{vn_size_info}{pipeline_info}\n\n"
                     f"Hardware: {self.hw_config.AH}×{self.hw_config.AW} NEST")
        messagebox.showinfo("Animation Generated", info_text)

    def _step_animation(self, delta: int):
        """Step animation"""
        if not self.animation_frames:
            return
        self.current_frame_idx = max(0, min(len(self.animation_frames) - 1, self.current_frame_idx + delta))
        self._update_animation_display()

    def _toggle_animation(self):
        """Toggle play/pause"""
        self.animation_running = not self.animation_running
        self.play_btn.config(text="Pause" if self.animation_running else "Play")

        if self.animation_running:
            self._run_animation()

    def _run_animation(self):
        """Run animation loop"""
        if not self.animation_running or not self.animation_frames:
            return

        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.animation_frames)
        self._update_animation_display()

        delay = int(1000 / self.speed_var.get())
        self.root.after(delay, self._run_animation)

    def _update_animation_display(self):
        """Update display for current frame"""
        if not self.animation_frames:
            self.anim_info_var.set("No animation generated")
            self.total_frames_var.set("/ 0")
            return

        frame = self.animation_frames[self.current_frame_idx]
        self.frame_var.set(self.current_frame_idx)
        self.total_frames_var.set(f"/ {len(self.animation_frames) - 1}")
        self.anim_info_var.set(f"Cycle {frame.cycle}: {frame.phase[:40]}...")

        self.nest_visualizer.update_from_frame(frame)

        # Update BIRRD visualization
        if frame.birrd_active_stage >= 0:
            # Set EGG configurations if available
            if frame.birrd_egg_configs:
                self.birrd_visualizer.set_all_egg_configs(frame.birrd_egg_configs)
            self.birrd_visualizer.highlight_active_stage(frame.birrd_active_stage)
        else:
            # Reset BIRRD to default state when not active
            self.birrd_visualizer.reset_display()

    def _on_instruction_select(self, event=None):
        """Handle instruction selection - show details and update derived M/K/N"""
        sel = self.isa_listbox.curselection()
        if not sel:
            return

        idx = sel[0]
        if idx < len(self.instructions):
            instr = self.instructions[idx]
            # Update M/K/N to reflect state at this program point
            self._derive_mkn_from_instructions(up_to_idx=idx)
            self._show_instruction_details(instr, idx)

    def _show_instruction_details(self, instr: ISAInstruction, idx: int):
        """Show detailed information about selected instruction"""
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)

        # Header
        self.metrics_text.insert(tk.END, f"Instruction {idx+1}: ", 'header')
        self.metrics_text.insert(tk.END, f"{instr.isa_type.value}\n\n")

        # Parameters
        self.metrics_text.insert(tk.END, "Parameters:\n", 'header')
        for key, value in instr.params.items():
            self.metrics_text.insert(tk.END, f"  {key}: ", 'param')
            self.metrics_text.insert(tk.END, f"{value}\n", 'value')

        # Show derived workload at this program point
        M = self.workload_config.M
        K = self.workload_config.K
        N = self.workload_config.N
        self.metrics_text.insert(tk.END, f"\nDerived GEMM: ", 'header')
        self.metrics_text.insert(tk.END, f"M={M}, K={K}, N={N}\n", 'value')

        # Order permutation description
        if instr.isa_type in [ISAType.SetIVNLayout, ISAType.SetWVNLayout, ISAType.SetOVNLayout]:
            order = instr.params.get('order', 0)
            operand_map = {
                ISAType.SetIVNLayout: 'I',
                ISAType.SetWVNLayout: 'W',
                ISAType.SetOVNLayout: 'O'
            }
            operand = operand_map.get(instr.isa_type, 'I')
            if order in ORDER_PERMUTATIONS:
                perm = ORDER_PERMUTATIONS[order][operand]
                self.metrics_text.insert(tk.END, f"\nLoop Order: ", 'header')
                self.metrics_text.insert(tk.END, f"{perm}\n", 'value')

        # Quick validation
        validator = ISAConfigValidator(self.hw_config, self.workload_config)
        is_valid, errors = validator.validate_instruction(instr)

        self.metrics_text.insert(tk.END, "\nValidation: ", 'header')
        if is_valid:
            self.metrics_text.insert(tk.END, "OK\n", 'value')
        else:
            self.metrics_text.insert(tk.END, "ERRORS\n", 'error')
            for err in errors[:3]:  # Show first 3 errors
                self.metrics_text.insert(tk.END, f"  • {err}\n", 'error')

        self.metrics_text.config(state=tk.DISABLED)

    def _update_metrics(self):
        """Update metrics display with trace summary"""
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)

        # Show trace summary
        self.metrics_text.insert(tk.END, "Trace Summary\n", 'header')
        self.metrics_text.insert(tk.END, f"Instructions: {len(self.instructions)}\n")
        self.metrics_text.insert(tk.END, f"NEST: {self.hw_config.AH}x{self.hw_config.AW}\n")
        self.metrics_text.insert(tk.END, f"SRAM: {self.sram_var.get()} MB\n\n")

        # Count by type
        type_counts = {}
        for instr in self.instructions:
            t = instr.isa_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        if type_counts:
            self.metrics_text.insert(tk.END, "By Type:\n", 'header')
            for t, count in type_counts.items():
                self.metrics_text.insert(tk.END, f"  {t}: ", 'param')
                self.metrics_text.insert(tk.END, f"{count}\n", 'value')

        if self.current_trace:
            self.metrics_text.insert(tk.END, f"\nGenerated trace: {len(self.current_trace.trace)} instrs\n")

        self.metrics_text.config(state=tk.DISABLED)

    def _add_instruction_from_dropdown(self):
        """Add instruction using the ISA type selected in dropdown"""
        isa_type_str = self.isa_type_var.get()
        isa_type_map = {
            "SetIVNLayout": ISAType.SetIVNLayout,
            "SetWVNLayout": ISAType.SetWVNLayout,
            "SetOVNLayout": ISAType.SetOVNLayout,
            "ExecuteMapping": ISAType.ExecuteMapping,
            "ExecuteStreaming": ISAType.ExecuteStreaming
        }
        isa_type = isa_type_map.get(isa_type_str)
        if isa_type:
            self._add_instruction(isa_type)

    def _add_instruction(self, isa_type: ISAType = None):
        """Add a new ISA instruction with configuration dialog"""
        if isa_type is None:
            messagebox.showwarning("Select Type", "Please select an ISA type from the dropdown")
            return

        # Update workload config from UI
        self.workload_config.M = self.m_var.get()
        self.workload_config.K = self.k_var.get()
        self.workload_config.N = self.n_var.get()

        # Open configuration dialog
        dialog = ISAConfigDialog(
            self.root, isa_type, self.hw_config, self.workload_config,
            title=f"Add {isa_type.value}"
        )
        self.root.wait_window(dialog)

        if dialog.result is not None:
            # Create new instruction and add to list
            new_instr = ISAInstruction(isa_type, dialog.result)

            # Insert at selected position or at end
            sel = self.isa_listbox.curselection()
            if sel:
                insert_idx = sel[0] + 1
                self.instructions.insert(insert_idx, new_instr)
            else:
                self.instructions.append(new_instr)

            self._update_isa_listbox()

    def _edit_instruction(self):
        """Edit selected ISA instruction"""
        sel = self.isa_listbox.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Please select an instruction to edit")
            return

        idx = sel[0]
        instr = self.instructions[idx]

        # Update workload config from UI
        self.workload_config.M = self.m_var.get()
        self.workload_config.K = self.k_var.get()
        self.workload_config.N = self.n_var.get()

        # Open configuration dialog with existing parameters
        dialog = ISAConfigDialog(
            self.root, instr.isa_type, self.hw_config, self.workload_config,
            existing_params=instr.params.copy(),
            title=f"Edit {instr.isa_type.value}"
        )
        self.root.wait_window(dialog)

        if dialog.result is not None:
            # Update instruction with new parameters
            self.instructions[idx] = ISAInstruction(instr.isa_type, dialog.result)
            self._update_isa_listbox()
            self.isa_listbox.selection_set(idx)

    def _delete_instruction(self):
        """Delete selected instruction with confirmation"""
        sel = self.isa_listbox.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Please select an instruction to delete")
            return

        idx = sel[0]
        instr = self.instructions[idx]

        if messagebox.askyesno("Confirm Delete",
                               f"Delete instruction {idx+1}?\n\n{instr.get_description()}"):
            del self.instructions[idx]
            self._update_isa_listbox()

            # Select next item if possible
            if self.instructions:
                new_idx = min(idx, len(self.instructions) - 1)
                self.isa_listbox.selection_set(new_idx)

    def _move_instruction_up(self):
        """Move selected instruction up in the trace"""
        sel = self.isa_listbox.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Please select an instruction to move")
            return

        idx = sel[0]
        if idx == 0:
            return  # Already at top

        # Swap with previous instruction
        self.instructions[idx], self.instructions[idx - 1] = \
            self.instructions[idx - 1], self.instructions[idx]
        self._update_isa_listbox()
        self.isa_listbox.selection_set(idx - 1)

    def _move_instruction_down(self):
        """Move selected instruction down in the trace"""
        sel = self.isa_listbox.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Please select an instruction to move")
            return

        idx = sel[0]
        if idx >= len(self.instructions) - 1:
            return  # Already at bottom

        # Swap with next instruction
        self.instructions[idx], self.instructions[idx + 1] = \
            self.instructions[idx + 1], self.instructions[idx]
        self._update_isa_listbox()
        self.isa_listbox.selection_set(idx + 1)

    def _validate_trace(self):
        """Validate the entire ISA trace sequence"""
        if not self.instructions:
            messagebox.showinfo("Validation", "No instructions to validate")
            return

        # Update workload config from UI
        self.workload_config.M = self.m_var.get()
        self.workload_config.K = self.k_var.get()
        self.workload_config.N = self.n_var.get()

        validator = ISAConfigValidator(self.hw_config, self.workload_config)
        is_valid, errors = validator.validate_trace_sequence(self.instructions)

        if is_valid:
            messagebox.showinfo("Validation Passed",
                               f"ISA trace is valid!\n\n"
                               f"Total instructions: {len(self.instructions)}\n"
                               f"Hardware: {self.hw_config.AH}x{self.hw_config.AW} NEST\n"
                               f"Workload: GEMM({self.workload_config.M}x{self.workload_config.K}) @ "
                               f"({self.workload_config.K}x{self.workload_config.N})")
        else:
            # Show error dialog with details
            self._show_validation_errors(errors)

    def _show_validation_errors(self, errors: List[str]):
        """Show validation errors in a dialog with detailed information"""
        error_window = tk.Toplevel(self.root)
        error_window.title("Validation Errors")
        error_window.geometry("600x400")
        error_window.transient(self.root)
        error_window.grab_set()

        # Center on parent
        error_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 600) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 400) // 2
        error_window.geometry(f"+{x}+{y}")

        main_frame = ttk.Frame(error_window, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Configuration Errors Detected",
                  font=('TkDefaultFont', 12, 'bold'), foreground='red').pack(pady=(0, 10))

        ttk.Label(main_frame, text=f"Found {len(errors)} error(s) in the ISA trace:").pack(anchor=tk.W)

        # Scrollable error list
        error_frame = ttk.Frame(main_frame)
        error_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        scrollbar = ttk.Scrollbar(error_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        error_text = tk.Text(error_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                            height=15, font=('TkDefaultFont', 10))
        error_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=error_text.yview)

        for i, error in enumerate(errors, 1):
            error_text.insert(tk.END, f"{i}. {error}\n\n")

        error_text.config(state=tk.DISABLED)

        # Close button
        ttk.Button(main_frame, text="Close", command=error_window.destroy, width=10).pack(pady=10)

    def _clear_all_instructions(self):
        """Clear all instructions from the trace"""
        if not self.instructions:
            return

        if messagebox.askyesno("Confirm Clear",
                               f"Clear all {len(self.instructions)} instructions from the trace?"):
            self.instructions.clear()
            self._update_isa_listbox()

    def _reset_default_instructions(self):
        """Reset to default ISA sequence"""
        if self.instructions:
            if not messagebox.askyesno("Confirm Reset",
                                       "Reset to default ISA sequence?\nThis will replace the current trace."):
                return

        self._create_default_instructions()
        messagebox.showinfo("Reset", "ISA trace reset to default sequence")

    def _load_trace(self):
        """Load ISA trace from JSON file"""
        filepath = filedialog.askopenfilename(
            title="Load ISA Trace",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            defaultextension=".json"
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Load instructions
            self.instructions.clear()
            for instr_data in data.get('instructions', []):
                isa_type = ISAType(instr_data['type'])
                params = instr_data.get('params', {})
                self.instructions.append(ISAInstruction(isa_type, params))

            # Load hardware config if present
            if 'hardware' in data:
                hw = data['hardware']
                self.hw_config.AH = hw.get('AH', self.hw_config.AH)
                self.hw_config.AW = hw.get('AW', self.hw_config.AW)
                self.hw_config.sram_mb = hw.get('sram_mb', self.hw_config.sram_mb)
                self.nest_size_var.set(f"{self.hw_config.AH}x{self.hw_config.AW}")
                self.sram_var.set(self.hw_config.sram_mb)
                self._on_nest_size_change()

            # Load workload config if present
            if 'workload' in data:
                wl = data['workload']
                self.m_var.set(wl.get('M', 16))
                self.k_var.set(wl.get('K', 16))
                self.n_var.set(wl.get('N', 16))

            self._update_isa_listbox()
            messagebox.showinfo("Load Success", f"Loaded {len(self.instructions)} instructions from:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load trace:\n{str(e)}")

    def _save_trace(self):
        """Save ISA trace to JSON file"""
        if not self.instructions:
            messagebox.showwarning("No Trace", "No instructions to save")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save ISA Trace",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            defaultextension=".json"
        )
        if not filepath:
            return

        try:
            # Prepare data
            data = {
                'version': '1.0',
                'description': 'MINISA ISA Trace',
                'hardware': {
                    'AH': self.hw_config.AH,
                    'AW': self.hw_config.AW,
                    'sram_mb': self.sram_var.get()
                },
                'workload': {
                    'M': self.m_var.get(),
                    'K': self.k_var.get(),
                    'N': self.n_var.get()
                },
                'instructions': [
                    {
                        'type': instr.isa_type.value,
                        'params': instr.params
                    }
                    for instr in self.instructions
                ]
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            messagebox.showinfo("Save Success", f"Saved {len(self.instructions)} instructions to:\n{filepath}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save trace:\n{str(e)}")

    def _reset_view(self):
        """Reset view"""
        self.nest_visualizer.reset_display()
        self.birrd_visualizer._draw_birrd_structure()

    def _run_tests(self):
        """Run verification tests"""
        run_all_tests()
        messagebox.showinfo("Tests", "Tests completed. Check console output.")

    def _show_about(self):
        """Show about dialog"""
        about_window = tk.Toplevel(self.root)
        about_window.title("About MINISA GUI")
        about_window.geometry("550x500")
        about_window.transient(self.root)
        about_window.grab_set()

        # Center on parent
        about_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 550) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 500) // 2
        about_window.geometry(f"+{x}+{y}")

        main_frame = ttk.Frame(about_window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="MINISA GUI",
                  font=('TkDefaultFont', 16, 'bold')).pack(pady=(0, 5))
        ttk.Label(main_frame, text="Interactive ISA Trace Editor for FEATHER+ Accelerator",
                  font=('TkDefaultFont', 10)).pack(pady=(0, 15))

        info_text = """MINISA (Minimal Instruction Set Architecture) is a VN-level
abstraction for programming FEATHER+ reconfigurable accelerator.

ISA Instructions:
• SetIVNLayout - Configure Input Virtual Neuron buffer layout
• SetWVNLayout - Configure Weight Virtual Neuron buffer layout
• SetOVNLayout - Configure Output Virtual Neuron buffer layout
• ExecuteMapping - Load WVN into NEST (paired with ExecuteStreaming)
• ExecuteStreaming - Stream inputs through NEST, trigger compute

Key Features:
• Configure all reconfigurable ISA knobs interactively
• Add, edit, delete, and reorder ISA trace instructions
• Real-time validation with error detection and reporting
• Save/Load ISA traces to JSON files
• Visualize NEST PE array and BIRRD reduction network

Reconfigurable Knobs (per instruction):
• Order: 3-bit permutation selector (0-5)
• Layout dimensions: L0 [1, AW], L1 parameters
• Mapping: r0, c0, Gr, Gc, sr, sc

Based on FEATHER+ RTL implementation and MINISA paper.
"""
        text_widget = tk.Text(main_frame, wrap=tk.WORD, height=18, width=60)
        text_widget.pack(fill=tk.BOTH, expand=True, pady=10)
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)

        ttk.Button(main_frame, text="Close", command=about_window.destroy, width=10).pack(pady=10)


# ============================================================================
# Demo Functions (from minisa_animation.py)
# ============================================================================

def demo_nest_animation():
    """Demo NEST animation"""
    print("NEST Animation Demo")
    print("=" * 50)

    gen = AccurateAnimationGenerator(AH=4, AW=4)
    frames = gen.generate_full_iteration(num_input_vns=4, num_output_rows=4)

    print(f"Generated {len(frames)} frames")

    for frame in frames[:10]:
        print(f"Cycle {frame.cycle}: {frame.phase}")

def demo_executemapping_animation():
    """Demo ExecuteMapping + ExecuteStreaming animation.

    In ISA 2.0, ExecuteMapping loads WVN into NEST and ExecuteStreaming
    streams IVN through NEST to trigger computation (EM->ES pairing).
    """
    demo_nest_animation()

def demo_layout_animation():
    """Demo layout animation (compatibility)"""
    print("Layout Animation Demo")
    print("=" * 50)
    print("Buffer layout visualization available in GUI")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    root = tk.Tk()
    app = MINISAGui(root)
    root.mainloop()

if __name__ == "__main__":
    main()
