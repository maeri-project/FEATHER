#!/usr/bin/env python3
"""Layout specifications: LayoutSpec, TABLE_II, choose_layout_*, choose_tile_sizes,
output-buffer port conflict detection, and inter-layer layout matching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import (
    FeatherPlusConfig, ceil_div, ceil_log2,
    canonical_n_ivn, canonical_n_wvn, canonical_n_ovn,
)


TABLE_II_OUTER_TO_INNER: Dict[int, Dict[str, Tuple[str, str, str]]] = {
    0: {"W": ("kL1", "nL0", "nL1"), "I": ("jL1", "mL0", "mL1"), "O": ("pL1", "pL0", "qL1")},
    1: {"W": ("kL1", "nL1", "nL0"), "I": ("jL1", "mL1", "mL0"), "O": ("pL1", "qL1", "pL0")},
    2: {"W": ("nL0", "kL1", "nL1"), "I": ("mL0", "jL1", "mL1"), "O": ("pL0", "pL1", "qL1")},
    3: {"W": ("nL0", "nL1", "kL1"), "I": ("mL0", "mL1", "jL1"), "O": ("pL0", "qL1", "pL1")},
    4: {"W": ("nL1", "kL1", "nL0"), "I": ("mL1", "jL1", "mL0"), "O": ("qL1", "pL1", "pL0")},
    5: {"W": ("nL1", "nL0", "kL1"), "I": ("mL1", "mL0", "jL1"), "O": ("qL1", "pL0", "pL1")},
}

# OVN→IVN order mapping for inter-layer ping-pong swap.
#
# After computing layer i, the output buffer (OVN) is swapped into the
# streaming buffer (IVN) for layer i+1.  For the physical addresses to
# match across the swap, the OVN and IVN permutation orders must produce
# identical traversal patterns over the shared dimensions:
#
#   OVN dims  →  IVN dims
#   pL0       →  mL0        (M-dimension bank factor)
#   pL1       →  mL1        (M-dimension row factor)
#   qL1       →  jL1        (K/N-dimension factor)
#
# Substituting these into each OVN permutation and matching to the IVN
# table yields:  ivn_order = 5 - ovn_order.
#
#   OVN order 0 (pL1,pL0,qL1) ↔ IVN order 5 (mL1,mL0,jL1)
#   OVN order 1 (pL1,qL1,pL0) ↔ IVN order 4 (mL1,jL1,mL0)
#   OVN order 2 (pL0,pL1,qL1) ↔ IVN order 3 (mL0,mL1,jL1)
#   OVN order 3 (pL0,qL1,pL1) ↔ IVN order 2 (mL0,jL1,mL1)
#   OVN order 4 (qL1,pL1,pL0) ↔ IVN order 1 (jL1,mL1,mL0)
#   OVN order 5 (qL1,pL0,pL1) ↔ IVN order 0 (jL1,mL0,mL1)
OVN_TO_IVN_ORDER: Dict[int, int] = {o: 5 - o for o in range(6)}
IVN_TO_OVN_ORDER: Dict[int, int] = {5 - o: o for o in range(6)}


@dataclass(frozen=True)
class LayoutSpec:
    operand: str
    order_id: int
    a0: int   # W: N_L0, I: M_L0, O: P_L0
    a1: int   # W: N_L1, I: M_L1, O: P_L1
    a2: int   # W: K_L1, I: J_L1, O: Q_L1
    AH: int
    AW: int

    def dims(self) -> Dict[str, int]:
        if self.operand == "W":
            return {"kL1": self.a2, "nL0": self.a0, "nL1": self.a1}
        if self.operand == "I":
            return {"jL1": self.a2, "mL0": self.a0, "mL1": self.a1}
        return {"pL0": self.a0, "pL1": self.a1, "qL1": self.a2}

    def vn_count(self) -> int:
        v = 1
        for k in self.dims().values():
            v *= int(k)
        return int(v)

    def linear_index(self, idxs: Dict[str, int]) -> int:
        perm = TABLE_II_OUTER_TO_INNER[self.order_id][self.operand]
        d = self.dims()
        i0, i1, i2 = int(idxs[perm[0]]), int(idxs[perm[1]]), int(idxs[perm[2]])
        D1, D2 = int(d[perm[1]]), int(d[perm[2]])
        return i0 * (D1 * D2) + i1 * D2 + i2


def choose_layout_W(cfg: FeatherPlusConfig, Kt: int, Nt: int, order_id: int = 0) -> LayoutSpec:
    AH, AW = cfg.AH, cfg.AW
    N_L0 = min(AW, max(1, Nt))
    N_L1 = int(math.ceil(Nt / N_L0))
    K_L1 = int(math.ceil(Kt / AH))
    spec = LayoutSpec("W", order_id, N_L0, N_L1, K_L1, AH, AW)
    expected = canonical_n_wvn(Kt, Nt, AH)
    actual = spec.vn_count()
    assert actual >= expected, (
        f"WVN layout vn_count ({actual}) < canonical_n_wvn ({expected}) "
        f"for Kt={Kt}, Nt={Nt}, AH={AH}")
    return spec


def choose_layout_I(cfg: FeatherPlusConfig, Mt: int, Kt: int, order_id: int = 0) -> LayoutSpec:
    AH, AW = cfg.AH, cfg.AW
    M_L0 = min(AW, max(1, Mt))
    M_L1 = int(math.ceil(Mt / M_L0))
    J_L1 = int(math.ceil(Kt / AH))
    spec = LayoutSpec("I", order_id, M_L0, M_L1, J_L1, AH, AW)
    expected = canonical_n_ivn(Mt, Kt, AH)
    actual = spec.vn_count()
    assert actual >= expected, (
        f"IVN layout vn_count ({actual}) < canonical_n_ivn ({expected}) "
        f"for Mt={Mt}, Kt={Kt}, AH={AH}")
    return spec


def choose_layout_O(cfg: FeatherPlusConfig, Mt: int, Nt: int, order_id: int = 0) -> LayoutSpec:
    AH, AW = cfg.AH, cfg.AW
    P_L0 = min(AW, max(1, Mt))
    P_L1 = int(math.ceil(Mt / P_L0))
    Q_L1 = int(math.ceil(Nt / AH))
    spec = LayoutSpec("O", order_id, P_L0, P_L1, Q_L1, AH, AW)
    expected = canonical_n_ovn(Mt, Nt, AH)
    actual = spec.vn_count()
    assert actual >= expected, (
        f"OVN layout vn_count ({actual}) < canonical_n_ovn ({expected}) "
        f"for Mt={Mt}, Nt={Nt}, AH={AH}")
    return spec


def choose_tile_sizes(M: int, K: int, N: int, cfg: FeatherPlusConfig) -> Tuple[int, int, int]:
    """Re-exported from vn.py for backward compatibility."""
    from .vn import choose_tile_sizes as _cts
    return _cts(M, K, N, cfg)


# ---------------------------------------------------------------------------
# Output-buffer port conflict detection (Step 4)
# ---------------------------------------------------------------------------

def check_ob_port_conflict(
    layout_o: LayoutSpec,
    cfg: FeatherPlusConfig,
    exec_params: List,
    combined_columns: List,
) -> bool:
    """Check output-buffer bank conflicts by simulating per-cycle OVN writes.

    The output buffer (OB) is organized as a D x AW buffer (D rows, AW
    column-banks).  At each streaming step within an ExecuteMapping, all
    AW PE columns simultaneously produce one BIRRD output each and write
    to the OB.  PE column a_w writes to OVN(p, q) where:
        p = m  (output row from the column's IVN sequence at this step)
        q = vn_subgroup  (N-subgroup index of this column)

    Address generation follows the VN layout permutation.  The OVN layout
    has three ordered ranks R = [P_L1, P_L0, Q_L1] with rank variables
    RV = [pL1, pL0, qL1].  Given permutation pi = (p0, p1, p2) from
    TABLE_II, the flattened VN index is:

        L = RV[pi[0]] * R[pi[1]] * R[pi[2]]
          + RV[pi[1]] * R[pi[2]]
          + RV[pi[2]]

    The physical buffer address is then:
        addr_col = L mod AW     (bank index / column address)
        addr_row = L // AW      (row address within the bank)

    A bank conflict occurs when two or more PE columns in the same cycle
    write to the same addr_col but at different addr_row values.

    Same addr_col + same addr_row is NOT a conflict — that is an
    accumulation (reduction) handled by BIRRD.

    Returns True if there IS a conflict, False if conflict-free.
    """
    AH, AW = cfg.AH, cfg.AW
    P_L0 = layout_o.a0
    P_L1 = layout_o.a1
    Q_L1 = layout_o.a2

    # Cycle-accurate simulation: for each EM, for each streaming step,
    # compute the OVN linear index L for each active column, then derive
    # addr_col = L % AW and addr_row = L // AW.  Check for bank conflicts
    # (same addr_col, different addr_row).
    for em_idx in range(len(exec_params)):
        start = em_idx * AW
        end = min(start + AW, len(combined_columns))
        cols_in_em = combined_columns[start:end]

        if not cols_in_em:
            continue

        max_steps = max(len(c.ivn_sequence) for c in cols_in_em)

        for step in range(max_steps):
            # addr_col -> addr_row seen first
            bank_to_row: Dict[int, int] = {}

            for col in cols_in_em:
                if step >= len(col.ivn_sequence):
                    continue  # column idle at this step — no OB write

                m_row, _j = col.ivn_sequence[step]
                sg = col.groups[0].vn_subgroup if col.groups else 0

                # Decompose output (p=m_row, q=sg) into OVN layout indices
                pL0 = m_row % P_L0 if P_L0 > 0 else 0
                pL1 = m_row // P_L0 if P_L0 > 0 else m_row
                qL1 = sg

                # Out-of-bounds: hardware zero-pads, no OB write
                if pL0 >= P_L0 or pL1 >= P_L1 or qL1 >= Q_L1:
                    continue

                L = layout_o.linear_index(
                    {"pL0": pL0, "pL1": pL1, "qL1": qL1})
                addr_col = L % AW
                addr_row = L // AW

                if addr_col in bank_to_row:
                    if bank_to_row[addr_col] != addr_row:
                        return True  # same bank, different row → conflict
                else:
                    bank_to_row[addr_col] = addr_row

    return False


# ---------------------------------------------------------------------------
# Inter-layer layout matching (Step 4)
# ---------------------------------------------------------------------------

def layouts_match(
    layout_o_prev: LayoutSpec,
    layout_i_next: LayoutSpec,
) -> bool:
    """Check whether SetOVNLayout^(i) matches SetIVNLayout^(i+1).

    For inter-layer data continuity, the output buffer (OVN) of layer i
    is ping-pong swapped into the streaming buffer (IVN) of layer i+1.
    For the physical addresses to match, the OVN and IVN must produce
    identical address layouts, which requires:

      - OVN order maps to IVN order via OVN_TO_IVN_ORDER (= 5 - order)
      - Compatible dimensions (P_L0↔M_L0, P_L1↔M_L1, Q_L1↔J_L1)
    """
    required_ivn_order = OVN_TO_IVN_ORDER[layout_o_prev.order_id]
    if layout_i_next.order_id != required_ivn_order:
        return False

    # O dimensions: P_L0, P_L1, Q_L1 must match I dimensions: M_L0, M_L1, J_L1
    o_dims = layout_o_prev.dims()
    i_dims = layout_i_next.dims()

    return (o_dims.get("pL0", 0) == i_dims.get("mL0", 0)
            and o_dims.get("pL1", 0) == i_dims.get("mL1", 0)
            and o_dims.get("qL1", 0) == i_dims.get("jL1", 0))


def derive_ovn_from_ivn(
    layout_i_next: LayoutSpec,
    cfg: FeatherPlusConfig,
    Mt_prev: int,
    Nt_prev: int,
) -> LayoutSpec:
    """Derive SetOVNLayout^(i) from SetIVNLayout^(i+1).

    The output layout of layer i must produce physical addresses that
    match the input layout of layer i+1 after the ping-pong swap.
    The OVN order is derived as IVN_TO_OVN_ORDER[ivn_order] (= 5 - ivn_order).

    OB port conflict checking is deferred to the caller, which must
    call check_ob_port_conflict() with the actual ExecuteMapping
    parameters and combined columns.
    """
    # IVN has (M_L0, M_L1, J_L1) → OVN needs (P_L0, P_L1, Q_L1)
    # where P maps to M (output rows) and Q maps to N (output columns → next K)
    i_dims = layout_i_next.dims()

    P_L0 = i_dims.get("mL0", 1)
    P_L1 = i_dims.get("mL1", 1)
    # Q_L1 for the output depends on the current layer's Nt, not the next layer's K
    Q_L1 = int(math.ceil(Nt_prev / cfg.AH))

    ovn_order = IVN_TO_OVN_ORDER[layout_i_next.order_id]
    return LayoutSpec(
        operand="O",
        order_id=ovn_order,
        a0=P_L0, a1=P_L1, a2=Q_L1,
        AH=cfg.AH, AW=cfg.AW,
    )


def check_inter_layer_layout(
    layout_o_prev: LayoutSpec,
    layout_i_next: LayoutSpec,
    layer_i_name: str = "",
    layer_next_name: str = "",
) -> List[str]:
    """Validate SetOVNLayout^(i) against SetIVNLayout^(i+1) and return errors.

    Returns a list of human-readable error strings.  An empty list means
    the layouts are compatible for ping-pong swap.
    """
    errors: List[str] = []
    prev_label = f"layer {layer_i_name}" if layer_i_name else "prev layer"
    next_label = f"layer {layer_next_name}" if layer_next_name else "next layer"

    # --- Order check ---
    required_ivn = OVN_TO_IVN_ORDER[layout_o_prev.order_id]
    actual_ivn = layout_i_next.order_id
    if actual_ivn != required_ivn:
        errors.append(
            f"Order mismatch: SetOVNLayout of {prev_label} has order={layout_o_prev.order_id}, "
            f"which requires SetIVNLayout of {next_label} to use order={required_ivn} "
            f"(got order={actual_ivn}).  "
            f"Rule: ivn_order = 5 - ovn_order."
        )

    # --- Dimension check ---
    o_dims = layout_o_prev.dims()
    i_dims = layout_i_next.dims()

    dim_pairs = [("P_L0", "pL0", "M_L0", "mL0"),
                 ("P_L1", "pL1", "M_L1", "mL1"),
                 ("Q_L1", "qL1", "J_L1", "jL1")]
    for o_name, o_key, i_name, i_key in dim_pairs:
        o_val = o_dims.get(o_key, 0)
        i_val = i_dims.get(i_key, 0)
        if o_val != i_val:
            errors.append(
                f"Dimension mismatch: {prev_label} OVN {o_name}={o_val} != "
                f"{next_label} IVN {i_name}={i_val}."
            )

    return errors
