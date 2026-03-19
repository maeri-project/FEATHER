#!/usr/bin/env python3
"""Workload loading utilities and CLI helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_ah_aw_pairs(ah_str: str, aw_str: str) -> List[Tuple[int, int]]:
    """Parse AH and AW values into (AH, AW) pairs.

    Supported formats:
      --aw same            → AW = AH for each entry (square arrays)
      --aw "8,16,32"       → 1:1 with --ah (must have same length)
      --aw "8,16,32/16,32" → Per-AH expansion: first AH gets first group,
                              second AH gets second group, etc. Groups are
                              separated by '/'. The number of groups must match
                              the number of AH values.
    """
    ah_list = [int(x) for x in ah_str.split(",") if x.strip()]
    if aw_str.strip().lower() == "same":
        return [(a, a) for a in ah_list]

    if "/" in aw_str:
        # Per-AH expansion: "8,16,32,64,128/16,32,64,128"
        groups = aw_str.split("/")
        if len(groups) != len(ah_list):
            raise ValueError(
                f"--aw has {len(groups)} groups (separated by /) but --ah has "
                f"{len(ah_list)} entries. They must match.")
        pairs = []
        for ah, group in zip(ah_list, groups):
            aw_vals = [int(x) for x in group.split(",") if x.strip()]
            for aw in aw_vals:
                pairs.append((ah, aw))
        return pairs

    aw_list = [int(x) for x in aw_str.split(",") if x.strip()]
    if len(ah_list) != len(aw_list):
        raise ValueError(
            f"--ah has {len(ah_list)} entries but --aw has {len(aw_list)} entries. "
            f"They must match, or use --aw same for square arrays, "
            f"or use '/' for per-AH expansion (e.g. '8,16,32/16,32').")
    return list(zip(ah_list, aw_list))


def parse_sram_map(s: str) -> Dict[int, float]:
    mp: Dict[int, float] = {}
    for part in s.split(","):
        k, v = part.split(":")
        mp[int(k)] = float(v)
    return mp


def sanitize_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in s)


def load_workload_csv(path: Path) -> pd.DataFrame:
    """Robustly load workload CSVs exported from spreadsheets.

    Supported formats:
      (A) Normal: columns include category,name,M,K,N
      (B) Spreadsheet-export: first data row contains headers 'M','K','N' and
          the actual column names are 'Unnamed: *'. In that case we assume:
            col0=category, col1=name, col2=M, col3=K, col4=N
          and drop the first row.

    Category/Name are forward-filled to support merged-cells style exports.
    """
    df = pd.read_csv(path)

    if {"M", "K", "N"}.issubset(df.columns):
        if "category" not in df.columns:
            df["category"] = ""
        if "name" not in df.columns:
            df["name"] = ""
    else:
        if len(df) == 0:
            raise ValueError(f"Empty CSV: {path}")
        row0 = [str(x).strip() for x in df.iloc[0].tolist()]
        if "M" in row0 and "K" in row0 and "N" in row0 and df.shape[1] >= 5:
            cols = ["category", "name", "M", "K", "N"] + [
                f"extra_{i}" for i in range(5, df.shape[1])
            ]
            df.columns = cols
            df = df.iloc[1:].reset_index(drop=True)
        else:
            raise ValueError(
                f"Unrecognized CSV format for {path}. "
                f"Need columns M/K/N or first row containing M/K/N."
            )

    df["category"] = df["category"].fillna("").replace("", np.nan).ffill().fillna("")
    df["name"] = df["name"].fillna("").replace("", np.nan).ffill().fillna("")
    df["name"] = df["name"].astype(str).str.replace("\n", " ").str.strip()

    df = df[df["M"].notna() & df["K"].notna() & df["N"].notna()].copy()
    df["M"] = df["M"].astype(int)
    df["K"] = df["K"].astype(int)
    df["N"] = df["N"].astype(int)
    return df
