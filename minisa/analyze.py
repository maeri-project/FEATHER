#!/usr/bin/env python3
"""Analysis + GPU/TPU comparison (Req 6).

Consolidates: memory reduction analysis, latency breakdown, speedup,
and GPU/TPU comparison from baseline data.

Usage:
    python -m minisa.analyze --bench-csv out/benchmark_summary.csv \
        --inst-csv out/inst_compare.csv --out-dir out/analysis
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import FeatherPlusConfig, make_cfg, ceil_div
from .search import co_search_layout_mapping
from .workload import parse_sram_map


_PKG_DIR = Path(__file__).resolve().parent


def _geom_mean(x):
    x = np.array(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def _config_label(ah: int, aw: int) -> str:
    """Human-readable config label."""
    if ah == aw:
        return f"AH=AW={ah}"
    return f"AH={ah} AW={aw}"


def _group_by_config(df: pd.DataFrame):
    """Yield (ah, aw, sub_df) groups from a DataFrame with ah/aw columns."""
    for (ah, aw), sub in df.groupby(["ah", "aw"]):
        yield int(ah), int(aw), sub


# ===================================================================
# FEATHER+ Scale-Out to Match TPUv6e8
# ===================================================================

def compute_scale_out_instances(
    feather_ah: int = 16, feather_aw: int = 16,
    tpu_engines: int = 8, tpu_pe_h: int = 256, tpu_pe_w: int = 256,
    tpu_8b_regs_per_pe: int = 1, tpu_32b_regs_per_pe: int = 1,
) -> Dict[str, Any]:
    """Compute number of FEATHER+ instances to match TPUv6e8 resources."""
    tpu_total_pe = tpu_engines * tpu_pe_h * tpu_pe_w
    feather_pe = feather_ah * feather_aw

    n_by_mult = tpu_total_pe // feather_pe

    tpu_reg_bits_per_pe = tpu_8b_regs_per_pe * 8 + tpu_32b_regs_per_pe * 32
    feather_reg_bits_per_pe = feather_ah * 8 + 1 * 32
    tpu_total_reg_bits = tpu_total_pe * tpu_reg_bits_per_pe
    feather_reg_bits_per_inst = feather_pe * feather_reg_bits_per_pe
    n_by_regs = tpu_total_reg_bits // feather_reg_bits_per_inst

    n_instances = int(math.sqrt(n_by_mult * n_by_regs))

    return {
        "n_instances": n_instances,
        "n_by_multipliers": n_by_mult,
        "n_by_register_bits": n_by_regs,
        "tpu_total_pe": tpu_total_pe,
        "feather_pe_per_instance": feather_pe,
        "feather_ah": feather_ah,
        "feather_aw": feather_aw,
    }


def split_workload(
    M: int, K: int, N: int, n_instances: int, min_tile: int = 16,
) -> Tuple[int, int, int, str]:
    """Split GEMM across instances, using 2D splitting when needed."""
    dims = {"M": M, "K": K, "N": N}
    highest = max(dims, key=dims.get)
    tile_1d = max(1, ceil_div(dims[highest], n_instances))

    if tile_1d >= min_tile:
        tile = dict(dims)
        tile[highest] = tile_1d
        return tile["M"], tile["K"], tile["N"], highest

    best_tile_M, best_tile_N, best_split_desc = M, N, highest
    best_min_dim = 0

    for split_m in range(1, n_instances + 1):
        if n_instances % split_m != 0:
            continue
        split_n = n_instances // split_m
        tm = max(1, ceil_div(M, split_m))
        tn = max(1, ceil_div(N, split_n))
        min_dim = min(tm, tn)
        if min_dim > best_min_dim:
            best_min_dim = min_dim
            best_tile_M, best_tile_N = tm, tn
            best_split_desc = f"M/{split_m}xN/{split_n}"

    return best_tile_M, K, best_tile_N, best_split_desc


def estimate_scaled_feather_latency(
    M: int, K: int, N: int,
    cfg: FeatherPlusConfig,
    n_instances: int,
) -> Dict[str, Any]:
    """Estimate FEATHER+ latency after distributing workload across instances."""
    Mt, Kt, Nt, split_dim = split_workload(M, K, N, n_instances, min_tile=cfg.AW)

    sr = co_search_layout_mapping(Mt, Kt, Nt, cfg, generate_trace=False)

    return {
        "tile_M": Mt, "tile_K": Kt, "tile_N": Nt,
        "split_dim": split_dim,
        "cycles_total": sr.cycles_total,
        "order_w": sr.order_w, "order_i": sr.order_i, "order_o": sr.order_o,
        "tile_Mt": sr.Mt, "tile_Kt": sr.Kt, "tile_Nt": sr.Nt,
    }


def _scaled_search_worker(
    category: str, name: str, M: int, K: int, N: int,
    cfg: FeatherPlusConfig, n_instances: int, ah: int, aw: int,
) -> Dict[str, Any]:
    """Worker for parallel scaled search. Top-level for pickling."""
    try:
        info = estimate_scaled_feather_latency(M, K, N, cfg, n_instances)
        freq_ghz = cfg.freq_ghz
        lat_us = info["cycles_total"] / (freq_ghz * 1e3)
        return {
            "category": category, "name": name,
            "M": M, "K": K, "N": N,
            "ah": ah, "aw": aw, "n_instances": n_instances,
            "scaled_cycles": info["cycles_total"],
            "feather_latency_us": lat_us,
            "tile_M": info["tile_M"], "tile_K": info["tile_K"],
            "tile_N": info["tile_N"], "split_dim": info["split_dim"],
        }
    except Exception as e:
        return {
            "category": category, "name": name,
            "M": M, "K": K, "N": N,
            "ah": ah, "aw": aw, "n_instances": n_instances,
            "scaled_cycles": 0, "feather_latency_us": float("inf"),
            "tile_M": M, "tile_K": K, "tile_N": N,
            "split_dim": f"error: {e}",
        }


def _run_scaled_searches_parallel(
    workloads: pd.DataFrame, cfg: FeatherPlusConfig,
    n_instances: int, ah: int, aw: int, jobs: int = 1,
) -> List[Dict[str, Any]]:
    """Run scaled searches for all workloads, optionally in parallel."""
    tasks = []
    for _, wl in workloads.iterrows():
        tasks.append((str(wl["category"]), str(wl["name"]),
                       int(wl["M"]), int(wl["K"]), int(wl["N"])))

    if jobs <= 1:
        return [_scaled_search_worker(cat, nm, M, K, N, cfg, n_instances, ah, aw)
                for cat, nm, M, K, N in tasks]

    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        fut_to_idx = {}
        for i, (cat, nm, M, K, N) in enumerate(tasks):
            f = ex.submit(_scaled_search_worker, cat, nm, M, K, N,
                          cfg, n_instances, ah, aw)
            fut_to_idx[f] = i
        for f in as_completed(fut_to_idx):
            results[fut_to_idx[f]] = f.result()
    return results


def load_gpu_baseline(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = _PKG_DIR / "baseline" / "gpu" / "gemm_profiling_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_tpu_baseline(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = _PKG_DIR / "baseline" / "tpu" / "jax_matmul_all_sharding_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def memory_reduction_summary(bench_csv: Path) -> pd.DataFrame:
    """Summarize MINISA vs config stream memory reduction per (AH, AW)."""
    df = pd.read_csv(bench_csv)
    if "compression_ratio" not in df.columns:
        return pd.DataFrame()

    valid = df[df["compression_ratio"].notna()]
    rows = []
    for ah, aw, sub in _group_by_config(valid):
        rows.append({
            "ah": ah, "aw": aw,
            "avg_compression_ratio": sub["compression_ratio"].mean(),
            "median_compression_ratio": sub["compression_ratio"].median(),
            "max_compression_ratio": sub["compression_ratio"].max(),
            "min_compression_ratio": sub["compression_ratio"].min(),
            "count": len(sub),
        })
    return pd.DataFrame(rows)


def instruction_reduction_summary(inst_csv: Path) -> pd.DataFrame:
    """Summarize instruction reduction (explicit/MINISA) per (AH, AW)."""
    df = pd.read_csv(inst_csv)
    col = "inst_bytes_ratio_explicit_vs_minisa"
    if col not in df.columns:
        return pd.DataFrame()

    valid = df[df[col].notna() & (df[col] > 0)]
    rows = []
    for ah, aw, sub in _group_by_config(valid):
        ratios = sub[col]
        rows.append({
            "ah": ah, "aw": aw,
            "geo_mean_reduction": _geom_mean(ratios.values),
            "median_reduction": float(ratios.median()),
            "max_reduction": float(ratios.max()),
            "count": len(sub),
        })
    return pd.DataFrame(rows)


def compute_utilization_summary(bench_csv: Path) -> pd.DataFrame:
    """Summarize compute utilization per (AH, AW)."""
    df = pd.read_csv(bench_csv)
    if "compute_util_avg" not in df.columns:
        return pd.DataFrame()

    valid = df[df["compute_util_avg"].notna()]
    rows = []
    for ah, aw, sub in _group_by_config(valid):
        rows.append({
            "ah": ah, "aw": aw,
            "avg_compute_util": sub["compute_util_avg"].mean(),
            "median_compute_util": sub["compute_util_avg"].median(),
            "avg_cycles_total": sub["cycles_total"].mean(),
            "count": len(sub),
        })
    return pd.DataFrame(rows)


def gpu_tpu_comparison(
    bench_csv: Path,
    ah: int, aw: int,
    freq_ghz: float = 1.0,
    gpu_csv: Optional[Path] = None,
    tpu_csv: Optional[Path] = None,
    scale_out: bool = True,
    sram_mb_map: Optional[Dict[int, float]] = None,
    instbuf_mb_map: Optional[Dict[int, float]] = None,
    alloc: Tuple[float, float, float] = (0.4, 0.4, 0.2),
    jobs: int = 1,
) -> pd.DataFrame:
    """Join FEATHER+ results with GPU/TPU baselines on (M, K, N)."""
    df_feather = pd.read_csv(bench_csv)
    df_feather = df_feather[(df_feather["ah"] == ah) & (df_feather["aw"] == aw)].copy()
    if df_feather.empty:
        return pd.DataFrame()

    # if sram_mb_map is None:
    #     sram_mb_map = {4: 0.25, 8: 1, 16: 4, 32: 8, 64: 32, 128: 128}
    # if instbuf_mb_map is None:
    #     instbuf_mb_map = {4: 0.0625, 8: 0.125, 16: 0.25, 32: 0.5, 64: 1, 128: 2}
    if sram_mb_map is None:
        sram_mb_map = {4: 4, 8: 16, 16: 64, 32: 256, 64: 1024, 128: 4096}
    if instbuf_mb_map is None:
        instbuf_mb_map = {4: 0.5, 8: 1, 16: 2, 32: 3, 64: 5, 128: 10}

    if scale_out:
        scale_info = compute_scale_out_instances(feather_ah=ah, feather_aw=aw)
        n_instances = scale_info["n_instances"]
        print(f"\n  Scale-out resource matching (FEATHER+ {ah}x{aw} vs TPUv6e8):")
        print(f"    TPU total PEs: {scale_info['tpu_total_pe']:,}")
        print(f"    FEATHER+ PEs/instance: {scale_info['feather_pe_per_instance']}")
        print(f"    N by multipliers: {scale_info['n_by_multipliers']:,}")
        print(f"    N by register bits: {scale_info['n_by_register_bits']:,}")
        print(f"    N instances (geo-mean): {n_instances:,}")
        print(f"    Splitting each workload along highest dimension by {n_instances}x\n")

        cfg = make_cfg(ah, aw, sram_mb_map, instbuf_mb_map, alloc, freq_ghz=freq_ghz)

        workloads = df_feather[["category", "name", "M", "K", "N"]].copy()
        scaled_rows = _run_scaled_searches_parallel(workloads, cfg, n_instances, ah, aw, jobs)

        df_scaled = pd.DataFrame(scaled_rows)[["M", "K", "N", "scaled_cycles", "tile_M", "tile_K", "tile_N", "split_dim"]]
        df_scaled.rename(columns={"N": "N"}, inplace=True)
        df_feather = df_feather.merge(df_scaled, on=["M", "K", "N"], how="left")
        df_feather["feather_latency_us"] = (
            df_feather["scaled_cycles"] / (freq_ghz * 1e3)
        )
        df_feather["n_instances"] = n_instances
    else:
        df_feather["feather_latency_us"] = (
            df_feather["cycles_total"] / (freq_ghz * 1e3)
        )

    df_gpu = load_gpu_baseline(gpu_csv)
    df_tpu = load_tpu_baseline(tpu_csv)

    cols = ["category", "name", "M", "K", "N", "ah", "aw",
            "cycles_total", "feather_latency_us", "compute_util_avg"]
    if scale_out:
        cols += ["n_instances", "scaled_cycles", "tile_M", "tile_K", "tile_N", "split_dim"]
    result = df_feather[cols].copy()

    if not df_gpu.empty and "Best_Mean_ms" in df_gpu.columns:
        gpu_sub = df_gpu[["M", "K", "N", "Best_Mean_ms"]].copy()
        gpu_sub["gpu_latency_us"] = gpu_sub["Best_Mean_ms"] * 1000.0
        result = result.merge(
            gpu_sub[["M", "K", "N", "gpu_latency_us"]],
            on=["M", "K", "N"], how="left",
        )

    if not df_tpu.empty and "Latency_us" in df_tpu.columns:
        tpu_min = df_tpu.groupby(["M", "K", "N"])["Latency_us"].min().reset_index()
        tpu_min.rename(columns={"Latency_us": "tpu_latency_us"}, inplace=True)
        result = result.merge(tpu_min, on=["M", "K", "N"], how="left")

    if "gpu_latency_us" in result.columns:
        result["speedup_vs_gpu"] = result["gpu_latency_us"] / result["feather_latency_us"]
    if "tpu_latency_us" in result.columns:
        result["speedup_vs_tpu"] = result["tpu_latency_us"] / result["feather_latency_us"]

    return result


def multi_config_gpu_tpu_comparison(
    bench_csv: Path,
    ah_aw_pairs: List[Tuple[int, int]],
    freq_ghz: float = 1.0,
    gpu_csv: Optional[Path] = None,
    tpu_csv: Optional[Path] = None,
    sram_mb_map: Optional[Dict[int, float]] = None,
    instbuf_mb_map: Optional[Dict[int, float]] = None,
    alloc: Tuple[float, float, float] = (0.4, 0.4, 0.2),
    jobs: int = 1,
) -> Tuple[pd.DataFrame, Tuple[int, int], pd.DataFrame]:
    """Run scale-out comparison for multiple (AH, AW) configs, pick best by avg latency."""
    if sram_mb_map is None:
        sram_mb_map = {4: 4, 8: 16, 16: 64, 32: 256, 64: 1024, 128: 4096}
    if instbuf_mb_map is None:
        instbuf_mb_map = {4: 0.5, 8: 1, 16: 2, 32: 3, 64: 5, 128: 10}

    df_bench = pd.read_csv(bench_csv)
    df_gpu = load_gpu_baseline(gpu_csv)
    df_tpu = load_tpu_baseline(tpu_csv)

    # Get unique workloads from the first available config
    first_ah, first_aw = ah_aw_pairs[0]
    df_sample = df_bench[(df_bench["ah"] == first_ah) & (df_bench["aw"] == first_aw)]
    workloads = df_sample[["category", "name", "M", "K", "N"]].copy()

    print(f"\n{'='*80}")
    print(f"Multi-config scale-out comparison: {[f'{ah}x{aw}' for ah, aw in ah_aw_pairs]}")
    print(f"{'='*80}")

    config_results: Dict[Tuple[int, int], pd.DataFrame] = {}
    config_avg_latency: Dict[Tuple[int, int], float] = {}

    for ah, aw in ah_aw_pairs:
        scale_info = compute_scale_out_instances(feather_ah=ah, feather_aw=aw)
        n_instances = scale_info["n_instances"]

        if ah not in sram_mb_map:
            print(f"  AH={ah} AW={aw}: skipped (no SRAM config)")
            continue

        cfg = make_cfg(ah, aw, sram_mb_map, instbuf_mb_map, alloc, freq_ghz=freq_ghz)

        print(f"\n  AH={ah} AW={aw}: {n_instances} instances "
              f"(n_mult={scale_info['n_by_multipliers']}, "
              f"n_regs={scale_info['n_by_register_bits']})")

        rows = _run_scaled_searches_parallel(workloads, cfg, n_instances, ah, aw, jobs)

        df_cfg = pd.DataFrame(rows)
        avg_lat = df_cfg["feather_latency_us"].mean()
        config_results[(ah, aw)] = df_cfg
        config_avg_latency[(ah, aw)] = avg_lat
        print(f"    Avg latency: {avg_lat:.2f} us")

    # Summary table
    print(f"\n  {'Config':>12s}  {'Instances':>10s}  {'Avg Latency (us)':>18s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*18}")
    best_key = min(config_avg_latency, key=config_avg_latency.get)
    for ah, aw in ah_aw_pairs:
        key = (ah, aw)
        if key in config_avg_latency:
            info = compute_scale_out_instances(feather_ah=ah, feather_aw=aw)
            tag = "  <-- best" if key == best_key else ""
            print(f"  {ah:>3d}x{aw:<7d}  {info['n_instances']:>10d}  "
                  f"{config_avg_latency[key]:>18.2f}{tag}")

    print(f"\n  Best configuration: AH={best_key[0]} AW={best_key[1]} "
          f"(avg latency={config_avg_latency[best_key]:.2f} us)")

    result = config_results[best_key].copy()

    # Add compute_util_avg from bench data
    df_bench_best = df_bench[
        (df_bench["ah"] == best_key[0]) & (df_bench["aw"] == best_key[1])
    ][["M", "K", "N", "compute_util_avg"]].copy()
    result = result.merge(df_bench_best, on=["M", "K", "N"], how="left")

    # Join GPU/TPU baselines
    if not df_gpu.empty and "Best_Mean_ms" in df_gpu.columns:
        gpu_sub = df_gpu[["M", "K", "N", "Best_Mean_ms"]].copy()
        gpu_sub["gpu_latency_us"] = gpu_sub["Best_Mean_ms"] * 1000.0
        result = result.merge(
            gpu_sub[["M", "K", "N", "gpu_latency_us"]],
            on=["M", "K", "N"], how="left")

    if not df_tpu.empty and "Latency_us" in df_tpu.columns:
        tpu_min = df_tpu.groupby(["M", "K", "N"])["Latency_us"].min().reset_index()
        tpu_min.rename(columns={"Latency_us": "tpu_latency_us"}, inplace=True)
        result = result.merge(tpu_min, on=["M", "K", "N"], how="left")

    if "gpu_latency_us" in result.columns:
        result["speedup_vs_gpu"] = result["gpu_latency_us"] / result["feather_latency_us"]
    if "tpu_latency_us" in result.columns:
        result["speedup_vs_tpu"] = result["tpu_latency_us"] / result["feather_latency_us"]

    df_all = pd.concat(list(config_results.values()), ignore_index=True)

    return result, best_key, df_all


# ===================================================================
# AW-Level Scaling
# ===================================================================

def compute_aw_scaled_dimensions(
    ah: int,
    tpu_pe_h: int = 256, tpu_pe_w: int = 256,
    tpu_8b_regs_per_pe: int = 1,
) -> Dict[str, Any]:
    """Compute AW to match a single TPU 256x256 engine's resources."""
    tpu_pe = tpu_pe_h * tpu_pe_w
    aw_by_macs = tpu_pe // ah
    aw_by_8b_regs = tpu_pe // (ah * ah)
    aw = int(math.sqrt(aw_by_macs * max(1, aw_by_8b_regs)))

    total_pe = ah * aw
    return {
        "ah": ah, "aw": aw,
        "aw_by_macs": aw_by_macs,
        "aw_by_8b_regs": aw_by_8b_regs,
        "total_pe": total_pe,
        "total_mults": total_pe,
        "total_8b_regs": total_pe * ah,
        "tpu_engine_pe": tpu_pe,
    }


def aw_scaling_gpu_tpu_comparison(
    bench_csv: Path,
    ah_values: List[int],
    n_instances: int = 8,
    freq_ghz: float = 1.0,
    gpu_csv: Optional[Path] = None,
    tpu_csv: Optional[Path] = None,
    jobs: int = 1,
) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    """AW-level scaling comparison."""
    df_bench = pd.read_csv(bench_csv)
    df_gpu = load_gpu_baseline(gpu_csv)
    df_tpu = load_tpu_baseline(tpu_csv)

    # Get unique workloads from any available config
    configs_in_bench = df_bench[["ah", "aw"]].drop_duplicates()
    first_ah = int(configs_in_bench.iloc[0]["ah"])
    first_aw = int(configs_in_bench.iloc[0]["aw"])
    df_sample = df_bench[(df_bench["ah"] == first_ah) & (df_bench["aw"] == first_aw)]
    workloads = df_sample[["category", "name", "M", "K", "N"]].copy()

    print(f"\n{'='*80}")
    print(f"AW-Level Scaling: AH fixed, AW widened to match single TPU 256x256 engine")
    print(f"Then x{n_instances} instances (matching TPUv6e8 {n_instances} engines)")
    print(f"{'='*80}")

    config_results: Dict[int, pd.DataFrame] = {}
    config_avg_latency: Dict[int, float] = {}

    for ah in ah_values:
        dims = compute_aw_scaled_dimensions(ah)
        aw = dims["aw"]

        sram_mb = max(1.0, ah * aw / 4.0)
        inst_buf_mb = max(0.5, sram_mb / 32.0)

        print(f"\n  AH={ah}, AW={aw} ({ah}x{aw} = {dims['total_pe']} PEs/instance)")
        print(f"    AW by MACs: {dims['aw_by_macs']}, AW by 8b regs: {dims['aw_by_8b_regs']}")
        print(f"    Total mults: {dims['total_mults']:,} (TPU engine: {dims['tpu_engine_pe']:,})")
        print(f"    Total 8b regs: {dims['total_8b_regs']:,} (TPU engine: {dims['tpu_engine_pe']:,})")
        print(f"    SRAM: {sram_mb:.0f} MB, inst buf: {inst_buf_mb:.1f} MB")
        print(f"    x{n_instances} instances")

        cfg = FeatherPlusConfig(
            ah=ah,
            aw=aw,
            total_sram_mb=sram_mb,
            inst_buf_mb=inst_buf_mb,
            bw_load_in=aw,
            bw_load_w=aw,
            bw_store_out=4 * aw,
            bw_onchip_move=4 * aw,
            freq_ghz=freq_ghz,
            is_birrd_plus=True,
            k_distribution=True,
            pe_flush_latency=3,
        )

        rows = _run_scaled_searches_parallel(workloads, cfg, n_instances, ah, aw, jobs)

        df_cfg = pd.DataFrame(rows)
        avg_lat = df_cfg["feather_latency_us"].replace([float("inf")], float("nan")).mean()
        config_results[ah] = df_cfg
        config_avg_latency[ah] = avg_lat
        print(f"    Avg latency: {avg_lat:.2f} us")

    # Summary table
    print(f"\n  {'Config':>12s}  {'PEs/inst':>10s}  {'x Inst':>8s}  {'Total PEs':>10s}  {'Avg Latency (us)':>18s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*18}")
    best_ah = min(config_avg_latency, key=config_avg_latency.get)
    for ah in ah_values:
        if ah in config_avg_latency:
            dims = compute_aw_scaled_dimensions(ah)
            tag = "  <-- best" if ah == best_ah else ""
            print(f"  {ah:>3d}x{dims['aw']:<7d}  {dims['total_pe']:>10,}  "
                  f"{n_instances:>8d}  {dims['total_pe']*n_instances:>10,}  "
                  f"{config_avg_latency[ah]:>18.2f}{tag}")

    print(f"\n  Best config: AH={best_ah}, AW={compute_aw_scaled_dimensions(best_ah)['aw']}")

    result = config_results[best_ah].copy()

    # Add compute_util_avg from bench data (use nearest config for reference)
    for try_ah, try_aw in [(best_ah, best_ah), (16, 16), (32, 32), (8, 8)]:
        df_bench_ref = df_bench[(df_bench["ah"] == try_ah) & (df_bench["aw"] == try_aw)]
        if not df_bench_ref.empty:
            result = result.merge(
                df_bench_ref[["M", "K", "N", "compute_util_avg"]],
                on=["M", "K", "N"], how="left")
            break

    # Join GPU/TPU baselines
    if not df_gpu.empty and "Best_Mean_ms" in df_gpu.columns:
        gpu_sub = df_gpu[["M", "K", "N", "Best_Mean_ms"]].copy()
        gpu_sub["gpu_latency_us"] = gpu_sub["Best_Mean_ms"] * 1000.0
        result = result.merge(
            gpu_sub[["M", "K", "N", "gpu_latency_us"]],
            on=["M", "K", "N"], how="left")

    if not df_tpu.empty and "Latency_us" in df_tpu.columns:
        tpu_min = df_tpu.groupby(["M", "K", "N"])["Latency_us"].min().reset_index()
        tpu_min.rename(columns={"Latency_us": "tpu_latency_us"}, inplace=True)
        result = result.merge(tpu_min, on=["M", "K", "N"], how="left")

    if "gpu_latency_us" in result.columns:
        result["speedup_vs_gpu"] = result["gpu_latency_us"] / result["feather_latency_us"]
    if "tpu_latency_us" in result.columns:
        result["speedup_vs_tpu"] = result["tpu_latency_us"] / result["feather_latency_us"]

    df_all = pd.concat(list(config_results.values()), ignore_index=True)

    return result, best_ah, df_all


def fixed_pe_gpu_tpu_comparison(
    bench_csv: Path,
    ah_values: List[int],
    total_pe_per_instance: int = 65536,
    n_instances: int = 8,
    freq_ghz: float = 1.0,
    gpu_csv: Optional[Path] = None,
    tpu_csv: Optional[Path] = None,
    jobs: int = 1,
) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    """Compare FEATHER+ vs GPU/TPU where each FEATHER instance has a fixed PE count.

    For each AH, AW is computed as total_pe_per_instance / AH. Then n_instances
    copies are used (e.g. 8 to match TPUv6e8's 8 engines).

    Parameters
    ----------
    ah_values : AH values to try (e.g. [8, 16, 32])
    total_pe_per_instance : PEs per FEATHER instance (default: 65536 = 256x256)
    n_instances : number of FEATHER copies (default: 8)
    """
    df_bench = pd.read_csv(bench_csv)
    df_gpu = load_gpu_baseline(gpu_csv)
    df_tpu = load_tpu_baseline(tpu_csv)

    # Get workloads from any available config
    configs_in_bench = df_bench[["ah", "aw"]].drop_duplicates()
    first_ah = int(configs_in_bench.iloc[0]["ah"])
    first_aw = int(configs_in_bench.iloc[0]["aw"])
    df_sample = df_bench[(df_bench["ah"] == first_ah) & (df_bench["aw"] == first_aw)]
    workloads = df_sample[["category", "name", "M", "K", "N"]].copy()

    print(f"\n{'='*80}")
    print(f"Fixed-PE Comparison: {total_pe_per_instance} PEs/instance x {n_instances} instances")
    print(f"Total PEs: {total_pe_per_instance * n_instances:,} "
          f"(matching TPUv6e8: 8 x 256x256 = {8*256*256:,})")
    print(f"{'='*80}")

    config_results: Dict[int, pd.DataFrame] = {}
    config_avg_latency: Dict[int, float] = {}

    for ah in ah_values:
        aw = total_pe_per_instance // ah
        if aw < ah:
            print(f"  AH={ah}: AW={aw} < AH, skipping (AW must be >= AH)")
            continue

        sram_mb = max(1.0, ah * aw / 4.0)
        inst_buf_mb = max(0.5, sram_mb / 32.0)

        print(f"\n  AH={ah}, AW={aw} ({ah}x{aw} = {ah*aw:,} PEs/instance)")
        print(f"    SRAM: {sram_mb:.0f} MB, inst buf: {inst_buf_mb:.1f} MB")
        print(f"    x{n_instances} instances = {ah*aw*n_instances:,} total PEs")

        cfg = FeatherPlusConfig(
            ah=ah, aw=aw,
            total_sram_mb=sram_mb,
            inst_buf_mb=inst_buf_mb,
            bw_load_in=aw,
            bw_load_w=aw,
            bw_store_out=4 * aw,
            bw_onchip_move=4 * aw,
            freq_ghz=freq_ghz,
            is_birrd_plus=True,
            k_distribution=True,
            pe_flush_latency=3,
        )

        rows = _run_scaled_searches_parallel(workloads, cfg, n_instances, ah, aw, jobs)
        # Add total_pe column
        for r in rows:
            r["total_pe"] = ah * aw * n_instances

        df_cfg = pd.DataFrame(rows)
        avg_lat = df_cfg["feather_latency_us"].replace([float("inf")], float("nan")).mean()
        config_results[ah] = df_cfg
        config_avg_latency[ah] = avg_lat
        print(f"    Avg latency: {avg_lat:.2f} us")

    # Summary
    print(f"\n  {'Config':>12s}  {'PEs/inst':>10s}  {'x Inst':>8s}  {'Total PEs':>10s}  {'Avg Latency (us)':>18s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*18}")
    best_ah = min(config_avg_latency, key=config_avg_latency.get) if config_avg_latency else ah_values[0]
    for ah in ah_values:
        if ah in config_avg_latency:
            aw = total_pe_per_instance // ah
            tag = "  <-- best" if ah == best_ah else ""
            print(f"  {ah:>3d}x{aw:<7d}  {ah*aw:>10,}  "
                  f"{n_instances:>8d}  {ah*aw*n_instances:>10,}  "
                  f"{config_avg_latency[ah]:>18.2f}{tag}")

    result = config_results.get(best_ah, pd.DataFrame()).copy()

    # Join GPU/TPU baselines
    if not result.empty:
        if not df_gpu.empty and "Best_Mean_ms" in df_gpu.columns:
            gpu_sub = df_gpu[["M", "K", "N", "Best_Mean_ms"]].copy()
            gpu_sub["gpu_latency_us"] = gpu_sub["Best_Mean_ms"] * 1000.0
            result = result.merge(
                gpu_sub[["M", "K", "N", "gpu_latency_us"]],
                on=["M", "K", "N"], how="left")

        if not df_tpu.empty and "Latency_us" in df_tpu.columns:
            tpu_min = df_tpu.groupby(["M", "K", "N"])["Latency_us"].min().reset_index()
            tpu_min.rename(columns={"Latency_us": "tpu_latency_us"}, inplace=True)
            result = result.merge(tpu_min, on=["M", "K", "N"], how="left")

        if "gpu_latency_us" in result.columns:
            result["speedup_vs_gpu"] = result["gpu_latency_us"] / result["feather_latency_us"]
        if "tpu_latency_us" in result.columns:
            result["speedup_vs_tpu"] = result["tpu_latency_us"] / result["feather_latency_us"]

    df_all = pd.concat(list(config_results.values()), ignore_index=True) if config_results else pd.DataFrame()

    return result, best_ah, df_all


def run_all_analyses(
    bench_csv: Path,
    inst_csv: Path,
    out_dir: Path,
    ah_for_comparison: int = 16,
    aw_for_comparison: int = 16,
    ah_aw_pairs_multi: Optional[List[Tuple[int, int]]] = None,
    aw_scaling_ahs: Optional[List[int]] = None,
    aw_instances: int = 8,
    fixed_pe_ahs: Optional[List[int]] = None,
    fixed_pe_count: int = 65536,
    fixed_pe_instances: int = 8,
    freq_ghz: float = 1.0,
    scale_out: bool = True,
    jobs: int = 1,
) -> None:
    """Run all analyses and save results."""
    out_dir.mkdir(parents=True, exist_ok=True)

    mem_df = memory_reduction_summary(bench_csv)
    if not mem_df.empty:
        mem_df.to_csv(out_dir / "memory_reduction_summary.csv", index=False)
        print("Memory reduction summary:")
        print(mem_df.to_string(index=False))
        print()

    inst_df = instruction_reduction_summary(inst_csv)
    if not inst_df.empty:
        inst_df.to_csv(out_dir / "instruction_reduction_summary.csv", index=False)
        print("Instruction reduction summary:")
        print(inst_df.to_string(index=False))
        print()

    util_df = compute_utilization_summary(bench_csv)
    if not util_df.empty:
        util_df.to_csv(out_dir / "compute_utilization_summary.csv", index=False)
        print("Compute utilization summary:")
        print(util_df.to_string(index=False))
        print()

    if fixed_pe_ahs:
        fpe_df, fpe_best_ah, fpe_all = fixed_pe_gpu_tpu_comparison(
            bench_csv, fixed_pe_ahs,
            total_pe_per_instance=fixed_pe_count,
            n_instances=fixed_pe_instances,
            freq_ghz=freq_ghz,
            jobs=jobs)
        if not fpe_df.empty:
            fpe_df.to_csv(out_dir / "gpu_tpu_fixed_pe.csv", index=False)
            fpe_all.to_csv(out_dir / "gpu_tpu_fixed_pe_all.csv", index=False)
            aw_best = fixed_pe_count // fpe_best_ah
            print(f"\nFixed-PE comparison (best: AH={fpe_best_ah}, AW={aw_best}):")
            cols_to_show = ["name", "M", "K", "N", "feather_latency_us"]
            if "tile_M" in fpe_df.columns:
                cols_to_show += ["tile_M", "tile_K", "tile_N"]
            if "gpu_latency_us" in fpe_df.columns:
                cols_to_show.append("speedup_vs_gpu")
            if "tpu_latency_us" in fpe_df.columns:
                cols_to_show.append("speedup_vs_tpu")
            print(fpe_df[cols_to_show].to_string(index=False))
            print()

    if aw_scaling_ahs:
        aw_df, best_ah, aw_all = aw_scaling_gpu_tpu_comparison(
            bench_csv, aw_scaling_ahs, n_instances=aw_instances, freq_ghz=freq_ghz,
            jobs=jobs)
        if not aw_df.empty:
            aw_df.to_csv(out_dir / "gpu_tpu_aw_scaling.csv", index=False)
            aw_all.to_csv(out_dir / "gpu_tpu_aw_scaling_all.csv", index=False)
            dims = compute_aw_scaled_dimensions(best_ah)
            print(f"\nAW-scaling comparison (best: AH={best_ah}, AW={dims['aw']}):")
            cols_to_show = ["name", "M", "K", "N", "feather_latency_us"]
            if "tile_M" in aw_df.columns:
                cols_to_show += ["tile_M", "tile_K", "tile_N"]
            if "gpu_latency_us" in aw_df.columns:
                cols_to_show.append("speedup_vs_gpu")
            if "tpu_latency_us" in aw_df.columns:
                cols_to_show.append("speedup_vs_tpu")
            print(aw_df[cols_to_show].to_string(index=False))
            print()

    if ah_aw_pairs_multi and scale_out:
        comp_df, best_key, df_all = multi_config_gpu_tpu_comparison(
            bench_csv, ah_aw_pairs_multi, freq_ghz, jobs=jobs)
        if not comp_df.empty:
            comp_df.to_csv(out_dir / "gpu_tpu_comparison.csv", index=False)
            df_all.to_csv(out_dir / "gpu_tpu_all_configs.csv", index=False)
            print(f"\nGPU/TPU comparison (best: AH={best_key[0]} AW={best_key[1]}):")
            cols_to_show = ["name", "M", "K", "N", "feather_latency_us"]
            if "tile_M" in comp_df.columns:
                cols_to_show += ["tile_M", "tile_K", "tile_N"]
            if "gpu_latency_us" in comp_df.columns:
                cols_to_show.append("speedup_vs_gpu")
            if "tpu_latency_us" in comp_df.columns:
                cols_to_show.append("speedup_vs_tpu")
            print(comp_df[cols_to_show].to_string(index=False))
            print()
    elif not aw_scaling_ahs:
        comp_df = gpu_tpu_comparison(
            bench_csv, ah_for_comparison, aw_for_comparison, freq_ghz,
            scale_out=scale_out, jobs=jobs)
        if not comp_df.empty:
            comp_df.to_csv(out_dir / "gpu_tpu_comparison.csv", index=False)
            print(f"GPU/TPU comparison (AH={ah_for_comparison} AW={aw_for_comparison}, "
                  f"scale_out={scale_out}):")
            cols_to_show = ["name", "M", "K", "N", "feather_latency_us"]
            if "tile_M" in comp_df.columns:
                cols_to_show += ["tile_M", "tile_K", "tile_N"]
            if "gpu_latency_us" in comp_df.columns:
                cols_to_show.append("speedup_vs_gpu")
            if "tpu_latency_us" in comp_df.columns:
                cols_to_show.append("speedup_vs_tpu")
            print(comp_df[cols_to_show].to_string(index=False))
            print()


def main():
    ap = argparse.ArgumentParser(description="MINISA Analysis + GPU/TPU Comparison")
    ap.add_argument("--bench-csv", type=str, required=True)
    ap.add_argument("--inst-csv", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--ah", type=int, default=16, help="AH for single-config comparison")
    ap.add_argument("--aw", type=int, default=-1, help="AW for single-config comparison (-1=same as AH)")
    ap.add_argument("--configs-multi", type=str, default=None,
                    help="Comma-separated AHxAW configs for multi-config comparison "
                         "(e.g., 8x8,16x16,32x32). Picks best by avg latency.")
    ap.add_argument("--aw-scaling", type=str, default=None,
                    help="Comma-separated AH values for AW-level scaling "
                         "(e.g., 16,32,64). Widens AW to match single TPU engine.")
    ap.add_argument("--aw-instances", type=int, default=8)
    ap.add_argument("--fixed-pe", type=str, default=None,
                    help="Comma-separated AH values for fixed-PE comparison "
                         "(e.g., 8,16,32). Each instance has --fixed-pe-count PEs.")
    ap.add_argument("--fixed-pe-count", type=int, default=65536,
                    help="PEs per FEATHER instance (default: 65536 = 256x256)")
    ap.add_argument("--fixed-pe-instances", type=int, default=8,
                    help="Number of FEATHER copies (default: 8)")
    ap.add_argument("--freq-ghz", type=float, default=1.0)
    ap.add_argument("--no-scale-out", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Number of parallel workers for search tasks")
    args = ap.parse_args()

    if args.aw < 0:
        args.aw = args.ah

    ah_aw_pairs_multi = None
    if args.configs_multi:
        ah_aw_pairs_multi = []
        for part in args.configs_multi.split(","):
            ah_s, aw_s = part.strip().split("x")
            ah_aw_pairs_multi.append((int(ah_s), int(aw_s)))

    aw_scaling_ahs = None
    if args.aw_scaling:
        aw_scaling_ahs = [int(x) for x in args.aw_scaling.split(",")]

    fixed_pe_ahs = None
    if args.fixed_pe:
        fixed_pe_ahs = [int(x) for x in args.fixed_pe.split(",")]

    run_all_analyses(
        bench_csv=Path(args.bench_csv),
        inst_csv=Path(args.inst_csv),
        out_dir=Path(args.out_dir),
        ah_for_comparison=args.ah,
        aw_for_comparison=args.aw,
        ah_aw_pairs_multi=ah_aw_pairs_multi,
        aw_scaling_ahs=aw_scaling_ahs,
        aw_instances=args.aw_instances,
        fixed_pe_ahs=fixed_pe_ahs,
        fixed_pe_count=args.fixed_pe_count,
        fixed_pe_instances=args.fixed_pe_instances,
        freq_ghz=args.freq_ghz,
        scale_out=not args.no_scale_out,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
