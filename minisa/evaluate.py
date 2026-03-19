#!/usr/bin/env python3
"""Full evaluation pipeline (Req 5): run_evaluation.

Usage:
    python -m minisa.evaluate --csv MINISA_Evaluation_Setup_Full.csv \
        --out-dir out --ah 8,16 --aw same --verify --jobs 1
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import FeatherPlusConfig, TraceBundle, CycleBreakdown, make_cfg, ceil_div, ceil_log2
from .workload import load_workload_csv, parse_ah_aw_pairs, parse_sram_map, sanitize_filename
from .vn import choose_tile_sizes
from .trace import (
    generate_trace_gemm, verify_trace, verify_config,
    estimate_minisa_inst_bytes, estimate_explicit_microinst_format,
)
from .cycles import estimate_cycles_for_gemm, model_instruction_fetch, estimate_latency_from_config_stream
from .isa import config_to_hw_params
from .to_config import convert_trace_to_config, compute_memory_comparison, compute_config_stream_summary
from .search import co_search_layout_mapping


def run_one_bench_point(
    category: str, name: str, M: int, K: int, N: int,
    ah: int, aw: int, sram_mb_map: Dict[int, float], instbuf_mb_map: Dict[int, float],
    alloc: Tuple[float, float, float], verify: bool, out_dir: Path,
    search_mode: bool = False, dump_trace: bool = False,
) -> Dict[str, Any]:
    cfg = make_cfg(ah, aw, sram_mb_map, instbuf_mb_map, alloc, freq_ghz=1.0)

    if search_mode:
        sr = co_search_layout_mapping(M, K, N, cfg, generate_trace=True)
        tb = sr.trace_bundle
        Mt, Kt, Nt = sr.Mt, sr.Kt, sr.Nt
        # Use the search's (possibly transposed) dimensions for the cycle
        # model so tile counts match the actual trace.  Input-stationary
        # dataflow swaps M↔N; the cycle model must see the same mapping.
        cyc_M, cyc_K, cyc_N = sr.search_M, sr.search_K, sr.search_N
    else:
        Mt, Kt, Nt = choose_tile_sizes(M, K, N, cfg)
        tb = generate_trace_gemm(M, K, N, cfg)
        cyc_M, cyc_K, cyc_N = M, K, N

    cyc = estimate_cycles_for_gemm(cyc_M, cyc_K, cyc_N, cfg, Mt, Kt, Nt, reuse_input_across_N=False)

    inst_bytes_total = int(cfg.minisa_trace_inst_bytes(tb.trace))
    base_cycles = int(cyc.total)
    inst_model = model_instruction_fetch(inst_bytes_total, base_cycles, cfg)
    cyc.inst_prefetch = int(inst_model["prefetch_cycles"])
    cyc.inst_stall = int(inst_model["stall_cycles"])
    cyc.load_inst = int(inst_model["total_extra_cycles"])
    cyc.total = int(base_cycles + cyc.load_inst)

    hw_params = config_to_hw_params(cfg)
    config_summary = compute_config_stream_summary(
        tb.trace, hw_params, cyc_M, cyc_K, cyc_N, Mt, Kt, Nt)
    comparison = compute_memory_comparison(tb.trace, config_summary, hw_params)
    config_latency = estimate_latency_from_config_stream(
        tb.trace, config_summary, cfg, cyc_M, cyc_K, cyc_N, Mt, Kt, Nt)

    offchip_bytes = int(cyc.bytes_in + cyc.bytes_w + cyc.bytes_out_store + inst_bytes_total)
    onchip_bytes = int(cyc.bytes_out_move)
    macs_total = int(M * K * N)
    time_s = cyc.total / (cfg.freq_ghz * 1e9)
    offchip_gbps = (offchip_bytes * 8.0) / max(1e-30, time_s) / 1e9
    onchip_gbps = (onchip_bytes * 8.0) / max(1e-30, time_s) / 1e9
    throughput_gfops = macs_total / max(1e-30, time_s) / 1e9

    peak = cfg.effective_peak_macs_per_cycle()
    compute_util_avg = macs_total / max(1, (cyc.compute + cyc.out_to_stream) * peak)
    compute_util_total = macs_total / max(1, cyc.total * peak)

    verify_ok = True
    max_err = 0.0
    config_verify_ok = True
    config_max_err = 0.0
    if verify:
        verify_ok, max_err = verify_trace(tb)
        config_verify_ok, config_max_err = verify_config(tb)

    trace_path = ""
    if dump_trace:
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = sanitize_filename(f"{category}_{name}_M{M}_K{K}_N{N}_AH{ah}_AW{aw}.json")
        trace_path = str(out_dir / fname)
        with open(trace_path, "w") as f:
            json.dump({
                "cfg": asdict(cfg), "workload": {"M": M, "K": K, "N": N},
                "chunk_strategy": tb.chunk_strategy, "order_ids": tb.order_ids,
                "trace": tb.trace, "cycle_estimate": asdict(cyc),
            }, f)

    return {
        "category": category, "name": name,
        "M": M, "K": K, "N": N, "ah": ah, "aw": aw,
        "sram_mb": cfg.total_sram_mb,
        "cap_stream_vn": cfg.cap_stream_vn(),
        "cap_stationary_vn": cfg.cap_stationary_vn(),
        "cap_output_vn": cfg.cap_output_vn(),
        "Mt": Mt, "Kt": Kt, "Nt": Nt,
        "trace_len": len(tb.trace),
        "cycles_total": cyc.total,
        "cycles_load_in": cyc.load_in,
        "cycles_load_w": cyc.load_w,
        "cycles_compute": cyc.compute,
        "cycles_out_to_stream": cyc.out_to_stream,
        "cycles_store_out": cyc.store_out,
        "cycles_load_inst": cyc.load_inst,
        "offchip_gbps": offchip_gbps,
        "onchip_gbps": onchip_gbps,
        "compute_util_avg": compute_util_avg,
        "compute_util_total": compute_util_total,
        "throughput_gfops": throughput_gfops,
        "offchip_bytes": offchip_bytes,
        "onchip_bytes": onchip_bytes,
        "inst_offchip_bytes": inst_bytes_total,
        "inst_bytes_mb": float(inst_bytes_total / (1024 * 1024)),
        "inst_buf_mb": float(cfg.inst_buf_mb),
        "inst_buf_bytes": int(cfg.inst_buf_bytes),
        "inst_prefetch_bytes": int(inst_model["prefetch_bytes"]),
        "inst_prefetch_cycles": int(cyc.inst_prefetch),
        "inst_stall_cycles": int(cyc.inst_stall),
        "inst_required_buf_no_stall_bytes": int(inst_model["required_buf_no_stall_bytes"]),
        "inst_no_stall_given_buf": bool(inst_model["required_buf_no_stall_bytes"] <= cfg.inst_buf_bytes),
        "macs": macs_total,
        "freq_ghz": cfg.freq_ghz,
        "verify_ok": verify_ok,
        "verify_max_abs_err": max_err,
        "config_verify_ok": config_verify_ok,
        "config_verify_max_abs_err": config_max_err,
        "trace_json": trace_path,
        "minisa_isa_bytes": comparison["minisa_bytes"],
        "config_stream_bytes": comparison["config_bytes"],
        "compression_ratio": comparison["compression_ratio"],
        "config_latency_total": config_latency.total,
        "config_latency_compute": config_latency.compute,
    }


def run_one_instcmp(
    category: str, name: str, M: int, K: int, N: int,
    ah: int, aw: int, sram_mb_map: Dict[int, float], instbuf_mb_map: Dict[int, float],
    alloc: Tuple[float, float, float],
) -> Dict[str, Any]:
    cfg = make_cfg(ah, aw, sram_mb_map, instbuf_mb_map, alloc, freq_ghz=1.0)
    Mt, Kt, Nt = choose_tile_sizes(M, K, N, cfg)

    base = estimate_cycles_for_gemm(M, K, N, cfg, Mt, Kt, Nt, reuse_input_across_N=False)
    base_cycles = int(base.total)

    minisa_ib = estimate_minisa_inst_bytes(M, K, N, cfg, Mt, Kt, Nt)
    minisa_im = model_instruction_fetch(minisa_ib, base_cycles, cfg)
    minisa_load_inst = int(minisa_im["total_extra_cycles"])
    minisa_total = base_cycles + minisa_load_inst

    fmt = estimate_explicit_microinst_format(cfg)
    explicit_ib_per = int(fmt["total_bytes"])
    explicit_ib = base_cycles * explicit_ib_per
    explicit_im = model_instruction_fetch(explicit_ib, base_cycles, cfg)
    explicit_load_inst = int(explicit_im["total_extra_cycles"])
    explicit_total = base_cycles + explicit_load_inst

    return {
        "category": category, "name": name,
        "M": int(M), "K": int(K), "N": int(N), "ah": int(ah), "aw": int(aw),
        "sram_mb": float(cfg.total_sram_mb),
        "inst_buf_mb": float(cfg.inst_buf_mb),
        "inst_buf_bytes": int(cfg.inst_buf_bytes),
        "cap_stream_vn": int(cfg.cap_stream_vn()),
        "cap_stationary_vn": int(cfg.cap_stationary_vn()),
        "cap_output_vn": int(cfg.cap_output_vn()),
        "Mt": int(Mt), "Kt": int(Kt), "Nt": int(Nt),
        "base_cycles": base_cycles,
        "minisa_inst_bytes": minisa_ib,
        "minisa_prefetch_cycles": int(minisa_im["prefetch_cycles"]),
        "minisa_stall_cycles": int(minisa_im["stall_cycles"]),
        "minisa_load_inst_cycles": minisa_load_inst,
        "minisa_total_cycles": minisa_total,
        "minisa_required_buf_no_stall_bytes": int(minisa_im["required_buf_no_stall_bytes"]),
        "minisa_no_stall_given_buf": bool(minisa_im["required_buf_no_stall_bytes"] <= cfg.inst_buf_bytes),
        "minisa_inst_bytes_mb": float(minisa_ib / (1024 * 1024)),
        "explicit_inst_bytes_per": explicit_ib_per,
        "explicit_inst_count": base_cycles,
        "explicit_inst_bytes": explicit_ib,
        "explicit_prefetch_cycles": int(explicit_im["prefetch_cycles"]),
        "explicit_stall_cycles": int(explicit_im["stall_cycles"]),
        "explicit_load_inst_cycles": explicit_load_inst,
        "explicit_total_cycles": explicit_total,
        "explicit_required_buf_no_stall_bytes": int(explicit_im["required_buf_no_stall_bytes"]),
        "explicit_no_stall_given_buf": bool(explicit_im["required_buf_no_stall_bytes"] <= cfg.inst_buf_bytes),
        "explicit_inst_bytes_mb": float(explicit_ib / (1024 * 1024)),
        "explicit_bits_birrd": fmt["birrd_bits"],
        "explicit_bits_pe": fmt["pe_bits"],
        "explicit_bits_addr_in": fmt["addr_in_bits"],
        "explicit_bits_addr_w": fmt["addr_w_bits"],
        "explicit_bits_addr_o": fmt["addr_o_bits"],
        "explicit_bits_overhead": fmt["overhead_bits"],
        "explicit_bits_total": fmt["total_bits"],
        "inst_bytes_ratio_explicit_vs_minisa": float(explicit_ib / max(1, minisa_ib)),
        "load_inst_cycles_ratio": float(explicit_load_inst / max(1, minisa_load_inst)),
        "total_cycles_ratio": float(explicit_total / max(1, minisa_total)),
    }


def _bench_worker(row: Dict[str, Any], ah_aw_pairs: List[Tuple[int, int]],
                  sram_mb_map: Dict[int, float], instbuf_mb_map: Dict[int, float],
                  alloc: Tuple[float, float, float], verify: bool,
                  out_dir: str, search_mode: bool = False,
                  dump_trace: bool = False) -> List[Dict[str, Any]]:
    """Process one workload across all (AH, AW) configs sequentially."""
    category = str(row.get("category", ""))
    name = str(row.get("name", ""))
    M, K, N = int(row["M"]), int(row["K"]), int(row["N"])
    results = []
    for ah, aw in ah_aw_pairs:
        results.append(_bench_single(
            category, name, M, K, N, ah, aw,
            sram_mb_map, instbuf_mb_map, alloc,
            verify, out_dir, search_mode, dump_trace))
    return results


def _bench_single(
    category: str, name: str, M: int, K: int, N: int,
    ah: int, aw: int,
    sram_mb_map: Dict[int, float], instbuf_mb_map: Dict[int, float],
    alloc: Tuple[float, float, float], verify: bool,
    out_dir: str, search_mode: bool = False, dump_trace: bool = False,
) -> Dict[str, Any]:
    """Process a single (workload, AH, AW) benchmark point."""
    try:
        r = run_one_bench_point(
            category, name, M, K, N, ah, aw,
            sram_mb_map, instbuf_mb_map, alloc,
            verify=verify, out_dir=Path(out_dir),
            search_mode=search_mode, dump_trace=dump_trace)
        isa_ok = "OK" if r["verify_ok"] else f"ISA_FAIL(err={r['verify_max_abs_err']:.2e})"
        cfg_ok = "OK" if r["config_verify_ok"] else f"CFG_FAIL(err={r['config_verify_max_abs_err']:.2e})"
        status = f"ISA:{isa_ok} CFG:{cfg_ok}"
        print(f"  [{category:15s}] {name:25s} M={M:6d} K={K:5d} N={N:5d} AH={ah:3d} AW={aw:3d} "
              f"cycles={r['cycles_total']:12d} util={r['compute_util_avg']:.4f} "
              f"ISA={r['minisa_isa_bytes']:8d}B cfg={r['config_stream_bytes']:10d}B "
              f"ratio={r['compression_ratio']:.1f}x {status}", flush=True)
        return r
    except Exception as e:
        print(f"  [{category:15s}] {name:25s} M={M:6d} K={K:5d} N={N:5d} AH={ah:3d} AW={aw:3d} ERROR: {e}", flush=True)
        return {
            "category": category, "name": name,
            "M": M, "K": K, "N": N, "ah": ah, "aw": aw, "error": str(e),
        }


def _instcmp_single(
    category: str, name: str, M: int, K: int, N: int,
    ah: int, aw: int,
    sram_mb_map: Dict[int, float], instbuf_mb_map: Dict[int, float],
    alloc: Tuple[float, float, float],
) -> Dict[str, Any]:
    """Process a single (workload, AH, AW) instruction comparison point."""
    try:
        return run_one_instcmp(category, name, M, K, N, ah, aw,
                               sram_mb_map, instbuf_mb_map, alloc)
    except Exception as e:
        print(f"  [instcmp] {category} {name} M={M} K={K} N={N} AH={ah} AW={aw} ERROR: {e}", flush=True)
        return {
            "category": category, "name": name,
            "M": M, "K": K, "N": N, "ah": ah, "aw": aw, "error": str(e),
        }


def _instcmp_worker(row: Dict[str, Any], ah_aw_pairs: List[Tuple[int, int]],
                    sram_mb_map: Dict[int, float], instbuf_mb_map: Dict[int, float],
                    alloc: Tuple[float, float, float]) -> List[Dict[str, Any]]:
    """Process one workload across all (AH, AW) configs sequentially."""
    category = str(row.get("category", ""))
    name = str(row.get("name", ""))
    M, K, N = int(row["M"]), int(row["K"]), int(row["N"])
    results = []
    for ah, aw in ah_aw_pairs:
        results.append(_instcmp_single(
            category, name, M, K, N, ah, aw,
            sram_mb_map, instbuf_mb_map, alloc))
    return results


def run_evaluation(args: argparse.Namespace) -> None:
    """Run the full evaluation pipeline."""
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ah_aw_pairs = parse_ah_aw_pairs(args.ah, args.aw)
    sram_map = parse_sram_map(args.sram_map)
    instbuf_map = parse_sram_map(args.instbuf_map)
    fs, fsta, fout = [float(x) for x in args.alloc.split(",")]
    alloc = (fs, fsta, fout)
    search_mode = getattr(args, "search_mode", False)
    dump_trace = getattr(args, "dump_trace", False)

    df = load_workload_csv(csv_path)
    rows = df.to_dict(orient="records")
    print(f"Loaded {len(rows)} workloads from {csv_path}")
    print(f"Array configs (AH, AW): {ah_aw_pairs}")
    print(f"Output directory: {out_dir}")

    # Step 1: Benchmark
    if not args.skip_bench:
        print("\n" + "=" * 80)
        print("STEP 1: Generating MINISA ISA traces + latency estimation + config comparison")
        print("=" * 80)

        bench_results: List[Dict[str, Any]] = []
        summary_path = out_dir / "benchmark_summary.csv"

        # Resume support: load existing results to skip completed points
        done_keys: set = set()
        if summary_path.exists():
            prev = pd.read_csv(summary_path)
            bench_results.extend(prev.to_dict(orient="records"))
            for _, r in prev.iterrows():
                done_keys.add((int(r["M"]), int(r["K"]), int(r["N"]),
                               int(r["ah"]), int(r["aw"])))
            print(f"  Resuming: {len(done_keys)} benchmark points already done")

        _written_header = [bool(done_keys)]  # skip header if appending

        def _save_result(r: Dict[str, Any]) -> None:
            bench_results.append(r)
            row_df = pd.DataFrame([r])
            if not _written_header[0]:
                row_df.to_csv(summary_path, index=False, mode="w")
                _written_header[0] = True
            else:
                row_df.to_csv(summary_path, index=False, mode="a", header=False)

        if args.jobs <= 1:
            for row in rows:
                for r in _bench_worker(row, ah_aw_pairs, sram_map, instbuf_map, alloc,
                                       args.verify, str(out_dir), search_mode,
                                       dump_trace):
                    _save_result(r)
        else:
            # Fine-grained parallelism: each (workload, AH, AW) is an independent task
            pending_points = []
            for row in rows:
                category = str(row.get("category", ""))
                name = str(row.get("name", ""))
                M, K, N = int(row["M"]), int(row["K"]), int(row["N"])
                for ah, aw in ah_aw_pairs:
                    if (M, K, N, ah, aw) in done_keys:
                        continue
                    pending_points.append((category, name, M, K, N, ah, aw))
            total_points = len(pending_points) + len(done_keys)
            print(f"  Parallelizing {len(pending_points)} remaining benchmark points"
                  f" across {args.jobs} workers ({total_points} total)")
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = []
                for category, name, M, K, N, ah, aw in pending_points:
                    futs.append(ex.submit(
                        _bench_single, category, name, M, K, N, ah, aw,
                        sram_map, instbuf_map, alloc,
                        args.verify, str(out_dir), search_mode,
                        dump_trace))
                n_done = 0
                n_fail = 0
                for fut in as_completed(futs):
                    try:
                        _save_result(fut.result())
                        n_done += 1
                    except Exception as e:
                        n_fail += 1
                        print(f"  WORKER FAILED ({n_fail}): {e}", flush=True)
                if n_fail:
                    print(f"  WARNING: {n_fail} benchmark points failed")

        # Re-read from CSV for clean column ordering
        df_bench = pd.read_csv(summary_path)
        print(f"\nBenchmark summary written to {summary_path} ({len(df_bench)} rows)")

        if args.verify and "verify_ok" in df_bench.columns:
            n_ok = int(df_bench["verify_ok"].sum())
            n_total = len(df_bench[df_bench["verify_ok"].notna()])
            print(f"  Checking Point 1 (ISA-level): {n_ok}/{n_total} passed")
        if args.verify and "config_verify_ok" in df_bench.columns:
            n_ok = int(df_bench["config_verify_ok"].sum())
            n_total = len(df_bench[df_bench["config_verify_ok"].notna()])
            print(f"  Checking Point 2 (Config-level): {n_ok}/{n_total} passed")

        if "compression_ratio" in df_bench.columns:
            valid = df_bench[df_bench["compression_ratio"].notna()]
            if len(valid) > 0:
                print(f"\n  Memory comparison (MINISA ISA vs Config Stream):")
                for ah, aw in ah_aw_pairs:
                    sub = valid[(valid["ah"] == ah) & (valid["aw"] == aw)]
                    if len(sub) > 0:
                        avg_ratio = sub["compression_ratio"].mean()
                        print(f"    AH={ah:3d} AW={aw:3d}: avg compression ratio = {avg_ratio:.1f}x")

        if "compute_util_avg" in df_bench.columns:
            valid = df_bench[df_bench["compute_util_avg"].notna()]
            if len(valid) > 0:
                print(f"\n  Average compute utilization and latency:")
                for ah, aw in ah_aw_pairs:
                    sub = valid[(valid["ah"] == ah) & (valid["aw"] == aw)]
                    if len(sub) > 0:
                        avg_util = sub["compute_util_avg"].mean()
                        avg_lat = sub["cycles_total"].mean()
                        print(f"    AH={ah:3d} AW={aw:3d}: avg_compute_util = {avg_util:.4f} "
                              f"({avg_util*100:.2f}%), avg_latency = {avg_lat:.0f} cycles")

    # Step 2: Instruction comparison
    if not args.skip_instcmp:
        print("\n" + "=" * 80)
        print("STEP 2: Instruction comparison (MINISA ISA vs explicit micro-instruction)")
        print("=" * 80)

        instcmp_results: List[Dict[str, Any]] = []
        ic_path = out_dir / "inst_compare.csv"
        _ic_header = [False]

        def _save_ic(r: Dict[str, Any]) -> None:
            instcmp_results.append(r)
            row_df = pd.DataFrame([r])
            if not _ic_header[0]:
                row_df.to_csv(ic_path, index=False, mode="w")
                _ic_header[0] = True
            else:
                row_df.to_csv(ic_path, index=False, mode="a", header=False)

        if args.jobs <= 1:
            for row in rows:
                for r in _instcmp_worker(row, ah_aw_pairs, sram_map, instbuf_map, alloc):
                    _save_ic(r)
        else:
            total_points = len(rows) * len(ah_aw_pairs)
            print(f"  Parallelizing {total_points} instcmp points across {args.jobs} workers")
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = []
                for row in rows:
                    category = str(row.get("category", ""))
                    name = str(row.get("name", ""))
                    M, K, N = int(row["M"]), int(row["K"]), int(row["N"])
                    for ah, aw in ah_aw_pairs:
                        futs.append(ex.submit(
                            _instcmp_single, category, name, M, K, N, ah, aw,
                            sram_map, instbuf_map, alloc))
                for fut in as_completed(futs):
                    _save_ic(fut.result())

        df_ic = pd.read_csv(ic_path)
        print(f"\nInstruction comparison written to {ic_path} ({len(df_ic)} rows)")

        valid = df_ic if "error" not in df_ic.columns else df_ic[df_ic["error"].isna()]
        if len(valid) > 0 and "inst_bytes_ratio_explicit_vs_minisa" in valid.columns:
            for ah, aw in ah_aw_pairs:
                sub = valid[(valid["ah"] == ah) & (valid["aw"] == aw)]
                if len(sub) > 0:
                    ratios = sub["inst_bytes_ratio_explicit_vs_minisa"]
                    geo_mean = float(np.exp(np.mean(np.log(ratios[ratios > 0]))))
                    print(f"  AH={ah:3d} AW={aw:3d}: inst reduction geo-mean = {geo_mean:.1f}x "
                          f"(explicit/MINISA)")

    # Step 3: Generate plots
    if not args.skip_plots:
        print("\n" + "=" * 80)
        print("STEP 3: Generating plots")
        print("=" * 80)

        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        minisa_pkg_dir = Path(__file__).resolve().parent
        fig_drawer_dir = minisa_pkg_dir / "figure_drawer"

        import subprocess

        ic_path = out_dir / "inst_compare.csv"
        if ic_path.exists():
            ratio_ah = args.ratio_ah
            ratio_aw = args.ratio_aw
            cmd = [
                sys.executable,
                str(fig_drawer_dir / "fig_instr_reduction.py"),
                "--csv", str(ic_path),
                "--out-dir", str(fig_dir),
                "--ratio-ah", str(ratio_ah),
                "--ratio-aw", str(ratio_aw),
            ]
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Instruction reduction plot saved to {fig_dir}/instr_reduction.pdf")
            else:
                print(f"  Plot failed: {result.stderr}")

        summary_path = out_dir / "benchmark_summary.csv"
        if summary_path.exists():
            for ah, aw in ah_aw_pairs:
                out_fig = fig_dir / f"latency_compute_utilization_AH{ah}_AW{aw}.pdf"
                cmd = [
                    sys.executable,
                    str(fig_drawer_dir / "fig_lat_breakdown_comp_utilization.py"),
                    "--csv", str(summary_path),
                    "--ah", str(ah),
                    "--aw", str(aw),
                    "--output", str(out_fig),
                ]
                print(f"  Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  Latency plot saved to {out_fig}")
                else:
                    print(f"  Plot failed (AH={ah} AW={aw}): {result.stderr}")

    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)

def main():
    ap = argparse.ArgumentParser(description="MINISA Full Evaluation Pipeline")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--ah", type=str, default="4,8,16",
                    help="Comma-separated AH (array height) values")
    ap.add_argument("--aw", type=str,
                    default="4,16,64/8,32,128/16,64,256",
                    help="AW values: 'same' for AW=AH, comma-separated for 1:1, "
                         "or per-AH groups separated by '/' "
                         "(e.g. '4,16,64/8,32,128/16,64,256')")
    # ap.add_argument("--sram-map", type=str,
    #                 default="4:0.25,8:1,16:4,32:8,64:32,128:128")
    # ap.add_argument("--instbuf-map", type=str,
    #                 default="4:0.0625,8:0.125,16:0.25,32:0.5,64:1,128:2")
    ap.add_argument("--sram-map", type=str,
                    default="4:4,8:16,16:64,32:64,64:64,128:128,256:256")
    ap.add_argument("--instbuf-map", type=str,
                    default="4:0.5,8:1,16:2,32:3,64:5,128:10,256:20")
    ap.add_argument("--alloc", type=str, default="0.4,0.4,0.2")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-instcmp", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--ratio-ah", type=int, default=16,
                    help="AH value for instruction reduction ratio lines")
    ap.add_argument("--ratio-aw", type=int, default=128,
                    help="AW value for ratio lines (-1 = same as ratio-ah)")
    ap.add_argument("--search-mode", action="store_true",
                    help="Use co-search to find best layout orders")
    ap.add_argument("--dump-trace", action="store_true",
                    help="Dump detailed trace JSON files (default: off to save memory)")
    args = ap.parse_args()
    if args.ratio_aw < 0:
        args.ratio_aw = args.ratio_ah
    run_evaluation(args)


if __name__ == "__main__":
    main()
