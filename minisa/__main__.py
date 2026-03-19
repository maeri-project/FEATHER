#!/usr/bin/env python3
"""Unified MINISA CLI — ``python -m minisa <command>``.

Subcommands
-----------
search      Mapping-layout cosearch (single workload, batch CSV, or JSON template)
instcmp     Instruction comparison (MINISA ISA vs explicit micro-instruction)
compare     GPU/TPU comparison and analysis
plot        Generate publication-quality figures
evaluate    Full pipeline (search + instcmp + plot)
gui         Launch interactive visualization
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ── helpers ──────────────────────────────────────────────────────────

def _sram_defaults():
    return "4:4,8:16,16:64,32:64,64:64,128:128,256:256"

def _instbuf_defaults():
    return "4:0.5,8:1,16:2,32:3,64:5,128:10,256:20"

def _add_hw_args(p: argparse.ArgumentParser) -> None:
    """Add common hardware-config arguments."""
    p.add_argument("--ah", type=str, default="16",
                   help="Comma-separated AH values")
    p.add_argument("--aw", type=str, default="same",
                   help="AW values: 'same' (=AH), comma-separated, "
                        "or per-AH groups via '/' (e.g. 4,16,64/8,32,128)")
    p.add_argument("--sram-map", type=str, default=_sram_defaults(),
                   help="AH:MB SRAM map")
    p.add_argument("--instbuf-map", type=str, default=_instbuf_defaults(),
                   help="AH:MB instruction buffer map")
    p.add_argument("--alloc", type=str, default="0.4,0.4,0.2",
                   help="SRAM allocation (stream,stationary,output)")


# =====================================================================
# search — mapping-layout cosearch
# =====================================================================

def _cmd_search(args: argparse.Namespace) -> None:
    from .config import make_cfg, ceil_div
    from .workload import parse_ah_aw_pairs, parse_sram_map, load_workload_csv, sanitize_filename
    from .search import co_search_layout_mapping, layout_constrained_search
    from .trace import generate_trace_gemm, verify_trace, verify_config, export_config_json

    sram_map = parse_sram_map(args.sram_map)
    instbuf_map = parse_sram_map(args.instbuf_map)
    fs, fsta, fout = [float(x) for x in args.alloc.split(",")]
    alloc = (fs, fsta, fout)

    # --- JSON template mode (ACT-style layout-constrained search) ---
    if args.input:
        _search_json_mode(args, sram_map, instbuf_map, alloc)
        return

    # --- Single-workload or CSV batch mode ---
    ah_aw_pairs = parse_ah_aw_pairs(args.ah, args.aw)

    if args.csv:
        _search_csv_mode(args, ah_aw_pairs, sram_map, instbuf_map, alloc)
        return

    # Single workload
    if args.M is None or args.K is None or args.N is None:
        print("Error: provide --M --K --N for single workload, "
              "--csv for batch, or --input for JSON template.", file=sys.stderr)
        sys.exit(1)

    M, K, N = args.M, args.K, args.N

    for ah, aw in ah_aw_pairs:
        cfg = make_cfg(ah, aw, sram_map, instbuf_map, alloc, freq_ghz=1.0)

        if args.layout_constrained:
            ow = getattr(args, 'order_w', 0)
            oi = getattr(args, 'order_i', 0)
            oo = getattr(args, 'order_o', 0)
            candidates = layout_constrained_search(M, K, N, cfg, ow, oi, oo)
            if not candidates:
                print(f"  AH={ah} AW={aw}: no valid mapping found")
                continue
            best = candidates[0]
            print(f"  AH={ah} AW={aw}: Mt={best.Mt} Kt={best.Kt} Nt={best.Nt} "
                  f"cycles={best.cycles_total} util={best.utilization:.4f}")
        else:
            sr = co_search_layout_mapping(M, K, N, cfg, generate_trace=True)
            tb = sr.trace_bundle
            print(f"  AH={ah} AW={aw}: orders=({sr.order_w},{sr.order_i},{sr.order_o}) "
                  f"Mt={sr.Mt} Kt={sr.Kt} Nt={sr.Nt} "
                  f"cycles={sr.cycles_total} dataflow={sr.dataflow}")

            if args.verify and tb is not None:
                ok1, e1 = verify_trace(tb)
                ok2, e2 = verify_config(tb)
                s1 = "PASS" if ok1 else f"FAIL(err={e1:.2e})"
                s2 = "PASS" if ok2 else f"FAIL(err={e2:.2e})"
                print(f"           ISA:{s1}  CFG:{s2}")

            if args.output and tb is not None:
                out_path = str(args.output)
                if out_path.endswith(".json"):
                    export_config_json(M, K, N, cfg, sr, out_path)
                    print(f"           -> {out_path}")


def _search_csv_mode(args, ah_aw_pairs, sram_map, instbuf_map, alloc):
    """Batch search over CSV workloads."""
    import pandas as pd
    from .config import make_cfg
    from .workload import load_workload_csv
    from .search import co_search_layout_mapping
    from .trace import verify_trace, verify_config

    df = load_workload_csv(Path(args.csv))
    rows = df.to_dict(orient="records")
    print(f"Loaded {len(rows)} workloads, {len(ah_aw_pairs)} configs")

    results = []
    for row in rows:
        cat = str(row.get("category", ""))
        name = str(row.get("name", ""))
        M, K, N = int(row["M"]), int(row["K"]), int(row["N"])
        for ah, aw in ah_aw_pairs:
            cfg = make_cfg(ah, aw, sram_map, instbuf_map, alloc, freq_ghz=1.0)
            try:
                sr = co_search_layout_mapping(M, K, N, cfg, generate_trace=args.verify)
                r = {"category": cat, "name": name, "M": M, "K": K, "N": N,
                     "ah": ah, "aw": aw,
                     "order_w": sr.order_w, "order_i": sr.order_i, "order_o": sr.order_o,
                     "Mt": sr.Mt, "Kt": sr.Kt, "Nt": sr.Nt,
                     "cycles_total": sr.cycles_total, "dataflow": sr.dataflow,
                     "inst_bytes": sr.inst_bytes}
                if args.verify and sr.trace_bundle:
                    ok1, _ = verify_trace(sr.trace_bundle)
                    ok2, _ = verify_config(sr.trace_bundle)
                    r["verify_ok"] = ok1 and ok2
                results.append(r)
                print(f"  [{cat:15s}] {name:25s} M={M:6d} K={K:5d} N={N:5d} "
                      f"AH={ah:3d} AW={aw:3d} cycles={sr.cycles_total}")
            except Exception as e:
                print(f"  [{cat:15s}] {name:25s} AH={ah} AW={aw} ERROR: {e}")

    if args.output:
        out = pd.DataFrame(results)
        out.to_csv(str(args.output), index=False)
        print(f"\nResults written to {args.output} ({len(results)} rows)")


def _search_json_mode(args, sram_map, instbuf_map, alloc):
    """ACT-style layout-constrained search from JSON template."""
    # Re-use the ACT compiler logic directly
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from compiler.ACT.launch_cost_model import (
        parse_input, extract_layer_params, feather_spec_to_config,
        process_layer, assemble_output,
    )

    input_data = parse_input(Path(args.input))
    feather_spec = input_data["FEATHER_spec"]
    print(f"FEATHER spec: AH={feather_spec['AH']}, AW={feather_spec['AW']}")

    layer_results = {}
    for layer_entry in input_data["layer"]:
        layer_name, layer_params = extract_layer_params(layer_entry)
        result = process_layer(layer_name, layer_params, feather_spec,
                               do_verify=args.verify)
        layer_results[layer_name] = result

    output_data = assemble_output(input_data, layer_results)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nOutput written to {args.output}")


# =====================================================================
# instcmp — instruction comparison
# =====================================================================

def _cmd_instcmp(args: argparse.Namespace) -> None:
    import pandas as pd
    from .workload import parse_ah_aw_pairs, parse_sram_map, load_workload_csv
    from .evaluate import run_one_instcmp

    ah_aw_pairs = parse_ah_aw_pairs(args.ah, args.aw)
    sram_map = parse_sram_map(args.sram_map)
    instbuf_map = parse_sram_map(args.instbuf_map)
    fs, fsta, fout = [float(x) for x in args.alloc.split(",")]
    alloc = (fs, fsta, fout)

    df = load_workload_csv(Path(args.csv))
    rows = df.to_dict(orient="records")
    print(f"Loaded {len(rows)} workloads, {len(ah_aw_pairs)} configs")

    import numpy as np
    results = []
    for row in rows:
        cat = str(row.get("category", ""))
        name = str(row.get("name", ""))
        M, K, N = int(row["M"]), int(row["K"]), int(row["N"])
        for ah, aw in ah_aw_pairs:
            try:
                r = run_one_instcmp(cat, name, M, K, N, ah, aw,
                                    sram_map, instbuf_map, alloc)
                results.append(r)
                ratio = r.get("inst_bytes_ratio_explicit_vs_minisa", 0)
                print(f"  [{cat:15s}] {name:25s} AH={ah:3d} AW={aw:3d} "
                      f"ratio={ratio:.1f}x")
            except Exception as e:
                print(f"  [{cat:15s}] {name:25s} AH={ah} AW={aw} ERROR: {e}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inst_compare.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nInstruction comparison written to {out_path} ({len(results)} rows)")

    # Print summary
    df_ic = pd.read_csv(out_path)
    for ah, aw in ah_aw_pairs:
        sub = df_ic[(df_ic["ah"] == ah) & (df_ic["aw"] == aw)]
        if len(sub) > 0 and "inst_bytes_ratio_explicit_vs_minisa" in sub.columns:
            ratios = sub["inst_bytes_ratio_explicit_vs_minisa"]
            geo = float(np.exp(np.mean(np.log(ratios[ratios > 0]))))
            print(f"  AH={ah:3d} AW={aw:3d}: inst reduction geo-mean = {geo:.1f}x")


# =====================================================================
# compare — GPU/TPU comparison + analysis
# =====================================================================

def _cmd_compare(args: argparse.Namespace) -> None:
    from .analyze import main as analyze_main

    # Build argv for the existing analyze CLI
    argv = ["--bench-csv", args.bench_csv,
            "--inst-csv", args.inst_csv,
            "--out-dir", args.out_dir]
    if args.ah:
        argv += ["--ah", str(args.ah)]
    if args.aw:
        argv += ["--aw", str(args.aw)]
    if hasattr(args, 'configs_multi') and args.configs_multi:
        argv += ["--configs-multi", args.configs_multi]
    if hasattr(args, 'aw_scaling') and args.aw_scaling:
        argv += ["--aw-scaling", args.aw_scaling]
    if hasattr(args, 'fixed_pe') and args.fixed_pe:
        argv += ["--fixed-pe", args.fixed_pe]
    if hasattr(args, 'jobs') and args.jobs:
        argv += ["--jobs", str(args.jobs)]

    sys.argv = ["minisa compare"] + argv
    analyze_main()


# =====================================================================
# plot — generate figures
# =====================================================================

def _cmd_plot(args: argparse.Namespace) -> None:
    import subprocess

    fig_dir = Path(args.out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    minisa_dir = Path(__file__).resolve().parent
    fig_drawer = minisa_dir / "figure_drawer"

    scripts_run = 0

    # Instruction reduction
    ic_path = args.inst_csv
    if ic_path and Path(ic_path).exists():
        cmd = [sys.executable, str(fig_drawer / "fig_instr_reduction.py"),
               "--csv", ic_path, "--out-dir", str(fig_dir),
               "--ratio-ah", str(args.ratio_ah), "--ratio-aw", str(args.ratio_aw)]
        print(f"  Running: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  -> {fig_dir}/instr_reduction.pdf")
            scripts_run += 1
        else:
            print(f"  FAILED: {r.stderr[:200]}")

    # Speedup over micro-instruction
    if ic_path and Path(ic_path).exists():
        cmd = [sys.executable, str(fig_drawer / "fig_speedup_over_micro_instr.py"),
               "--csv", ic_path, "--out-dir", str(fig_dir)]
        print(f"  Running: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  -> {fig_dir}/speedup_over_micro_instr.pdf")
            scripts_run += 1
        else:
            print(f"  FAILED: {r.stderr[:200]}")

    # Latency breakdown + compute utilization (per config)
    bench_path = args.bench_csv
    if bench_path and Path(bench_path).exists():
        import pandas as pd
        from .workload import parse_ah_aw_pairs
        dfb = pd.read_csv(bench_path)
        configs = dfb[["ah", "aw"]].drop_duplicates().values.tolist()
        for ah, aw in configs:
            out_fig = fig_dir / f"latency_compute_utilization_AH{ah}_AW{aw}.pdf"
            cmd = [sys.executable,
                   str(fig_drawer / "fig_lat_breakdown_comp_utilization.py"),
                   "--csv", bench_path,
                   "--ah", str(int(ah)), "--aw", str(int(aw)),
                   "--output", str(out_fig)]
            print(f"  Running: {' '.join(cmd)}")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"  -> {out_fig}")
                scripts_run += 1
            else:
                print(f"  FAILED (AH={ah} AW={aw}): {r.stderr[:200]}")

    # GPU/TPU comparison
    gpu_tpu_csv = Path(args.out_dir) / "analysis" / "gpu_tpu_comparison.csv"
    if gpu_tpu_csv.exists():
        out_fig = fig_dir / "latency_vs_gpu_tpu.pdf"
        cmd = [sys.executable, str(fig_drawer / "fig_latency_vs_gpu_tpu.py"),
               "--csv", str(gpu_tpu_csv), "--output", str(out_fig)]
        print(f"  Running: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  -> {out_fig}")
            scripts_run += 1
        else:
            print(f"  FAILED: {r.stderr[:200]}")

    print(f"\n{scripts_run} figures generated in {fig_dir}/")


# =====================================================================
# evaluate — full pipeline
# =====================================================================

def _cmd_evaluate(args: argparse.Namespace) -> None:
    from .evaluate import main as evaluate_main
    # Forward to the existing evaluate CLI
    sys.argv = ["minisa evaluate"] + sys.argv[2:]
    evaluate_main()


# =====================================================================
# gui — interactive visualization
# =====================================================================

def _cmd_gui(args: argparse.Namespace) -> None:
    from .minisa_gui import main as gui_main
    gui_main()


# =====================================================================
# top-level parser
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m minisa",
        description="MINISA ISA Toolchain — unified CLI",
    )
    sub = ap.add_subparsers(dest="command", help="Available commands")

    # ── search ───────────────────────────────────────────────────────
    p_search = sub.add_parser(
        "search",
        help="Mapping-layout cosearch (single workload, batch CSV, or JSON template)",
        description="Run MINISA mapping-layout cosearch.\n\n"
                    "Three modes:\n"
                    "  Single workload:  -M 24 -K 48 -N 512 --ah 16 --aw 16\n"
                    "  Batch CSV:        --csv workloads.csv --ah 4,8,16 --aw ...\n"
                    "  JSON template:    --input template.json (layout-constrained)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_search.add_argument("-M", type=int, default=None, help="M dimension")
    p_search.add_argument("-K", type=int, default=None, help="K dimension")
    p_search.add_argument("-N", type=int, default=None, help="N dimension")
    p_search.add_argument("--csv", type=str, default=None,
                          help="CSV workload file for batch search")
    p_search.add_argument("--input", type=str, default=None,
                          help="JSON template (ACT-style layout-constrained search)")
    p_search.add_argument("--output", "-o", type=str, default=None,
                          help="Output file (.json for single, .csv for batch)")
    p_search.add_argument("--layout-constrained", action="store_true",
                          help="Fix layout orders (use --order-w/i/o)")
    p_search.add_argument("--order-w", type=int, default=0, help="Weight layout order (0-5)")
    p_search.add_argument("--order-i", type=int, default=0, help="Input layout order (0-5)")
    p_search.add_argument("--order-o", type=int, default=0, help="Output layout order (0-5)")
    p_search.add_argument("--verify", action="store_true",
                          help="Run functional verification")
    _add_hw_args(p_search)
    p_search.set_defaults(func=_cmd_search)

    # ── instcmp ──────────────────────────────────────────────────────
    p_ic = sub.add_parser(
        "instcmp",
        help="Instruction comparison (MINISA ISA vs explicit micro-instruction)",
    )
    p_ic.add_argument("--csv", type=str, required=True,
                      help="CSV workload file")
    p_ic.add_argument("--out-dir", type=str, required=True,
                      help="Output directory")
    _add_hw_args(p_ic)
    p_ic.set_defaults(func=_cmd_instcmp)

    # ── compare ──────────────────────────────────────────────────────
    p_cmp = sub.add_parser(
        "compare",
        help="GPU/TPU comparison and analysis",
    )
    p_cmp.add_argument("--bench-csv", type=str, required=True,
                       help="Benchmark summary CSV")
    p_cmp.add_argument("--inst-csv", type=str, required=True,
                       help="Instruction comparison CSV")
    p_cmp.add_argument("--out-dir", type=str, required=True,
                       help="Output directory")
    p_cmp.add_argument("--ah", type=int, default=16)
    p_cmp.add_argument("--aw", type=int, default=-1,
                       help="AW (-1 = same as AH)")
    p_cmp.add_argument("--configs-multi", type=str, default=None,
                       help="Multi-config: comma-separated AHxAW pairs")
    p_cmp.add_argument("--aw-scaling", type=str, default=None,
                       help="AW-level scaling: per-AH widening")
    p_cmp.add_argument("--fixed-pe", type=str, default=None,
                       help="Fixed-PE comparison: per-AH configs")
    p_cmp.add_argument("--jobs", type=int, default=1)
    p_cmp.set_defaults(func=_cmd_compare)

    # ── plot ─────────────────────────────────────────────────────────
    p_plot = sub.add_parser(
        "plot",
        help="Generate publication-quality figures",
    )
    p_plot.add_argument("--bench-csv", type=str, default=None,
                        help="Benchmark summary CSV")
    p_plot.add_argument("--inst-csv", type=str, default=None,
                        help="Instruction comparison CSV")
    p_plot.add_argument("--out-dir", type=str, required=True,
                        help="Output directory (figures/ subdirectory created)")
    p_plot.add_argument("--ratio-ah", type=int, default=16,
                        help="AH for instruction reduction ratio lines")
    p_plot.add_argument("--ratio-aw", type=int, default=128,
                        help="AW for ratio lines")
    p_plot.set_defaults(func=_cmd_plot)

    # ── evaluate ─────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "evaluate",
        help="Full pipeline (search + instcmp + plot)",
        description="Run the full evaluation pipeline. "
                    "Delegates to minisa.evaluate with all arguments forwarded.",
        add_help=False,
    )
    p_eval.set_defaults(func=_cmd_evaluate)

    # ── gui ──────────────────────────────────────────────────────────
    p_gui = sub.add_parser("gui", help="Launch interactive visualization")
    p_gui.set_defaults(func=_cmd_gui)

    # ── dispatch ─────────────────────────────────────────────────────
    args = ap.parse_args()
    if args.command is None:
        ap.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
