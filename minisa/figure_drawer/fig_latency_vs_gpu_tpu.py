#!/usr/bin/env python3
"""Grouped bar chart: FEATHER+ vs GPU vs TPU latency comparison.

Usage:
    python -m minisa.figure_drawer.fig_latency_vs_gpu_tpu \
        --csv out/analysis/gpu_tpu_comparison.csv --output fig_latency_gpu_tpu.pdf
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('axes', labelsize=22)
plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)
plt.rc('font', size=22)


def main():
    ap = argparse.ArgumentParser(description="FEATHER+ vs GPU vs TPU latency comparison")
    ap.add_argument("--csv", required=True, help="gpu_tpu_comparison.csv from analyze.py")
    ap.add_argument("--output", required=True, help="Output figure path (pdf/png)")
    ap.add_argument("--log", action="store_true", help="Use log scale for y-axis")
    ap.add_argument("--fig-w", type=float, default=18.0)
    ap.add_argument("--fig-h", type=float, default=8.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    has_gpu = "gpu_latency_us" in df.columns
    has_tpu = "tpu_latency_us" in df.columns

    if not has_gpu and not has_tpu:
        print("No GPU or TPU latency columns found; nothing to plot.")
        return

    df["case"] = df.apply(
        lambda r: f"{int(r['M'])},{int(r['K'])},{int(r['N'])}", axis=1)
    x = np.arange(len(df))

    n_bars = 1 + int(has_gpu) + int(has_tpu)
    group_w = 0.8
    bw = group_w / n_bars

    fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h))

    bar_idx = 0
    if has_gpu:
        vals = df["gpu_latency_us"].fillna(0)
        ax.bar(x - group_w / 2 + (bar_idx + 0.5) * bw, vals,
               width=bw, color="#6C8EBF", edgecolor="black", label="GPU (RTX5090)")
        bar_idx += 1

    if has_tpu:
        vals = df["tpu_latency_us"].fillna(0)
        ax.bar(x - group_w / 2 + (bar_idx + 0.5) * bw, vals,
               width=bw, color="#82B366", edgecolor="black", label="TPUv6e8")
        bar_idx += 1

    # FEATHER+ (rightmost) — label includes config if available
    feather_label = "FEATHER+"
    if "ah" in df.columns and "aw" in df.columns:
        ah = int(df["ah"].iloc[0])
        aw = int(df["aw"].iloc[0])
        n_inst = int(df["n_instances"].iloc[0]) if "n_instances" in df.columns else 1
        feather_label = f"FEATHER+ ({ah}×{aw})×{n_inst}"
    elif "n_instances" in df.columns:
        n_inst = int(df["n_instances"].iloc[0])
        feather_label = f"FEATHER+ ×{n_inst}"
    ax.bar(x - group_w / 2 + (bar_idx + 0.5) * bw, df["feather_latency_us"],
           width=bw, color="#990000", edgecolor="black", label=feather_label)
    bar_idx += 1

    if args.log:
        ax.set_yscale("log")

    ax.set_ylabel("Latency (us)")
    ax.set_xlabel("GEMM shape (M,K,N)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["case"].tolist(), rotation=90, ha="center")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=22, ncol=n_bars, loc="upper left", frameon=True,
              bbox_to_anchor=(0.0, 0.92))

    # Annotate speedups
    annotations = []
    if has_gpu and "speedup_vs_gpu" in df.columns:
        valid = df["speedup_vs_gpu"].dropna()
        pos = valid.values[valid.values > 0]
        if len(pos):
            geo = float(np.exp(np.mean(np.log(pos))))
            annotations.append(f"vs GPU: {geo:.1f}x")
    if has_tpu and "speedup_vs_tpu" in df.columns:
        valid = df["speedup_vs_tpu"].dropna()
        pos = valid.values[valid.values > 0]
        if len(pos):
            geo = float(np.exp(np.mean(np.log(pos))))
            annotations.append(f"vs TPU: {geo:.1f}x")
    if annotations:
        label = "geo-mean speedup: " + ", ".join(annotations)
        if "ah" in df.columns and "aw" in df.columns:
            ah = int(df["ah"].iloc[0])
            aw = int(df["aw"].iloc[0])
            n_inst = int(df["n_instances"].iloc[0]) if "n_instances" in df.columns else 1
            label = f"{n_inst}x FEATHER+ ({ah}×{aw}) — {label}"
        elif "n_instances" in df.columns:
            n_inst = int(df["n_instances"].iloc[0])
            label = f"{n_inst}x FEATHER+ instances — {label}"
        ax.text(0.01, 0.98, label,
                transform=ax.transAxes, ha="left", va="top", fontsize=22,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="black", alpha=0.9))

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
