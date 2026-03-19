#!/usr/bin/env python3
"""
Unified Fig1: merge the matrix multiplication cases from Fig1_FHE + Fig1_ZKP + Fig1_ChatGPT_OSS
into ONE plot (single x-axis), keeping x-ticks as 90 "M,K,N".

What it draws:
- Left y-axis (log): MINISA instruction bytes, grouped by (AH,AW) config (grayscale bars).
- Right y-axis: instruction-bytes / data-bytes ratio lines at a chosen (AH,AW)
  (MINISA = #990000, micro = black).
- Category separators and labels (FHE / ZKP / ChatGPT OSS).

Usage:
  python3 fig_instr_reduction.py --csv inst_compare.csv --out-dir out --ratio-ah 16 --ratio-aw 16
"""
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rc('axes', labelsize=25)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)
plt.rc('font', size=25)

CAT_ORDER  = ["FHE", "ZKP", "ChatGPT OSS"]

def geom_mean(x):
    x = np.array(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")

def log_ylim_for_75pct(ymin, ymax_data, target=0.75, extra=1.0):
    ymin = max(1e-12, float(ymin))
    ymax_data = max(ymin * 1.001, float(ymax_data))
    log_ymax = math.log(ymin) + (math.log(ymax_data) - math.log(ymin)) / target
    return math.exp(log_ymax) * extra

def data_bytes(M, K, N):
    return M*K + K*N + 4*M*N

def greys_for_configs(configs):
    vals = np.linspace(0.95, 0.15, len(configs)) if len(configs) > 1 else [0.5]
    return {cfg: str(v) for cfg, v in zip(configs, vals)}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ratio-ah", type=int, default=16)
    ap.add_argument("--ratio-aw", type=int, default=-1)
    ap.add_argument("--configs", default=None,
                    help="Filter to specific AH,AW pairs, e.g. '4x4,4x16,8x8'")
    args = ap.parse_args()
    if args.ratio_aw < 0:
        args.ratio_aw = args.ratio_ah

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # Build unique (ah, aw) configs sorted by ah then aw
    config_cols = df[["ah", "aw"]].drop_duplicates().sort_values(["ah", "aw"])
    configs = list(config_cols.itertuples(index=False, name=None))  # list of (ah, aw)

    if args.configs:
        allowed = set()
        for tok in args.configs.split(","):
            ah_s, aw_s = tok.strip().split("x")
            allowed.add((int(ah_s), int(aw_s)))
        configs = [c for c in configs if c in allowed]
        df = df[df.apply(lambda r: (int(r["ah"]), int(r["aw"])) in allowed, axis=1)]

    # collect all shapes across categories in a stable order
    parts = []
    for cat in CAT_ORDER:
        d = df[df["category"] == cat][["category", "name", "M", "K", "N"]].drop_duplicates()
        d = d.sort_values(["name", "M", "K", "N"])
        parts.append(d)
    shapes = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["category","name","M","K","N"]).reset_index(drop=True)
    x = np.arange(len(shapes))

    cfg_color = greys_for_configs(configs)

    fig, ax = plt.subplots(figsize=(25, 9))
    group_w = 0.92
    bw = group_w / len(configs)

    # Bars: MINISA instruction bytes per (AH, AW) config
    all_y = []
    for j, (ah, aw) in enumerate(configs):
        sub = df[(df["ah"] == ah) & (df["aw"] == aw)].merge(
            shapes, on=["category","name","M","K","N"], how="right")
        y = sub["minisa_inst_bytes"].to_numpy(dtype=float)
        all_y.append(y)
        xpos = x - group_w/2 + (j + 0.5) * bw
        label = f"AH={ah} AW={aw}" if ah != aw else f"AH=AW={ah}"
        ax.bar(xpos, y, width=bw, edgecolor="black", color=cfg_color[(ah, aw)], label=label)

    ax.set_yscale("log")
    ax.set_ylabel("MINISA instruction bytes (B)")
    ax.set_xlabel("GEMM shape (M,N,K): (M,K)x(K,N)->(M,N)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r.M)},{int(r.K)},{int(r.N)}" for r in shapes.itertuples(index=False)],
                       rotation=90, ha="center", fontsize=22)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    all_y = np.concatenate(all_y)
    y_max = float(np.nanmax(all_y))
    y_min = float(np.nanmin(all_y[all_y > 0])) if np.any(all_y > 0) else 1.0
    y_min_plot = max(1.0, y_min/2.0)
    ax.set_ylim(y_min_plot, log_ylim_for_75pct(y_min_plot, y_max, target=1, extra=1.35))

    # Right axis: inst/data ratio lines at chosen (AH, AW)
    ax2 = ax.twinx()
    ratio_ah, ratio_aw = args.ratio_ah, args.ratio_aw
    # Find the closest matching config
    available = set(configs)
    if (ratio_ah, ratio_aw) not in available:
        # Fallback to largest config
        ratio_ah, ratio_aw = configs[-1]
    ratio_label = f"AH={ratio_ah} AW={ratio_aw}" if ratio_ah != ratio_aw else f"AH=AW={ratio_ah}"

    subr = df[(df["ah"] == ratio_ah) & (df["aw"] == ratio_aw)].merge(
        shapes, on=["category","name","M","K","N"], how="right")
    db = np.array([data_bytes(int(r.M), int(r.K), int(r.N)) for r in shapes.itertuples(index=False)], dtype=float)
    r_minisa = subr["minisa_inst_bytes"].to_numpy(dtype=float) / db
    r_micro  = subr["explicit_inst_bytes"].to_numpy(dtype=float) / db

    ax2.plot(x, r_minisa, linewidth=3, marker="o", markersize=4, color="#990000",
             label=f"MINISA instruction/data @{ratio_label}")
    ax2.plot(x, r_micro,  linewidth=3, marker="X", markersize=4, color="black",
             label=f"Micro instruction/data @{ratio_label}")
    ax2.set_ylabel("Instruction bytes / data bytes")
    y2min, y2max = ax2.get_ylim()
    ax2.set_ylim(y2min, y2max * 1.25)

    # Category separators + labels
    bounds = []
    start = 0
    for cat in CAT_ORDER:
        cnt = len(shapes[shapes["category"] == cat])
        if cnt == 0:
            continue
        end = start + cnt
        bounds.append((cat, start, end))
        start = end

    for (cat, s, e) in bounds[:-1]:
        ax.axvline(e - 0.5, color="k", linestyle=":", linewidth=1, alpha=0.6)

    red = (subr["explicit_inst_bytes"] / subr["minisa_inst_bytes"]).to_numpy(dtype=float)
    txt = f"@ {ratio_label}: reduction (micro/MINISA)  geo-mean={geom_mean(red):.1f}x  median={np.median(red):.1f}x  max={np.max(red):.1f}x"
    ax.text(
        0.01, 0.98, txt,
        transform=ax.transAxes, ha="left", va="top", fontsize=22,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.9)
    )

    bar_handles = [Patch(facecolor=cfg_color[c], edgecolor="black",
                         label=(f"AH={c[0]} AW={c[1]}" if c[0] != c[1] else f"AH=AW={c[0]}"))
                   for c in configs]
    line_handles, line_labels = ax2.get_legend_handles_labels()
    handles = bar_handles + line_handles
    labels = [h.get_label() for h in bar_handles] + line_labels
    ax.legend(handles, labels, fontsize=22, ncol=4,
              columnspacing=0.25, labelspacing=0.10, handletextpad=0.25, borderpad=0.25,
              loc="upper left", bbox_to_anchor=(0, 0.92), frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "instr_reduction.pdf"), bbox_inches="tight", transparent=True)
    plt.close(fig)

if __name__ == "__main__":
    main()
