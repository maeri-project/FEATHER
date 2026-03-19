#!/usr/bin/env python3
"""
Plot MINISA cycle breakdown (stacked bars, log-scale) + compute utilization (line).

Bar heights use log y-axis for overall cycle count.  Within each bar the
segments are sized by their *ratio* of total cycles (proportional in visual
log-space), so the coloring accurately reflects latency breakdown regardless
of log-scale distortion.

Input: benchmark_summary.csv from evaluate.py bench output.

Example:
  python fig_lat_breakdown_comp_utilization.py --csv benchmark_summary.csv --ah 16 --aw 16 --output fig.pdf
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

COLOR_LIST = ["#6C8EBF", "#82B366", "#D79B00", "#B85450", "#9673A6", "#B46504", "#D6B656", "#23445D"]

CYCLE_COLS = [
    ("cycles_load_in", "Load In"),
    ("cycles_load_w", "Load W"),
    ("cycles_compute", "Compute"),
    ("cycles_out_to_stream", "Out->Stream"),
    ("cycles_store_out", "Store Out"),
    ("cycles_load_inst", "Load Inst"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to benchmark_summary*.csv")
    ap.add_argument("--ah", type=int, required=True, help="AH (array height) to plot")
    ap.add_argument("--aw", type=int, default=-1, help="AW (array width) to plot (-1 = same as AH)")
    ap.add_argument("--category", default=None, help="Optional category filter (exact match)")
    ap.add_argument("--sort", choices=["category_name","compute_util","total_cycles","none"], default="category_name")
    ap.add_argument("--rotate3", action="store_true", help="Move left-most 3 cases to the end (paper ordering)")
    ap.add_argument("--output", required=True, help="Output figure path (pdf/png)")
    ap.add_argument("--fig_w", type=float, default=16.0)
    ap.add_argument("--fig_h", type=float, default=9.0)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    if args.aw < 0:
        args.aw = args.ah

    df = pd.read_csv(args.csv)

    # Required columns
    need = {"ah","aw","category","name","M","K","N","cycles_total","compute_util_avg"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {missing}")

    df = df[(df["ah"] == args.ah) & (df["aw"] == args.aw)].copy()
    if args.category is not None:
        df = df[df["category"] == args.category].copy()

    if df.empty:
        raise SystemExit(f"No rows after filtering; check --ah/--aw/--category.")

    # label as "M,K,N"
    df["case"] = df.apply(lambda r: f"{int(r['M'])},{int(r['K'])},{int(r['N'])}", axis=1)

    # sorting
    if args.sort == "compute_util":
        df = df.sort_values("compute_util_avg", ascending=False)
    elif args.sort == "total_cycles":
        df = df.sort_values("cycles_total", ascending=False)
    elif args.sort == "category_name":
        df = df.sort_values(["category","name","M","K","N"])

    df = df.reset_index(drop=True)

    # default behavior: rotate first 3 to end (paper ordering)
    if args.rotate3:
        if len(df) > 3:
            df = pd.concat([df.iloc[3:], df.iloc[:3]], ignore_index=True)

    x = np.arange(len(df))
    bar_width = 0.7

    fig, ax1 = plt.subplots(figsize=(args.fig_w, args.fig_h))
    ax1.set_yscale("log")

    # --- Ratio-based stacked bars in log space ---
    # For each workload, total bar height = cycles_total (on log axis).
    # Within each bar, segments are sized proportionally to their ratio
    # of total cycles. In log-space this means geometric interpolation:
    #   boundary_i = y_floor * (total / y_floor) ^ cumulative_ratio_i
    # where y_floor is the bottom of the log axis.

    # Collect all cycle columns
    all_vals = {}
    for col, label in CYCLE_COLS:
        if col not in df.columns:
            raise SystemExit(f"Missing cycle column: {col}")
        all_vals[col] = df[col].astype(float).values

    totals = np.zeros(len(df))
    for col, _ in CYCLE_COLS:
        totals += all_vals[col]

    # y_floor: minimum bar bottom (must be > 0 for log scale)
    y_floor = 1.0

    # Draw bars as individual rectangles with geometric interpolation
    for i in range(len(df)):
        total = max(totals[i], y_floor + 1)
        cum_ratio = 0.0
        for idx, (col, label) in enumerate(CYCLE_COLS):
            val = all_vals[col][i]
            ratio = val / total if total > 0 else 0.0
            if ratio <= 0:
                cum_ratio += ratio
                continue

            # Geometric interpolation in log space
            bot = y_floor * (total / y_floor) ** cum_ratio
            cum_ratio += ratio
            top = y_floor * (total / y_floor) ** cum_ratio

            rect = Rectangle(
                (x[i] - bar_width / 2, bot),
                bar_width,
                top - bot,
                facecolor=COLOR_LIST[idx % len(COLOR_LIST)],
                edgecolor="k",
                linewidth=0.5,
            )
            ax1.add_patch(rect)

    # Add ratio percentage labels inside each segment (for segments > 8% of total)
    for i in range(len(df)):
        total = max(totals[i], y_floor + 1)
        cum_ratio = 0.0
        for idx, (col, label) in enumerate(CYCLE_COLS):
            val = all_vals[col][i]
            ratio = val / total if total > 0 else 0.0
            if ratio <= 0:
                cum_ratio += ratio
                continue
            bot = y_floor * (total / y_floor) ** cum_ratio
            cum_ratio += ratio
            top = y_floor * (total / y_floor) ** cum_ratio
            if ratio >= 0.08:
                mid_y = np.sqrt(bot * top)  # geometric midpoint
                pct_text = f"{ratio*100:.0f}%"
                ax1.text(x[i], mid_y, pct_text, ha="center", va="center",
                         fontsize=7, color="white", fontweight="bold")

    # Create legend handles manually (rectangles don't auto-legend)
    legend_handles = []
    for idx, (col, label) in enumerate(CYCLE_COLS):
        handle = plt.Rectangle((0, 0), 1, 1,
                                facecolor=COLOR_LIST[idx % len(COLOR_LIST)],
                                edgecolor="k", linewidth=0.5, label=label)
        legend_handles.append(handle)

    ax1.set_ylabel("Cycles", fontsize=22)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["case"].tolist(), rotation=90, fontsize=22)
    ax1.tick_params(axis='y', labelsize=22)

    # Set y limits to encompass all bars
    y_max = totals.max() * 3 if len(totals) > 0 else 1e6
    ax1.set_ylim(y_floor, y_max)
    ax1.set_xlim(-0.5, len(df) - 0.5)

    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    # compute utilization on right axis
    ax2 = ax1.twinx()
    util_vals = df["compute_util_avg"].astype(float).values * 100
    ax2.plot(
        x,
        util_vals,
        marker="o",
        linewidth=1.8,
        color="#990000",
        label="Compute utilization",
    )
    ax2.set_ylabel("Compute utilization (%)", fontsize=22)
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis='y', labelsize=22)

    # combined legend
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(legend_handles + h2,
               [h.get_label() for h in legend_handles] + l2,
               loc="upper center", ncol=4, frameon=True, fontsize=22,
               columnspacing=0.3, labelspacing=0.1, handletextpad=0.2, borderpad=0.2)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
