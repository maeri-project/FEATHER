#!/usr/bin/env python3
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rc('axes', labelsize=25)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)
plt.rc('font', size=25)


def geom_mean(x):
    x = np.array(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--configs", default=None,
                    help="Filter to specific AH,AW pairs, e.g. '4x4,4x16,8x8'")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # Build unique (ah, aw) configs sorted by total PEs
    config_cols = df[["ah", "aw"]].drop_duplicates().sort_values(["ah", "aw"])
    configs = list(config_cols.itertuples(index=False, name=None))

    if args.configs:
        allowed = set()
        for tok in args.configs.split(","):
            ah_s, aw_s = tok.strip().split("x")
            allowed.add((int(ah_s), int(aw_s)))
        configs = [c for c in configs if c in allowed]
        df = df[df.apply(lambda r: (int(r["ah"]), int(r["aw"])) in allowed, axis=1)]

    speedups = []
    for ah, aw in configs:
        dfn = df[(df["ah"] == ah) & (df["aw"] == aw)]
        speedups.append(geom_mean(dfn["total_cycles_ratio"].values))

    micro_norm = np.ones(len(configs))
    minisa_norm = np.array([1.0/s if (np.isfinite(s) and s > 0) else np.nan for s in speedups], dtype=float)

    x = np.arange(len(configs))
    bar_w = 0.35

    # 16:9 ratio
    fig, ax = plt.subplots(figsize=(16, 5))

    ax.bar(x - bar_w/2, micro_norm,  width=bar_w, edgecolor="black", color="black",
           label="Micro-instruction")
    ax.bar(x + bar_w/2, minisa_norm, width=bar_w, edgecolor="black", color="#990000",
           label="MINISA")

    # Curved arrow from the top of each black bar to the top of the red bar
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="k")
    for i in range(len(configs)):
        yb = float(micro_norm[i])
        yr = float(minisa_norm[i])
        if not (np.isfinite(yb) and np.isfinite(yr)):
            continue
        p1 = (x[i] - bar_w/2, yb)
        p2 = (x[i] + bar_w/2, yr)
        ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle="arc3,rad=-.15", **kw))
        s = float(speedups[i])
        ax.text(x[i], max(yb, yr) + 0.02, f"{s:.1f}x", ha="center", va="bottom",
                fontsize=20, fontweight="bold")

    # Geomean speedup text in top-left corner
    all_speedups = [s for s in speedups if np.isfinite(s) and s > 0]
    if all_speedups:
        overall_geo = geom_mean(all_speedups)
        ax.text(0.02, 0.90, f"Arrows: geomean speedup",
                transform=ax.transAxes, ha="left", va="center", fontsize=22, fontweight="bold")

    ax.set_xticks(x)
    labels = []
    for ah, aw in configs:
        labels.append(f"AH={ah}\nAW={aw}")
    ax.set_xticklabels(labels, fontsize=25)
    ax.set_ylabel("Normalized total cycles", fontsize=25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    # Headroom for arrows + annotations
    ymax = np.nanmax([np.nanmax(micro_norm), np.nanmax(minisa_norm), 1.0])
    ax.set_ylim(0, ymax * 1.35)

    ax.legend(fontsize=22, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.0),
              columnspacing=0.4, labelspacing=0.15, handletextpad=0.3, borderpad=0.15, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "speedup_over_micro_instr.pdf"), bbox_inches="tight", transparent=True)
    plt.close(fig)

if __name__ == "__main__":
    main()
