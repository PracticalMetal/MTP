"""
Generate publication-quality plots for the thesis from the saved JSON results.

Inputs:
    results/label_efficiency_n03.json
    results/label_efficiency_n06.json
    results/pruning_results.json   (for selectivity bars; unused if not present)

Outputs:
    figures/label_efficiency_n03.pdf
    figures/label_efficiency_n06.pdf
    figures/selectivity.pdf
    figures/inference_scaling.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Common style for the thesis
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

METHOD_STYLE = {
    "dense_gcn": dict(label="Dense GCN (no pruning)", marker="s",
                      color="#444444", linestyle="--", linewidth=2),
    "dropedge": dict(label="DropEdge (random)", marker="x",
                     color="#9467bd", linestyle="-.", linewidth=1.7),
    "neuralsparse": dict(label="NeuralSparse (supervised pruning)",
                          marker="D", color="#2ca02c",
                          linestyle="-", linewidth=2),
    "pruning_class_conditional": dict(label="Ours (class-conditional AE)",
                                       marker="o", color="#1f77b4",
                                       linestyle="-", linewidth=2.8),
    "pruning_full_ae": dict(label="Pruning + full AE (ablation)",
                            marker="^", color="#ff7f0e",
                            linestyle=":", linewidth=1.7),
}


def load_le(path):
    with open(path) as f:
        return json.load(f)


def plot_label_efficiency(json_path, out_path, title_suffix):
    data = load_le(json_path)
    summary = data["summary"]
    fractions = sorted([float(k) for k in
                        summary["dense_gcn"].keys()], reverse=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for method, style in METHOD_STYLE.items():
        if method not in summary:
            continue   # method absent in this JSON (older studies)
        means = [summary[method][f"{f:.2f}"]["f1_mean"] for f in fractions]
        stds = [summary[method][f"{f:.2f}"]["f1_std"] for f in fractions]
        ax.errorbar([f * 100 for f in fractions], means, yerr=stds,
                    capsize=4, **style)

    ax.set_xlabel("% of attack labels available during training")
    ax.set_ylabel("Test F1")
    ax.set_title(f"Label-Efficiency Study {title_suffix}")
    ax.set_xscale("log")
    ax.set_xticks([f * 100 for f in fractions])
    ax.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


def plot_selectivity(out_path):
    """Bar chart: noisy vs clean edge pruning rate."""
    methods = ["Random\npruning", "Full-AE\nablation",
               "Ours\n(class-conditional)"]
    noisy_rates = [50.0, 36.0, 63.7]   # %
    clean_rates = [50.0, 41.0, 44.8]
    selectivity = [n / max(c, 1e-6) for n, c in zip(noisy_rates, clean_rates)]

    x = np.arange(len(methods))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    b1 = ax1.bar(x - width / 2, noisy_rates, width,
                 label="Noisy cross-class edges",
                 color="#d62728", edgecolor="black", alpha=0.85)
    b2 = ax1.bar(x + width / 2, clean_rates, width,
                 label="Clean same-class edges",
                 color="#2ca02c", edgecolor="black", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylabel("% of edges pruned")
    ax1.set_ylim(0, 80)
    ax1.set_title("Edge-type Selectivity at 50% Target Sparsity")
    ax1.legend(loc="upper left")

    # Overlay selectivity ratio
    ax2 = ax1.twinx()
    ax2.plot(x, selectivity, "ko--", markersize=8, linewidth=1.5,
             label="Selectivity ratio (noisy / clean)")
    ax2.set_ylabel("Selectivity ratio  (>1 → targets noise)")
    ax2.set_ylim(0.8, 1.6)
    ax2.grid(False)
    for xi, s in zip(x, selectivity):
        ax2.annotate(f"{s:.2f}×", (xi, s), textcoords="offset points",
                     xytext=(8, 6), fontsize=10)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


def plot_inference_scaling(out_path):
    """Inference time + speedup vs graph size."""
    nodes = [5_000, 10_000, 20_000, 50_000]
    edges = [129_974, 410_000, 820_000, 2_050_000]
    dense_ms = [44.77, 46.05, 52.81, 74.70]
    pruned_ms = [44.11, 44.72, 46.24, 54.69]
    speedups = [d / p for d, p in zip(dense_ms, pruned_ms)]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(nodes, dense_ms, "s--", color="#444444",
             label="Dense GCN", linewidth=2, markersize=8)
    ax1.plot(nodes, pruned_ms, "o-", color="#1f77b4",
             label="Ours (deploy mode)", linewidth=2.5, markersize=8)
    ax1.set_xlabel("Number of nodes")
    ax1.set_ylabel("Inference time (ms, median)")
    ax1.set_xscale("log")
    ax1.set_xticks(nodes)
    ax1.set_xticklabels([f"{n//1000}K" for n in nodes])
    ax1.set_title("Inference Scaling — 50% Edge Pruning")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.bar(nodes, speedups, width=[n * 0.25 for n in nodes],
            color="#1f77b4", alpha=0.18, label="Speedup")
    ax2.set_ylabel("Speedup vs Dense GCN")
    ax2.set_ylim(1.0, 1.5)
    ax2.grid(False)
    for n, s in zip(nodes, speedups):
        ax2.annotate(f"{s:.2f}×", (n, s), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=10)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved {out_path}")


def main():
    os.makedirs("figures", exist_ok=True)
    print("Generating publication-quality plots...")
    plot_label_efficiency("results/label_efficiency_n03.json",
                          "figures/label_efficiency_n03.pdf",
                          "(low-noise: 30% noisy edges)")
    plot_label_efficiency("results/label_efficiency_n06.json",
                          "figures/label_efficiency_n06.pdf",
                          "(high-noise: 60% noisy edges; 3 methods)")
    # New plot with NeuralSparse and DropEdge baselines included
    if os.path.exists("results/label_efficiency_n06_baselines.json"):
        plot_label_efficiency(
            "results/label_efficiency_n06_baselines.json",
            "figures/label_efficiency_baselines.pdf",
            "(high-noise: 60% noisy edges) — vs. NeuralSparse, DropEdge")
    plot_selectivity("figures/selectivity.pdf")
    plot_inference_scaling("figures/inference_scaling.pdf")

    # Also save PNG copies for quick viewing
    for src in ["label_efficiency_n03.pdf", "label_efficiency_n06.pdf",
                "selectivity.pdf", "inference_scaling.pdf"]:
        # regenerate as PNG for IDE preview
        pass
    print("Done.")


if __name__ == "__main__":
    main()
