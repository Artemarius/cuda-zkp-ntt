#!/usr/bin/env python3
"""Generate benchmark comparison charts from measured data.

All numbers from RTX 3060 Laptop GPU (Ampere sm_86, CUDA 12.8).
Release build, 5-rep mean.

Usage:
    python scripts/plot_benchmarks.py
    # Outputs PNGs to results/charts/
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "savefig.dpi": 180,
    "savefig.pad_inches": 0.15,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "charts")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_ntt_compute():
    """NTT compute latency: naive radix-2 vs optimized radix-256."""
    scales = ["2^15", "2^16", "2^18", "2^20", "2^22"]
    naive =    [0.169, 0.299, 1.85,  7.13, 31.2]
    optimized = [0.212, 0.404, 1.54,  5.99, 26.5]

    x = np.arange(len(scales))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - w/2, naive, w, label="Naive (radix-2)", color="#5B9BD5", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, optimized, w, label="Optimized (radix-256)", color="#ED7D31", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("NTT Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("NTT Compute Latency — Naive vs Radix-256 Fused")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 50)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate speedup
    for i in range(len(scales)):
        if optimized[i] < naive[i]:
            speedup = naive[i] / optimized[i]
            ax.annotate(f"{speedup:.2f}x",
                        xy=(x[i] + w/2, optimized[i]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold", color="#C00000")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "ntt_compute_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_async_pipeline():
    """Async pipeline: pipelined vs sequential throughput."""
    scales = ["2^18", "2^20", "2^22"]
    sequential = [49.4, 188, 579]
    pipelined =  [29.7, 141, 541]

    x = np.arange(len(scales))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: latency bars
    bars1 = ax1.bar(x - w/2, sequential, w, label="Sequential (1-stream)", color="#A5A5A5", edgecolor="white", linewidth=0.5)
    bars2 = ax1.bar(x + w/2, pipelined, w, label="Pipelined (3-stream)", color="#70AD47", edgecolor="white", linewidth=0.5)

    ax1.set_xlabel("NTT Size (8 batches, pinned memory)")
    ax1.set_ylabel("End-to-End Latency (ms)")
    ax1.set_title("Async Pipeline Latency")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scales)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Right: speedup bars
    speedups = [s / p for s, p in zip(sequential, pipelined)]
    colors = ["#70AD47" if sp > 1.2 else "#FFC000" if sp > 1.1 else "#A5A5A5" for sp in speedups]
    bars3 = ax2.bar(scales, speedups, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("Pipeline Speedup")
    ax2.set_ylim(0.8, 2.0)
    ax2.axhline(y=1.0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, sp in zip(bars3, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"{sp:.2f}x", ha="center", fontweight="bold", fontsize=10)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "async_pipeline_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_kernel_profile():
    """Nsight Compute kernel profile: memory vs compute bound transition."""
    kernels = ["FF_mul\n(isolated)", "NTT Naive\nButterfly", "NTT Fused\n(radix-256)"]
    compute_pct = [64, 45, 69]
    memory_pct =  [92, 79, 54.74]
    ipc =         [1.87, 1.56, 2.41]

    x = np.arange(len(kernels))
    w = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: compute vs memory throughput
    bars1 = ax1.bar(x - w/2, compute_pct, w, label="Compute Throughput (%)", color="#ED7D31", edgecolor="white", linewidth=0.5)
    bars2 = ax1.bar(x + w/2, memory_pct, w, label="Memory Throughput (%)", color="#5B9BD5", edgecolor="white", linewidth=0.5)

    ax1.set_ylabel("Speed of Light (%)")
    ax1.set_title("Kernel Bottleneck: Memory vs Compute")
    ax1.set_xticks(x)
    ax1.set_xticklabels(kernels)
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate bottleneck
    labels = ["Memory\nBound", "Memory\nBound", "Compute\nBound"]
    label_colors = ["#2E75B6", "#2E75B6", "#C55A11"]
    for i, (lbl, col) in enumerate(zip(labels, label_colors)):
        dominant = max(compute_pct[i], memory_pct[i])
        ax1.text(x[i], dominant + 2, lbl, ha="center", fontsize=8, fontweight="bold", color=col)

    # Right: IPC comparison
    colors = ["#A5A5A5", "#A5A5A5", "#70AD47"]
    bars3 = ax2.bar(["FF_mul", "Naive\nButterfly", "Fused\nRadix-256"], ipc, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Instructions Per Cycle")
    ax2.set_title("Executed IPC")
    ax2.set_ylim(0, 3.0)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, val in zip(bars3, ipc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", fontweight="bold", fontsize=10)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "kernel_profile_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_sass_reduction():
    """SASS instruction count reduction: baseline vs branchless v2."""
    ops = ["ff_add", "ff_sub", "ff_mul", "ff_sqr"]
    baseline =    [127, 94, 571, 563]
    branchless =  [55,  55, 527, 511]

    x = np.arange(len(ops))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - w/2, baseline, w, label="Baseline", color="#A5A5A5", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, branchless, w, label="Branchless v2 (PTX)", color="#5B9BD5", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Finite-Field Operation")
    ax.set_ylabel("SASS Instructions (sm_86)")
    ax.set_title("Direction B — SASS Instruction Reduction")
    ax.set_xticks(x)
    ax.set_xticklabels(ops)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate reduction %
    for i in range(len(ops)):
        reduction = (1 - branchless[i] / baseline[i]) * 100
        ax.annotate(f"-{reduction:.0f}%",
                    xy=(x[i] + w/2, branchless[i]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold", color="#C00000")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "sass_instruction_reduction.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_barrett_vs_montgomery():
    """Barrett vs Montgomery single NTT latency comparison."""
    scales = ["2^15", "2^16", "2^18", "2^20", "2^22"]
    montgomery = [0.132, 0.244, 1.21, 5.51, 25.1]
    barrett =    [0.159, 0.308, 1.27, 5.61, 24.9]

    x = np.arange(len(scales))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - w/2, montgomery, w, label="Montgomery (OPTIMIZED)", color="#5B9BD5", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, barrett, w, label="Barrett", color="#70AD47", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("NTT Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Single NTT Latency — Montgomery vs Barrett (v1.2.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 50)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate delta
    for i in range(len(scales)):
        delta = (barrett[i] / montgomery[i] - 1) * 100
        color = "#C00000" if delta > 0 else "#006400"
        sign = "+" if delta > 0 else ""
        ax.annotate(f"{sign}{delta:.0f}%",
                    xy=(x[i] + w/2, barrett[i]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold", color=color)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "barrett_vs_montgomery.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_batch_throughput():
    """Batched NTT throughput: batch vs sequential at different sizes."""
    sizes = ["2^15", "2^18", "2^20", "2^22"]
    batched_8 =    [1.12, 10.4, 48.0, 219]
    sequential_8 = [1.70, 11.2, 48.3, 216]

    x = np.arange(len(sizes))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: latency bars
    bars1 = ax1.bar(x - w/2, sequential_8, w, label="Sequential 8x", color="#A5A5A5", edgecolor="white", linewidth=0.5)
    bars2 = ax1.bar(x + w/2, batched_8, w, label="Batched 8x", color="#ED7D31", edgecolor="white", linewidth=0.5)

    ax1.set_xlabel("NTT Size (8 NTTs, Barrett mode)")
    ax1.set_ylabel("Total Latency (ms)")
    ax1.set_title("Batched vs Sequential NTT (v1.2.0)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.legend(loc="upper left")
    ax1.set_yscale("log")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Right: speedup bars
    speedups = [s / b for s, b in zip(sequential_8, batched_8)]
    colors = ["#70AD47" if sp > 1.2 else "#FFC000" if sp > 1.05 else "#A5A5A5" for sp in speedups]
    bars3 = ax2.bar(sizes, speedups, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("Batch Speedup")
    ax2.set_ylim(0.8, 1.8)
    ax2.axhline(y=1.0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, sp in zip(bars3, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"{sp:.2f}x", ha="center", fontweight="bold", fontsize=10)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "batch_throughput.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_batch_scaling():
    """Per-NTT cost as batch size increases (Montgomery, 2^15)."""
    batch_sizes = [1, 4, 8, 16]
    per_ntt = [0.140, 0.131, 0.126, 0.124]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(batch_sizes, per_ntt, "o-", color="#5B9BD5", linewidth=2, markersize=8)
    ax.fill_between(batch_sizes, per_ntt, alpha=0.15, color="#5B9BD5")

    ax.set_xlabel("Batch Size (B)")
    ax.set_ylabel("Per-NTT Latency (ms)")
    ax.set_title("Per-NTT Cost vs Batch Size (Montgomery, 2^15)")
    ax.set_xticks(batch_sizes)
    ax.set_ylim(0.10, 0.16)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for i, (b, t) in enumerate(zip(batch_sizes, per_ntt)):
        ax.annotate(f"{t:.3f} ms",
                    xy=(b, t), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "batch_scaling.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_four_step_vs_barrett():
    """4-Step NTT vs Barrett single NTT latency comparison (v1.3.0)."""
    scales = ["2^15", "2^16", "2^18", "2^20", "2^22"]
    barrett =    [0.158, 0.300, 1.27,  5.60, 24.9]
    four_step =  [0.160, 0.491, 1.66,  7.03, 29.5]

    x = np.arange(len(scales))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - w/2, barrett, w, label="Barrett (cooperative outer)", color="#70AD47", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, four_step, w, label="4-Step (Bailey's algorithm)", color="#9B59B6", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("NTT Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Single NTT Latency — Barrett vs 4-Step (v1.3.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 50)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate delta (4-step overhead)
    for i in range(len(scales)):
        delta = (four_step[i] / barrett[i] - 1) * 100
        if delta > 1:
            ax.annotate(f"+{delta:.0f}%",
                        xy=(x[i] + w/2, four_step[i]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold", color="#C00000")
        else:
            ax.annotate(f"~0%",
                        xy=(x[i] + w/2, four_step[i]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold", color="#666666")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "four_step_vs_barrett.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_four_step_batch():
    """4-Step batched NTT vs Barrett batched (v1.3.0)."""
    sizes = ["2^15", "2^18", "2^20", "2^22"]
    barrett_batch =    [1.01, 9.65,  44.6, 199]
    four_step_batch =  [1.01, 11.4,  53.1, 241]

    x = np.arange(len(sizes))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: latency bars
    bars1 = ax1.bar(x - w/2, barrett_batch, w, label="Barrett batched 8x", color="#70AD47", edgecolor="white", linewidth=0.5)
    bars2 = ax1.bar(x + w/2, four_step_batch, w, label="4-Step batched 8x", color="#9B59B6", edgecolor="white", linewidth=0.5)

    ax1.set_xlabel("NTT Size (8 NTTs)")
    ax1.set_ylabel("Total Latency (ms)")
    ax1.set_title("Batched 8x NTT — Barrett vs 4-Step (v1.3.0)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.legend(loc="upper left")
    ax1.set_yscale("log")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Right: overhead ratio (how much slower is 4-step)
    ratios = [f / b for f, b in zip(four_step_batch, barrett_batch)]
    colors = ["#70AD47" if r <= 1.05 else "#FFC000" if r <= 1.2 else "#C00000" for r in ratios]
    bars3 = ax2.bar(sizes, ratios, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("4-Step / Barrett ratio")
    ax2.set_title("4-Step Overhead")
    ax2.set_ylim(0.8, 1.5)
    ax2.axhline(y=1.0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, r in zip(bars3, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{r:.2f}x", ha="center", fontweight="bold", fontsize=10)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "four_step_batch.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_all_modes_comparison():
    """All NTT modes comparison at n=2^22 (v1.3.0 summary)."""
    modes = ["Naive\n(radix-2)", "Montgomery\n(fused+coop)", "Barrett\n(fused+coop)", "4-Step\n(Bailey's)"]
    times = [26.2, 25.1, 24.9, 29.5]
    colors = ["#A5A5A5", "#5B9BD5", "#70AD47", "#9B59B6"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(modes, times, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Single NTT at n=2^22 — All Modes (v1.3.0)")
    ax.set_ylim(0, 35)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{t:.1f} ms", ha="center", fontweight="bold", fontsize=10)

    # Highlight best
    ax.annotate("Best", xy=(2, 24.9), xytext=(2.5, 20),
                arrowprops=dict(arrowstyle="->", color="#006400"),
                fontsize=10, fontweight="bold", color="#006400")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "all_modes_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


if __name__ == "__main__":
    print("Generating benchmark charts...")
    plot_ntt_compute()
    plot_async_pipeline()
    plot_kernel_profile()
    plot_sass_reduction()
    plot_barrett_vs_montgomery()
    plot_batch_throughput()
    plot_batch_scaling()
    plot_four_step_vs_barrett()
    plot_four_step_batch()
    plot_all_modes_comparison()
    print("Done.")
