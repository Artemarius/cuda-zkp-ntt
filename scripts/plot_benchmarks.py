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


def plot_v150_vs_v140():
    """v1.5.0 vs v1.4.0 single NTT latency comparison at all sizes."""
    scales = ["2^15", "2^16", "2^18", "2^20", "2^22"]

    # v1.4.0 medians
    mont_v14 = [0.132, 0.244, 0.952, 4.11, 17.1]
    # v1.5.0 medians
    mont_v15 = [0.259, 0.624, 1.75,  5.53, 15.5]

    x = np.arange(len(scales))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - w/2, mont_v14, w, label="v1.4.0 Montgomery (radix-4)", color="#5B9BD5", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, mont_v15, w, label="v1.5.0 Montgomery (radix-8)", color="#ED7D31", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("NTT Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Single NTT Latency — v1.4.0 vs v1.5.0 (Montgomery)")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 30)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate delta at 2^22
    delta = (mont_v15[-1] / mont_v14[-1] - 1) * 100
    color = "#006400" if delta < 0 else "#C00000"
    sign = "" if delta < 0 else "+"
    ax.annotate(f"{sign}{delta:.0f}%",
                xy=(x[-1] + w/2, mont_v15[-1]),
                xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold", color=color)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "v150_vs_v140.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_version_history():
    """NTT latency progression across all versions at n=2^22."""
    versions = ["v1.1\nMont", "v1.2\nBarrett", "v1.3\n4-Step", "v1.4\nRadix-4", "v1.5\nRadix-8"]
    times = [25.1, 24.9, 29.5, 17.1, 15.5]
    colors = ["#A5A5A5", "#70AD47", "#9B59B6", "#5B9BD5", "#ED7D31"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(versions, times, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("NTT at n=2^22 — Version History (Best Mode Each Release)")
    ax.set_ylim(0, 35)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{t:.1f} ms", ha="center", fontweight="bold", fontsize=10)

    # Highlight best
    ax.annotate("Best\n-38% vs v1.1", xy=(4, 15.5), xytext=(3.2, 8),
                arrowprops=dict(arrowstyle="->", color="#006400"),
                fontsize=9, fontweight="bold", color="#006400")

    # Mark negative results
    ax.annotate("Negative\nresult", xy=(2, 29.5), xytext=(2.5, 33),
                arrowprops=dict(arrowstyle="->", color="#C00000"),
                fontsize=8, color="#C00000")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "version_history.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_v150_batched():
    """v1.5.0 batched NTT: Montgomery radix-8 vs Barrett radix-4."""
    sizes = ["2^15", "2^18", "2^20", "2^22"]
    mont_batch =    [1.54, 7.25, 31.2, 130]
    barrett_batch = [1.81, 7.48, 33.1, 148]

    x = np.arange(len(sizes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - w/2, mont_batch, w, label="Montgomery (radix-8)", color="#ED7D31", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, barrett_batch, w, label="Barrett (radix-4)", color="#70AD47", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("NTT Size (batch of 8)")
    ax.set_ylabel("Total Latency (ms)")
    ax.set_title("Batched 8x NTT — Montgomery vs Barrett (v1.5.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate delta at 2^22
    delta = (barrett_batch[-1] / mont_batch[-1] - 1) * 100
    ax.annotate(f"+{delta:.0f}%",
                xy=(x[-1] + w/2, barrett_batch[-1]),
                xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold", color="#C00000")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "v150_batched.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_multifield_latency():
    """3-field NTT latency comparison (v1.6.0)."""
    scales = ["2^10", "2^12", "2^15", "2^16", "2^18", "2^20", "2^22"]
    bls = [0.181, 0.199, 0.288, 0.366, 1.046, 3.953, 15.073]
    gl  = [0.013, 0.024, 0.033, 0.056, 0.139, 0.762,  3.603]
    bb  = [0.009, 0.016, 0.025, 0.039, 0.093, 0.398,  2.446]

    x = np.arange(len(scales))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, bls, w, label="BLS12-381 (256-bit)", color="#5B9BD5", edgecolor="white", linewidth=0.5)
    ax.bar(x,     gl,  w, label="Goldilocks (64-bit)", color="#ED7D31", edgecolor="white", linewidth=0.5)
    ax.bar(x + w, bb,  w, label="BabyBear (31-bit)",   color="#70AD47", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("NTT Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Single NTT Latency — 3-Field Comparison (v1.6.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.set_ylim(0.005, 30)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate at 2^22
    for i, (val, label) in enumerate([(bls[-1], "15.1"), (gl[-1], "3.6"), (bb[-1], "2.4")]):
        ax.annotate(f"{label} ms",
                    xy=(x[-1] + (i-1)*w, val),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "multifield_latency.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_multifield_speedup():
    """Goldilocks and BabyBear speedup vs BLS12-381 (v1.6.0)."""
    scales = ["2^10", "2^12", "2^15", "2^16", "2^18", "2^20", "2^22"]
    gl_speedup = [13.62, 8.43, 8.78, 6.49, 7.50, 5.19, 4.18]
    bb_speedup = [19.67, 12.12, 11.71, 9.39, 11.22, 9.92, 6.16]

    x = np.arange(len(scales))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, gl_speedup, w, label="Goldilocks / BLS12-381", color="#ED7D31", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, bb_speedup, w, label="BabyBear / BLS12-381",   color="#70AD47", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("NTT Size")
    ax.set_ylabel("Speedup (x faster than BLS12-381)")
    ax.set_title("Multi-Field NTT Speedup vs BLS12-381 (v1.6.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 25)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, sp in zip(bars1, gl_speedup):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{sp:.1f}x", ha="center", fontsize=7, fontweight="bold", color="#C55A11")
    for bar, sp in zip(bars2, bb_speedup):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{sp:.1f}x", ha="center", fontsize=7, fontweight="bold", color="#2E7D32")

    # Annotation: explain convergence
    ax.annotate("Speedup decreases at large sizes:\nouter stages are memory-bound,\nnot arithmetic-bound",
                xy=(6, 6.16), xytext=(4.5, 18),
                arrowprops=dict(arrowstyle="->", color="#666666"),
                fontsize=8, color="#666666", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="#cccccc"))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "multifield_speedup.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_multifield_throughput():
    """Elements/sec throughput for all 3 fields (v1.6.0)."""
    scales = ["2^10", "2^12", "2^15", "2^16", "2^18", "2^20", "2^22"]
    n_vals = [1024, 4096, 32768, 65536, 262144, 1048576, 4194304]

    bls_ms = [0.181, 0.199, 0.288, 0.366, 1.046, 3.953, 15.073]
    gl_ms  = [0.013, 0.024, 0.033, 0.056, 0.139, 0.762,  3.603]
    bb_ms  = [0.009, 0.016, 0.025, 0.039, 0.093, 0.398,  2.446]

    # Throughput in millions of elements per second
    bls_tp = [n / (t * 1e3) for n, t in zip(n_vals, bls_ms)]  # M elem/s
    gl_tp  = [n / (t * 1e3) for n, t in zip(n_vals, gl_ms)]
    bb_tp  = [n / (t * 1e3) for n, t in zip(n_vals, bb_ms)]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(scales))
    ax.plot(x, bls_tp, "o-", color="#5B9BD5", linewidth=2, markersize=7, label="BLS12-381 (256-bit)")
    ax.plot(x, gl_tp,  "s-", color="#ED7D31", linewidth=2, markersize=7, label="Goldilocks (64-bit)")
    ax.plot(x, bb_tp,  "^-", color="#70AD47", linewidth=2, markersize=7, label="BabyBear (31-bit)")

    ax.set_xlabel("NTT Size")
    ax.set_ylabel("Throughput (M elements/sec)")
    ax.set_title("NTT Throughput — 3-Field Comparison (v1.6.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", alpha=0.15, linestyle="--")

    # Annotate peak throughputs at 2^22
    for tp, label, color in [(bls_tp[-1], "BLS", "#5B9BD5"),
                              (gl_tp[-1], "GL", "#ED7D31"),
                              (bb_tp[-1], "BB", "#70AD47")]:
        ax.annotate(f"{tp:.0f} M/s",
                    xy=(x[-1], tp), xytext=(8, 0), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=color, va="center")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "multifield_throughput.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_multifield_batch():
    """Batch efficiency per field: batch/sequential ratio (v1.6.0)."""
    scales = ["2^15", "2^16", "2^18", "2^20", "2^22"]

    # Single NTT × 8 (sequential cost estimate)
    bls_s1 = [0.288, 0.366, 1.046, 3.953, 15.073]
    gl_s1  = [0.033, 0.056, 0.139, 0.762,  3.603]
    bb_s1  = [0.025, 0.039, 0.093, 0.398,  2.446]

    # Batched 8×
    bls_b8 = [0.952, 1.708, 6.807, 28.558, 120.223]
    gl_b8  = [0.126, 0.271, 1.194,  5.557,  31.408]
    bb_b8  = [0.083, 0.144, 0.691,  3.149,  21.077]

    # Batch efficiency = (8 × single) / batched
    bls_eff = [8 * s / b for s, b in zip(bls_s1, bls_b8)]
    gl_eff  = [8 * s / b for s, b in zip(gl_s1, gl_b8)]
    bb_eff  = [8 * s / b for s, b in zip(bb_s1, bb_b8)]

    x = np.arange(len(scales))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w, bls_eff, w, label="BLS12-381", color="#5B9BD5", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x,     gl_eff,  w, label="Goldilocks", color="#ED7D31", edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + w, bb_eff,  w, label="BabyBear",   color="#70AD47", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("NTT Size")
    ax.set_ylabel("Batch Efficiency (8x single / batched)")
    ax.set_title("Batched 8x NTT Efficiency — 3-Field Comparison (v1.6.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 3.5)
    ax.axhline(y=1.0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Label: >1 = batching is beneficial, <1 = overhead
    ax.text(0.02, 0.95, "> 1.0 = batching saves time\n< 1.0 = batching adds overhead",
            transform=ax.transAxes, fontsize=8, color="#666666", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="#cccccc"))

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                    f"{h:.2f}", ha="center", fontsize=7, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "multifield_batch.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_version_history_v160():
    """NTT latency progression across all versions at n=2^22 (updated for v1.6.0)."""
    versions = ["v1.1\nBLS Mont", "v1.2\nBLS Barrett", "v1.3\nBLS 4-Step", "v1.4\nBLS Radix-4",
                "v1.5\nBLS Radix-8", "v1.6\nGoldilocks", "v1.6\nBabyBear"]
    times = [25.1, 24.9, 29.5, 17.1, 15.5, 3.6, 2.4]
    colors = ["#A5A5A5", "#70AD47", "#9B59B6", "#5B9BD5", "#ED7D31", "#ED7D31", "#70AD47"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(versions, times, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("NTT at n=2^22 — Version History + Multi-Field (v1.6.0)")
    ax.set_ylim(0, 35)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{t:.1f} ms", ha="center", fontweight="bold", fontsize=9)

    # Divider between BLS versions and multi-field
    ax.axvline(x=4.5, color="#999999", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(2.0, 33, "BLS12-381 optimization history", ha="center", fontsize=9, color="#555555")
    ax.text(5.5, 33, "Multi-field (v1.6.0)", ha="center", fontsize=9, color="#555555")

    # Mark negative result
    ax.annotate("Negative\nresult", xy=(2, 29.5), xytext=(2.5, 32),
                arrowprops=dict(arrowstyle="->", color="#C00000"),
                fontsize=8, color="#C00000")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "version_history_v160.png")
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
    plot_v150_vs_v140()
    plot_version_history()
    plot_v150_batched()
    plot_multifield_latency()
    plot_multifield_speedup()
    plot_multifield_throughput()
    plot_multifield_batch()
    plot_version_history_v160()
    print("Done.")
