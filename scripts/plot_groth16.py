#!/usr/bin/env python3
"""Generate v2.0.0 benchmark charts: MSM scaling, Groth16 pipeline breakdown.

Reads JSON from results/data/bench_v200_msm.json and bench_v200_groth16.json.
Falls back to hardcoded placeholder data if files not found.

Usage:
    python scripts/plot_groth16.py
    # Outputs PNGs to results/charts/
"""

import os
import json
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "results", "charts")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "results", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def load_json(filename):
    """Load JSON benchmark data, return None if not found."""
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def plot_msm_scaling():
    """MSM latency vs number of points."""
    data = load_json("bench_v200_msm.json")
    if data:
        results = data["results"]
        labels = [f"2^{r['log_n']}" for r in results]
        times = [r["msm_ms"] for r in results]
        windows = [r["window_bits"] for r in results]
    else:
        # Placeholder data (will be replaced after benchmarking)
        labels = ["2^10", "2^12", "2^14", "2^15", "2^16", "2^18"]
        times = [5.0, 12.0, 35.0, 65.0, 130.0, 550.0]
        windows = [4, 4, 4, 4, 4, 5]
        print("  [!] Using placeholder MSM data")

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, times, color="#5B9BD5", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Number of Points")
    ax.set_ylabel("MSM Latency (ms)")
    ax.set_title("GPU MSM (Pippenger) Scaling — BLS12-381 G1 (v2.0.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, t, c in zip(bars, times, windows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f"{t:.1f} ms\n(c={c})", ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "msm_scaling.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_msm_throughput():
    """MSM throughput (points per millisecond) vs size."""
    data = load_json("bench_v200_msm.json")
    if data:
        results = data["results"]
        labels = [f"2^{r['log_n']}" for r in results]
        throughput = [r["points_per_ms"] for r in results]
    else:
        labels = ["2^10", "2^12", "2^14", "2^15", "2^16", "2^18"]
        throughput = [205, 341, 468, 503, 504, 476]
        print("  [!] Using placeholder MSM throughput data")

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, throughput, "o-", color="#ED7D31", linewidth=2.5, markersize=8)
    ax.fill_between(x, throughput, alpha=0.15, color="#ED7D31")

    ax.set_xlabel("Number of Points")
    ax.set_ylabel("Points / ms")
    ax.set_title("MSM Throughput — BLS12-381 G1 (v2.0.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", alpha=0.15, linestyle="--")

    for i, (lbl, tp) in enumerate(zip(labels, throughput)):
        ax.annotate(f"{tp:.0f}", xy=(x[i], tp), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "msm_throughput.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_groth16_pipeline():
    """Groth16 pipeline phase breakdown (stacked bar)."""
    data = load_json("bench_v200_groth16.json")
    if data:
        results = data["results"]
        labels = [f"n={r['domain_size']}" for r in results]
        setup = [r["setup_ms"] for r in results]
        gpu = [r["gpu_prove_ms"] for r in results]
    else:
        labels = ["n=256", "n=512", "n=1024"]
        setup = [800, 1600, 3200]
        gpu = [15, 20, 30]
        print("  [!] Using placeholder Groth16 data")

    x = np.arange(len(labels))
    w = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x, setup, w, label="Trusted Setup (CPU)", color="#A5A5A5",
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, gpu, w, bottom=setup, label="Proof Generation (GPU+CPU)",
                   color="#70AD47", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("NTT Domain Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Groth16 Pipeline Breakdown — x^3 + x + 5 = y (v2.0.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate totals
    for i in range(len(labels)):
        total = setup[i] + gpu[i]
        ax.text(x[i], total + total * 0.03,
                f"{total:.0f} ms", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "groth16_pipeline.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_groth16_gpu_vs_cpu():
    """GPU vs CPU proof generation time."""
    data = load_json("bench_v200_groth16.json")
    if data:
        results = data["results"]
        labels = [f"n={r['domain_size']}" for r in results]
        gpu = [r["gpu_prove_ms"] for r in results]
        cpu = [r["cpu_prove_ms"] for r in results]
    else:
        labels = ["n=256", "n=512", "n=1024"]
        gpu = [15, 20, 30]
        cpu = [12, 18, 28]
        print("  [!] Using placeholder GPU vs CPU data")

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w / 2, gpu, w, label="GPU Proof", color="#5B9BD5",
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w / 2, cpu, w, label="CPU Proof", color="#ED7D31",
                   edgecolor="white", linewidth=0.5)

    ax.set_xlabel("NTT Domain Size")
    ax.set_ylabel("Proof Generation Time (ms)")
    ax.set_title("Groth16 Proof: GPU vs CPU — x^3 + x + 5 = y (v2.0.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Annotate values
    for bar, t in zip(bars1, gpu):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{t:.1f}", ha="center", fontsize=9, fontweight="bold",
                color="#2E75B6")
    for bar, t in zip(bars2, cpu):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{t:.1f}", ha="center", fontsize=9, fontweight="bold",
                color="#C55A11")

    # Note about toy circuit
    ax.text(0.98, 0.95,
            "Toy circuit (4 constraints):\nsetup dominates, GPU\noverhead > benefit at\nthis scale",
            transform=ax.transAxes, fontsize=8, color="#666666",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="#cccccc"))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "groth16_gpu_vs_cpu.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def plot_version_history_v200():
    """Full version history including v2.0.0 primitives."""
    versions = [
        "v1.1\nBLS NTT", "v1.4\nRadix-4", "v1.5\nRadix-8",
        "v1.6\nGL NTT", "v1.6\nBB NTT",
        "v2.0\nFq Arith", "v2.0\nEC G1/G2", "v2.0\nMSM", "v2.0\nGroth16"
    ]
    # Values: NTT latency for NTT versions, then feature markers for v2.0
    # Use a different approach: show capabilities added per version
    capabilities = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = [
        "#A5A5A5", "#5B9BD5", "#ED7D31",
        "#ED7D31", "#70AD47",
        "#9B59B6", "#9B59B6", "#9B59B6", "#9B59B6"
    ]

    labels_desc = [
        "BLS12-381\n256-bit NTT\n25.1 ms",
        "Radix-4\nouter\n17.1 ms",
        "Radix-8\nouter\n15.5 ms",
        "Goldilocks\n64-bit NTT\n3.6 ms",
        "BabyBear\n31-bit NTT\n2.4 ms",
        "Fq + Fq2\n381-bit\narithmetic",
        "G1 + G2\nEC point\noperations",
        "Pippenger\nGPU MSM\n(G1)",
        "Groth16\ntoy prover\nend-to-end",
    ]

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, (ver, desc, col) in enumerate(zip(versions, labels_desc, colors)):
        ax.barh(0, 1, left=i, height=0.6, color=col, edgecolor="white",
                linewidth=1.5, alpha=0.85)
        ax.text(i + 0.5, 0, desc, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")
        ax.text(i + 0.5, -0.5, ver, ha="center", va="top",
                fontsize=8, fontweight="bold")

    # Dividers
    ax.axvline(x=5, color="#333333", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.text(2.5, 0.55, "v1.x: NTT Optimization", ha="center",
            fontsize=10, fontweight="bold", color="#555555")
    ax.text(7, 0.55, "v2.0: Groth16 Primitives", ha="center",
            fontsize=10, fontweight="bold", color="#555555")

    ax.set_xlim(-0.2, 9.2)
    ax.set_ylim(-0.8, 0.8)
    ax.axis("off")
    ax.set_title("cuda-zkp-ntt: Feature Timeline (v1.1 - v2.0.0)", fontsize=14)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "version_timeline_v200.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


if __name__ == "__main__":
    print("Generating v2.0.0 benchmark charts...")
    plot_msm_scaling()
    plot_msm_throughput()
    plot_groth16_pipeline()
    plot_groth16_gpu_vs_cpu()
    plot_version_history_v200()
    print("Done.")
