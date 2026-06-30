#!/usr/bin/env python3
"""Generate the report figures directly from the committed result JSONs.

Reads results/ and results-ext/, writes figures/*.pdf. Reproduces every plot in
template.tex. Run: ``uv run --group analysis python scripts/report/make_figures.py``.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R, RE = os.path.join(ROOT, "results"), os.path.join(ROOT, "results-ext")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

V = ["bf16", "bnb-int8", "bnb-nf4", "gptq-official", "awq-official"]
LBL = ["bf16", "bnb-int8", "bnb-nf4", "gptq-off", "awq-off"]
CLR = ["#444444", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]


def jload(p):
    with open(p) as f:
        return json.load(f)


def collect():
    m = {}
    for v in V:
        mmlu = jload(f"{R}/_meta/mmlu_group_{v}.json")
        gsm = jload(f"{R}/{v}/gsm8k/result.json")
        eff = jload(f"{R}/{v}/efficiency.json")
        m[v] = {
            "mmlu": mmlu["acc"],
            "gsm8k": gsm["extra_metrics"]["exact_match,flexible-extract"],
            "humaneval": jload(f"{R}/{v}/humaneval/result.json")["metric_value"],
            "ifeval": jload(f"{R}/{v}/ifeval/result.json")["metric_value"],
            "disk_gib": eff["disk"]["weights_bytes"] / 2**30,
            "generate_peak_gib": eff["memory"]["generate_peak_bytes"] / 2**30,
            "decode_long": eff["throughput"]["long"]["decode_tok_s"],
        }
    j = {}
    for v in V:
        p = f"{RE}/{v}/judgebench/result.json"
        if os.path.exists(p):
            d = jload(p)
            j[v] = {"acc": d["metric_value"], "consistency": d["meta"]["consistency_rate"]}
    return m, j


def main():
    plt.rcParams.update({
        "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
        "axes.axisbelow": True, "figure.dpi": 130, "savefig.bbox": "tight",
        "font.family": "serif",
    })
    M, J = collect()

    benches = ["mmlu", "gsm8k", "humaneval", "ifeval"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(benches))
    for i, v in enumerate(V):
        ax.bar(x + (i - 2) * 0.16, [M[v][b] * 100 for b in benches], 0.16,
               label=LBL[i], color=CLR[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["MMLU", "GSM8K", "HumanEval", "IFEval"])
    ax.set_ylabel("Accuracy / pass@1 (\\%)")
    ax.set_ylim(60, 90)
    ax.legend(ncol=5, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              frameon=False, columnspacing=1.0)
    ax.set_title("Quality by benchmark and quantization variant", pad=22)
    fig.savefig(f"{FIG}/fig_quality.pdf")
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    dec = [M[v]["decode_long"] for v in V]
    a1.bar(LBL, dec, color=CLR)
    a1.axhline(M["bf16"]["decode_long"], ls="--", c="#444444", lw=1, alpha=0.7)
    a1.set_ylabel("Decode throughput (tok/s)")
    a1.set_title("(a) Decode speed (batch 1)")
    a1.tick_params(axis="x", rotation=30)
    for i, d in enumerate(dec):
        a1.text(i, d + 0.7, f"{d:.0f}", ha="center", fontsize=8)
    xp = np.arange(len(V))
    a2.bar(xp - 0.2, [M[v]["generate_peak_gib"] for v in V], 0.4, color=CLR)
    a2.bar(xp + 0.2, [M[v]["disk_gib"] for v in V], 0.4, color=CLR, alpha=0.45, hatch="//")
    a2.set_xticks(xp)
    a2.set_xticklabels(LBL, rotation=30)
    a2.set_ylabel("GiB")
    a2.set_title("(b) Generate VRAM vs disk size")
    a2.legend(handles=[Patch(fc="#888", label="gen VRAM"),
                       Patch(fc="#888", alpha=0.45, hatch="//", label="disk")],
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_efficiency.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, v in enumerate(V):
        ax.scatter(M[v]["decode_long"], M[v]["mmlu"] * 100,
                   s=M[v]["generate_peak_gib"] * 60, color=CLR[i],
                   edgecolor="k", lw=0.6, alpha=0.85, zorder=3)
        ax.annotate(LBL[i], (M[v]["decode_long"], M[v]["mmlu"] * 100),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Decode throughput (tok/s)  $\\rightarrow$ faster")
    ax.set_ylabel("MMLU accuracy (\\%)")
    ax.set_title("Quality–speed–memory trade-off (marker area $\\propto$ generate VRAM)")
    ax.set_ylim(71.5, 74.2)
    fig.savefig(f"{FIG}/fig_tradeoff.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    xp = np.arange(len(V))
    ax.bar(xp - 0.2, [J[v]["acc"] * 100 for v in V], 0.4, color=CLR)
    ax.bar(xp + 0.2, [J[v]["consistency"] * 100 for v in V], 0.4, color=CLR,
           alpha=0.45, hatch="xx")
    ax.set_xticks(xp)
    ax.set_xticklabels(LBL, rotation=30)
    ax.set_ylabel("\\%")
    ax.set_title("EXT-2 JudgeBench: pairwise judge accuracy \\& position consistency")
    ax.legend(handles=[Patch(fc="#888", label="judge accuracy"),
                       Patch(fc="#888", alpha=0.45, hatch="xx", label="position consistency")],
              fontsize=8.5, frameon=False)
    fig.savefig(f"{FIG}/fig_judge.pdf")
    plt.close(fig)

    # EXT-4 numbers come from results-ext/_meta/ext4_kvcache.json
    kvd = jload(f"{RE}/_meta/ext4_kvcache.json")["comparison"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.8))
    labels = ["fp16 KV\n(bf16)", "int4 KV\n(bf16-kvq4)"]
    kv = [c["kv_plus_act_gb"] for c in kvd]
    spd = [c["decode_tok_per_s"] for c in kvd]
    a1.bar(labels, kv, color=["#444444", "#d95f02"])
    a1.set_ylabel("KV+activation transient (GiB)")
    a1.set_title("(a) KV memory $\\downarrow$22\\%")
    for i, xx in enumerate(kv):
        a1.text(i, xx + 0.01, f"{xx:.2f}", ha="center", fontsize=9)
    a2.bar(labels, spd, color=["#444444", "#d95f02"])
    a2.set_ylabel("Decode throughput (tok/s)")
    a2.set_title("(b) Decode speed $\\downarrow$20$\\times$")
    for i, xx in enumerate(spd):
        a2.text(i, xx + 0.6, f"{xx:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_kvcache.pdf")
    plt.close(fig)
    print("wrote 5 figures to", FIG)


if __name__ == "__main__":
    main()
