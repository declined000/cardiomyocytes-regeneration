"""
Video Contractility Report: BF Optical Flow Analysis
=====================================================
Generates figures, statistics, and a text summary for the BF video
contractility arm of the cardiomyocyte-on-piezoelectric-substrate study.

Cross-references video-derived beating metrics with existing phalloidin
morphology data to build a structure-function picture.

Methods aligned with:
  - Czirok/Bhatt et al. Sci Rep 2017 (convergence, contractile foci)
  - Huebsch et al. Tissue Eng 2015 (automated video-based CM analysis)
  - Sala et al. Circ Res 2018 / MUSCLEMOTION (beat rate, amplitude, duration)

Outputs go to final_report/video_* to sit alongside existing staining figs.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from scipy.stats import mannwhitneyu
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

OUT = Path("final_report")
OUT.mkdir(exist_ok=True)

VIDEO_CSV = Path("video_results") / "batch_results_v3_final.csv"
STAINING_CSV = OUT / "phalloidin_per_cell_data.csv"

CLS_ORDER = ["active_beating", "individual_cells_beating",
             "non_contractile", "no_beating"]
CLS_COLORS = {
    "active_beating": "#2ca02c",
    "individual_cells_beating": "#1f77b4",
    "non_contractile": "#ff7f0e",
    "no_beating": "#d62728",
}
CLS_LABELS = {
    "active_beating": "Active beating",
    "individual_cells_beating": "Individual cells beating",
    "non_contractile": "Non-contractile motion",
    "no_beating": "No beating",
}

SUBSTRATE_ORDER = ["plastic", "non-poled_PVDF", "gold_PVDF"]
SUBSTRATE_COLORS = {
    "plastic": "#7fbfff",
    "non-poled_PVDF": "#a8d5a2",
    "gold_PVDF": "#ff9966",
}
SUBSTRATE_LABELS = {
    "plastic": "Bare plastic",
    "non-poled_PVDF": "Non-poled PVDF",
    "gold_PVDF": "Gold-coated PVDF",
}

CELLLINE_COLORS = {"XR1": "#6baed6", "GCaMP6f": "#fd8d3c"}

STAINING_CONDS = [
    "Au-PVDF Poled Pulsed (device)",
    "Au-PVDF Poled Un-pulsed (control)",
    "β-PVDF Nonpoled Pulsed (device)",
    "Cells only (baseline)",
]
STAINING_COLORS = {
    "Au-PVDF Poled Pulsed (device)": "#ff9966",
    "Au-PVDF Poled Un-pulsed (control)": "#ffd966",
    "β-PVDF Nonpoled Pulsed (device)": "#a8d5a2",
    "Cells only (baseline)": "#7fbfff",
}
STAINING_SHORT = {
    "Au-PVDF Poled Pulsed (device)": "Poled Pulsed\n(device)",
    "Au-PVDF Poled Un-pulsed (control)": "Poled Un-pulsed\n(control)",
    "β-PVDF Nonpoled Pulsed (device)": "Nonpoled Pulsed\n(device)",
    "Cells only (baseline)": "Cells only\n(baseline)",
}

# Staining CSV uses newlines in condition labels; video CSV uses flat strings
STAINING_LABEL_MAP = {
    "Au-PVDF Poled Pulsed (device)": "Au-PVDF Poled\nPulsed (device)",
    "Au-PVDF Poled Un-pulsed (control)": "Au-PVDF Poled\nUn-pulsed (control)",
    "β-PVDF Nonpoled Pulsed (device)": "β-PVDF Nonpoled\nPulsed (device)",
    "Cells only (baseline)": "Cells only\n(baseline)",
}


def load_data():
    vdf = pd.read_csv(VIDEO_CSV)
    sdf = pd.read_csv(STAINING_CSV) if STAINING_CSV.exists() else None
    return vdf, sdf


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ══════════════════════════════════════════════════════════════════════════
# FIGURE V1: Contractility Classification Overview
# ══════════════════════════════════════════════════════════════════════════
def fig_v1_classification(vdf):
    print("  Fig V1: Classification overview...")

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    def _stacked_bar(ax, groups, group_labels, title):
        x = np.arange(len(groups))
        bottom = np.zeros(len(groups))
        for cls in CLS_ORDER:
            vals = np.array([
                (vdf[vdf[groups.name] == g]["manual_classification"] == cls).sum()
                for g in groups
            ], dtype=float)
            label = CLS_LABELS[cls] if ax is axes[0] else None
            ax.bar(x, vals, bottom=bottom, color=CLS_COLORS[cls],
                   label=label, edgecolor="white", linewidth=0.5)
            for i, v in enumerate(vals):
                if v > 0:
                    ax.text(i, bottom[i] + v / 2, str(int(v)), ha="center",
                            va="center", fontsize=11, fontweight="bold")
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=10)
        ax.set_ylabel("Number of videos", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

    # Panel A: by substrate
    subs = pd.Series(SUBSTRATE_ORDER, name="substrate")
    _stacked_bar(axes[0], subs,
                 [SUBSTRATE_LABELS[s] for s in SUBSTRATE_ORDER],
                 "A) By Substrate")
    axes[0].legend(fontsize=9, loc="upper right")

    # Panel B: by cell line
    cls_s = pd.Series(["XR1", "GCaMP6f"], name="cell_type")
    _stacked_bar(axes[1], cls_s, ["XR1", "GCaMP6f"], "B) By Cell Line")

    # Panel C: by recording type
    rts = pd.Series(["baseline", "on-device", "control"], name="recording_type")
    rt_labels = ["Baseline\n(day 1-2)", "On-device\n(day 6-8)",
                 "Control\n(day 6-8)"]
    _stacked_bar(axes[2], rts, rt_labels, "C) By Recording Type")

    fig.suptitle("Figure V1: Contractility Classification Overview\n"
                 "Manual expert classification of 23 brightfield videos",
                 fontsize=14, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(OUT / "video_Fig_V1_classification.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE V2: Temporal Progression
# ══════════════════════════════════════════════════════════════════════════
def fig_v2_temporal(vdf):
    print("  Fig V2: Temporal progression...")

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    day_order = ["day1", "day2", "day6", "day8"]
    day_labels = ["Day 1", "Day 2", "Day 6", "Day 8"]
    beating = vdf[vdf["manual_classification"] == "active_beating"]

    # Panel A: classification distribution over time
    ax = axes[0]
    x = np.arange(len(day_order))
    bottom = np.zeros(len(day_order))
    for cls in CLS_ORDER:
        vals = np.array([(vdf[vdf["day"] == d]["manual_classification"] == cls).sum()
                         for d in day_order], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=CLS_COLORS[cls],
               label=CLS_LABELS[cls], edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(i, bottom[i] + v / 2, str(int(v)), ha="center",
                        va="center", fontsize=10, fontweight="bold")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, fontsize=11)
    ax.set_ylabel("Number of videos", fontsize=11)
    ax.set_title("A) Classification by Day", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel B: amplitude over time
    ax = axes[1]
    for i, d in enumerate(day_order):
        d_vals = beating[beating["day"] == d]["roi_amplitude"]
        if len(d_vals) > 0:
            jitter = np.random.normal(0, 0.06, size=len(d_vals))
            ax.scatter(np.full(len(d_vals), i) + jitter, d_vals,
                       c="#2ca02c", s=60, alpha=0.7, edgecolors="black",
                       linewidth=0.5, zorder=5)
            ax.plot([i - 0.2, i + 0.2], [d_vals.median(), d_vals.median()],
                    color="black", linewidth=2.5, zorder=6)
    ax.set_xticks(range(len(day_order)))
    ax.set_xticklabels(day_labels, fontsize=11)
    ax.set_ylabel("Contraction amplitude (px)", fontsize=11)
    ax.set_title("B) Amplitude Over Time\n(active beating only)",
                 fontsize=13, fontweight="bold")

    # Panel C: BPM over time
    ax = axes[2]
    for i, d in enumerate(day_order):
        d_vals = beating[beating["day"] == d]["roi_bpm"]
        if len(d_vals) > 0:
            jitter = np.random.normal(0, 0.06, size=len(d_vals))
            ax.scatter(np.full(len(d_vals), i) + jitter, d_vals,
                       c="#2ca02c", s=60, alpha=0.7, edgecolors="black",
                       linewidth=0.5, zorder=5)
            ax.plot([i - 0.2, i + 0.2], [d_vals.median(), d_vals.median()],
                    color="black", linewidth=2.5, zorder=6)
    ax.set_xticks(range(len(day_order)))
    ax.set_xticklabels(day_labels, fontsize=11)
    ax.set_ylabel("Beat rate (BPM)", fontsize=11)
    ax.set_title("C) Beat Rate Over Time\n(active beating only)",
                 fontsize=13, fontweight="bold")

    fig.suptitle("Figure V2: Temporal Progression of Contractility\n"
                 "Days 1-2 = baseline (bare plastic only); "
                 "Days 6-8 = treatment phase (device + control)",
                 fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(OUT / "video_Fig_V2_temporal.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE V3: Beating Metrics by Substrate (active_beating only)
# ══════════════════════════════════════════════════════════════════════════
def fig_v3_substrate(vdf):
    print("  Fig V3: Substrate comparison...")

    beating = vdf[vdf["manual_classification"] == "active_beating"].copy()
    metrics = [
        ("roi_amplitude", "Contraction\nAmplitude (px)"),
        ("roi_bpm", "Beat Rate\n(BPM)"),
        ("roi_contraction_time_s", "Contraction\nTime (s)"),
        ("roi_relaxation_time_s", "Relaxation\nTime (s)"),
        ("roi_relax_contract_ratio", "Relaxation /\nContraction Ratio"),
        ("conv_disp_ratio", "Convergence-Disp.\nRatio"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    axes_flat = axes.flatten()

    for idx, (col, title) in enumerate(metrics):
        ax = axes_flat[idx]
        data = []
        labels = []
        colors = []
        for s in SUBSTRATE_ORDER:
            vals = beating[beating["substrate"] == s][col].dropna().values
            data.append(vals)
            labels.append(SUBSTRATE_LABELS[s])
            colors.append(SUBSTRATE_COLORS[s])

        has_data = [len(d) > 0 for d in data]
        if any(has_data):
            bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                            showfliers=False)
            for j, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(colors[j])
                patch.set_alpha(0.7)
            for j, d in enumerate(data):
                if len(d) > 0:
                    jitter = np.random.normal(j + 1, 0.05, size=len(d))
                    ax.scatter(jitter, d, alpha=0.6, s=40,
                               color="black", zorder=5)

            y_max = max((np.max(d) for d in data if len(d) > 0), default=1)
            offset = 0
            for (i1, i2) in [(0, 1), (0, 2), (1, 2)]:
                if len(data[i1]) >= 2 and len(data[i2]) >= 2:
                    _, p = mannwhitneyu(data[i1], data[i2],
                                        alternative="two-sided")
                    stars = sig_stars(p)
                    if stars != "ns":
                        y = y_max * (1.08 + 0.09 * offset)
                        ax.plot([i1 + 1, i2 + 1], [y, y], "k-", lw=1)
                        ax.text((i1 + i2 + 2) / 2, y, stars, ha="center",
                                va="bottom", fontsize=10, fontweight="bold")
                        offset += 1

        ax.set_xticklabels(labels, fontsize=9, rotation=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

    n_counts = {s: len(beating[beating["substrate"] == s])
                for s in SUBSTRATE_ORDER}
    fig.suptitle(
        "Figure V3: Beating Metrics by Substrate (Active Beating Videos Only)\n"
        f"Bare plastic: n={n_counts['plastic']}, "
        f"Non-poled PVDF: n={n_counts['non-poled_PVDF']}, "
        f"Gold PVDF: n={n_counts['gold_PVDF']}  |  "
        "Mann-Whitney U (* p<0.05, ** p<0.01, *** p<0.001)",
        fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(OUT / "video_Fig_V3_substrate.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE V4: Cell Line Comparison
# ══════════════════════════════════════════════════════════════════════════
def fig_v4_cellline(vdf):
    print("  Fig V4: Cell line comparison...")

    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    beating = vdf[vdf["manual_classification"] == "active_beating"]

    # Panels A-B: pie charts
    for i, cl in enumerate(["XR1", "GCaMP6f"]):
        ax = axes[i]
        cl_df = vdf[vdf["cell_type"] == cl]
        sizes = [(cl_df["manual_classification"] == c).sum()
                 for c in CLS_ORDER]
        nonzero = [(s, CLS_COLORS[c], CLS_LABELS[c])
                   for s, c in zip(sizes, CLS_ORDER) if s > 0]
        if nonzero:
            sz, co, la = zip(*nonzero)
            wedges, texts, autotexts = ax.pie(
                sz, colors=co, labels=la, autopct="%1.0f%%",
                startangle=90, textprops={"fontsize": 9})
            for at in autotexts:
                at.set_fontweight("bold")
        ax.set_title(f"{'A' if i == 0 else 'B'}) {cl} (n={len(cl_df)})",
                     fontsize=13, fontweight="bold")

    # Panel C: amplitude
    ax = axes[2]
    for i, cl in enumerate(["XR1", "GCaMP6f"]):
        vals = beating[beating["cell_type"] == cl]["roi_amplitude"].dropna()
        if len(vals) > 0:
            bp = ax.boxplot([vals.values], positions=[i],
                            patch_artist=True, widths=0.5, showfliers=False)
            bp["boxes"][0].set_facecolor(CELLLINE_COLORS[cl])
            bp["boxes"][0].set_alpha(0.7)
            jitter = np.random.normal(i, 0.05, size=len(vals))
            ax.scatter(jitter, vals, alpha=0.6, s=40, color="black", zorder=5)
        else:
            ax.text(i, 0.5, "No actively\nbeating videos",
                    ha="center", va="center", fontsize=10,
                    style="italic", color="gray")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["XR1", "GCaMP6f"], fontsize=11)
    ax.set_ylabel("Contraction amplitude (px)", fontsize=11)
    ax.set_title("C) Amplitude\n(active beating only)",
                 fontsize=13, fontweight="bold")

    # Panel D: BPM
    ax = axes[3]
    for i, cl in enumerate(["XR1", "GCaMP6f"]):
        vals = beating[beating["cell_type"] == cl]["roi_bpm"].dropna()
        if len(vals) > 0:
            bp = ax.boxplot([vals.values], positions=[i],
                            patch_artist=True, widths=0.5, showfliers=False)
            bp["boxes"][0].set_facecolor(CELLLINE_COLORS[cl])
            bp["boxes"][0].set_alpha(0.7)
            jitter = np.random.normal(i, 0.05, size=len(vals))
            ax.scatter(jitter, vals, alpha=0.6, s=40, color="black", zorder=5)
        else:
            ax.text(i, 5, "No actively\nbeating videos",
                    ha="center", va="center", fontsize=10,
                    style="italic", color="gray")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["XR1", "GCaMP6f"], fontsize=11)
    ax.set_ylabel("Beat rate (BPM)", fontsize=11)
    ax.set_title("D) Beat Rate\n(active beating only)",
                 fontsize=13, fontweight="bold")

    n_tammy = len(vdf[vdf["cell_type"] == "XR1"])
    n_jaz = len(vdf[vdf["cell_type"] == "GCaMP6f"])
    fig.suptitle(
        "Figure V4: Cell Line Comparison (XR1 vs GCaMP6f)\n"
        f"XR1 = rows A, B ({n_tammy} videos); "
        f"GCaMP6f = row C ({n_jaz} videos)",
        fontsize=14, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(OUT / "video_Fig_V4_cellline.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE V5: On-Device vs Control Paired Comparison
# ══════════════════════════════════════════════════════════════════════════
def fig_v5_paired(vdf):
    print("  Fig V5: On-device vs control paired...")

    treatment = vdf[vdf["phase"] == "treatment"].copy()
    pairs = []
    for w in treatment["well"].unique():
        for d in treatment["day"].unique():
            dev = treatment[(treatment["well"] == w) &
                            (treatment["day"] == d) &
                            (treatment["recording_type"] == "on-device")]
            ctrl = treatment[(treatment["well"] == w) &
                             (treatment["day"] == d) &
                             (treatment["recording_type"] == "control")]
            if len(dev) == 1 and len(ctrl) == 1:
                pairs.append((dev.iloc[0], ctrl.iloc[0]))

    if not pairs:
        print("    No device-control pairs found, skipping.")
        return

    metrics = [
        ("roi_amplitude", "Contraction Amplitude (px)"),
        ("roi_bpm", "Beat Rate (BPM)"),
        ("n_foci", "Contractile Foci Count"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 7))
    if len(metrics) == 1:
        axes = [axes]

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        for dev_row, ctrl_row in pairs:
            cl = dev_row["cell_type"]
            color = CELLLINE_COLORS.get(cl, "gray")
            label = f"{dev_row['well']}-{dev_row['day']}"

            ax.plot([0, 1], [dev_row[col], ctrl_row[col]], "o-",
                    color=color, markersize=8, linewidth=1.5, alpha=0.7)
            ax.text(1.05, ctrl_row[col], label, fontsize=8, va="center",
                    color=color)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["On-device\n(pulsed)", "Control\n(unpulsed)"],
                           fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.3, 1.6)

    legend_elements = [Patch(facecolor=CELLLINE_COLORS[cl], label=cl)
                       for cl in ["XR1", "GCaMP6f"]]
    axes[-1].legend(handles=legend_elements, fontsize=10)

    fig.suptitle(
        f"Figure V5: On-Device vs Control Paired Comparison\n"
        f"{len(pairs)} matched well-day pairs (same well, same day, "
        "device vs control plate)",
        fontsize=14, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(OUT / "video_Fig_V5_paired.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE V6: Contractile Foci Landscape
# ══════════════════════════════════════════════════════════════════════════
def fig_v6_foci(vdf):
    print("  Fig V6: Contractile foci landscape...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Panel A: bubble scatter
    ax = axes[0]
    for _, row in vdf.iterrows():
        cls = row["manual_classification"]
        color = CLS_COLORS.get(cls, "gray")
        amp = row.get("foci_mean_amplitude", 0)
        size = max(20, amp * 150)
        ax.scatter(row["n_foci"], row["foci_coverage_pct"],
                   s=size, c=color, alpha=0.6, edgecolors="black",
                   linewidth=0.5)

    legend_handles = [Patch(facecolor=CLS_COLORS[c], label=CLS_LABELS[c])
                      for c in CLS_ORDER]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left")
    ax.set_xlabel("Number of contractile foci detected", fontsize=11)
    ax.set_ylabel("Foci coverage (% of FOV)", fontsize=11)
    ax.set_title("A) Foci Detection Landscape\n"
                 "(bubble size = mean foci amplitude)",
                 fontsize=13, fontweight="bold")

    # Panel B: foci metrics by substrate
    ax = axes[1]
    x = np.arange(len(SUBSTRATE_ORDER))
    width = 0.35

    mean_nfoci = [vdf[vdf["substrate"] == s]["n_foci"].mean()
                  for s in SUBSTRATE_ORDER]
    mean_cov = [vdf[vdf["substrate"] == s]["foci_coverage_pct"].mean()
                for s in SUBSTRATE_ORDER]

    ax.bar(x - width / 2, mean_nfoci, width,
           color=[SUBSTRATE_COLORS[s] for s in SUBSTRATE_ORDER],
           edgecolor="black", label="Mean foci count")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, mean_cov, width,
            color=[SUBSTRATE_COLORS[s] for s in SUBSTRATE_ORDER],
            edgecolor="black", alpha=0.5, hatch="//",
            label="Mean coverage (%)")

    ax.set_xticks(x)
    ax.set_xticklabels([SUBSTRATE_LABELS[s] for s in SUBSTRATE_ORDER],
                       fontsize=10)
    ax.set_ylabel("Mean foci count", fontsize=11)
    ax2.set_ylabel("Mean foci coverage (%)", fontsize=11)
    ax.set_title("B) Foci Metrics by Substrate", fontsize=13,
                 fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    fig.suptitle(
        "Figure V6: Contractile Foci Analysis\n"
        "Per-center detection via convergence thresholding "
        "(Czirok/Bhatt et al. 2017)",
        fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(OUT / "video_Fig_V6_foci.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE V7: Adhesion and Flow Analysis
# ══════════════════════════════════════════════════════════════════════════
def fig_v7_adhesion(vdf):
    print("  Fig V7: Adhesion and flow...")

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel A: flow by substrate
    ax = axes[0]
    flow_pcts = []
    for s in SUBSTRATE_ORDER:
        s_df = vdf[vdf["substrate"] == s]
        pct = 100 * s_df["has_flow"].sum() / len(s_df) if len(s_df) > 0 else 0
        flow_pcts.append(pct)
    bars = ax.bar(range(len(SUBSTRATE_ORDER)), flow_pcts,
                  color=[SUBSTRATE_COLORS[s] for s in SUBSTRATE_ORDER],
                  edgecolor="black")
    for i, v in enumerate(flow_pcts):
        n = len(vdf[vdf["substrate"] == SUBSTRATE_ORDER[i]])
        n_flow = vdf[(vdf["substrate"] == SUBSTRATE_ORDER[i]) &
                     (vdf["has_flow"] == True)].shape[0]
        ax.text(i, v + 2, f"{n_flow}/{n}", ha="center", fontsize=11,
                fontweight="bold")
    ax.set_xticks(range(len(SUBSTRATE_ORDER)))
    ax.set_xticklabels([SUBSTRATE_LABELS[s] for s in SUBSTRATE_ORDER],
                       fontsize=10)
    ax.set_ylabel("Videos with flow/drift (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("A) Flow/Drift by Substrate", fontsize=13, fontweight="bold")

    # Panel B: flow by cell line
    ax = axes[1]
    for i, cl in enumerate(["XR1", "GCaMP6f"]):
        cl_df = vdf[vdf["cell_type"] == cl]
        pct = 100 * cl_df["has_flow"].sum() / len(cl_df) if len(cl_df) > 0 else 0
        n_flow = cl_df["has_flow"].sum()
        ax.bar(i, pct, color=CELLLINE_COLORS[cl], edgecolor="black")
        ax.text(i, pct + 2, f"{n_flow}/{len(cl_df)}", ha="center",
                fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["XR1", "GCaMP6f"], fontsize=11)
    ax.set_ylabel("Videos with flow/drift (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("B) Flow/Drift by Cell Line", fontsize=13, fontweight="bold")

    # Panel C: flow uniformity vs amplitude
    ax = axes[2]
    for _, row in vdf.iterrows():
        cls = row["manual_classification"]
        color = CLS_COLORS.get(cls, "gray")
        marker = "^" if row["has_flow"] else "o"
        ax.scatter(row["flow_uniformity"], row["roi_amplitude"],
                   c=color, marker=marker, s=60, alpha=0.7,
                   edgecolors="black", linewidth=0.5)
    legend_handles = [Patch(facecolor=CLS_COLORS[c], label=CLS_LABELS[c])
                      for c in CLS_ORDER]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax.set_xlabel("Flow uniformity (0 = none, 1 = pure drift)", fontsize=11)
    ax.set_ylabel("ROI contraction amplitude (px)", fontsize=11)
    ax.set_title("C) Flow Uniformity vs Amplitude\n"
                 "(triangle = flow detected)", fontsize=13, fontweight="bold")

    fig.suptitle(
        "Figure V7: Cell Adhesion and Flow/Drift Analysis\n"
        "Flow = unattached cells/debris via spatial displacement uniformity",
        fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(OUT / "video_Fig_V7_adhesion.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE V8: Cross-Reference -- Morphology vs Function
# ══════════════════════════════════════════════════════════════════════════
def fig_v8_crossref(vdf, sdf):
    print("  Fig V8: Cross-reference morphology vs function...")

    if sdf is None:
        print("    Staining data not found, skipping.")
        return None

    # Filter video to XR1 only for cross-reference with staining
    # (all staining images are from XR1 wells: B2, A3, B3)
    vdf_t = vdf[vdf["cell_type"] == "XR1"]

    rows = []
    for sc in STAINING_CONDS:
        # Video side (XR1 only)
        sc_v = vdf_t[vdf_t["staining_condition"] == sc]
        beat_v = sc_v[sc_v["manual_classification"] == "active_beating"]
        n_v = len(sc_v)
        n_b = len(beat_v)
        beat_frac = n_b / n_v if n_v > 0 else 0
        med_amp = beat_v["roi_amplitude"].median() if n_b > 0 else 0
        med_bpm = beat_v["roi_bpm"].median() if n_b > 0 else 0

        # Staining side (labels have \n in CSV)
        sc_nl = STAINING_LABEL_MAP[sc]
        sc_s = sdf[sdf["condition"] == sc_nl]
        n_cells = len(sc_s)
        med_factin = sc_s["mean_phalloidin_intensity"].median() if n_cells > 0 else np.nan
        med_coh = sc_s["actin_coherency"].median() if n_cells > 0 else np.nan
        med_circ = sc_s["circularity"].median() if n_cells > 0 else np.nan
        med_area = sc_s["cell_area_px"].median() if n_cells > 0 else np.nan

        rows.append({
            "condition": sc,
            "n_videos": n_v, "n_beating": n_b,
            "beating_fraction": beat_frac,
            "median_amplitude": med_amp, "median_bpm": med_bpm,
            "n_cells": n_cells, "median_factin": med_factin,
            "median_coherency": med_coh, "median_circularity": med_circ,
            "median_area": med_area,
        })
    merged = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel A: beating fraction vs F-actin
    ax = axes[0, 0]
    for _, r in merged.iterrows():
        color = STAINING_COLORS[r["condition"]]
        ax.scatter(r["median_factin"], r["beating_fraction"] * 100,
                   s=220, c=color, edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(STAINING_SHORT[r["condition"]].replace("\n", " "),
                    (r["median_factin"], r["beating_fraction"] * 100),
                    textcoords="offset points", xytext=(12, 5), fontsize=8)
    ax.set_xlabel("Median F-actin intensity (staining)", fontsize=11)
    ax.set_ylabel("% videos with active beating", fontsize=11)
    ax.set_title("A) F-actin Intensity vs Beating Fraction",
                 fontsize=13, fontweight="bold")

    # Panel B: beating fraction vs actin coherency
    ax = axes[0, 1]
    for _, r in merged.iterrows():
        color = STAINING_COLORS[r["condition"]]
        ax.scatter(r["median_coherency"], r["beating_fraction"] * 100,
                   s=220, c=color, edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(STAINING_SHORT[r["condition"]].replace("\n", " "),
                    (r["median_coherency"], r["beating_fraction"] * 100),
                    textcoords="offset points", xytext=(12, 5), fontsize=8)
    ax.set_xlabel("Median actin coherency (staining)", fontsize=11)
    ax.set_ylabel("% videos with active beating", fontsize=11)
    ax.set_title("B) Actin Coherency vs Beating Fraction",
                 fontsize=13, fontweight="bold")

    # Panel C: amplitude vs cell area
    ax = axes[1, 0]
    for _, r in merged.iterrows():
        color = STAINING_COLORS[r["condition"]]
        ax.scatter(r["median_area"], r["median_amplitude"],
                   s=220, c=color, edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(STAINING_SHORT[r["condition"]].replace("\n", " "),
                    (r["median_area"], r["median_amplitude"]),
                    textcoords="offset points", xytext=(12, 5), fontsize=8)
    ax.set_xlabel("Median cell area (px, staining)", fontsize=11)
    ax.set_ylabel("Median contraction amplitude (px, video)", fontsize=11)
    ax.set_title("C) Cell Size vs Contraction Amplitude",
                 fontsize=13, fontweight="bold")

    # Panel D: summary table
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for _, r in merged.iterrows():
        short_cond = STAINING_SHORT[r["condition"]].replace("\n", " ")
        table_data.append([
            short_cond,
            str(r["n_videos"]),
            f"{r['n_beating']}/{r['n_videos']}",
            f"{r['median_amplitude']:.1f}" if r["median_amplitude"] > 0 else "-",
            str(r["n_cells"]),
            f"{r['median_factin']:.3f}" if not np.isnan(r["median_factin"]) else "-",
            f"{r['median_coherency']:.3f}" if not np.isnan(r["median_coherency"]) else "-",
        ])
    tbl = ax.table(
        cellText=table_data,
        colLabels=["Condition", "Videos\n(XR1)", "Beating\n(XR1)", "Amp\n(px)",
                    "Cells\n(stain)", "F-actin\nintens.",
                    "Actin\ncoher."],
        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_facecolor("#d9e2f3")
            cell.set_text_props(fontweight="bold")
    ax.set_title("D) Unified Summary Table", fontsize=13,
                 fontweight="bold", pad=20)

    fig.suptitle(
        "Figure V8: Cross-Reference -- Staining Morphology vs "
        "Video Contractility\n"
        "XR1 only (both modalities from same cell line)",
        fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(OUT / "video_Fig_V8_crossref.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    return merged


# ══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════
def generate_video_stats(vdf):
    print("  Generating video statistics tables...")

    beating = vdf[vdf["manual_classification"] == "active_beating"]

    metrics = [
        "roi_amplitude", "roi_bpm", "roi_contraction_time_s",
        "roi_relaxation_time_s", "roi_relax_contract_ratio",
        "roi_contraction_vel", "roi_relaxation_vel",
        "conv_disp_ratio", "n_foci", "foci_coverage_pct",
        "foci_mean_bpm", "foci_mean_amplitude",
    ]
    metric_labels = {
        "roi_amplitude": "Contraction amplitude (px)",
        "roi_bpm": "Beat rate (BPM)",
        "roi_contraction_time_s": "Contraction time (s)",
        "roi_relaxation_time_s": "Relaxation time (s)",
        "roi_relax_contract_ratio": "Relaxation/contraction ratio",
        "roi_contraction_vel": "Contraction velocity (px/s)",
        "roi_relaxation_vel": "Relaxation velocity (px/s)",
        "conv_disp_ratio": "Convergence-displacement ratio",
        "n_foci": "Contractile foci count",
        "foci_coverage_pct": "Foci FOV coverage (%)",
        "foci_mean_bpm": "Foci mean BPM",
        "foci_mean_amplitude": "Foci mean amplitude (px)",
    }

    # Descriptive stats
    desc_rows = []
    groups = {"All active_beating": beating}
    for sub in SUBSTRATE_ORDER:
        sub_df = beating[beating["substrate"] == sub]
        groups[f"active_beating | {SUBSTRATE_LABELS[sub]}"] = sub_df

    for grp_name, grp_df in groups.items():
        for m in metrics:
            vals = grp_df[m].dropna()
            if len(vals) > 0:
                desc_rows.append({
                    "group": grp_name,
                    "metric": metric_labels.get(m, m),
                    "n": len(vals),
                    "mean": round(float(vals.mean()), 4),
                    "std": round(float(vals.std()), 4),
                    "median": round(float(vals.median()), 4),
                    "q25": round(float(vals.quantile(0.25)), 4),
                    "q75": round(float(vals.quantile(0.75)), 4),
                })
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(OUT / "video_descriptive_statistics.csv", index=False)

    # Classification contingency
    cont_rows = []
    for sub in SUBSTRATE_ORDER:
        for cls in CLS_ORDER:
            n = len(vdf[(vdf["substrate"] == sub) &
                        (vdf["manual_classification"] == cls)])
            cont_rows.append({
                "grouping": "substrate",
                "group": SUBSTRATE_LABELS[sub],
                "classification": CLS_LABELS[cls],
                "count": n,
            })
    for cl in ["XR1", "GCaMP6f"]:
        for cls in CLS_ORDER:
            n = len(vdf[(vdf["cell_type"] == cl) &
                        (vdf["manual_classification"] == cls)])
            cont_rows.append({
                "grouping": "cell_line",
                "group": cl,
                "classification": CLS_LABELS[cls],
                "count": n,
            })
    cont_df = pd.DataFrame(cont_rows)
    cont_df.to_csv(OUT / "video_classification_contingency.csv", index=False)

    # Pairwise Mann-Whitney U
    pw_rows = []
    for s1, s2 in combinations(SUBSTRATE_ORDER, 2):
        d1 = beating[beating["substrate"] == s1]
        d2 = beating[beating["substrate"] == s2]
        for m in metrics:
            v1 = d1[m].dropna()
            v2 = d2[m].dropna()
            if len(v1) >= 2 and len(v2) >= 2:
                stat, p = mannwhitneyu(v1, v2, alternative="two-sided")
                pw_rows.append({
                    "comparison": "substrate",
                    "group_1": SUBSTRATE_LABELS[s1],
                    "group_2": SUBSTRATE_LABELS[s2],
                    "metric": metric_labels.get(m, m),
                    "n_1": len(v1), "n_2": len(v2),
                    "median_1": round(float(v1.median()), 4),
                    "median_2": round(float(v2.median()), 4),
                    "U_statistic": float(stat),
                    "p_value": float(p),
                    "significance": sig_stars(p),
                })

    # On-device vs control
    treat_beating = beating[beating["phase"] == "treatment"]
    dev = treat_beating[treat_beating["recording_type"] == "on-device"]
    ctrl = treat_beating[treat_beating["recording_type"] == "control"]
    for m in metrics:
        v1 = dev[m].dropna()
        v2 = ctrl[m].dropna()
        if len(v1) >= 2 and len(v2) >= 2:
            stat, p = mannwhitneyu(v1, v2, alternative="two-sided")
            pw_rows.append({
                "comparison": "device_vs_control",
                "group_1": "On-device (pulsed)",
                "group_2": "Control (unpulsed)",
                "metric": metric_labels.get(m, m),
                "n_1": len(v1), "n_2": len(v2),
                "median_1": round(float(v1.median()), 4),
                "median_2": round(float(v2.median()), 4),
                "U_statistic": float(stat),
                "p_value": float(p),
                "significance": sig_stars(p),
            })

    pw_df = pd.DataFrame(pw_rows)
    pw_df.to_csv(OUT / "video_pairwise_comparisons.csv", index=False)

    return desc_df, cont_df, pw_df


# ══════════════════════════════════════════════════════════════════════════
# TEXT SUMMARY
# ══════════════════════════════════════════════════════════════════════════
def write_video_summary(vdf, desc_df, pw_df, crossref_df):
    print("  Writing video report summary...")

    beating = vdf[vdf["manual_classification"] == "active_beating"]
    n_total = len(vdf)
    n_beating = len(beating)
    L = []  # line accumulator

    L.append("=" * 80)
    L.append("VIDEO CONTRACTILITY REPORT: BF OPTICAL FLOW ANALYSIS")
    L.append("Piezoelectric Substrate Effects on hiPSC-Derived Cardiomyocytes")
    L.append("=" * 80)
    L.append("")

    # ── METHODS ──
    L.append("METHODS")
    L.append("-" * 40)
    L.append("Analysis: Farneback optical flow (OpenCV) on 2x-downsampled")
    L.append("  brightfield video frames. Traditional CV, no deep learning.")
    L.append("")
    L.append("Motion decomposition:")
    L.append("  1. Whole-field displacement (frame-to-frame pixel magnitude)")
    L.append("  2. Convergence map (neg. divergence of displacement field)")
    L.append("     -> localizes active contraction (Czirok et al. Sci Rep 2017)")
    L.append("  3. Convergence-ROI (top 30% convergence = contracting region)")
    L.append("  4. Contractile foci (thresholded convergence + connected")
    L.append("     component labeling for individual beating centers)")
    L.append("")
    L.append("Beat detection: scipy.signal.find_peaks on detrended traces,")
    L.append("  1.5s settling-period exclusion, auto peak-polarity detection.")
    L.append("")
    L.append("Classification (expert-validated):")
    L.append("  active_beating        - synchronized whole-field contraction")
    L.append("  individual_cells_beating - localized foci, no whole-field synchrony")
    L.append("  non_contractile       - periodic passive displacement, no")
    L.append("                          sarcomeric contraction (loosely attached)")
    L.append("  no_beating            - no contractile or periodic motion")
    L.append("")
    L.append("All automated classifications validated by manual expert review.")
    L.append("")
    L.append("Key metrics (cf. Huebsch 2015, Sala/MUSCLEMOTION 2018):")
    L.append("  contraction amplitude, beat rate (BPM),")
    L.append("  contraction/relaxation time & velocity,")
    L.append("  relaxation/contraction ratio (maturation marker),")
    L.append("  convergence-displacement ratio, foci count & coverage,")
    L.append("  flow uniformity (adhesion quality)")
    L.append("")

    # ── RESULTS ──
    L.append("=" * 80)
    L.append("RESULTS")
    L.append("=" * 80)
    L.append("")

    L.append("1. OVERALL CONTRACTILITY")
    L.append("-" * 40)
    L.append(f"Total BF videos analyzed: {n_total}")
    for cls in CLS_ORDER:
        n = (vdf["manual_classification"] == cls).sum()
        L.append(f"  {CLS_LABELS[cls]:30s}: {n:2d} ({100*n/n_total:.0f}%)")
    L.append("")

    L.append("2. SUBSTRATE EFFECTS")
    L.append("-" * 40)
    for sub in SUBSTRATE_ORDER:
        s_df = vdf[vdf["substrate"] == sub]
        s_beat = s_df[s_df["manual_classification"] == "active_beating"]
        flow_n = s_df["has_flow"].sum()
        L.append(f"  {SUBSTRATE_LABELS[sub]}  (n={len(s_df)}):")
        L.append(f"    Active beating: {len(s_beat)}/{len(s_df)}")
        if len(s_beat) > 0:
            L.append(f"    Median amplitude: {s_beat['roi_amplitude'].median():.2f} px")
            L.append(f"    Median BPM: {s_beat['roi_bpm'].median():.1f}")
        L.append(f"    Flow/drift detected: {flow_n}/{len(s_df)} videos")
        L.append("")

    L.append("  Non-poled PVDF yielded the strongest and most consistent")
    L.append("  beating. Gold-coated PVDF showed poor cell adhesion (floating")
    L.append("  cells, flow/drift), reducing functional contractility.")
    L.append("")

    L.append("3. CELL LINE DIFFERENCES")
    L.append("-" * 40)
    for cl in ["XR1", "GCaMP6f"]:
        c_df = vdf[vdf["cell_type"] == cl]
        c_beat = c_df[c_df["manual_classification"] == "active_beating"]
        L.append(f"  {cl}  (n={len(c_df)}):")
        L.append(f"    Active beating: {len(c_beat)} ({100*len(c_beat)/len(c_df):.0f}%)")
        if len(c_beat) > 0:
            L.append(f"    Median amplitude: {c_beat['roi_amplitude'].median():.2f} px")
            L.append(f"    Median BPM: {c_beat['roi_bpm'].median():.1f}")
    L.append("")
    L.append("  XR1: robust spontaneous beating from day 2 onward.")
    L.append("  GCaMP6f: no synchronized beating at any timepoint;")
    L.append("  only C3-day8 (non-GCaMP6f) showed individual cell beating.")
    L.append("")

    L.append("4. TEMPORAL PROGRESSION")
    L.append("-" * 40)
    for d in ["day1", "day2", "day6", "day8"]:
        d_df = vdf[vdf["day"] == d]
        d_beat = d_df[d_df["manual_classification"] == "active_beating"]
        label = d.replace("day", "Day ")
        L.append(f"  {label}  (n={len(d_df)}):")
        L.append(f"    Active beating: {len(d_beat)}")
        if len(d_beat) > 0:
            L.append(f"    Median amplitude: {d_beat['roi_amplitude'].median():.2f} px")
            L.append(f"    Median BPM: {d_beat['roi_bpm'].median():.1f}")
    L.append("")
    L.append("  Days 1-2: baseline on bare plastic (control plate only).")
    L.append("  Days 6-8: treatment phase (device + control recordings).")
    L.append("  Amplitude increased from baseline to treatment, consistent")
    L.append("  with CM maturation and/or substrate-mediated effects.")
    L.append("")

    L.append("5. ON-DEVICE vs CONTROL")
    L.append("-" * 40)
    treat = vdf[vdf["phase"] == "treatment"]
    dev_b = treat[(treat["recording_type"] == "on-device") &
                  (treat["manual_classification"] == "active_beating")]
    ctrl_b = treat[(treat["recording_type"] == "control") &
                   (treat["manual_classification"] == "active_beating")]
    L.append(f"  On-device (pulsed): {len(dev_b)} actively beating")
    if len(dev_b) > 0:
        L.append(f"    Median amplitude: {dev_b['roi_amplitude'].median():.2f} px")
        L.append(f"    Median BPM: {dev_b['roi_bpm'].median():.1f}")
    L.append(f"  Control (unpulsed): {len(ctrl_b)} actively beating")
    if len(ctrl_b) > 0:
        L.append(f"    Median amplitude: {ctrl_b['roi_amplitude'].median():.2f} px")
        L.append(f"    Median BPM: {ctrl_b['roi_bpm'].median():.1f}")
    L.append("")

    # Significant pairwise results
    L.append("6. STATISTICAL COMPARISONS")
    L.append("-" * 40)
    if pw_df is not None and len(pw_df) > 0:
        sig_pw = pw_df[pw_df["significance"] != "ns"]
        if len(sig_pw) > 0:
            for _, r in sig_pw.iterrows():
                L.append(f"  {r['group_1']} vs {r['group_2']}:")
                L.append(f"    {r['metric']}: p={r['p_value']:.4f} {r['significance']}")
                L.append(f"    (median {r['median_1']:.4f} vs {r['median_2']:.4f})")
        else:
            L.append("  No pairwise comparisons reached statistical significance")
            L.append("  (all p > 0.05, Mann-Whitney U). This is expected given the")
            L.append("  small sample sizes (n=2-6 per substrate group). See")
            L.append("  video_pairwise_comparisons.csv for all test results.")
    L.append("")

    L.append("7. ADHESION AND NON-CONTRACTILE MOTION")
    L.append("-" * 40)
    n_nc = (vdf["manual_classification"] == "non_contractile").sum()
    n_nb = (vdf["manual_classification"] == "no_beating").sum()
    L.append(f"  Non-contractile motion: {n_nc} videos")
    L.append(f"  No beating: {n_nb} videos")
    L.append("")
    L.append("  Non-contractile motion = periodic passive displacement without")
    L.append("  sarcomeric contraction, in loosely attached cells. Violates")
    L.append("  the elastic-plate model (Czirok 2017); manual expert review")
    L.append("  was essential to correctly classify these cases.")
    L.append("")
    L.append("  Gold-PVDF showed highest non-contractile/no-beating prevalence,")
    L.append("  suggesting surface functionalization is needed for adhesion.")
    L.append("")

    # Cross-reference
    if crossref_df is not None:
        L.append("8. CROSS-REFERENCE: VIDEO vs STAINING")
        L.append("-" * 40)
        for _, r in crossref_df.iterrows():
            cond_short = STAINING_SHORT[r["condition"]].replace("\n", " ")
            L.append(f"  {cond_short}:")
            L.append(f"    Videos: {r['n_videos']}, Beating: {r['n_beating']}")
            if r["median_amplitude"] > 0:
                L.append(f"    Median amplitude: {r['median_amplitude']:.2f} px")
            L.append(f"    Staining cells: {r['n_cells']}")
            if not np.isnan(r.get("median_factin", np.nan)):
                L.append(f"    F-actin intensity: {r['median_factin']:.3f}")
                L.append(f"    Actin coherency: {r['median_coherency']:.3f}")
            L.append("")
        L.append("  Note: all staining images are from XR1 wells (B2, A3, B3).")
        L.append("  Video data above filtered to XR1 only for apples-to-apples comparison.")
        L.append("")

    # ── DISCUSSION ──
    L.append("=" * 80)
    L.append("DISCUSSION")
    L.append("=" * 80)
    L.append("")

    L.append("1. SUBSTRATE-DEPENDENT ADHESION AND FUNCTION")
    L.append("-" * 40)
    L.append("  Non-poled PVDF supported high-amplitude, regular beating")
    L.append("  (strongest contractions in the dataset). Gold-coated PVDF")
    L.append("  showed pervasive poor adhesion with floating cells/debris,")
    L.append("  consistent with known challenges of cell adhesion on metallic")
    L.append("  thin films. Gold surfaces typically require protein coating")
    L.append("  (fibronectin, laminin, Matrigel) for cardiomyocyte attachment.")
    L.append("")
    L.append("  The convergence-based analysis was key: it distinguished")
    L.append("  contractile regions from passively displaced floating cells,")
    L.append("  preventing false-positive beating classification.")
    L.append("")

    L.append("2. CELL LINE VARIABILITY: XR1 vs GCaMP6f")
    L.append("-" * 40)
    L.append("  XR1-derived CMs beat spontaneously from day 2, with")
    L.append("  increasing amplitude through day 8 (functional maturation).")
    L.append("  GCaMP6f-derived CMs showed NO synchronized beating at any point.")
    L.append("")
    L.append("  MECHANISTIC EXPLANATION: The GCaMP6f line carries a genetically")
    L.append("  encoded calcium indicator whose calmodulin (CaM) moiety")
    L.append("  interferes with L-type calcium channel (CaV1) gating")
    L.append("  (Yang et al. Nat Commun 2018). In cardiomyocytes, CaV1.2 is")
    L.append("  the primary trigger for excitation-contraction coupling.")
    L.append("  Constitutive GCaMP6f expression may perturb CaV1.2 function,")
    L.append("  directly impairing calcium transients needed for synchronized")
    L.append("  beating. GCaMP2 transgenic mice developed cardiomegaly.")
    L.append("  This provides a mechanistic basis beyond generic iPSC")
    L.append("  line-to-line variability (Burridge et al. 2014).")
    L.append("")

    L.append("3. NON-CONTRACTILE MOTION")
    L.append("-" * 40)
    L.append("  Periodic passive displacement without sarcomeric contraction")
    L.append("  was observed primarily in loosely attached cells on gold-PVDF")
    L.append("  and in early-stage cultures (day 1). This distinction matters")
    L.append("  because convergence analysis assumes substrate-adhered cells;")
    L.append("  floating cells produce false-positive convergence signals.")
    L.append("  Expert manual validation was essential for classification.")
    L.append("")

    L.append("4. PIEZOELECTRIC STIMULATION EFFECTS")
    L.append("-" * 40)
    if len(dev_b) > 0 and len(ctrl_b) > 0:
        L.append(f"  On-device median amplitude: {dev_b['roi_amplitude'].median():.2f} px")
        L.append(f"  Control median amplitude: {ctrl_b['roi_amplitude'].median():.2f} px")
    L.append("")
    L.append("  Interpretation requires caution: on-device wells have different")
    L.append("  substrates (non-poled or gold PVDF) vs some control wells")
    L.append("  (plastic). Paired within-well comparisons (Figure V5) provide")
    L.append("  the most reliable assessment of pulsation effects.")
    L.append("")

    L.append("5. STRUCTURE-FUNCTION RELATIONSHIP")
    L.append("-" * 40)
    L.append("  Cross-referencing with phalloidin staining (Figure V8) shows")
    L.append("  condition-level associations. The existing staining analysis")
    L.append("  found Au-PVDF Poled Pulsed (device) has significantly lower")
    L.append("  actin coherency (p<0.001 vs all conditions), interpreted as")
    L.append("  maturation-associated cytoskeletal remodeling (stress fibers")
    L.append("  -> sarcomeric actin). Direct cell-level correlation is not")
    L.append("  possible (different measurement modalities/populations).")
    L.append("")

    L.append("6. METHODOLOGICAL NOTES")
    L.append("-" * 40)
    L.append("  - Farneback optical flow: established for BF CM video at")
    L.append("    ~10-12 FPS (Maddah et al. 2015)")
    L.append("  - Convergence measure: discriminates active contraction from")
    L.append("    passive deformation, validated by myosin IIa co-localization")
    L.append("    (Czirok et al. Sci Rep 2017)")
    L.append("  - Per-center foci detection: extends whole-field analysis to")
    L.append("    sparse/asynchronous cultures (Huebsch et al. 2015)")
    L.append("  - MUSCLEMOTION (Sala et al. 2018): our metrics align with")
    L.append("    their beat rate, amplitude, contraction/relaxation duration")
    L.append("")

    # ── LIMITATIONS ──
    L.append("=" * 80)
    L.append("LIMITATIONS")
    L.append("=" * 80)
    L.append("")
    L.append("  1. N=1 biological replicate per condition")
    L.append("  2. Unbalanced design (no 'Nonpoled Un-pulsed' control)")
    L.append("  3. Low sample size (1-5 videos per substrate group)")
    L.append("  4. No usable gold-substrate videos at day 6 (< 2 FPS)")
    L.append("  5. Fluorescence/calcium imaging not yet analyzed")
    L.append("  6. Staining images are all from XR1 wells; GCaMP6f not imaged")
    L.append("  7. 10-12 FPS limits temporal resolution of kinetics")
    L.append("  8. 2x spatial downsampling may reduce sensitivity")
    L.append("  9. Old cells from external lab")
    L.append("  10. B3-day8 PVDF film detached (1 data point lost)")
    L.append("")

    # ── FILES ──
    L.append("=" * 80)
    L.append("FILES PRODUCED")
    L.append("=" * 80)
    L.append("  Figures:")
    L.append("    video_Fig_V1_classification.png  - Classification overview")
    L.append("    video_Fig_V2_temporal.png         - Temporal progression")
    L.append("    video_Fig_V3_substrate.png        - Substrate comparison")
    L.append("    video_Fig_V4_cellline.png         - Cell line comparison")
    L.append("    video_Fig_V5_paired.png           - On-device vs control")
    L.append("    video_Fig_V6_foci.png             - Contractile foci")
    L.append("    video_Fig_V7_adhesion.png         - Adhesion/flow analysis")
    L.append("    video_Fig_V8_crossref.png         - Morphology vs function")
    L.append("")
    L.append("  Data:")
    L.append("    video_descriptive_statistics.csv  - Beating metric stats")
    L.append("    video_classification_contingency.csv - Classification counts")
    L.append("    video_pairwise_comparisons.csv    - Mann-Whitney U tests")
    L.append("")
    L.append("  Source:")
    L.append("    video_results/batch_results_v3_final.csv - Per-video metrics")
    L.append("")

    text = "\n".join(L)
    (OUT / "VIDEO_REPORT_SUMMARY.txt").write_text(text, encoding="utf-8")
    return text


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("GENERATING VIDEO CONTRACTILITY REPORT")
    print("=" * 60)

    vdf, sdf = load_data()
    print(f"\nVideo data: {len(vdf)} videos, "
          f"{vdf['manual_classification'].nunique()} classifications")
    print(f"Staining data: {'loaded (' + str(len(sdf)) + ' cells)' if sdf is not None else 'NOT FOUND'}")
    print()

    # Figures
    fig_v1_classification(vdf)
    fig_v2_temporal(vdf)
    fig_v3_substrate(vdf)
    fig_v4_cellline(vdf)
    fig_v5_paired(vdf)
    fig_v6_foci(vdf)
    fig_v7_adhesion(vdf)
    crossref_df = fig_v8_crossref(vdf, sdf)

    # Statistics
    desc_df, cont_df, pw_df = generate_video_stats(vdf)

    # Text summary
    text = write_video_summary(vdf, desc_df, pw_df, crossref_df)

    print("\n" + "=" * 60)
    print("VIDEO REPORT COMPLETE")
    print(f"All files saved to: {OUT}/")
    print("=" * 60)
    print()
    print(text)


if __name__ == "__main__":
    main()
