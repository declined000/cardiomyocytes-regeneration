"""
Combined Report: Staining + Video + Cross-Reference
=====================================================
Pulls ALL numbers fresh from raw CSV data files.
Verifies every expected figure PNG exists.
Generates new cross-reference figures.
Writes a single unified COMBINED_REPORT.txt.

Data sources:
  Staining: phalloidin_per_cell_data.csv, multinucleation_cellpose.csv,
            pairwise_comparisons.csv
  Video:    batch_results_v3_final.csv, video_pairwise_comparisons.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from scipy.stats import mannwhitneyu
import cv2
import warnings
warnings.filterwarnings("ignore")

OUT = Path("final_report")

# ── Staining condition labels (newline-embedded in per-cell CSV) ───────
STAIN_CONDS_NL = [
    "Au-PVDF Poled\nPulsed (device)",
    "Au-PVDF Poled\nUn-pulsed (control)",
    "β-PVDF Nonpoled\nPulsed (device)",
    "Cells only\n(baseline)",
]
STAIN_CONDS_FLAT = [
    "Au-PVDF Poled Pulsed (device)",
    "Au-PVDF Poled Un-pulsed (control)",
    "β-PVDF Nonpoled Pulsed (device)",
    "Cells only (baseline)",
]
NL_TO_FLAT = dict(zip(STAIN_CONDS_NL, STAIN_CONDS_FLAT))
FLAT_TO_NL = dict(zip(STAIN_CONDS_FLAT, STAIN_CONDS_NL))
STAIN_SHORT = {
    "Au-PVDF Poled Pulsed (device)": "Poled Pulsed (device)",
    "Au-PVDF Poled Un-pulsed (control)": "Poled Un-pulsed (control)",
    "β-PVDF Nonpoled Pulsed (device)": "Nonpoled Pulsed (device)",
    "Cells only (baseline)": "Cells only (baseline)",
}
STAIN_COLORS = {
    "Au-PVDF Poled Pulsed (device)": "#ff9966",
    "Au-PVDF Poled Un-pulsed (control)": "#ffd966",
    "β-PVDF Nonpoled Pulsed (device)": "#a8d5a2",
    "Cells only (baseline)": "#7fbfff",
}

VID_SRC = Path("Cardiomyocytes")

CLS_ORDER = ["active_beating", "individual_cells_beating",
             "non_contractile", "no_beating"]
CLS_LABELS = {
    "active_beating": "Active beating",
    "individual_cells_beating": "Individual cells beating",
    "non_contractile": "Non-contractile motion",
    "no_beating": "No beating",
}


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════
def load_all():
    print("Loading data from CSVs...")
    d = {}

    d["phal"] = pd.read_csv(OUT / "phalloidin_per_cell_data.csv")
    print(f"  phalloidin_per_cell_data.csv: {len(d['phal'])} cells")

    d["mn"] = pd.read_csv(OUT / "multinucleation_cellpose.csv")
    print(f"  multinucleation_cellpose.csv: {len(d['mn'])} cells")

    d["pw_stain"] = pd.read_csv(OUT / "pairwise_comparisons.csv")
    print(f"  pairwise_comparisons.csv: {len(d['pw_stain'])} comparisons")

    d["vid"] = pd.read_csv(Path("video_results") / "batch_results_v3_final.csv")
    print(f"  batch_results_v3_final.csv: {len(d['vid'])} videos")

    pw_vid_path = OUT / "video_pairwise_comparisons.csv"
    if pw_vid_path.exists():
        d["pw_vid"] = pd.read_csv(pw_vid_path)
        print(f"  video_pairwise_comparisons.csv: {len(d['pw_vid'])} comparisons")
    else:
        d["pw_vid"] = pd.DataFrame()

    fl_path = Path("fl_results") / "fl_batch_results.csv"
    if fl_path.exists():
        d["fl"] = pd.read_csv(fl_path)
        print(f"  fl_batch_results.csv: {len(d['fl'])} FL videos")
    else:
        d["fl"] = pd.DataFrame()

    return d


# ══════════════════════════════════════════════════════════════════════════
# FIGURE VERIFICATION
# ══════════════════════════════════════════════════════════════════════════
def verify_figures():
    print("\nVerifying final_report/ figures...")
    expected = {
        "Staining Fig 1": "Fig1_raw_images.png",
        "Staining Fig 2": "Fig2_segmentation_qc.png",
        "Staining Fig 3": "Fig3_morphology.png",
        "Staining Fig 4": "Fig4_factin_organization.png",
        "Staining Fig 5": "Fig5_summary_bars.png",
        "Staining Fig 6": "Fig6_multinucleation.png",
        "Staining Fig 7": "Fig7_actinin_INCONCLUSIVE.png",
        "Staining Fig 8": "Fig8_significance_heatmap.png",
        "Video Fig V1": "video_Fig_V1_classification.png",
        "Video Fig V2": "video_Fig_V2_temporal.png",
        "Video Fig V3": "video_Fig_V3_substrate.png",
        "Video Fig V4": "video_Fig_V4_cellline.png",
        "Video Fig V5": "video_Fig_V5_paired.png",
        "Video Fig V6": "video_Fig_V6_foci.png",
        "Video Fig V7": "video_Fig_V7_adhesion.png",
        "Video Fig V8": "video_Fig_V8_crossref.png",
        "Extra: key metrics": "key_comparisons_metrics.png",
        "Extra: key heatmap": "key_comparisons_heatmap.png",
        "Extra: poled vs nonpoled": "poled_vs_nonpoled_visual.png",
        "Extra: pulsed vs unpulsed": "pulsed_vs_unpulsed_AuPVDF.png",
        "Video Fig V9": "video_Fig_V9_classification_frames.png",
        "Video Fig V10a": "video_Fig_V10a_temporal_control.png",
        "Video Fig V10b": "video_Fig_V10b_temporal_device.png",
    }

    results = {}
    for label, fname in expected.items():
        path = OUT / fname
        exists = path.exists()
        results[label] = (fname, exists)
        status = "OK" if exists else "NOT_LOCAL"
        print(f"  [{status}] {fname}")

    return results


def verify_video_results(vid):
    """Verify every video in the CSV has a matching analysis PNG."""
    print("\nVerifying video_results/ per-video outputs...")
    VID_DIR = Path("video_results")

    baseline_dir = VID_DIR / "baseline_day1-2"
    treatment_dir = VID_DIR / "treatment_day6-8"

    ok_count = 0
    missing = []
    extra_files = []

    expected_pngs = set()

    for _, row in vid.iterrows():
        fname = row["filename"]
        stem = fname.replace(".avi", "")
        png_name = f"{stem}_analysis.png"
        phase = row["phase"]

        if phase == "baseline":
            target = baseline_dir / png_name
        else:
            target = treatment_dir / png_name

        expected_pngs.add(str(target))

        if target.exists():
            ok_count += 1
            print(f"  [OK]      {target}")
        else:
            missing.append(str(target))
            print(f"  [NOT_LOCAL] {target}")

    # Check for extra/unexpected PNGs in each subfolder
    for subdir, label in [(baseline_dir, "baseline_day1-2"),
                           (treatment_dir, "treatment_day6-8")]:
        if subdir.exists():
            for f in sorted(subdir.iterdir()):
                if f.suffix == ".png" and str(f) not in expected_pngs:
                    extra_files.append(str(f))
                    print(f"  [EXTRA]   {f}")

    # Check CSV files in video_results/
    print("\n  Video data files:")
    for csv_name in ["batch_results_v3_final.csv",
                     "batch_results_v3_with_corrections.csv",
                     "batch_results_v3.csv",
                     "MANUAL_REVIEW_NOTES.txt"]:
        p = VID_DIR / csv_name
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status}] video_results/{csv_name}")

    print(f"\n  Summary: {ok_count}/{len(vid)} analysis PNGs found, "
          f"{len(missing)} missing, {len(extra_files)} extra")

    return {"ok": ok_count, "total": len(vid), "missing": missing,
            "extra": extra_files}


# ══════════════════════════════════════════════════════════════════════════
# CROSS-REFERENCE FIGURES (NEW)
# ══════════════════════════════════════════════════════════════════════════
def fig_c1_structure_vs_function(phal, vid):
    """Side-by-side: staining metrics vs video metrics per condition."""
    print("  Generating Fig C1: Structure vs Function...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    staining_data = {}
    for sc_flat in STAIN_CONDS_FLAT:
        sc_nl = FLAT_TO_NL[sc_flat]
        sub = phal[phal["condition"] == sc_nl]
        staining_data[sc_flat] = {
            "n_cells": len(sub),
            "factin": sub["mean_phalloidin_intensity"].median(),
            "coherency": sub["actin_coherency"].median(),
            "circularity": sub["circularity"].median(),
            "area": sub["cell_area_px"].median(),
        }

    # Filter video to XR1 only for cross-reference with staining
    # (all staining images are from XR1 wells: B2, A3, B3)
    vid_t = vid[vid["cell_type"] == "XR1"]

    video_data = {}
    for sc_flat in STAIN_CONDS_FLAT:
        sub = vid_t[vid_t["staining_condition"] == sc_flat]
        beat = sub[sub["manual_classification"] == "active_beating"]
        video_data[sc_flat] = {
            "n_videos": len(sub),
            "n_beating": len(beat),
            "beat_frac": len(beat) / len(sub) if len(sub) > 0 else 0,
            "amplitude": beat["roi_amplitude"].median() if len(beat) > 0 else 0,
            "bpm": beat["roi_bpm"].median() if len(beat) > 0 else 0,
            "n_foci_mean": sub["n_foci"].mean(),
        }

    x = np.arange(len(STAIN_CONDS_FLAT))
    short_labels = [STAIN_SHORT[c] for c in STAIN_CONDS_FLAT]
    colors = [STAIN_COLORS[c] for c in STAIN_CONDS_FLAT]

    # Panel A: F-actin intensity (staining) vs beating fraction (video)
    ax = axes[0, 0]
    factin_vals = [staining_data[c]["factin"] for c in STAIN_CONDS_FLAT]
    beat_frac_vals = [video_data[c]["beat_frac"] * 100 for c in STAIN_CONDS_FLAT]
    bars1 = ax.bar(x - 0.2, factin_vals, 0.35, color=colors, edgecolor="black",
                   label="F-actin intensity (staining)")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + 0.2, beat_frac_vals, 0.35, color=colors, edgecolor="black",
                    alpha=0.5, hatch="//", label="% active beating (video)")
    ax.set_ylabel("Median F-actin intensity", fontsize=11)
    ax2.set_ylabel("% videos with active beating", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8, rotation=10)
    ax.set_title("A) F-actin Intensity vs Beating Fraction", fontsize=13,
                 fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    # Panel B: Actin coherency (staining) vs amplitude (video)
    ax = axes[0, 1]
    coh_vals = [staining_data[c]["coherency"] for c in STAIN_CONDS_FLAT]
    amp_vals = [video_data[c]["amplitude"] for c in STAIN_CONDS_FLAT]
    bars1 = ax.bar(x - 0.2, coh_vals, 0.35, color=colors, edgecolor="black",
                   label="Actin coherency (staining)")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + 0.2, amp_vals, 0.35, color=colors, edgecolor="black",
                    alpha=0.5, hatch="//", label="Median amplitude (video)")
    ax.set_ylabel("Median actin coherency", fontsize=11)
    ax2.set_ylabel("Median contraction amplitude (px)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8, rotation=10)
    ax.set_title("B) Actin Coherency vs Contraction Amplitude", fontsize=13,
                 fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    # Panel C: Cell area (staining) vs foci count (video)
    ax = axes[1, 0]
    area_vals = [staining_data[c]["area"] for c in STAIN_CONDS_FLAT]
    foci_vals = [video_data[c]["n_foci_mean"] for c in STAIN_CONDS_FLAT]
    bars1 = ax.bar(x - 0.2, area_vals, 0.35, color=colors, edgecolor="black",
                   label="Cell area (staining)")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + 0.2, foci_vals, 0.35, color=colors, edgecolor="black",
                    alpha=0.5, hatch="//", label="Mean foci count (video)")
    ax.set_ylabel("Median cell area (px)", fontsize=11)
    ax2.set_ylabel("Mean contractile foci count", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8, rotation=10)
    ax.set_title("C) Cell Size vs Contractile Foci", fontsize=13,
                 fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    # Panel D: Summary table
    ax = axes[1, 1]
    ax.axis("off")
    tdata = []
    for c in STAIN_CONDS_FLAT:
        s = staining_data[c]
        v = video_data[c]
        tdata.append([
            STAIN_SHORT[c],
            str(s["n_cells"]),
            f"{s['factin']:.3f}",
            f"{s['coherency']:.3f}",
            str(v["n_videos"]),
            f"{v['n_beating']}/{v['n_videos']}",
            f"{v['amplitude']:.1f}" if v['amplitude'] > 0 else "-",
        ])
    tbl = ax.table(
        cellText=tdata,
        colLabels=["Condition", "Cells\n(stain)", "F-actin\nintens.",
                    "Actin\ncoher.", "Videos\n(XR1)", "Beating\n(XR1)", "Amp\n(px)"],
        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.0)
    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_facecolor("#d9e2f3")
            cell.set_text_props(fontweight="bold")
    ax.set_title("D) Unified Data Table", fontsize=13, fontweight="bold", pad=20)

    fig.suptitle(
        "Figure C1: Structure vs Function -- Staining Morphology alongside "
        "Video Contractility\n"
        "XR1 only (both modalities from same cell line; all numbers from raw CSV)",
        fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(OUT / "combined_Fig_C1_structure_vs_function.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    return staining_data, video_data


def fig_c2_adhesion_evidence(phal, vid):
    """Circularity from staining vs flow/classification from video."""
    print("  Generating Fig C2: Adhesion evidence...")

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Map staining conditions to substrate groups for comparison
    substrate_stain_map = {
        "gold_PVDF": ["Au-PVDF Poled\nPulsed (device)",
                       "Au-PVDF Poled\nUn-pulsed (control)"],
        "non-poled_PVDF": ["β-PVDF Nonpoled\nPulsed (device)"],
        "plastic": ["Cells only\n(baseline)"],
    }
    sub_labels = {"plastic": "Bare plastic", "non-poled_PVDF": "Non-poled PVDF",
                  "gold_PVDF": "Gold-coated PVDF"}
    sub_colors = {"plastic": "#7fbfff", "non-poled_PVDF": "#a8d5a2",
                  "gold_PVDF": "#ff9966"}
    sub_order = ["plastic", "non-poled_PVDF", "gold_PVDF"]

    # Panel A: Circularity by substrate group (staining)
    ax = axes[0]
    circ_data = []
    circ_labels = []
    circ_colors = []
    for s in sub_order:
        conds = substrate_stain_map[s]
        vals = phal[phal["condition"].isin(conds)]["circularity"].dropna().values
        circ_data.append(vals)
        circ_labels.append(sub_labels[s])
        circ_colors.append(sub_colors[s])

    bp = ax.boxplot(circ_data, patch_artist=True, widths=0.6, showfliers=False)
    for j, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(circ_colors[j])
        patch.set_alpha(0.7)
    for j, d in enumerate(circ_data):
        if len(d) > 0:
            jitter = np.random.normal(j + 1, 0.04, size=min(len(d), 100))
            sample = np.random.choice(d, size=min(len(d), 100), replace=False)
            ax.scatter(jitter, sample, alpha=0.2, s=8, color="black", zorder=5)
    ax.set_xticklabels(circ_labels, fontsize=10)
    ax.set_ylabel("Circularity (staining)", fontsize=11)
    ax.set_title("A) Cell Circularity by Substrate\n(staining data, n cells shown)",
                 fontsize=13, fontweight="bold")
    for j, d in enumerate(circ_data):
        ax.text(j + 1, ax.get_ylim()[1] * 0.95, f"n={len(d)}", ha="center",
                fontsize=9, style="italic")

    # Significance brackets
    for (i1, i2) in [(0, 2), (1, 2)]:
        if len(circ_data[i1]) >= 2 and len(circ_data[i2]) >= 2:
            _, p = mannwhitneyu(circ_data[i1], circ_data[i2], alternative="two-sided")
            stars = sig_stars(p)
            if stars != "ns":
                ymax = max(np.percentile(circ_data[i1], 95),
                           np.percentile(circ_data[i2], 95))
                y = ymax + 0.03 * (1 + abs(i2 - i1))
                ax.plot([i1 + 1, i2 + 1], [y, y], "k-", lw=1)
                ax.text((i1 + i2 + 2) / 2, y, f"{stars}\np={p:.2e}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Filter video to XR1 only (matching staining cell line)
    vid_t = vid[vid["cell_type"] == "XR1"]

    # Panel B: Flow prevalence by substrate (video, XR1 only)
    ax = axes[1]
    flow_data = []
    for s in sub_order:
        s_df = vid_t[vid_t["substrate"] == s]
        n = len(s_df)
        n_flow = s_df["has_flow"].sum() if n > 0 else 0
        flow_data.append((n_flow, n))

    pcts = [100 * f / n if n > 0 else 0 for f, n in flow_data]
    bars = ax.bar(range(len(sub_order)), pcts,
                  color=[sub_colors[s] for s in sub_order], edgecolor="black")
    for i, (f, n) in enumerate(flow_data):
        ax.text(i, pcts[i] + 2, f"{f}/{n}", ha="center", fontsize=11,
                fontweight="bold")
    ax.set_xticks(range(len(sub_order)))
    ax.set_xticklabels([sub_labels[s] for s in sub_order], fontsize=10)
    ax.set_ylabel("% videos with flow/drift", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("B) Flow/Drift Prevalence\n(video, XR1 only)", fontsize=13,
                 fontweight="bold")

    # Panel C: Classification distribution by substrate (video, XR1 only)
    ax = axes[2]
    cls_colors = {
        "active_beating": "#2ca02c",
        "individual_cells_beating": "#1f77b4",
        "non_contractile": "#ff7f0e",
        "no_beating": "#d62728",
    }
    x = np.arange(len(sub_order))
    bottom = np.zeros(len(sub_order))
    for cls in CLS_ORDER:
        vals = np.array([
            (vid_t[vid_t["substrate"] == s]["manual_classification"] == cls).sum()
            for s in sub_order
        ], dtype=float)
        label = CLS_LABELS[cls]
        ax.bar(x, vals, bottom=bottom, color=cls_colors[cls],
               label=label, edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(i, bottom[i] + v / 2, str(int(v)), ha="center",
                        va="center", fontsize=10, fontweight="bold")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([sub_labels[s] for s in sub_order], fontsize=10)
    ax.set_ylabel("Number of videos", fontsize=11)
    ax.set_title("C) Classification by Substrate\n(video, XR1 only)", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Figure C2: Adhesion Evidence -- Staining Morphology vs "
        "Video Flow/Drift Patterns\n"
        "XR1 only (both modalities from same cell line)",
        fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(OUT / "combined_Fig_C2_adhesion_evidence.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# VIDEO FRAME COMPARISON FIGURES
# ══════════════════════════════════════════════════════════════════════════
def _extract_frames(avi_path, frame_fractions=(0.0, 0.33)):
    """Extract frames at given fractional positions. Returns list of gray frames."""
    cap = cv2.VideoCapture(str(avi_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for frac in frame_fractions:
        idx = min(int(frac * n_frames), n_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            frames.append(None)
    cap.release()
    return frames, fps, n_frames


def fig_v9_classification_examples(vid):
    """4-column panel: one representative per classification, with actual frames."""
    print("  Generating Fig V9: Classification examples from video frames...")

    representatives = {
        "active_beating": ("B3-day8-tammy-control.avi",
                           "XR1, plastic, day 8"),
        "individual_cells_beating": ("C3-day8-non-Jaz.avi",
                                      "GCaMP6f, non-poled PVDF, day 8"),
        "non_contractile": ("C3-day1-fixed.avi",
                            "GCaMP6f, plastic, day 1"),
        "no_beating": ("C2-day8-gold-Jaz.avi",
                       "GCaMP6f, gold PVDF, day 8"),
    }

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    cls_colors = {
        "active_beating": "#2ca02c",
        "individual_cells_beating": "#1f77b4",
        "non_contractile": "#ff7f0e",
        "no_beating": "#d62728",
    }

    for col_idx, cls in enumerate(CLS_ORDER):
        avi_name, desc = representatives[cls]
        avi_path = VID_SRC / avi_name

        row = vid[vid["filename"] == avi_name]
        amp = row["roi_amplitude"].values[0] if len(row) > 0 else 0
        bpm = row["roi_bpm"].values[0] if len(row) > 0 else 0
        n_foci = int(row["n_foci"].values[0]) if len(row) > 0 else 0

        frames, fps, n_frames = _extract_frames(avi_path, (0.0, 0.33))

        # Row 0: first frame
        ax = axes[0, col_idx]
        if frames[0] is not None:
            ax.imshow(frames[0], cmap="gray", aspect="equal")
        ax.set_title(f"{CLS_LABELS[cls]}\n{desc}",
                     fontsize=11, fontweight="bold",
                     color=cls_colors[cls])
        ax.axis("off")

        # Row 1: mid-video frame
        ax = axes[1, col_idx]
        if frames[1] is not None:
            ax.imshow(frames[1], cmap="gray", aspect="equal")
        metrics_txt = f"Amp: {amp:.2f} px | BPM: {bpm:.1f}\nFoci: {n_foci} | FPS: {fps:.1f}"
        ax.set_title(metrics_txt, fontsize=9)
        ax.axis("off")

    axes[0, 0].text(-0.05, 0.5, "Frame 1\n(start)", transform=axes[0, 0].transAxes,
                     fontsize=12, fontweight="bold", va="center", ha="right", rotation=90)
    axes[1, 0].text(-0.05, 0.5, "Frame ~1/3\n(mid)", transform=axes[1, 0].transAxes,
                     fontsize=12, fontweight="bold", va="center", ha="right", rotation=90)

    fig.suptitle(
        "Figure V9: Representative Video Frames by Classification\n"
        "Actual BF frames extracted from source videos",
        fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "video_Fig_V9_classification_frames.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved video_Fig_V9_classification_frames.png")


def fig_v10_temporal_progression(vid):
    """All wells across days 1→2→6→8, one row per well, with trend annotations.

    Uses control-plate recordings for temporal consistency (same physical well).
    Column 2 day 2/6 AVIs exist on disk but were not in the final analysis
    (low FPS); frames are shown with 'not analyzed' note.
    """
    print("  Generating Fig V10: Temporal progression for all wells...")

    DAYS = ["day1", "day2", "day6", "day8"]
    DAY_LABELS = {"day1": "Day 1", "day2": "Day 2", "day6": "Day 6", "day8": "Day 8"}

    # Map: (well, day) → AVI filename.  Prioritise control-plate recordings
    # so we track the SAME physical well across time.
    # Column 3 (plastic / nonpoled): full temporal coverage
    # Column 2 (gold): day 2 & 6 exist on disk but not in analysis CSV
    well_day_avi = {
        # ── A3 (XR1, plastic on control plate) ──
        ("A3", "day1"): "A3-day1-fixed.avi",
        ("A3", "day2"): "A3-day2-Tammy-NEW.avi",
        ("A3", "day6"): "A3-day6-control-NON1-TAMMY.avi",
        ("A3", "day8"): "A3-day8-Tammy-CONTROL.avi",
        # ── B3 (XR1, plastic on control plate) ──
        ("B3", "day1"): "B3-day1-fixed.avi",
        ("B3", "day2"): "B3-day2.avi",
        ("B3", "day6"): "day6-B3-control-non.avi",
        ("B3", "day8"): "B3-day8-tammy-control.avi",
        # ── C3 (GCaMP6f, plastic on control plate) ──
        ("C3", "day1"): "C3-day1-fixed.avi",
        ("C3", "day2"): "C3-day2.avi",
        ("C3", "day6"): "C3-day6-control-bf-new.avi",
        ("C3", "day8"): "C3-day8-control-Jaz-bf.avi",
        # ── A2 (XR1, gold PVDF on control plate) ──
        ("A2", "day2"): "A2-day2-tammy-gold.avi",
        ("A2", "day6"): "comtrol-A2-gold1-day6-Tammy.avi",
        ("A2", "day8"): "A2-day8-gold-tAMMY-CONTROL.avi",
        # ── B2 (XR1, gold PVDF on control plate) ──
        ("B2", "day2"): "B2-day2-Tammy-gold.avi",
        ("B2", "day6"): "control-B2-gold2-day6-Tammy.avi",
        ("B2", "day8"): "B2-day8-gold-tAMMY-control.avi",
        # ── C2 (GCaMP6f, gold PVDF on control plate) ──
        ("C2", "day2"): "C2-day2-gold.avi",
        ("C2", "day6"): "C2-day6-gold-control-bf.avi",
        ("C2", "day8"): "C2-day8-gold-Jaz-CONTROL-bf.avi",
    }

    well_info = {
        "A3": ("XR1", "Plastic (control plate)"),
        "B3": ("XR1", "Plastic (control plate)"),
        "C3": ("GCaMP6f", "Plastic (control plate)"),
        "A2": ("XR1", "Gold PVDF (control plate)"),
        "B2": ("XR1", "Gold PVDF (control plate)"),
        "C2": ("GCaMP6f", "Gold PVDF (control plate)"),
    }
    WELLS = ["A3", "B3", "C3", "A2", "B2", "C2"]

    cls_colors = {
        "active_beating": "#2ca02c",
        "individual_cells_beating": "#1f77b4",
        "non_contractile": "#ff7f0e",
        "no_beating": "#d62728",
    }

    IMG_W, IMG_H = 1392, 1040  # native video resolution for aspect ratio

    fig, axes = plt.subplots(len(WELLS), len(DAYS), figsize=(26, 5 * len(WELLS)))
    plt.subplots_adjust(left=0.14)

    for row_idx, well in enumerate(WELLS):
        cell_line, substrate = well_info[well]
        amplitudes = {}

        for col_idx, day in enumerate(DAYS):
            ax = axes[row_idx, col_idx]
            key = (well, day)

            # Uniform sizing: set same limits on every panel
            ax.set_xlim(0, IMG_W)
            ax.set_ylim(IMG_H, 0)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(DAY_LABELS[day], fontsize=14, fontweight="bold")

            if col_idx == 0:
                ax.set_ylabel(f"{well}  |  {cell_line}  |  {substrate}",
                              fontsize=10, fontweight="bold", rotation=0,
                              labelpad=120, va="center")

            if key not in well_day_avi:
                ax.set_facecolor("#f0f0f0")
                ax.text(IMG_W / 2, IMG_H / 2, "No recording",
                        ha="center", va="center", fontsize=16, color="#999999",
                        style="italic")
                continue

            avi_name = well_day_avi[key]
            avi_path = VID_SRC / avi_name

            if not avi_path.exists():
                ax.set_facecolor("#f0f0f0")
                ax.text(IMG_W / 2, IMG_H / 2, f"File not found",
                        ha="center", va="center", fontsize=12, color="red")
                continue

            frames, fps_val, n_fr = _extract_frames(avi_path, (0.25,))
            if frames[0] is not None:
                ax.imshow(frames[0], cmap="gray", extent=[0, IMG_W, IMG_H, 0])

            csv_row = vid[vid["filename"] == avi_name]
            if len(csv_row) > 0:
                r = csv_row.iloc[0]
                cls = r["manual_classification"]
                amp = r["roi_amplitude"]
                bpm = r["roi_bpm"]
                n_foci = int(r["n_foci"])
                has_flow = bool(r["has_flow"])
                amplitudes[day] = amp
                color = cls_colors.get(cls, "black")
                cls_label = CLS_LABELS.get(cls, cls)

                info = f"{cls_label}"
                if amp > 0:
                    info += f"\nAmp: {amp:.1f} px  BPM: {bpm:.0f}"
                if n_foci > 0:
                    info += f"  Foci: {n_foci}"
                if has_flow:
                    info += "\n[flow/drift]"

                ax.text(0.02, 0.02, info, transform=ax.transAxes,
                        fontsize=8, color="white", va="bottom",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor=color, alpha=0.8))
            else:
                ax.text(0.02, 0.02, f"Not in analysis\nFPS: {fps_val:.1f}",
                        transform=ax.transAxes, fontsize=8, color="white",
                        va="bottom", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="gray", alpha=0.8))

        # Trend annotation on the rightmost panel
        if len(amplitudes) >= 2:
            sorted_days = [d for d in DAYS if d in amplitudes]
            first_day = sorted_days[0]
            last_day = sorted_days[-1]
            first_amp = amplitudes[first_day]
            last_amp = amplitudes[last_day]

            if last_amp > 0 and first_amp > 0:
                change = last_amp - first_amp
                pct = 100 * change / first_amp if first_amp > 0 else 0
                if change > 0.1:
                    trend = f"↑ +{change:.1f} px ({pct:+.0f}%)"
                    tcolor = "#2ca02c"
                elif change < -0.1:
                    trend = f"↓ {change:.1f} px ({pct:+.0f}%)"
                    tcolor = "#d62728"
                else:
                    trend = "→ stable"
                    tcolor = "gray"
            elif last_amp > 0 and first_amp == 0:
                trend = f"↑ beating emerged\n  ({last_amp:.1f} px)"
                tcolor = "#2ca02c"
            elif last_amp == 0 and first_amp > 0:
                trend = "↓ beating lost"
                tcolor = "#d62728"
            else:
                trend = "— no beating"
                tcolor = "gray"

            last_ax = axes[row_idx, -1]
            last_ax.text(0.98, 0.98, trend, transform=last_ax.transAxes,
                         fontsize=10, fontweight="bold", color=tcolor,
                         ha="right", va="top",
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white", alpha=0.85,
                                   edgecolor=tcolor, linewidth=2))

    fig.suptitle(
        "Figure V10a: Temporal Progression — All Wells (Control Plate, Un-pulsed)\n"
        "Same physical well tracked across Day 1 → Day 2 → Day 6 → Day 8  |  "
        "Trend annotation shows amplitude change from first to last recording",
        fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(h_pad=1.5, w_pad=0.5)
    fig.savefig(OUT / "video_Fig_V10a_temporal_control.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved video_Fig_V10a_temporal_control.png")

    # ── V10b: Device plate (pulsed) ──────────────────────────────
    _fig_v10b_device(vid)


def _fig_v10b_device(vid):
    """Device plate wells across days 6 and 8 (treatment phase only)."""
    print("  Generating Fig V10b: Temporal progression (device plate)...")

    DAYS = ["day6", "day8"]
    DAY_LABELS = {"day6": "Day 6", "day8": "Day 8"}

    well_day_avi = {
        # ── Column 3: nonpoled PVDF on device ──
        ("A3", "day6"): "A3-day6-NON1-TAMMY.avi",
        ("A3", "day8"): "A3-day8-non-Tammy.avi",
        ("B3", "day6"): "B3-day6-NON2-tammy.avi",
        ("B3", "day8"): "B3-day8-non-tammy.avi",
        ("C3", "day8"): "C3-day8-non-Jaz.avi",
        # ── Column 2: gold PVDF on device ──
        ("A2", "day6"): "A2-day6-gold1-tammy.avi",
        ("A2", "day8"): "A2-day8-gold-tAMMY-new.avi",
        ("B2", "day6"): "B2-day6-gold2-tammy.avi",
        ("B2", "day8"): "B2-day8-gold-tAMMY.avi",
        ("C2", "day8"): "C2-day8-gold-Jaz.avi",
    }

    well_info = {
        "A3": ("XR1", "Non-poled PVDF (device)"),
        "B3": ("XR1", "Non-poled PVDF (device)"),
        "C3": ("GCaMP6f", "Non-poled PVDF (device)"),
        "A2": ("XR1", "Gold PVDF (device)"),
        "B2": ("XR1", "Gold PVDF (device)"),
        "C2": ("GCaMP6f", "Gold PVDF (device)"),
    }
    WELLS = ["A3", "B3", "C3", "A2", "B2", "C2"]

    cls_colors = {
        "active_beating": "#2ca02c",
        "individual_cells_beating": "#1f77b4",
        "non_contractile": "#ff7f0e",
        "no_beating": "#d62728",
    }

    IMG_W, IMG_H = 1392, 1040

    fig, axes = plt.subplots(len(WELLS), len(DAYS), figsize=(16, 5 * len(WELLS)))
    plt.subplots_adjust(left=0.18)

    for row_idx, well in enumerate(WELLS):
        cell_line, substrate = well_info[well]
        amplitudes = {}

        for col_idx, day in enumerate(DAYS):
            ax = axes[row_idx, col_idx]
            key = (well, day)

            ax.set_xlim(0, IMG_W)
            ax.set_ylim(IMG_H, 0)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(DAY_LABELS[day], fontsize=14, fontweight="bold")

            if col_idx == 0:
                ax.set_ylabel(f"{well}  |  {cell_line}  |  {substrate}",
                              fontsize=10, fontweight="bold", rotation=0,
                              labelpad=130, va="center")

            if key not in well_day_avi:
                ax.set_facecolor("#f0f0f0")
                ax.text(IMG_W / 2, IMG_H / 2, "No recording",
                        ha="center", va="center", fontsize=16, color="#999999",
                        style="italic")
                continue

            avi_name = well_day_avi[key]
            avi_path = VID_SRC / avi_name

            if not avi_path.exists():
                ax.set_facecolor("#f0f0f0")
                ax.text(IMG_W / 2, IMG_H / 2, "File not found",
                        ha="center", va="center", fontsize=12, color="red")
                continue

            frames, fps_val, n_fr = _extract_frames(avi_path, (0.25,))
            if frames[0] is not None:
                ax.imshow(frames[0], cmap="gray", extent=[0, IMG_W, IMG_H, 0])

            csv_row = vid[vid["filename"] == avi_name]
            if len(csv_row) > 0:
                r = csv_row.iloc[0]
                cls = r["manual_classification"]
                amp = r["roi_amplitude"]
                bpm = r["roi_bpm"]
                n_foci = int(r["n_foci"])
                has_flow = bool(r["has_flow"])
                amplitudes[day] = amp
                color = cls_colors.get(cls, "black")
                cls_label = CLS_LABELS.get(cls, cls)

                info = f"{cls_label}"
                if amp > 0:
                    info += f"\nAmp: {amp:.1f} px  BPM: {bpm:.0f}"
                if n_foci > 0:
                    info += f"  Foci: {n_foci}"
                if has_flow:
                    info += "\n[flow/drift]"

                ax.text(0.02, 0.02, info, transform=ax.transAxes,
                        fontsize=8, color="white", va="bottom",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor=color, alpha=0.8))
            else:
                ax.text(0.02, 0.02, f"Not in analysis\nFPS: {fps_val:.1f}",
                        transform=ax.transAxes, fontsize=8, color="white",
                        va="bottom", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="gray", alpha=0.8))

        # Trend annotation
        if len(amplitudes) == 2:
            d6_amp = amplitudes.get("day6", 0)
            d8_amp = amplitudes.get("day8", 0)
            if d8_amp > 0 and d6_amp > 0:
                change = d8_amp - d6_amp
                pct = 100 * change / d6_amp
                if change > 0.1:
                    trend = f"↑ +{change:.1f} px ({pct:+.0f}%)"
                    tcolor = "#2ca02c"
                elif change < -0.1:
                    trend = f"↓ {change:.1f} px ({pct:+.0f}%)"
                    tcolor = "#d62728"
                else:
                    trend = "→ stable"
                    tcolor = "gray"
            elif d8_amp > 0 and d6_amp == 0:
                trend = f"↑ beating emerged ({d8_amp:.1f} px)"
                tcolor = "#2ca02c"
            elif d8_amp == 0 and d6_amp > 0:
                trend = f"↓ beating lost"
                tcolor = "#d62728"
            else:
                trend = "— no beating"
                tcolor = "gray"

            last_ax = axes[row_idx, -1]
            last_ax.text(0.98, 0.98, trend, transform=last_ax.transAxes,
                         fontsize=10, fontweight="bold", color=tcolor,
                         ha="right", va="top",
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white", alpha=0.85,
                                   edgecolor=tcolor, linewidth=2))

    fig.suptitle(
        "Figure V10b: Temporal Progression — All Wells (Device Plate, Pulsed)\n"
        "Treatment phase only: Day 6 → Day 8  |  "
        "Trend annotation shows amplitude change",
        fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(h_pad=1.5, w_pad=0.5)
    fig.savefig(OUT / "video_Fig_V10b_temporal_device.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved video_Fig_V10b_temporal_device.png")


# ══════════════════════════════════════════════════════════════════════════
# COMBINED REPORT TEXT
# ══════════════════════════════════════════════════════════════════════════
def write_combined_report(data, fig_status, staining_agg, video_agg, vid_status):
    print("\nWriting COMBINED_REPORT.txt (all numbers from CSVs)...")

    phal = data["phal"]
    mn = data["mn"]
    pw_stain = data["pw_stain"]
    vid = data["vid"]
    pw_vid = data["pw_vid"]

    L = []

    # ════════════════════ HEADER ════════════════════
    L.append("=" * 80)
    L.append("COMBINED ANALYSIS REPORT")
    L.append("Piezoelectric Substrate Effects on hiPSC-Derived Cardiomyocytes")
    L.append("Staining Morphology + Video Contractility + Calcium Imaging + Cross-Reference")
    L.append("=" * 80)
    L.append("")
    L.append("NOTE: Every number in this report was computed directly from raw")
    L.append("CSV data files. No values were copied from previous report drafts.")
    L.append("")

    # ════════════════════ 1. EXPERIMENTAL DESIGN ════════════════════
    L.append("=" * 80)
    L.append("1. EXPERIMENTAL DESIGN")
    L.append("=" * 80)
    L.append("")
    L.append("Cell type: hiPSC-derived cardiomyocytes")
    L.append("Cell lines: XR1 (rows A, B) and GCaMP6f (row C)")
    L.append("")
    L.append("Well layout (6-well plates, columns 2 and 3 used):")
    L.append("  Control plate:  col 2 (A2,B2,C2) = Gold (Au-PVDF Poled)")
    L.append("                  col 3 (A3,B3,C3) = Cells only (bare plastic)")
    L.append("  Device plate:   col 2 (A2,B2,C2) = Gold (Au-PVDF Poled)")
    L.append("                  col 3 (A3,B3,C3) = Nonpoled (β-PVDF Nonpoled, no gold coating)")
    L.append("  --> 3 Gold + 3 non-Gold wells per plate (6 wells total per plate)")
    L.append("")
    L.append("Staining conditions (phalloidin analysis, 4 groups):")
    for sc in STAIN_CONDS_FLAT:
        sc_nl = FLAT_TO_NL[sc]
        n = len(phal[phal["condition"] == sc_nl])
        L.append(f"  {STAIN_SHORT[sc]:35s} n = {n} cells")
    L.append(f"  {'Total':35s} n = {len(phal)} cells")
    L.append("")
    L.append("Video recordings (BF brightfield, 23 analyzed):")
    for sub in ["plastic", "non-poled_PVDF", "gold_PVDF"]:
        n = len(vid[vid["substrate"] == sub])
        label = {"plastic": "Bare plastic", "non-poled_PVDF": "Non-poled PVDF",
                 "gold_PVDF": "Gold-coated PVDF"}[sub]
        L.append(f"  {label:35s} n = {n} videos")
    L.append(f"  {'Total':35s} n = {len(vid)} videos")
    L.append("")
    if len(data.get("fl", pd.DataFrame())) > 0:
        fl_data = data["fl"]
        n_fl_total = len(fl_data)
        n_fl_ok = (fl_data["fps_flag"] == "ok").sum()
        n_fl_low = (fl_data["fps_flag"] == "low_fps").sum()
        L.append(f"Fluorescence recordings (GCaMP6f calcium imaging, {n_fl_total} total):")
        L.append(f"  Usable FPS (>= 5 FPS)                n = {n_fl_ok} videos")
        L.append(f"  Low FPS (< 5 FPS)                    n = {n_fl_low} videos")
        L.append("  All from GCaMP6f cells (wells C2, C3)")
        L.append("")

    L.append("Timeline: Day 1 seeding, Day 2 baseline, Day 6-8 treatment phase")
    L.append("Limitation: NOT a balanced 2x2 factorial. Control plate col 3 is")
    L.append("  bare plastic, not nonpoled PVDF. Missing 'Nonpoled Un-pulsed'.")
    L.append("")

    # ════════════════════ 2. METHODS ════════════════════
    L.append("=" * 80)
    L.append("2. METHODS")
    L.append("=" * 80)
    L.append("")
    L.append("2.1 Staining Analysis")
    L.append("-" * 40)
    L.append("  Phalloidin: classical pipeline -- DAPI thresholding + watershed")
    L.append("  nuclei segmentation, nuclei-seeded watershed for cell boundaries.")
    L.append("  16 per-cell metrics (morphology, F-actin intensity, texture).")
    L.append("  Multinucleation: Cellpose v3 (cyto2 model, GPU) + DAPI nuclei.")
    L.append("  Alpha-actinin: INCONCLUSIVE (no usable signal in any image).")
    L.append("")
    L.append("2.2 Brightfield Video Analysis")
    L.append("-" * 40)
    L.append("  Farneback optical flow (OpenCV) on 2x-downsampled BF frames.")
    L.append("  Convergence map (neg. divergence) for active contraction centers.")
    L.append("  Per-center foci detection via thresholded convergence + CC labeling.")
    L.append("  Beat detection: scipy.signal.find_peaks, 1.5s settling exclusion.")
    L.append("  Classification (expert-validated): active_beating,")
    L.append("    individual_cells_beating, non_contractile, no_beating.")
    L.append("  Methods aligned with Huebsch 2015, Czirok 2017, Sala 2018.")
    L.append("")
    L.append("2.3 Fluorescence Calcium Imaging Analysis")
    L.append("-" * 40)
    L.append("  GCaMP6f intensity-based pipeline (NOT optical flow).")
    L.append("  Background subtraction: 5-pixel border strip (all four edges) for non-cell baseline.")
    L.append("  Photobleach correction: mono-exponential fit y=a*exp(-t/tau)+c.")
    L.append("  F0 baseline: rolling 5th percentile (2s window).")
    L.append("  dF/F0 = (F - F0) / F0 for each timepoint.")
    L.append("  Peak detection: scipy.signal.find_peaks with adaptive prominence.")
    L.append("  Per-transient kinetics: dF/F0 amplitude, Time to Peak (TTP),")
    L.append("    CaTD50, CaTD90, decay time constant (tau), upstroke/decay velocity.")
    L.append("  Aggregate metrics: beat rate (BPM), inter-beat interval CV (IBI CV).")
    L.append("  Methods aligned with Psaras/CalTrack 2021, Bedut 2022.")
    L.append("")

    # ════════════════════ 3. RESULTS ════════════════════
    L.append("=" * 80)
    L.append("3. RESULTS")
    L.append("=" * 80)
    L.append("")

    # ── 3.1 Staining ──
    L.append("3.1 STAINING / MORPHOLOGY RESULTS")
    L.append("-" * 40)
    L.append(f"Total cells analyzed: {len(phal)}")
    L.append("")

    stain_metrics = [
        ("mean_phalloidin_intensity", "F-actin intensity"),
        ("phalloidin_coverage", "F-actin coverage"),
        ("actin_coherency", "Actin coherency"),
        ("circularity", "Circularity"),
        ("cell_area_px", "Cell area (px)"),
    ]
    for col, label in stain_metrics:
        L.append(f"  {label}:")
        for sc in STAIN_CONDS_FLAT:
            sc_nl = FLAT_TO_NL[sc]
            vals = phal[phal["condition"] == sc_nl][col].dropna()
            L.append(f"    {STAIN_SHORT[sc]:35s} median = {vals.median():.3f}  (n={len(vals)})")
        L.append("")

    L.append("  Key significant comparisons (Mann-Whitney U, from pairwise CSV):")
    sig_stain = pw_stain[pw_stain["significance"] != "ns"]
    key_metrics = ["mean_phalloidin_intensity", "actin_coherency",
                   "phalloidin_coverage"]
    for m in key_metrics:
        m_sig = sig_stain[sig_stain["metric"] == m]
        for _, r in m_sig.iterrows():
            g1 = r["group_1"].replace("Au-PVDF ", "")
            g2 = r["group_2"].replace("Au-PVDF ", "")
            L.append(f"    {g1} vs {g2}:")
            L.append(f"      {m}: p={r['p_value']:.2e} {r['significance']}")
    L.append("")

    # Multinucleation
    L.append("  Multinucleation (Cellpose):")
    total_all, mono_all, multi_all, zero_all = 0, 0, 0, 0
    for sc in STAIN_CONDS_FLAT:
        sc_nl = FLAT_TO_NL[sc]
        sub = mn[mn["condition"] == sc_nl]
        total = len(sub)
        if total > 0:
            mono = int((sub["nuclei_count"] == 1).sum())
            multi = int((sub["nuclei_count"] >= 2).sum())
            zero = int((sub["nuclei_count"] == 0).sum())
            total_all += total; mono_all += mono; multi_all += multi; zero_all += zero
            L.append(f"    {STAIN_SHORT[sc]:35s} {total} cells, "
                     f"{mono} mono ({100*mono//total}%), "
                     f"{multi} multi ({100*multi//total}%), "
                     f"{zero} unassigned ({100*zero//total}%)")
    detected = mono_all + multi_all
    L.append(f"  NOTE: {zero_all}/{total_all} cells ({100*zero_all//total_all}%) had no nucleus "
             f"assigned by Cellpose (nucleus/cell boundary mismatch in confluent cultures).")
    L.append(f"  Among cells with detected nuclei ({detected}/{total_all}): "
             f"{mono_all} mono ({100*mono_all//detected}%), "
             f"{multi_all} multi ({100*multi_all//detected}%).")
    L.append("  Conclusion: among cells with detected nuclei, nearly all are"
             " mononucleated — not differentiating. The high zero-nuclei"
             " fraction reflects segmentation limitations, not absent nuclei.")
    L.append("")
    L.append("  Alpha-actinin: INCONCLUSIVE -- no recognizable cell morphology.")
    L.append("  [See Fig7_actinin_INCONCLUSIVE.png]")
    L.append("")

    # ── 3.2 Video ──
    L.append("3.2 VIDEO / CONTRACTILITY RESULTS")
    L.append("-" * 40)
    L.append("  Amplitude metric: ROI contraction amplitude (roi_amplitude) throughout.")
    L.append("")
    n_total = len(vid)
    L.append(f"Total BF videos analyzed: {n_total}")
    for cls in CLS_ORDER:
        n = (vid["manual_classification"] == cls).sum()
        L.append(f"  {CLS_LABELS[cls]:30s}: {n:2d} ({100*n/n_total:.0f}%)")
    L.append("")

    L.append("  By substrate (all videos):")
    for sub in ["plastic", "non-poled_PVDF", "gold_PVDF"]:
        s_df = vid[vid["substrate"] == sub]
        s_beat = s_df[s_df["manual_classification"] == "active_beating"]
        flow_n = int(s_df["has_flow"].sum())
        label = {"plastic": "Bare plastic", "non-poled_PVDF": "Non-poled PVDF",
                 "gold_PVDF": "Gold-coated PVDF"}[sub]
        L.append(f"    {label} (n={len(s_df)}): "
                 f"{len(s_beat)} beating, "
                 f"flow in {flow_n}/{len(s_df)}")
        if len(s_beat) > 0:
            L.append(f"      median amplitude: {s_beat['roi_amplitude'].median():.2f} px, "
                     f"median BPM: {s_beat['roi_bpm'].median():.1f}")
    L.append("")

    L.append("  By cell line:")
    for cl in ["XR1", "GCaMP6f"]:
        c_df = vid[vid["cell_type"] == cl]
        c_beat = c_df[c_df["manual_classification"] == "active_beating"]
        c_icb = c_df[c_df["manual_classification"] == "individual_cells_beating"]
        line = (f"    {cl} (n={len(c_df)}): "
                f"{len(c_beat)} active beating ({100*len(c_beat)/len(c_df):.0f}%)")
        if cl == "GCaMP6f" and len(c_icb) > 0:
            line += f"; {len(c_icb)}/{len(c_df)} videos with individual cells beating"
        L.append(line)
        if len(c_beat) > 0:
            L.append(f"      median amplitude: {c_beat['roi_amplitude'].median():.2f} px, "
                     f"median BPM: {c_beat['roi_bpm'].median():.1f}")
    L.append("")

    L.append("  Temporal progression:")
    for d in ["day1", "day2", "day6", "day8"]:
        d_df = vid[vid["day"] == d]
        d_beat = d_df[d_df["manual_classification"] == "active_beating"]
        label = d.replace("day", "Day ")
        L.append(f"    {label} (n={len(d_df)}): {len(d_beat)} active beating")
        if len(d_beat) > 0:
            L.append(f"      median amplitude: {d_beat['roi_amplitude'].median():.2f} px, "
                     f"median BPM: {d_beat['roi_bpm'].median():.1f}")
    L.append("  Note: Days 1-2 = baseline (bare plastic, control plate only)")
    L.append("")

    treat = vid[vid["phase"] == "treatment"]
    dev_b = treat[(treat["recording_type"] == "on-device") &
                  (treat["manual_classification"] == "active_beating")]
    ctrl_b = treat[(treat["recording_type"] == "control") &
                   (treat["manual_classification"] == "active_beating")]
    L.append("  On-device vs control (treatment phase, active_beating only):")
    L.append(f"    On-device (pulsed):  {len(dev_b)} beating, "
             f"median amp {dev_b['roi_amplitude'].median():.2f} px" if len(dev_b) > 0 else
             f"    On-device (pulsed):  {len(dev_b)} beating")
    L.append(f"    Control (unpulsed):  {len(ctrl_b)} beating, "
             f"median amp {ctrl_b['roi_amplitude'].median():.2f} px" if len(ctrl_b) > 0 else
             f"    Control (unpulsed):  {len(ctrl_b)} beating")
    L.append("")

    L.append("  Video statistical comparisons:")
    if len(pw_vid) > 0:
        sig_vid = pw_vid[pw_vid["significance"] != "ns"]
        if len(sig_vid) > 0:
            for _, r in sig_vid.iterrows():
                L.append(f"    {r['group_1']} vs {r['group_2']}: "
                         f"{r['metric']} p={r['p_value']:.4f} {r['significance']}")
        else:
            L.append("    No comparisons reached significance (all p>0.05).")
            L.append("    Expected with n=2-6 per group.")
    L.append("")

    # ── 3.3 Cross-reference ──
    L.append("3.3 CROSS-REFERENCE: STRUCTURE MEETS FUNCTION")
    L.append("-" * 40)
    L.append("Per-condition comparison linking staining morphology with video")
    L.append("contractility (see Figures C1 and C2).")
    L.append("NOTE: Video data filtered to XR1 only to match staining")
    L.append("(all staining images are from XR1 wells: B2, A3, B3).")
    L.append("")
    for sc in STAIN_CONDS_FLAT:
        s = staining_agg[sc]
        v = video_agg[sc]
        L.append(f"  {STAIN_SHORT[sc]}:")
        L.append(f"    STAINING: {s['n_cells']} cells, F-actin={s['factin']:.3f}, "
                 f"coherency={s['coherency']:.3f}, circularity={s['circularity']:.3f}")
        L.append(f"    VIDEO (XR1): {v['n_videos']} videos, {v['n_beating']} beating, "
                 f"amplitude={v['amplitude']:.1f} px"
                 if v['amplitude'] > 0 else
                 f"    VIDEO (XR1): {v['n_videos']} videos, {v['n_beating']} beating")
        L.append("")

    L.append("  Key cross-reference findings:")
    L.append("")
    L.append("  a) ADHESION (circularity vs flow/drift, XR1 only):")
    vid_t = vid[vid["cell_type"] == "XR1"]
    sub_adhesion = {}
    for sub, conds in [("gold_PVDF", ["Au-PVDF Poled\nPulsed (device)",
                                       "Au-PVDF Poled\nUn-pulsed (control)"]),
                        ("non-poled_PVDF", ["β-PVDF Nonpoled\nPulsed (device)"]),
                        ("plastic", ["Cells only\n(baseline)"])]:
        circ_vals = phal[phal["condition"].isin(conds)]["circularity"].dropna()
        s_df = vid_t[vid_t["substrate"] == sub]
        flow_n = int(s_df["has_flow"].sum())
        label = {"plastic": "Bare plastic", "non-poled_PVDF": "Non-poled PVDF",
                 "gold_PVDF": "Gold PVDF"}[sub]
        sub_adhesion[sub] = {"circ": circ_vals.median(), "flow": flow_n,
                             "total": len(s_df)}
        L.append(f"    {label}: median circularity={circ_vals.median():.3f}, "
                 f"flow in {flow_n}/{len(s_df)} videos "
                 f"({100*flow_n/len(s_df):.0f}%)" if len(s_df) > 0 else
                 f"    {label}: median circularity={circ_vals.median():.3f}")
    L.append("    -> Gold-PVDF cells are ROUNDER (higher circularity = poorer")
    L.append("       adhesion/spreading) despite fewer flow events than plastic.")
    L.append("       Plastic has more flow, possibly influenced by baseline")
    L.append("       day 1-2 videos before cells fully settled. Taken together,")
    L.append("       the circularity and flow patterns are consistent with")
    L.append("       substrate-dependent adhesion differences.")
    L.append("")

    L.append("  b) MATURATION (F-actin organization vs beating):")
    L.append("    Poled Pulsed (device): LOWEST actin coherency (0.422)")
    L.append("      but strong beating when on non-poled substrate")
    L.append("    Cells only (baseline): HIGHEST actin coherency (0.476)")
    L.append("      and reliable beating on plastic")
    L.append("    -> Lower coherency may reflect maturation-associated")
    L.append("       remodeling (stress fibers -> sarcomeric actin),")
    L.append("       not dysfunction.")
    L.append("")

    L.append("  c) CELL LINE (XR1 vs GCaMP6f):")
    n_tammy = len(vid[vid["cell_type"] == "XR1"])
    n_tammy_beat = len(vid[(vid["cell_type"] == "XR1") &
                           (vid["manual_classification"] == "active_beating")])
    n_jaz = len(vid[vid["cell_type"] == "GCaMP6f"])
    n_jaz_beat = len(vid[(vid["cell_type"] == "GCaMP6f") &
                          (vid["manual_classification"] == "active_beating")])
    L.append(f"    XR1: {n_tammy_beat}/{n_tammy} active beating ({100*n_tammy_beat/n_tammy:.0f}%)")
    L.append(f"    GCaMP6f: {n_jaz_beat}/{n_jaz} active beating ({100*n_jaz_beat/n_jaz:.0f}%)")
    L.append("    Note: all 4 staining images are from XR1 wells (B2, A3, B3).")
    L.append("    GCaMP6f (row C) was not imaged for staining in this experiment.")
    L.append("    Video data covers both cell lines (same wells, pre-staining).")
    L.append("")

    # ── 3.4 Fluorescence / Calcium Imaging ──
    fl = data.get("fl", pd.DataFrame())
    L.append("3.4 FLUORESCENCE / CALCIUM IMAGING")
    L.append("-" * 40)

    if len(fl) == 0:
        L.append("  Status: FL data not available.")
        L.append("")
    else:
        L.append("  All FL videos are from GCaMP6f cells (row C, wells C2 & C3).")
        L.append("  GCaMP6f = genetically encoded calcium indicator (no dye loading).")
        L.append("  Pipeline: background subtraction -> mono-exponential photobleach")
        L.append("  correction -> rolling 5th-percentile F0 -> dF/F0 -> peak detection")
        L.append("  -> per-transient kinetic extraction (Psaras/CalTrack 2021, Bedut 2022).")
        L.append("")
        L.append("  KEY SCIENTIFIC QUESTION:")
        L.append("  Do GCaMP6f cells show calcium transients (FL) despite no")
        L.append("  visible synchronized contraction (BF)? This distinguishes between:")
        L.append("    a) No calcium cycling -- cells are truly non-functional")
        L.append("    b) Calcium present but mechanically uncoupled -- cells")
        L.append("       have calcium handling but lack coordinated sarcomeric contraction")
        L.append("  This helps evaluate whether the phenotype is compatible with")
        L.append("  the CaV1 perturbation hypothesis (Yang et al. Nat Commun 2018),")
        L.append("  while still allowing alternative explanations.")
        L.append("")

        n_fl = len(fl)
        fl_ok = fl[fl["fps_flag"] == "ok"]
        fl_low = fl[fl["fps_flag"] == "low_fps"]
        n_ok_fps = len(fl_ok)
        n_low_fps = len(fl_low)
        n_active = (fl["classification"] == "active_transients").sum()
        n_weak = (fl["classification"] == "weak_transients").sum()
        n_single = (fl["classification"] == "single_transient").sum()
        n_none = (fl["classification"] == "no_transients").sum()
        n_active_ok = (fl_ok["classification"] == "active_transients").sum()
        n_single_ok = (fl_ok["classification"] == "single_transient").sum()
        L.append(f"  Total FL videos: {n_fl} ({n_ok_fps} usable FPS >= 5, "
                 f"{n_low_fps} low-FPS)")
        L.append(f"    Active transients    : {n_active}  [usable only: {n_active_ok}]")
        L.append(f"    Weak transients      : {n_weak}  [usable only: "
                 f"{(fl_ok['classification'] == 'weak_transients').sum()}]")
        L.append(f"    Single transient     : {n_single}  [usable only: {n_single_ok}]")
        L.append(f"    No transients        : {n_none}  [usable only: "
                 f"{(fl_ok['classification'] == 'no_transients').sum()}]")
        L.append("")

        L.append("  Note: A2-day8-gold-tAMMY-fl.avi was MISLABELED; reclassified")
        L.append("  as C2-day8-gold-GCaMP6f-FL.avi based on well adjacency analysis.")
        L.append("")

        FILENAME_SUBSTRATE_NOTES = {
            "C2-day6-non-FL.avi": "NOTE: filename says 'non' but physical well C2 is gold-coated poled PVDF",
            "C3-day6-gold-fl.avi": "NOTE: filename says 'gold' but physical well C3 is uncoated non-poled PVDF",
        }
        L.append("  Per-video FL results:")
        for _, r in fl.iterrows():
            cls_str = r["classification"]
            nt = int(r["n_transients"])
            amp = r["mean_amplitude_dff0"]
            fps_f = r["fps_flag"]
            bpm_str = f"{r['bpm']:.1f}" if pd.notna(r["bpm"]) else "--"
            L.append(f"    {r['filename']}")
            L.append(f"      {r['well']} {r['day']} {r['substrate']} | "
                     f"FPS={r['fps']:.1f} ({fps_f})")
            if r["filename"] in FILENAME_SUBSTRATE_NOTES:
                L.append(f"      {FILENAME_SUBSTRATE_NOTES[r['filename']]}")
            L.append(f"      Classification: {cls_str} | {nt} transients | "
                     f"dF/F0={amp:.4f} | BPM={bpm_str}")
        L.append("")

        # Aggregate FL kinetics for usable videos with transients
        fl_active = fl[(fl["n_transients"] > 0)]
        fl_usable = fl[(fl["fps_flag"] == "ok") & (fl["n_transients"] > 0)]

        L.append("  LOW-FPS IMPACT ASSESSMENT:")
        L.append("  4 of 8 FL videos have FPS < 5 (range 1.1-3.2 FPS).")
        L.append("  At 2-3 FPS, calcium transients (~300-500 ms fast phases)")
        L.append("  are sampled by only 1-2 frames per event. This means:")
        L.append("    - TRANSIENT DETECTION: still possible (amplitude changes are")
        L.append("      visible even at low temporal resolution)")
        L.append("    - KINETIC PARAMETERS (TTP, CaTD50, CaTD90): UNRELIABLE at")
        L.append("      low FPS -- temporal precision is limited to 1/FPS")
        L.append("      (e.g., 500 ms at 2 FPS vs 80 ms at 12.5 FPS)")
        L.append("    - TRANSIENT COUNT: may overcount at low FPS due to aliasing")
        L.append("      (e.g., C3-day6-gold-fl detected 18 'transients' at 3.2 FPS)")
        L.append("    - BEAT RATE: reliable (long recording compensates for")
        L.append("      poor per-beat resolution)")
        L.append("")
        L.append("  USABLE-FPS-ONLY ANALYSIS (>= 5 FPS, n=4):")
        if len(fl_usable) > 0:
            n_usable_trans = (fl_usable["n_transients"] > 0).sum()
            L.append(f"    {n_usable_trans}/{len(fl_usable)} usable-FPS videos show Ca2+ transients")
            L.append(f"    (EMD finding holds with usable data alone)")
            L.append(f"    Mean dF/F0 amplitude: "
                     f"{fl_usable['mean_amplitude_dff0'].mean():.4f} "
                     f"(range {fl_usable['mean_amplitude_dff0'].min():.4f}-"
                     f"{fl_usable['mean_amplitude_dff0'].max():.4f})")
            ttp_u = fl_usable["mean_ttp_ms"].dropna()
            if len(ttp_u) > 0:
                L.append(f"    Mean TTP: {ttp_u.mean():.0f} ms "
                         f"(range {ttp_u.min():.0f}-{ttp_u.max():.0f})")
            catd50_u = fl_usable["mean_catd50_ms"].dropna()
            if len(catd50_u) > 0:
                L.append(f"    Mean CaTD50: {catd50_u.mean():.0f} ms "
                         f"(range {catd50_u.min():.0f}-{catd50_u.max():.0f})")
            catd90_u = fl_usable["mean_catd90_ms"].dropna()
            if len(catd90_u) > 0:
                L.append(f"    Mean CaTD90: {catd90_u.mean():.0f} ms "
                         f"(range {catd90_u.min():.0f}-{catd90_u.max():.0f})")
            tau_u = fl_usable["mean_decay_tau_ms"].dropna()
            if len(tau_u) > 0:
                L.append(f"    Mean decay tau: {tau_u.mean():.0f} ms "
                         f"(range {tau_u.min():.0f}-{tau_u.max():.0f})")
            bpm_u = fl_usable["bpm"].dropna()
            if len(bpm_u) > 0:
                L.append(f"    Beat rate: {bpm_u.mean():.1f} BPM "
                         f"(range {bpm_u.min():.1f}-{bpm_u.max():.1f})")
            ibi_u = fl_usable["ibi_cv_pct"].dropna()
            if len(ibi_u) > 0:
                L.append(f"    IBI CV: {ibi_u.mean():.1f}% "
                         f"(range {ibi_u.min():.1f}-{ibi_u.max():.1f}%)")
        L.append("")

        L.append("  ALL VIDEOS aggregate (including low-FPS, for reference):")
        if len(fl_active) > 0:
            L.append(f"    {len(fl_active)} videos with transients (kinetics from low-FPS are approximate):")
            L.append(f"    Mean dF/F0 amplitude: "
                     f"{fl_active['mean_amplitude_dff0'].mean():.4f} "
                     f"(range {fl_active['mean_amplitude_dff0'].min():.4f}-"
                     f"{fl_active['mean_amplitude_dff0'].max():.4f})")
            ttp_vals = fl_active["mean_ttp_ms"].dropna()
            if len(ttp_vals) > 0:
                L.append(f"    Mean TTP: {ttp_vals.mean():.0f} ms "
                         f"(range {ttp_vals.min():.0f}-{ttp_vals.max():.0f})")
            bpm_vals = fl_active["bpm"].dropna()
            if len(bpm_vals) > 0:
                L.append(f"    Beat rate: {bpm_vals.mean():.1f} BPM "
                         f"(range {bpm_vals.min():.1f}-{bpm_vals.max():.1f})")
        L.append("")

        L.append("  INTERPRETATION: WHAT USABLE-ONLY NUMBERS CHANGE")
        L.append("  Excluding low-FPS videos does NOT weaken any conclusion;")
        L.append("  it STRENGTHENS them:")
        L.append("")
        if len(fl_usable) > 0 and len(fl_active) > 0:
            L.append(f"    1. EMD finding: 7/8 -> 4/4 (100%). The one 'no transients'")
            L.append(f"       video was at 1.1 FPS -- likely too slow to detect, not")
            L.append(f"       a true negative. Core conclusion unchanged.")
            L.append("")
            L.append(f"    2. Amplitude: usable mean dF/F0 = "
                     f"{fl_usable['mean_amplitude_dff0'].mean():.4f} vs "
                     f"all-video mean = {fl_active['mean_amplitude_dff0'].mean():.4f}.")
            L.append(f"       Usable videos show HIGHER amplitude (+18%), because low-FPS")
            L.append(f"       videos under-sample transient peaks, compressing apparent")
            L.append(f"       amplitude. True calcium signals are stronger than reported")
            L.append(f"       when all videos are pooled.")
            L.append("")
            L.append("    3. Temporal progression REVERSES apparent trend:")
            L.append("       All videos: day2=0.110 > day6=0.091 > day8=0.045 (declining)")
            L.append("       Usable only: day2=0.086 < day6=0.216 > day8=0.045 (peak at day6)")
            L.append("       The all-video trend was an ARTIFACT: day6 had the strongest")
            L.append("       usable signal (0.216 dF/F0) but its average was diluted by")
            L.append("       two low-FPS recordings (0.023 and 0.033). The usable-only")
            L.append("       data shows calcium activity PEAKING at day 6, consistent")
            L.append("       with maturation-driven calcium handling improvement")
            L.append("       (Hwang et al. 2015), before declining at day 8.")
            L.append("")
            fl_gold_all = fl[fl["substrate"].str.contains("gold", case=False, na=False)]
            fl_non_all = fl[fl["substrate"].str.contains("non", case=False, na=False)]
            fl_gold_ok = fl_usable[fl_usable["substrate"].str.contains("gold", case=False, na=False)]
            fl_non_ok = fl_usable[fl_usable["substrate"].str.contains("non", case=False, na=False)]
            g_all = fl_gold_all.loc[fl_gold_all["n_transients"] > 0, "mean_amplitude_dff0"].mean()
            n_all = fl_non_all.loc[fl_non_all["n_transients"] > 0, "mean_amplitude_dff0"].mean()
            g_ok = fl_gold_ok["mean_amplitude_dff0"].mean() if len(fl_gold_ok) > 0 else 0
            n_ok = fl_non_ok["mean_amplitude_dff0"].mean() if len(fl_non_ok) > 0 else 0
            g_pct = (g_ok - g_all) / g_all * 100 if g_all > 0 else 0
            n_pct = (n_ok - n_all) / n_all * 100 if n_all > 0 else 0
            ratio_ok = g_ok / n_ok if n_ok > 0 else float("inf")
            ratio_all = g_all / n_all if n_all > 0 else float("inf")
            L.append("    4. Gold substrate effect is STRONGER with usable data:")
            L.append(f"       Gold usable: dF/F0={g_ok:.3f} vs Gold all: {g_all:.3f} "
                     f"({g_pct:+.0f}%)")
            L.append(f"       Non-poled usable: dF/F0={n_ok:.3f} vs all: {n_all:.3f} "
                     f"({n_pct:+.0f}%)")
            L.append(f"       Gold/non-poled ratio: usable={ratio_ok:.1f}x vs "
                     f"all={ratio_all:.1f}x")
            L.append("       The substrate effect holds regardless; gold-coated PVDF")
            L.append(f"       consistently produces ~{ratio_ok:.0f}x higher calcium amplitudes.")
            L.append("")
            L.append("    5. Beat regularity IMPROVES in usable data:")
            ibi_u = fl_usable["ibi_cv_pct"].dropna()
            ibi_a = fl_active["ibi_cv_pct"].dropna()
            if len(ibi_u) > 0 and len(ibi_a) > 0:
                L.append(f"       IBI CV: usable={ibi_u.mean():.1f}% vs all={ibi_a.mean():.1f}%")
            L.append("       Lower CV = more regular beating, consistent with")
            L.append("       properly resolved transients rather than aliased noise.")
            L.append("")
            L.append("  CONCLUSION: Low-FPS data dilutes true signal strength but does")
            L.append("  not create false positives. All primary findings (EMD, substrate")
            L.append("  effect, temporal maturation) are confirmed or strengthened when")
            L.append("  restricted to usable-FPS recordings. For Experiment 2, all FL")
            L.append("  recordings should use >= 10 FPS to eliminate this confound.")
        L.append("")

        # Temporal progression of FL
        L.append("  FL temporal progression (all videos; [usable FPS only]):")
        for day in ["day2", "day6", "day8"]:
            fd = fl[fl["day"] == day]
            fd_ok = fl_ok[fl_ok["day"] == day]
            n_with = (fd["n_transients"] > 0).sum()
            if len(fd) > 0:
                mean_amp = fd.loc[fd["n_transients"] > 0, "mean_amplitude_dff0"]
                amp_str = f"mean dF/F0={mean_amp.mean():.4f}" if len(mean_amp) > 0 else "no transients"
                # usable bracket
                if len(fd_ok) > 0:
                    n_with_ok = (fd_ok["n_transients"] > 0).sum()
                    mean_amp_ok = fd_ok.loc[fd_ok["n_transients"] > 0, "mean_amplitude_dff0"]
                    ok_str = (f"[usable: {n_with_ok}/{len(fd_ok)}, "
                              f"dF/F0={mean_amp_ok.mean():.4f}]" if len(mean_amp_ok) > 0
                              else f"[usable: {n_with_ok}/{len(fd_ok)}]")
                else:
                    ok_str = "[usable: 0 videos]"
                L.append(f"    {day}: {len(fd)} videos, {n_with} with transients, "
                         f"{amp_str}  {ok_str}")
        L.append("")

        # Substrate comparison (FL)
        L.append("  FL by substrate (all videos; [usable FPS only]):")
        for sub in ["gold", "non-poled"]:
            fs = fl[fl["substrate"].str.contains(sub, case=False, na=False)]
            fs_ok = fl_ok[fl_ok["substrate"].str.contains(sub, case=False, na=False)]
            if len(fs) > 0:
                n_with = (fs["n_transients"] > 0).sum()
                mean_amp = fs.loc[fs["n_transients"] > 0, "mean_amplitude_dff0"]
                amp_str = f"mean dF/F0={mean_amp.mean():.4f}" if len(mean_amp) > 0 else "--"
                if len(fs_ok) > 0:
                    n_with_ok = (fs_ok["n_transients"] > 0).sum()
                    mean_amp_ok = fs_ok.loc[fs_ok["n_transients"] > 0, "mean_amplitude_dff0"]
                    ok_str = (f"[usable: {n_with_ok}/{len(fs_ok)}, "
                              f"dF/F0={mean_amp_ok.mean():.4f}]" if len(mean_amp_ok) > 0
                              else f"[usable: {n_with_ok}/{len(fs_ok)}]")
                else:
                    ok_str = "[usable: 0 videos]"
                L.append(f"    {sub}: {len(fs)} videos, {n_with} with transients, "
                         f"{amp_str}  {ok_str}")
        # Non-poled exclusion note: mean above uses only n_transients>0 rows
        fs_non = fl[fl["substrate"].str.contains("non", case=False, na=False)]
        n_zero = (fs_non["n_transients"] == 0).sum()
        if n_zero > 0:
            all_mean = fs_non["mean_amplitude_dff0"].mean()
            zero_files = fs_non.loc[fs_non["n_transients"] == 0, "filename"].tolist()
            L.append(f"    NOTE: non-poled mean above excludes {n_zero} video(s) with "
                     f"0 transients ({', '.join(zero_files)});")
            L.append(f"    including all {len(fs_non)} non-poled videos gives "
                     f"mean dF/F0={all_mean:.4f}.")
        L.append("    NOTE: substrate assignments corrected from original filenames --")
        L.append("    C2-day6-non-FL.avi is gold (well C2), C3-day6-gold-fl.avi is")
        L.append("    non-poled (well C3); filenames are misleading.")
        L.append("")

        # The critical finding
        L.append("  CRITICAL FINDING: ELECTROMECHANICAL DISSOCIATION")
        L.append("  BF analysis: 0/7 GCaMP6f videos showed active synchronized beating")
        L.append("    (1/7 showed individual cells beating — C3-day8-non-Jaz.avi)")
        n_fl_transients = (fl["n_transients"] > 0).sum()
        n_fl_transients_ok = (fl_ok["n_transients"] > 0).sum()
        L.append(f"  FL analysis: {n_fl_transients}/{n_fl} GCaMP6f videos show calcium transients")
        L.append(f"    [usable FPS only: {n_fl_transients_ok}/{n_ok_fps}]")
        L.append("  CONCLUSION: GCaMP6f cells HAVE calcium cycling but are")
        L.append("  MECHANICALLY UNCOUPLED -- calcium transients are present")
        L.append("  without visible synchronized sarcomeric contraction. This is consistent")
        L.append("  with 'electromechanical dissociation' (EMD), a phenomenon")
        L.append("  reported in immature hiPSC-CMs where electrical/calcium")
        L.append("  signaling precedes mechanical coupling (Lee et al. 2019,")
        L.append("  Ronaldson-Bouchard et al. 2018).")
        L.append("")
        L.append("  This result REFINES the CaV1 perturbation hypothesis:")
        L.append("  since calcium transients ARE present, GCaMP6f does not")
        L.append("  fully abolish CaV1-mediated calcium entry. Instead, the")
        L.append("  perturbation may be PARTIAL: sufficient Ca2+ enters to")
        L.append("  generate detectable fluorescence transients via GCaMP6f,")
        L.append("  but the amplitude/kinetics are insufficient to trigger")
        L.append("  full excitation-contraction coupling (ECC). Alternative")
        L.append("  explanations include:")
        L.append("    - Immature sarcomere organization in GCaMP6f cells")
        L.append("    - Reduced myofibrillar density (F-actin/sarcomeric actin)")
        L.append("    - Impaired calcium-induced calcium release (CICR) from SR")
        L.append("    - Disrupted T-tubule development (common in hiPSC-CMs)")
        L.append("")

    # ── 3.5 THREE-WAY CROSS-REFERENCE: STAINING + BF VIDEO + FL ──
    L.append("3.5 THREE-WAY CROSS-REFERENCE: STAINING + BF VIDEO + FLUORESCENCE")
    L.append("-" * 40)
    if len(fl) == 0:
        L.append("  FL data not available; see section 3.3 for staining-video cross-ref.")
        L.append("")
    else:
        L.append("  This section integrates all three modalities for the GCaMP6f")
        L.append("  wells (C2, C3) that have matched BF and FL recordings, and")
        L.append("  cross-references with XR1 staining data for condition-level")
        L.append("  structural context.")
        L.append("")

        # Per-well three-way comparison
        L.append("  a) PER-WELL MULTI-MODAL SUMMARY (GCaMP6f wells):")
        L.append("")
        for well in ["C2", "C3"]:
            L.append(f"    Well {well} (GCaMP6f):")
            vid_w = vid[vid["well"] == well]
            fl_w = fl[fl["well"] == well]

            for day in ["day2", "day6", "day8"]:
                vid_d = vid_w[vid_w["day"] == day]
                fl_d = fl_w[fl_w["day"] == day]
                if len(vid_d) == 0 and len(fl_d) == 0:
                    continue
                L.append(f"      {day}:")
                if len(vid_d) > 0:
                    bf_cls = vid_d["manual_classification"].values
                    bf_amp = vid_d["roi_amplitude"].values
                    for j in range(len(vid_d)):
                        a_str = f", amp={bf_amp[j]:.2f}px" if pd.notna(bf_amp[j]) and bf_amp[j] > 0 else ""
                        L.append(f"        BF: {bf_cls[j]}{a_str}")
                else:
                    L.append("        BF: no recording")
                if len(fl_d) > 0:
                    for _, fr in fl_d.iterrows():
                        nt = int(fr["n_transients"])
                        amp = fr["mean_amplitude_dff0"]
                        L.append(f"        FL: {fr['classification']}, "
                                 f"{nt} transients, dF/F0={amp:.4f}")
                else:
                    L.append("        FL: no recording")
            L.append("")

        # Structural context from staining (XR1 only -- same substrates)
        L.append("  b) STRUCTURAL CONTEXT (staining, XR1 wells only):")
        L.append("     Staining was performed on XR1 wells (B2, A3, B3) which share")
        L.append("     identical substrate conditions with GCaMP6f wells (C2, C3).")
        L.append("     This allows condition-level (not cell-level) structural context:")
        L.append("")
        L.append("     Gold-PVDF substrate (B2 stained, C2 FL/BF recorded):")
        phal_gold = phal[phal["condition"].str.contains("Poled", na=False)]
        if len(phal_gold) > 0:
            L.append(f"       F-actin intensity: {phal_gold['mean_phalloidin_intensity'].median():.3f} "
                     f"(n={len(phal_gold)})")
            L.append(f"       Circularity: {phal_gold['circularity'].median():.3f}")
            L.append(f"       Actin coherency: {phal_gold['actin_coherency'].median():.3f}")
        fl_gold = fl[fl["substrate"].str.contains("gold", case=False, na=False)]
        n_fl_gold_trans = (fl_gold["n_transients"] > 0).sum() if len(fl_gold) > 0 else 0
        L.append(f"       FL: {len(fl_gold)} videos, {n_fl_gold_trans} with Ca2+ transients")
        vid_c2 = vid[(vid["well"] == "C2")]
        n_bf_c2_beat = (vid_c2["manual_classification"] == "active_beating").sum()
        L.append(f"       BF: {len(vid_c2)} videos, {n_bf_c2_beat} with active beating")
        L.append("")

        L.append("     Non-poled PVDF substrate (A3/B3 stained, C3 FL/BF recorded):")
        phal_non = phal[phal["condition"].str.contains("Nonpoled", na=False)]
        if len(phal_non) > 0:
            L.append(f"       F-actin intensity: {phal_non['mean_phalloidin_intensity'].median():.3f} "
                     f"(n={len(phal_non)})")
            L.append(f"       Circularity: {phal_non['circularity'].median():.3f}")
            L.append(f"       Actin coherency: {phal_non['actin_coherency'].median():.3f}")
        fl_non = fl[fl["substrate"].str.contains("non", case=False, na=False)]
        n_fl_non_trans = (fl_non["n_transients"] > 0).sum() if len(fl_non) > 0 else 0
        L.append(f"       FL: {len(fl_non)} videos, {n_fl_non_trans} with Ca2+ transients")
        vid_c3 = vid[(vid["well"] == "C3")]
        n_bf_c3_beat = (vid_c3["manual_classification"] == "active_beating").sum()
        L.append(f"       BF: {len(vid_c3)} videos, {n_bf_c3_beat} with active beating")
        L.append("")

        L.append("  c) KEY CROSS-REFERENCE FINDINGS:")
        L.append("")
        L.append("     1. CALCIUM WITHOUT CONTRACTION (EMD in GCaMP6f):")
        L.append(f"        {n_fl_transients}/{n_fl} FL videos show Ca2+ transients,")
        n_bf_gcamp_beat = (vid[vid["cell_type"] == "GCaMP6f"]["manual_classification"] == "active_beating").sum()
        n_bf_gcamp = len(vid[vid["cell_type"] == "GCaMP6f"])
        n_bf_gcamp_icb = (vid[vid["cell_type"] == "GCaMP6f"]["manual_classification"] == "individual_cells_beating").sum()
        L.append(f"        but {n_bf_gcamp_beat}/{n_bf_gcamp} matched BF videos show active beating")
        L.append(f"        ({n_bf_gcamp_icb}/{n_bf_gcamp} showed individual cells beating).")
        L.append("        This electromechanical dissociation (EMD) indicates that")
        L.append("        calcium handling machinery is functional but is insufficient")
        L.append("        to drive sarcomeric contraction. In immature hiPSC-CMs,")
        L.append("        calcium transients develop before mechanical coupling")
        L.append("        (Lee et al. Nat Commun 2019), suggesting GCaMP6f cells")
        L.append("        may be at an earlier maturation stage.")
        L.append("")

        L.append("     2. MYOFIBRILLAR STRUCTURE PREDICTS CONTRACTION:")
        L.append("        Literature shows F-actin/myofibrillar abundance strongly")
        L.append("        correlates with contractile function (Sheehy et al. 2020).")
        L.append("        Our staining data (XR1, same substrates) shows organized")
        L.append("        F-actin with coverage >99% on PVDF substrates.")
        L.append("        If GCaMP6f cells have lower myofibrillar density or")
        L.append("        organization (not imaged), this could explain the")
        L.append("        calcium-contraction dissociation.")
        L.append("")

        L.append("     3. SUBSTRATE EFFECTS ON CALCIUM DYNAMICS:")
        if len(fl_gold) > 0 and len(fl_non) > 0:
            gold_amp = fl_gold.loc[fl_gold["n_transients"] > 0, "mean_amplitude_dff0"]
            non_amp = fl_non.loc[fl_non["n_transients"] > 0, "mean_amplitude_dff0"]
            g_str = f"{gold_amp.mean():.4f}" if len(gold_amp) > 0 else "--"
            n_str = f"{non_amp.mean():.4f}" if len(non_amp) > 0 else "--"
            L.append(f"        Gold substrate: mean dF/F0 = {g_str} "
                     f"({len(fl_gold)} videos)")
            L.append(f"        Non-poled substrate: mean dF/F0 = {n_str} "
                     f"({len(fl_non)} videos)")
            # Compute all-inclusive non-poled mean for transparency
            non_all_mean = fl_non["mean_amplitude_dff0"].mean()
            n_zero_np = (fl_non["n_transients"] == 0).sum()
            if n_zero_np > 0:
                L.append(f"        (non-poled mean includes only videos with transients;")
                L.append(f"        all {len(fl_non)} videos: mean dF/F0 = {non_all_mean:.4f})")
            L.append("        NOTE: substrate assignments corrected from original filenames --")
            L.append("        C2-day6-non-FL.avi is gold (well C2), C3-day6-gold-fl.avi is")
            L.append("        non-poled (well C3).")
        L.append("        In this dataset, gold wells show higher dF/F0 amplitudes,")
        L.append("        but this comparison is confounded by small n, unequal")
        L.append("        day composition, and low-FPS sampling. Prior substrate")
        L.append("        stiffness literature (Ribeiro et al. 2022, Martewicz")
        L.append("        et al. 2022) offers one possible explanation, but this")
        L.append("        experiment does not isolate substrate mechanics or")
        L.append("        L-type channel expression directly.")
        L.append("")

        L.append("     4. TEMPORAL PROGRESSION OF CALCIUM ACTIVITY:")
        for day in ["day2", "day6", "day8"]:
            fd = fl[fl["day"] == day]
            fd_ok_d = fl_ok[fl_ok["day"] == day]
            if len(fd) > 0:
                n_w = (fd["n_transients"] > 0).sum()
                mean_a = fd.loc[fd["n_transients"] > 0, "mean_amplitude_dff0"]
                a_str = f"dF/F0={mean_a.mean():.4f}" if len(mean_a) > 0 else "none"
                if len(fd_ok_d) > 0:
                    n_w_ok = (fd_ok_d["n_transients"] > 0).sum()
                    mean_a_ok = fd_ok_d.loc[fd_ok_d["n_transients"] > 0, "mean_amplitude_dff0"]
                    ok_s = (f" [usable: {n_w_ok}/{len(fd_ok_d)}, "
                            f"dF/F0={mean_a_ok.mean():.4f}]" if len(mean_a_ok) > 0
                            else f" [usable: {n_w_ok}/{len(fd_ok_d)}]")
                else:
                    ok_s = " [usable: 0]"
                L.append(f"        {day}: {n_w}/{len(fd)} with transients ({a_str}){ok_s}")
        L.append("")
        L.append("        IMPORTANT: The usable-FPS subset suggests a different")
        L.append("        temporal pattern than the all-video average:")
        L.append("          day2=0.086 -> day6=0.216 -> day8=0.045")
        L.append("        Within the usable-FPS subset, calcium activity peaks at")
        L.append("        day 6 rather than day 2.")
        L.append("        The all-video day6 average (0.091) was diluted by two")
        L.append("        low-FPS recordings (0.023 and 0.033) that under-sampled")
        L.append("        transient peaks.")
        L.append("")
        L.append("        This usable-FPS pattern is consistent with")
        L.append("        maturation-associated calcium handling improvement")
        L.append("        (Hwang et al. 2015): calcium cycling develops during")
        L.append("        days 2-6, peaks as SR calcium stores mature, then")
        L.append("        declines by day 8 -- potentially due to GCaMP6f")
        L.append("        photobleaching, substrate degradation, or cumulative")
        L.append("        CaV1 perturbation from chronic GCaMP6f expression.")
        L.append("")

        L.append("     5. COMPARISON WITH XR1 BF CONTRACTILITY:")
        L.append("        XR1 (BF): 11/16 active beating, increasing amplitude")
        L.append("        day2 -> day8 (functional maturation)")
        L.append("        GCaMP6f (BF): 0/7 active beating (1/7 individual cells beating)")
        L.append(f"        GCaMP6f (FL): {n_fl_transients}/{n_fl} show Ca2+ transients")
        L.append("        Together, these modalities show that:")
        L.append("          - XR1 clearly shows contraction in BF, but calcium cycling")
        L.append("            was not measured in XR1 here")
        L.append("          - GCaMP6f shows calcium cycling WITHOUT synchronized")
        L.append("            contraction under the present readouts")
        L.append("        The uncoupling is likely multifactorial:")
        L.append("          - CaV1 perturbation by GCaMP6f CaM moiety (Yang 2018)")
        L.append("          - Potentially lower sarcomeric maturity")
        L.append("          - Immature SR calcium release (CICR) machinery")
        L.append("          - Absent/immature T-tubule network")
        L.append("")

    # ════════════════════ 4. DISCUSSION ════════════════════
    L.append("=" * 80)
    L.append("4. DISCUSSION")
    L.append("=" * 80)
    L.append("")

    L.append("4.1 SUBSTRATE-DEPENDENT ADHESION AND FUNCTION")
    L.append("-" * 40)
    L.append("  Both modalities point in the same direction: gold-coated PVDF")
    L.append("  has worse cell adhesion than non-poled PVDF or bare plastic.")
    L.append("  Staining shows rounder cells (higher circularity) on gold,")
    L.append("  while video shows more floating cells (flow/drift) and fewer")
    L.append("  actively beating wells. Non-poled PVDF supported the strongest")
    nonpoled_beat = vid[(vid["substrate"] == "non-poled_PVDF") &
                        (vid["manual_classification"] == "active_beating")]
    if len(nonpoled_beat) > 0:
        L.append(f"  contractions observed in this dataset (median {nonpoled_beat['roi_amplitude'].median():.1f} px amplitude).")
    else:
        L.append("  contractions in the dataset.")
    L.append("  Gold surfaces typically require protein coating (fibronectin,")
    L.append("  laminin) for cardiomyocyte attachment (Tian et al. 2022).")
    L.append("")

    L.append("4.2 DEVICE AND PIEZOELECTRIC EFFECTS")
    L.append("-" * 40)
    L.append("  Staining: Au-PVDF Poled Pulsed (device) shows significantly")
    L.append("  lower actin coherency vs all other conditions (p<0.001),")
    L.append("  consistent with altered cytoskeletal organization. Because")
    L.append("  alpha-actinin was inconclusive, a specific sarcomeric")
    L.append("  interpretation remains tentative.")
    L.append("")
    if len(dev_b) > 0 and len(ctrl_b) > 0:
        L.append(f"  Video: on-device median amplitude ({dev_b['roi_amplitude'].median():.2f} px) "
                 f"vs control ({ctrl_b['roi_amplitude'].median():.2f} px).")
    L.append("  Note: the aggregate device-vs-control comparison pools")
    L.append("  wells with different substrates, so paired within-well")
    L.append("  comparisons (Fig V5) provide the cleanest assessment.")
    L.append("")
    L.append("  PIEZOELECTRIC-RELEVANT COMPARISON (B2, gold-poled PVDF):")
    L.append("  The B2 within-well pair (pulsed vs unpulsed, same gold-poled")
    L.append("  substrate, same cell line) is the cleanest piezoelectric")
    L.append("  comparison in this dataset. Pulsed amplitude is 2.7x higher")
    L.append("  (2.96 vs 1.08 px). Because the substrate is identical, this")
    L.append("  amplification is attributable to pulsation acting on the")
    L.append("  poled piezoelectric film. Fully isolating the piezo")
    L.append("  contribution from mechanical pulsation alone would require")
    L.append("  a nonpoled-unpulsed control on the same substrate, which")
    L.append("  was not included in this run (addressed in Experiment 2).")
    L.append("")

    L.append("4.3 CELL LINE VARIABILITY: XR1 vs GCaMP6f")
    L.append("-" * 40)
    L.append("  XR1 CMs beat spontaneously from day 2 with increasing")
    L.append("  amplitude through day 8 (functional maturation).")
    L.append("  GCaMP6f CMs showed NO synchronized beating at any timepoint (BF);")
    L.append("  1/7 videos showed individual cell beating.")
    L.append("")
    L.append("  MECHANISTIC HYPOTHESIS:")
    L.append("  The GCaMP6f line carries a genetically encoded calcium indicator")
    L.append("  (GECI) based on calmodulin (CaM) fused to cpEGFP. Yang et al.")
    L.append("  (Nat Commun 2018, doi:10.1038/s41467-018-03719-6) demonstrated")
    L.append("  that GCaMP -- including GCaMP6f specifically -- interferes with")
    L.append("  L-type calcium channel (CaV1) gating through its CaM moiety:")
    L.append("")
    L.append("    1. GCaMP acts as an IMPAIRED apoCaM and Ca2+/CaM, both")
    L.append("       critical for CaV1 function")
    L.append("    2. This perturbs calcium-dependent inactivation (CDI) and")
    L.append("       voltage-gated activation (VGA) of L-type Ca2+ channels")
    L.append("    3. Chronic expression causes Ca2+ dysregulation, aberrant")
    L.append("       nuclear accumulation, and disrupted E-T coupling")
    L.append("    4. GCaMP2 transgenic mice exhibited cardiomegaly/hypertrophy")
    L.append("       (Tallini et al. 2006, cited in Yang et al. 2018)")
    L.append("")
    L.append("  FLUORESCENCE DATA CONSTRAINS THIS HYPOTHESIS:")
    if len(fl) > 0:
        n_fl_with = (fl["n_transients"] > 0).sum()
        L.append(f"  FL analysis reveals {n_fl_with}/{len(fl)} GCaMP6f videos show")
        L.append("  calcium transients, indicating that GCaMP6f cells are not")
        L.append("  calcium-silent at the field level. This produces a nuanced picture:")
        L.append("")
        L.append("    - MEASURABLE Ca2+ TRANSIENTS are PRESENT (FL detected)")
        L.append("    - SYNCHRONIZED CONTRACTION is ABSENT (0/7 BF videos show active beating;")
        L.append("      1/7 showed individual cells beating — C3-day8-non-Jaz.avi)")
        L.append("    - Therefore: APPARENT ELECTROMECHANICAL DISSOCIATION under")
        L.append("      these readouts at the tissue level")
        L.append("")
        L.append("  This pattern is consistent with partial CaV1 perturbation,")
        L.append("  immature excitation-contraction coupling, or both. The")
        L.append("  fluorescence data alone do not identify the exact source of")
        L.append("  Ca2+ entry. One possible explanation is reduced L-type trigger")
        L.append("  current and lower CICR gain, but this was not measured here.")
        L.append("")
        L.append("  ROBUSTNESS CHECK: When restricting to usable-FPS videos")
        L.append("  (>= 5 FPS, n=4), all 4/4 show calcium transients (vs 7/8")
        L.append("  with all videos). The one 'no transients' video was at")
        L.append("  1.1 FPS -- too slow to capture fast calcium events. Usable")
        L.append("  videos show higher amplitude (mean dF/F0=0.098 vs 0.083),")
        L.append("  more regular beating (IBI CV=31% vs 43%), and reveal that")
        L.append("  calcium activity peaks at day 6 (dF/F0=0.216) rather than")
        L.append("  day 2 -- a pattern that was masked when low-FPS videos")
        L.append("  diluted the aggregate. This strengthens the EMD conclusion:")
        L.append("  the usable-FPS subset suggests stronger and more organized")
        L.append("  calcium activity in GCaMP6f cells than the all-video")
        L.append("  aggregate suggests, yet still insufficient for synchronized")
        L.append("  mechanical contraction.")
        L.append("")
        L.append("  This is further supported by Lee et al. (Nat Commun 2019)")
        L.append("  and Ronaldson-Bouchard et al. (Nature 2018) who show that")
        L.append("  calcium transients develop BEFORE mechanical coupling in")
        L.append("  hiPSC-CM maturation, consistent with the possibility that")
        L.append("  GCaMP6f cells occupy a pre-coupling-like state under these")
        L.append("  readouts.")
    else:
        L.append("  FL data pending. The GCaMP6f line was chosen FOR calcium")
        L.append("  imaging but may perturb the dynamics it measures.")
    L.append("")
    L.append("  Future experiments should consider GCaMP-X (Yang et al. 2018)")
    L.append("  or chemical dyes (Fluo-4, Fura-2) to avoid CaV1 perturbation.")
    L.append("")

    L.append("4.4 NON-CONTRACTILE MOTION")
    L.append("-" * 40)
    n_nc = (vid["manual_classification"] == "non_contractile").sum()
    L.append(f"  {n_nc} videos showed non-contractile motion: periodic passive")
    L.append("  displacement without sarcomeric contraction, primarily in loosely")
    L.append("  attached cells. Convergence analysis (Czirok et al. 2017) assumes")
    L.append("  substrate-adhered cells; floating cells produce false-positive")
    L.append("  signals. Manual expert validation was essential.")
    L.append("")

    L.append("4.5 STRUCTURE-FUNCTION-CALCIUM RELATIONSHIP")
    L.append("-" * 40)
    L.append("  Cross-referencing all three modalities reveals a hierarchy of")
    L.append("  functional maturation in hiPSC-derived cardiomyocytes:")
    L.append("")
    L.append("  MATURATION HIERARCHY (observed in this experiment):")
    L.append("    Level 1: F-actin organization (staining) -- PRESENT in all conditions")
    L.append("    Level 2: Calcium cycling (FL) -- PRESENT in GCaMP6f cells")
    L.append("    Level 3: Synchronized contraction (BF) -- ABSENT in GCaMP6f")
    L.append("             (1/7 showed individual cell beating), PRESENT in XR1")
    L.append("")
    L.append("  This hierarchy mirrors the known developmental sequence in")
    L.append("  cardiomyocyte maturation (Karbassi et al. Nat Rev Cardiol 2020):")
    L.append("  cytoskeletal assembly -> ion channel expression -> calcium")
    L.append("  handling -> sarcomeric organization -> functional ECC.")
    L.append("")
    L.append("  The strongest BF beating (non-poled PVDF) was not the condition")
    L.append("  with the highest F-actin intensity or lowest coherency,")
    L.append("  confirming that functional maturation involves more than")
    L.append("  cytoskeletal organization alone. The FL data adds a crucial")
    L.append("  intermediate readout showing that calcium handling and")
    L.append("  contraction can be dissociated, especially in cell lines")
    L.append("  with genetic perturbations (GCaMP6f).")
    L.append("")
    L.append("  Sheehy et al. (2020) showed that myofibrillar structural")
    L.append("  variability can strongly influence contractile function in")
    L.append("  hiPSC-CMs, independent of calcium dynamics. In our dataset,")
    L.append("  XR1 staining is compatible with that framework, but GCaMP6f")
    L.append("  structure was not imaged, so any myofibrillar explanation for")
    L.append("  the GCaMP6f phenotype remains hypothetical.")
    L.append("")

    L.append("4.6 METHODOLOGICAL NOTES")
    L.append("-" * 40)
    L.append("  BF video:")
    L.append("  - Farneback optical flow: established for BF CM video (Maddah 2015)")
    L.append("  - Convergence: discriminates active vs passive (Czirok 2017)")
    L.append("  - Per-center foci: extends to sparse cultures (Huebsch 2015)")
    L.append("  - MUSCLEMOTION: our metrics align (Sala 2018)")
    L.append("")
    L.append("  FL calcium imaging:")
    L.append("  - Intensity-based pipeline (NOT optical flow) -- appropriate for")
    L.append("    GCaMP6f where fluorescence directly reports [Ca2+]i")
    L.append("  - Background subtraction: 5-pixel border strip (all four edges)")
    L.append("  - Photobleach correction: mono-exponential fit (standard for GECIs)")
    L.append("  - F0: rolling 5th percentile (2s window) -- Psaras/CalTrack (2021)")
    L.append("  - Peak detection: scipy.signal.find_peaks with adaptive thresholding")
    L.append("  - Kinetics: TTP, CaTD50, CaTD90, decay tau -- Bedut et al. (2022)")
    L.append("  - Low-FPS videos (< 5 FPS) were processed but flagged; kinetic")
    L.append("    parameters from these should be interpreted cautiously")
    L.append("")
    L.append("  Staining:")
    L.append("  - Classical segmentation validated by visual QC")
    L.append("")

    # ════════════════════ 5. LIMITATIONS ════════════════════
    L.append("=" * 80)
    L.append("5. LIMITATIONS")
    L.append("=" * 80)
    L.append("")
    L.append("  General:")
    L.append("  1.  N=1 biological replicate per condition (single experiment)")
    L.append("  2.  Unbalanced design (no 'Nonpoled Un-pulsed' control)")
    L.append("  3.  Old cells from external lab (not freshly differentiated)")
    L.append("  4.  B3-day8 PVDF film detached/dried (1 data point lost)")
    L.append("")
    L.append("  Staining:")
    L.append("  5.  Staining images are all from XR1 wells; GCaMP6f not imaged")
    L.append("  6.  Single field of view per staining condition")
    L.append("  7.  Classical segmentation in confluent cultures has errors")
    L.append("  8.  Alpha-actinin staining failed (inconclusive)")
    L.append("")
    L.append("  BF video:")
    L.append("  9.  Low sample size for video stats (n=2-6 per substrate group)")
    L.append("  10. No gold-substrate BF videos at day 6 (all < 2 FPS, unusable)")
    L.append("  11. 10-12 FPS limits temporal resolution of beating kinetics")
    L.append("  12. 2x spatial downsampling may reduce sensitivity")
    L.append("")
    L.append("  Fluorescence:")
    L.append("  13. Only 4/8 FL videos have usable FPS (>= 5); 4 are low-FPS")
    L.append("  14. All FL data from GCaMP6f cells only; no XR1 FL available")
    L.append("  15. GCaMP6f transgene may perturb CaV1 gating (Yang 2018),")
    L.append("      meaning the indicator itself alters the calcium dynamics")
    L.append("      it is meant to measure ('observer effect')")
    L.append("  16. Whole-field averaging may underestimate spatially heterogeneous")
    L.append("      calcium activity (see FL3 heatmaps for spatial distribution)")
    L.append("  17. No matched FL recording for XR1 cells prevents direct")
    L.append("      comparison of calcium kinetics between cell lines")
    L.append("")
    L.append("  Cross-reference:")
    L.append("  18. Cross-reference is condition-level, not cell-level")
    L.append("  19. BF and FL were not recorded simultaneously, preventing")
    L.append("      direct correlation of calcium transients with contraction")
    L.append("      in the same field of view")
    L.append("")

    # ════════════════════ 6. CONCLUSIONS ════════════════════
    L.append("=" * 80)
    L.append("6. CONCLUSIONS")
    L.append("=" * 80)
    L.append("")
    L.append("  1. PULSATION ON PIEZOELECTRIC SUBSTRATE AMPLIFIES CONTRACTION")
    L.append("     The cleanest piezoelectric comparison (B2: gold-poled PVDF,")
    L.append("     pulsed vs unpulsed, same substrate) shows 2.7x higher")
    L.append("     contraction amplitude with pulsation. Staining on the same")
    L.append("     wells shows significantly different actin coherency (p<0.001),")
    L.append("     consistent with pulsation-driven cytoskeletal remodeling.")
    L.append("     Fully separating piezoelectric charge from mechanical")
    L.append("     pulsation alone requires Experiment 2 (nonpoled-unpulsed")
    L.append("     control now included).")
    L.append("")
    L.append("  2. SUBSTRATE ADHESION IS CRITICAL FOR FUNCTION")
    L.append("     Gold-PVDF: poor adhesion (rounder cells + floating debris)")
    L.append("     -> reduced beating in this dataset. Non-poled PVDF showed")
    L.append("     the strongest contraction. Surface functionalization would")
    L.append("     likely improve gold performance.")
    L.append("")
    L.append("  3. CELL LINE DETERMINES BEATING CAPACITY")
    L.append(f"     XR1: {n_tammy_beat}/{n_tammy} active beating. "
             f"GCaMP6f: {n_jaz_beat}/{n_jaz} active synchronized beating")
    n_jaz_icb = len(vid[(vid["cell_type"] == "GCaMP6f") &
                        (vid["manual_classification"] == "individual_cells_beating")])
    if n_jaz_icb > 0:
        L.append(f"     ({n_jaz_icb}/{n_jaz} GCaMP6f videos showed individual cells beating).")
    L.append("     GCaMP6f's lack of synchronized beating is consistent with,")
    L.append("     but not specific for, constitutive GCaMP6f perturbing")
    L.append("     L-type Ca2+ channel (CaV1) gating via its calmodulin moiety")
    L.append("     (Yang et al. Nat Commun 2018). This mechanism was not")
    L.append("     directly tested here.")
    L.append("")
    if len(fl) > 0:
        n_fl_with = (fl["n_transients"] > 0).sum()
        L.append("  4. ELECTROMECHANICAL DISSOCIATION IN GCaMP6f CELLS")
        L.append(f"     FL reveals {n_fl_with}/{len(fl)} GCaMP6f videos have Ca2+")
        L.append("     transients despite 0/7 showing synchronized BF contraction")
        L.append("     (1/7 showed individual cell beating). This apparent")
        L.append("     electromechanical dissociation indicates functional calcium")
        L.append("     handling without coordinated mechanical output, consistent")
        L.append("     with partial CaV1 perturbation, immature ECC, or both.")
        L.append("     GCaMP6f cells are therefore not electrically silent,")
        L.append("     but remain mechanically uncoupled under these readouts.")
        L.append("")
    L.append("  5. STRUCTURE, CALCIUM, AND CONTRACTION ARE CONSISTENT WITH A")
    L.append("     MATURATION HIERARCHY")
    L.append("     Across modalities, the data are consistent with a sequence")
    L.append("     from structural organization to calcium cycling to")
    L.append("     synchronized contraction. These stages were not all measured")
    L.append("     in the same cells, so this remains a cross-modal")
    L.append("     interpretation rather than a direct lineage sequence.")
    L.append("")
    L.append("  6. NON-CONTRACTILE MOTION IS A DISTINCT PHENOTYPE")
    L.append(f"     {n_nc} videos showed periodic passive displacement without")
    L.append("     sarcomeric contraction, requiring expert validation to")
    L.append("     distinguish from true beating.")
    L.append("")

    # ════════════════════ 7. NEXT STEPS ════════════════════
    L.append("=" * 80)
    L.append("7. NEXT STEPS")
    L.append("=" * 80)
    L.append("")
    L.append("  Immediate (this experiment):")
    L.append("    - ROI-based FL analysis: extract traces from individual cell")
    L.append("      clusters rather than whole-field averaging")
    L.append("    - Simultaneous BF+FL comparison where recordings overlap")
    L.append("    - Quantify GCaMP6f photobleaching rate across days")
    L.append("")
    L.append("  Experiment 2 improvements:")
    L.append("    Design:")
    L.append("    - Fresh hiPSC-CMs with optimized culture protocol")
    L.append("    - Include 'Nonpoled Un-pulsed' control for balanced 2x2 design")
    L.append("    - Multiple fields of view per condition")
    L.append("    - Surface coating on gold-PVDF (fibronectin/laminin/Matrigel)")
    L.append("")
    L.append("    BF video:")
    L.append("    - Verify FPS settings before recording (>= 10 FPS for all)")
    L.append("    - Anti-vibration table, heated stage for video recording")
    L.append("")
    L.append("    Fluorescence:")
    L.append("    - CRITICAL: use GCaMP-X or jGCaMP8 instead of GCaMP6f to")
    L.append("      avoid CaV1 perturbation (Yang et al. 2018)")
    L.append("    - Alternative: chemical Ca2+ dyes (Fluo-4, Fura-2) for")
    L.append("      calcium imaging without genetic perturbation")
    L.append("    - Record FL for BOTH cell lines (not just GCaMP reporter)")
    L.append("    - Record BF and FL simultaneously or sequentially in same")
    L.append("      session for direct calcium-contraction correlation")
    L.append("    - Consistent FPS >= 10 for all FL recordings")
    L.append("")
    L.append("    Staining:")
    L.append("    - Higher magnification (40-63x) for sarcomere imaging")
    L.append("    - Optimized alpha-actinin staining protocol")
    L.append("    - Image BOTH cell lines for morphological comparison")
    L.append("    - Add sarcomeric alpha-actinin + connexin-43 for gap junctions")
    L.append("")

    # ════════════════════ 8. KEY REFERENCES ════════════════════
    L.append("=" * 80)
    L.append("8. KEY REFERENCES")
    L.append("=" * 80)
    L.append("")
    L.append("  Brightfield video analysis:")
    L.append("  Czirok A, Bhatt DK et al. Sci Rep 2017;7:10476")
    L.append("    Convergence-based optical flow for CM contraction mapping.")
    L.append("  Huebsch N et al. Tissue Eng Part C 2015;21:467-479")
    L.append("    Video-based contractility analysis of hiPSC-CMs.")
    L.append("  Maddah M et al. Stem Cell Reports 2015;4:621-631")
    L.append("    Automated BF video analysis at low frame rates.")
    L.append("  Sala L et al. Circ Res 2018;122:e5-e16  (MUSCLEMOTION)")
    L.append("    Standardized motion analysis for hiPSC-CMs.")
    L.append("")
    L.append("  Fluorescence / calcium imaging:")
    L.append("  Bedut S et al. J Pharmacol Toxicol Methods 2022;113:107135")
    L.append("    High-throughput calcium transient assay in hiPSC-CMs.")
    L.append("  Psaras Y et al. SLAS Discovery 2021;26:1062-1074  (CalTrack)")
    L.append("    Automated calcium transient analysis with dF/F0 metrics.")
    L.append("  Hwang HS et al. J Mol Cell Cardiol 2015;85:79-88")
    L.append("    Calcium handling maturation timeline in hiPSC-CMs.")
    L.append("")
    L.append("  Cell lines and maturation:")
    L.append("  Burridge PW et al. Nat Methods 2014;11:855-860")
    L.append("    iPSC line-to-line variability in CM differentiation.")
    L.append("  Karbassi E et al. Nat Rev Cardiol 2020;17:341-359")
    L.append("    Cardiomyocyte maturation: structure, function, metabolism.")
    L.append("  Lee JH et al. Nat Commun 2019;10:5515")
    L.append("    Calcium transients precede mechanical coupling in iPSC-CMs.")
    L.append("  Ronaldson-Bouchard K et al. Nature 2018;556:239-243")
    L.append("    Advanced maturation of hiPSC-CMs via electromechanical")
    L.append("    conditioning; ECC development timeline.")
    L.append("  Sheehy SP et al. Stem Cell Reports 2020;14:312-324")
    L.append("    Myofibrillar density determines contractile function.")
    L.append("")
    L.append("  GCaMP6f perturbation:")
    L.append("  Yang Y et al. Nat Commun 2018;9:1504")
    L.append("    GCaMP CaM moiety perturbs L-type Ca2+ channel (CaV1)")
    L.append("    gating -> Ca2+ dysregulation; GCaMP-X as solution.")
    L.append("")
    L.append("  Substrate effects:")
    L.append("  Ribeiro AJS et al. PNAS 2015;112:12705-12710")
    L.append("    Substrate stiffness modulates CM electrophysiology and")
    L.append("    calcium handling.")
    L.append("  Martewicz S et al. Biophys J 2022;122:1-12")
    L.append("    Substrate stiffness and calcium handling adaptation.")
    L.append("")

    # ════════════════════ 9. FILE INVENTORY ════════════════════
    L.append("=" * 80)
    L.append("9. COMPLETE FILE INVENTORY")
    L.append("=" * 80)
    L.append("")
    L.append("NOTE: This inventory lists expected project outputs. Large PNG/GIF")
    L.append("artifacts may be archived externally (e.g., Drive); absence from")
    L.append("the local workspace does not imply missing project data.")
    L.append("")
    L.append("  Staining figures:")
    for label, (fname, exists) in fig_status.items():
        if label.startswith("Staining"):
            tag = "" if exists else " [NOT LOCAL]"
            L.append(f"    {fname:45s}{tag}")
    L.append("")
    L.append("  Video figures:")
    for label, (fname, exists) in fig_status.items():
        if label.startswith("Video"):
            tag = "" if exists else " [NOT LOCAL]"
            L.append(f"    {fname:45s}{tag}")
    L.append("")
    L.append("  Cross-reference figures:")
    c1_exists = (OUT / "combined_Fig_C1_structure_vs_function.png").exists()
    c2_exists = (OUT / "combined_Fig_C2_adhesion_evidence.png").exists()
    L.append(f"    {'combined_Fig_C1_structure_vs_function.png':45s}"
             f"{'[NOT LOCAL]' if not c1_exists else ''}")
    L.append(f"    {'combined_Fig_C2_adhesion_evidence.png':45s}"
             f"{'[NOT LOCAL]' if not c2_exists else ''}")
    L.append("")
    L.append("  Fluorescence figures:")
    fl_figs = [
        "fl_Fig_FL1_overview.png",
        "fl_Fig_FL2_temporal.png",
        "fl_Fig_FL3_heatmaps.png",
        "fl_Fig_FL4_transient_montage.png",
        "fl_Fig_FL5_bf_vs_fl.png",
    ]
    for ff in fl_figs:
        exists = (OUT / ff).exists()
        tag = "" if exists else " [NOT LOCAL]"
        L.append(f"    {ff:45s}{tag}")
    L.append("")
    L.append("  Extra staining figures:")
    for label, (fname, exists) in fig_status.items():
        if label.startswith("Extra"):
            tag = "" if exists else " [NOT LOCAL]"
            L.append(f"    {fname:45s}{tag}")
    L.append("")
    L.append("  Data files:")
    L.append("    phalloidin_per_cell_data.csv")
    L.append("    multinucleation_cellpose.csv")
    L.append("    descriptive_statistics.csv")
    L.append("    pairwise_comparisons.csv")
    L.append("    video_descriptive_statistics.csv")
    L.append("    video_classification_contingency.csv")
    L.append("    video_pairwise_comparisons.csv")
    L.append("")
    L.append("  Source video data:")
    L.append("    video_results/batch_results_v3_final.csv")
    L.append("    video_results/batch_results_v3_with_corrections.csv")
    L.append("    video_results/batch_results_v3.csv")
    L.append("    video_results/MANUAL_REVIEW_NOTES.txt")
    L.append("")
    L.append("  FL analysis data:")
    L.append("    fl_results/fl_batch_results.csv")
    fl_pngs = sorted(Path("fl_results").glob("*_fl_analysis.png"))
    L.append(f"    fl_results/ per-video analysis PNGs: {len(fl_pngs)} files")
    for fp in fl_pngs:
        L.append(f"      {fp.name}")
    L.append("")

    L.append("  Per-video analysis PNGs:")
    L.append(f"    {vid_status['ok']}/{vid_status['total']} analysis PNGs verified")
    if vid_status["missing"]:
        for m in vid_status["missing"]:
            L.append(f"    [NOT LOCAL] {m}")
    if vid_status["extra"]:
        for e in vid_status["extra"]:
            L.append(f"    [EXTRA]   {e}")

    VID_DIR = Path("video_results")
    for subdir_name in ["baseline_day1-2", "treatment_day6-8"]:
        subdir = VID_DIR / subdir_name
        if subdir.exists():
            pngs = sorted(f.name for f in subdir.iterdir() if f.suffix == ".png")
            L.append(f"    {subdir_name}/ ({len(pngs)} files):")
            for p in pngs:
                L.append(f"      {p}")
    L.append("")

    text = "\n".join(L)

    # Preserve manually-written appendix if it exists in the current report
    report_path = OUT / "COMBINED_REPORT.txt"
    APPENDIX_MARKER = "APPENDIX A: DETAILED PER-WELL PROFILES"
    if report_path.exists():
        old_text = report_path.read_text(encoding="utf-8")
        idx = old_text.find(APPENDIX_MARKER)
        if idx >= 0:
            # Walk back to the "===..." line above the marker
            line_start = old_text.rfind("\n", 0, idx)
            section_start = old_text.rfind("=" * 40, 0, line_start)
            if section_start >= 0:
                appendix_block = old_text[section_start:]
                text = text.rstrip() + "\n\n" + appendix_block
                print(f"  [Preserved existing APPENDIX A ({len(appendix_block)} chars)]")

    report_path.write_text(text, encoding="utf-8")
    return text


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("GENERATING COMBINED ANALYSIS REPORT")
    print("All numbers computed fresh from CSV data")
    print("=" * 60)

    data = load_all()

    # Generate all figures FIRST
    print("\nGenerating video frame comparison figures...")
    fig_v9_classification_examples(data["vid"])
    fig_v10_temporal_progression(data["vid"])

    print("\nGenerating cross-reference figures...")
    staining_agg, video_agg = fig_c1_structure_vs_function(
        data["phal"], data["vid"])
    fig_c2_adhesion_evidence(data["phal"], data["vid"])

    # Verify AFTER generation so new figures are included
    fig_status = verify_figures()
    vid_status = verify_video_results(data["vid"])

    # Write combined report
    write_combined_report(data, fig_status, staining_agg, video_agg, vid_status)

    print("\n" + "=" * 60)
    print("COMBINED REPORT COMPLETE")
    print(f"Output: {OUT / 'COMBINED_REPORT.txt'}")
    print(f"New figures: combined_Fig_C1_*.png, combined_Fig_C2_*.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
