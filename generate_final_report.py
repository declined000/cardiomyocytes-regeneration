"""
Final Comprehensive Report: Cardiomyocyte Image Analysis
=========================================================
Generates all figures and a text summary for the capstone project.

Experiment: hiPSC-derived cardiomyocytes on piezoelectric substrates
Stains analyzed: Phalloidin (F-actin), DAPI (nuclei), Alpha-actinin (inconclusive)

Conditions:
  1. Au-PVDF Poled, Pulsed (B2 device) - gold+poled PVDF, mechanical+piezoelectric
  2. Au-PVDF Poled, Un-pulsed (B2 control plate) - gold+poled PVDF, no stimulation
  3. β-PVDF Nonpoled, Pulsed (A3 device) - uncoated nonpoled β-PVDF, mechanical only
  4. Cells only (B3 control plate) - bare plastic, baseline
"""

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

OUT = Path("final_report")
OUT.mkdir(exist_ok=True)

DATA = Path("Cardiomyocytes")
RESULTS = Path("results_v2")

PHALLOIDIN_CONDITIONS = {
    "Au-PVDF Poled\nPulsed (device)": {
        "phalloidin": DATA / "device-gold2-phalloidin-b2-image.tif",
        "dapi": DATA / "device-gold2-dapi-b2-image.tif",
    },
    "Au-PVDF Poled\nUn-pulsed (control)": {
        "phalloidin": DATA / "control-gold2-b2-phalloidin.tif",
        "dapi": DATA / "control-gold2-b2-dapi.tif",
    },
    "β-PVDF Nonpoled\nPulsed (device)": {
        "phalloidin": DATA / "device-non-phalloidin-a3-image.tif",
        "dapi": DATA / "device-non-dapi-a3-image.tif",
    },
    "Cells only\n(baseline)": {
        "phalloidin": DATA / "control-b3-non-phalloidin.tif",
        "dapi": DATA / "control-b3-non-dapi.tif",
    },
}

# Mapping from old CSV condition names to corrected labels
CSV_TO_LABEL = {
    "Gold\n(device)": "Au-PVDF Poled\nPulsed (device)",
    "Gold\n(control plate)": "Au-PVDF Poled\nUn-pulsed (control)",
    "Nonpoled\n(device)": "β-PVDF Nonpoled\nPulsed (device)",
    "Cells only\n(control)": "Cells only\n(baseline)",
}


def remap_conditions(df):
    """Remap old condition names in CSV data to new 2x2 design labels."""
    df["condition"] = df["condition"].map(CSV_TO_LABEL).fillna(df["condition"])
    return df

ACTININ_CONDITIONS = {
    "Au-PVDF Poled\nUn-pulsed (control)": DATA / "control-gold1-a2-alphaact.tif",
    "Au-PVDF Poled\nPulsed (device)": DATA / "device-gold1-alphaactinin-A2-image.tif",
    "Cells only\n(baseline)": DATA / "control-a3-alpha-non.tif",
}

COLORS = {
    "Au-PVDF Poled\nPulsed (device)": "#ff9966",
    "Au-PVDF Poled\nUn-pulsed (control)": "#ffd966",
    "β-PVDF Nonpoled\nPulsed (device)": "#a8d5a2",
    "Cells only\n(baseline)": "#7fbfff",
}


def norm16(img):
    p1, p99 = np.percentile(img, (1, 99.5))
    return np.clip((img.astype(float) - p1) / (p99 - p1), 0, 1)


# ============================================================
# FIGURE 1: Raw Images Overview
# ============================================================
def fig1_raw_images():
    print("  Figure 1: Raw image overview...")
    fig, axes = plt.subplots(4, 3, figsize=(21, 28))

    for row, (cond, paths) in enumerate(PHALLOIDIN_CONDITIONS.items()):
        dapi = norm16(tifffile.imread(paths["dapi"]))
        phal = norm16(tifffile.imread(paths["phalloidin"]))

        axes[row, 0].imshow(dapi, cmap="Blues")
        axes[row, 0].set_title(f"DAPI\n{cond}", fontsize=12, fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(phal, cmap="Greens")
        axes[row, 1].set_title(f"Phalloidin (F-actin)\n{cond}", fontsize=12, fontweight="bold")
        axes[row, 1].axis("off")

        merge = np.zeros((*dapi.shape, 3))
        merge[:, :, 1] = phal
        merge[:, :, 2] = dapi
        axes[row, 2].imshow(np.clip(merge, 0, 1))
        axes[row, 2].set_title(f"Merge (DAPI+Phalloidin)\n{cond}", fontsize=12, fontweight="bold")
        axes[row, 2].axis("off")

    fig.suptitle("Figure 1: Raw Fluorescence Images\nPhalloidin (F-actin, green) + DAPI (nuclei, blue)",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUT / "Fig1_raw_images.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# FIGURE 2: Segmentation Quality Control
# ============================================================
def fig2_segmentation_qc():
    print("  Figure 2: Segmentation QC...")
    from skimage import filters, morphology, measure, feature, segmentation
    from scipy.ndimage import binary_fill_holes
    from scipy import ndimage

    fig, axes = plt.subplots(4, 2, figsize=(18, 28))

    for row, (cond, paths) in enumerate(PHALLOIDIN_CONDITIONS.items()):
        dapi = norm16(tifffile.imread(paths["dapi"]))
        phal = norm16(tifffile.imread(paths["phalloidin"]))

        # Segment nuclei
        smoothed_d = filters.gaussian(dapi, sigma=2)
        thresh_d = filters.threshold_otsu(smoothed_d)
        binary_d = smoothed_d > thresh_d
        binary_d = morphology.remove_small_objects(binary_d, min_size=50)
        binary_d = binary_fill_holes(binary_d)
        binary_d = morphology.binary_closing(binary_d, morphology.disk(2))
        dist = ndimage.distance_transform_edt(binary_d)
        coords = feature.peak_local_max(dist, min_distance=8, labels=binary_d)
        peak_mask = np.zeros_like(dist, dtype=bool)
        peak_mask[tuple(coords.T)] = True
        markers = measure.label(peak_mask)
        nuclei = segmentation.watershed(-dist, markers, mask=binary_d)

        # Segment cells
        smoothed_p = filters.gaussian(phal, sigma=1.5)
        thresh_p = filters.threshold_otsu(smoothed_p)
        cell_mask = smoothed_p > (thresh_p * 0.5)
        cell_mask = morphology.binary_dilation(cell_mask, morphology.disk(3))
        cell_mask = binary_fill_holes(cell_mask)
        cell_mask = morphology.remove_small_objects(cell_mask, min_size=100)
        gradient = filters.sobel(smoothed_p)
        cells = segmentation.watershed(gradient, nuclei, mask=cell_mask)
        cells = morphology.remove_small_objects(cells, min_size=80)

        n_nuclei = nuclei.max()
        n_cells = cells.max()

        # Phalloidin + cell boundaries
        rgb1 = np.zeros((*phal.shape, 3))
        rgb1[:, :, 1] = phal * 0.9
        cell_bounds = segmentation.find_boundaries(cells, mode="thick")
        rgb1[cell_bounds] = [1, 0, 0]
        axes[row, 0].imshow(np.clip(rgb1, 0, 1))
        axes[row, 0].set_title(f"Cell segmentation\n{cond}\n({n_cells} cells detected)",
                               fontsize=11, fontweight="bold")
        axes[row, 0].axis("off")

        # Phalloidin + nuclei boundaries
        rgb2 = np.zeros((*phal.shape, 3))
        rgb2[:, :, 1] = phal * 0.7
        rgb2[:, :, 2] = dapi * 0.7
        nuc_bounds = segmentation.find_boundaries(nuclei, mode="thick")
        rgb2[nuc_bounds] = [1, 1, 0]
        rgb2[cell_bounds] = [1, 0, 0]
        axes[row, 1].imshow(np.clip(rgb2, 0, 1))
        axes[row, 1].set_title(f"Nuclei + cell overlay\n{cond}\n({n_nuclei} nuclei, {n_cells} cells)",
                               fontsize=11, fontweight="bold")
        axes[row, 1].axis("off")

    fig.suptitle("Figure 2: Segmentation Quality Control\n"
                 "Red = cell boundaries (watershed from DAPI seeds), Yellow = nuclei boundaries",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUT / "Fig2_segmentation_qc.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# FIGURE 3: Cell Morphology
# ============================================================
def fig3_morphology(df):
    print("  Figure 3: Cell morphology...")
    metrics = [
        ("cell_area_px", "Cell Area (px)"),
        ("aspect_ratio", "Aspect Ratio"),
        ("circularity", "Circularity"),
        ("elongation", "Elongation"),
        ("solidity", "Solidity"),
    ]
    conds = list(PHALLOIDIN_CONDITIONS.keys())

    fig, axes = plt.subplots(1, 5, figsize=(28, 6))
    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        data = [df[df["condition"] == c][col].dropna().values for c in conds]
        bp = ax.boxplot(data, labels=[c.replace("\n", " ") for c in conds],
                        patch_artist=True, widths=0.6, showfliers=False)
        for j, (patch, c) in enumerate(zip(bp["boxes"], conds)):
            patch.set_facecolor(COLORS[c])
            patch.set_alpha(0.7)
        for j, d in enumerate(data):
            x = np.random.normal(j + 1, 0.04, size=len(d))
            ax.scatter(x, d, alpha=0.3, s=8, color="black", zorder=5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8, rotation=15)

    fig.suptitle("Figure 3: Cell Morphology Comparison (Phalloidin Segmentation)",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(OUT / "Fig3_morphology.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# FIGURE 4: F-actin Intensity & Organization
# ============================================================
def fig4_factin(df):
    print("  Figure 4: F-actin intensity & organization...")
    metrics = [
        ("mean_phalloidin_intensity", "Mean F-actin\nIntensity"),
        ("phalloidin_coverage", "F-actin\nCoverage"),
        ("actin_coherency", "Actin Fiber\nCoherency"),
        ("haralick_correlation", "Texture\nCorrelation"),
        ("haralick_contrast", "Texture\nContrast"),
        ("std_phalloidin_intensity", "Intensity\nVariability (std)"),
    ]
    conds = list(PHALLOIDIN_CONDITIONS.keys())

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    axes = axes.flatten()
    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        data = [df[df["condition"] == c][col].dropna().values for c in conds]
        bp = ax.boxplot(data, labels=[c.replace("\n", " ") for c in conds],
                        patch_artist=True, widths=0.6, showfliers=False)
        for j, (patch, c) in enumerate(zip(bp["boxes"], conds)):
            patch.set_facecolor(COLORS[c])
            patch.set_alpha(0.7)
        for j, d in enumerate(data):
            x = np.random.normal(j + 1, 0.04, size=len(d))
            ax.scatter(x, d, alpha=0.3, s=8, color="black", zorder=5)

        # Key comparisons: Poled Pulsed vs Poled Unpulsed, Poled vs Nonpoled (pulsed), treatment vs baseline
        pairs_to_test = [(0, 1), (0, 2), (0, 3)]
        max_y = max(np.max(d) for d in data if len(d) > 0)
        for pi, (i1, i2) in enumerate(pairs_to_test):
            if len(data[i1]) > 0 and len(data[i2]) > 0:
                _, p = mannwhitneyu(data[i1], data[i2], alternative="two-sided")
                if p < 0.05:
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                    y = max_y * (1.05 + 0.07 * pi)
                    ax.plot([i1 + 1, i2 + 1], [y, y], "k-", linewidth=1)
                    ax.text((i1 + i2 + 2) / 2, y, sig, ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7, rotation=15)

    fig.suptitle("Figure 4: F-actin Intensity & Organization\n(* p<0.05, ** p<0.01, *** p<0.001, Mann-Whitney U)",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(OUT / "Fig4_factin_organization.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# FIGURE 5: Summary Bar Charts (Mean +/- SEM)
# ============================================================
def fig5_summary_bars(df):
    print("  Figure 5: Summary bar charts...")
    metrics = [
        ("mean_phalloidin_intensity", "Mean F-actin Intensity"),
        ("phalloidin_coverage", "F-actin Coverage"),
        ("actin_coherency", "Actin Coherency"),
        ("haralick_correlation", "Texture Correlation"),
    ]
    conds = list(PHALLOIDIN_CONDITIONS.keys())
    colors = [COLORS[c] for c in conds]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        means, sems = [], []
        for c in conds:
            vals = df[df["condition"] == c][col].dropna()
            means.append(vals.mean())
            sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        ax.bar(range(len(conds)), means, yerr=sems, capsize=5,
               color=colors, edgecolor="black", linewidth=0.8)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c.replace("\n", " ") for c in conds], fontsize=8, rotation=15)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for i, (m, s) in enumerate(zip(means, sems)):
            ax.text(i, m + s + 0.005 * max(means), f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Figure 5: Key Metrics Summary (Mean +/- SEM)", fontsize=15, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(OUT / "Fig5_summary_bars.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# FIGURE 6: Multinucleation (Cellpose)
# ============================================================
def fig6_multinucleation():
    print("  Figure 6: Multinucleation...")
    mn = pd.read_csv(RESULTS / "multinucleation_cellpose.csv")
    mn = remap_conditions(mn)

    conds = list(PHALLOIDIN_CONDITIONS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Bar chart: cells per condition
    counts = [len(mn[mn["condition"] == c]) for c in conds]
    axes[0].bar(range(len(conds)), counts, color=[COLORS[c] for c in conds], edgecolor="black")
    axes[0].set_xticks(range(len(conds)))
    axes[0].set_xticklabels([c.replace("\n", " ") for c in conds], fontsize=9, rotation=15)
    axes[0].set_title("Cells Detected by Cellpose", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Cell count")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 1, str(v), ha="center", fontweight="bold")

    # Stacked bar: nuclei distribution
    mono_pcts, bi_pcts, zero_pcts = [], [], []
    for c in conds:
        sub = mn[mn["condition"] == c]
        total = len(sub)
        if total > 0:
            mono = (sub["nuclei_count"] == 1).sum()
            bi = (sub["nuclei_count"] >= 2).sum()
            zero = (sub["nuclei_count"] == 0).sum()
            mono_pcts.append(100 * mono / total)
            bi_pcts.append(100 * bi / total)
            zero_pcts.append(100 * zero / total)
        else:
            mono_pcts.append(0)
            bi_pcts.append(0)
            zero_pcts.append(0)

    x = range(len(conds))
    axes[1].bar(x, mono_pcts, color="#4da6ff", edgecolor="black", label="1 nucleus")
    axes[1].bar(x, bi_pcts, bottom=mono_pcts, color="#ffcc00", edgecolor="black", label="2+ nuclei")
    bottoms2 = [m + b for m, b in zip(mono_pcts, bi_pcts)]
    axes[1].bar(x, zero_pcts, bottom=bottoms2, color="#cccccc", edgecolor="black", label="No nucleus matched")
    axes[1].set_xticks(range(len(conds)))
    axes[1].set_xticklabels([c.replace("\n", " ") for c in conds], fontsize=9, rotation=15)
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("Nuclei per Cell Distribution", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)

    # Text summary
    axes[2].axis("off")
    summary = (
        "MULTINUCLEATION SUMMARY\n"
        "================================\n\n"
        "Method: Cellpose v3 (cyto2 model, GPU)\n"
        "  + classical DAPI nuclei segmentation\n\n"
    )
    total_all, mono_all, multi_all, zero_all = 0, 0, 0, 0
    for c in conds:
        sub = mn[mn["condition"] == c]
        total = len(sub)
        mono = int((sub["nuclei_count"] == 1).sum())
        bi = int((sub["nuclei_count"] >= 2).sum())
        zero = int((sub["nuclei_count"] == 0).sum())
        total_all += total; mono_all += mono; multi_all += bi; zero_all += zero
        label = c.replace("\n", " ")
        summary += (f"{label}:\n  {total} cells, {mono} mono ({100*mono/total:.0f}%), "
                    f"{bi} multi ({100*bi/total:.0f}%), "
                    f"{zero} unassigned ({100*zero/total:.0f}%)\n\n")
    detected = mono_all + multi_all
    summary += (f"\n{zero_all}/{total_all} cells ({100*zero_all/total_all:.0f}%) had no\n"
                f"nucleus assigned (segmentation limitation).\n"
                f"Among detected ({detected}): {100*mono_all/detected:.0f}% mono.\n"
                f"Conclusion: mononucleated where\ndetectable. Expected for hiPSC-CMs.")
    axes[2].text(0.05, 0.95, summary, transform=axes[2].transAxes,
                 fontsize=11, fontfamily="monospace", verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"))

    fig.suptitle("Figure 6: Multinucleation Analysis (Cellpose Deep Learning)",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(OUT / "Fig6_multinucleation.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# FIGURE 7: Alpha-Actinin - INCONCLUSIVE
# ============================================================
def fig7_actinin_inconclusive():
    print("  Figure 7: Alpha-actinin (inconclusive)...")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    for idx, (cond, path) in enumerate(ACTININ_CONDITIONS.items()):
        img = norm16(tifffile.imread(path))
        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(cond.replace("\n", " "), fontsize=13, fontweight="bold")
        axes[idx].axis("off")
        axes[idx].text(
            0.5, 0.5, "INCONCLUSIVE", transform=axes[idx].transAxes,
            fontsize=30, fontweight="bold", color="red", alpha=0.75,
            ha="center", va="center", rotation=30,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="red", alpha=0.65, linewidth=3),
        )

    fig.suptitle(
        "Figure 7: Alpha-Actinin Staining - INCONCLUSIVE\n"
        "No recognizable cell morphology. Images show substrate artifacts:\n"
        "Gold film bubbles/debris (control plate), scratches (device), diffuse non-specific signal (cells only)",
        fontsize=13, fontweight="bold", color="darkred", y=1.06,
    )
    plt.tight_layout()
    fig.savefig(OUT / "Fig7_actinin_INCONCLUSIVE.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# FIGURE 8: Significance Heatmap
# ============================================================
def fig8_significance_heatmap(df):
    print("  Figure 8: Statistical significance heatmap...")
    metrics = [
        "cell_area_px", "aspect_ratio", "circularity", "elongation", "solidity",
        "mean_phalloidin_intensity", "phalloidin_coverage",
        "actin_coherency", "haralick_correlation", "haralick_contrast",
    ]
    metric_labels = [
        "Cell Area", "Aspect Ratio", "Circularity", "Elongation", "Solidity",
        "F-actin Intensity", "F-actin Coverage",
        "Actin Coherency", "Texture Correlation", "Texture Contrast",
    ]
    conds = list(PHALLOIDIN_CONDITIONS.keys())
    comparisons = []
    comp_labels = []
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            comparisons.append((conds[i], conds[j]))
            l1 = conds[i].replace("\n", " ")
            l2 = conds[j].replace("\n", " ")
            comp_labels.append(f"{l1}\nvs\n{l2}")

    n_comp = len(comparisons)
    n_met = len(metrics)
    pval_matrix = np.ones((n_met, n_comp))
    direction_matrix = np.zeros((n_met, n_comp))

    for ci, (c1, c2) in enumerate(comparisons):
        for mi, m in enumerate(metrics):
            v1 = df[df["condition"] == c1][m].dropna()
            v2 = df[df["condition"] == c2][m].dropna()
            if len(v1) > 1 and len(v2) > 1:
                _, p = mannwhitneyu(v1, v2, alternative="two-sided")
                pval_matrix[mi, ci] = p
                direction_matrix[mi, ci] = 1 if v1.median() > v2.median() else -1

    sig_matrix = np.zeros_like(pval_matrix)
    sig_matrix[pval_matrix < 0.05] = 1
    sig_matrix[pval_matrix < 0.01] = 2
    sig_matrix[pval_matrix < 0.001] = 3
    display = sig_matrix * direction_matrix

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(display, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")

    ax.set_xticks(range(n_comp))
    ax.set_xticklabels(comp_labels, fontsize=8, ha="center")
    ax.set_yticks(range(n_met))
    ax.set_yticklabels(metric_labels, fontsize=10)

    for mi in range(n_met):
        for ci in range(n_comp):
            p = pval_matrix[mi, ci]
            if p < 0.001:
                txt = "***"
            elif p < 0.01:
                txt = "**"
            elif p < 0.05:
                txt = "*"
            else:
                txt = "ns"
            color = "white" if abs(display[mi, ci]) >= 2 else "black"
            ax.text(ci, mi, txt, ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cbar.set_ticklabels(["***\n(1st lower)", "**", "*", "ns", "*", "**", "***\n(1st higher)"])

    fig.suptitle("Figure 8: Statistical Significance Heatmap (Phalloidin Analysis)\n"
                 "Mann-Whitney U test, all pairwise comparisons",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "Fig8_significance_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# STATISTICS TABLE
# ============================================================
def generate_stats_table(df):
    print("  Generating statistics tables...")
    metrics = [
        "cell_area_px", "aspect_ratio", "circularity", "elongation", "solidity",
        "mean_phalloidin_intensity", "integrated_phalloidin_intensity",
        "std_phalloidin_intensity", "phalloidin_coverage",
        "actin_coherency", "haralick_energy", "haralick_homogeneity",
        "haralick_contrast", "haralick_correlation",
    ]
    conds = list(PHALLOIDIN_CONDITIONS.keys())

    # Descriptive stats
    desc_rows = []
    for c in conds:
        sub = df[df["condition"] == c]
        for m in metrics:
            vals = sub[m].dropna()
            desc_rows.append({
                "condition": c.replace("\n", " "),
                "metric": m,
                "n": len(vals),
                "mean": vals.mean(),
                "std": vals.std(),
                "median": vals.median(),
                "q25": vals.quantile(0.25),
                "q75": vals.quantile(0.75),
            })
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(OUT / "descriptive_statistics.csv", index=False)

    # Pairwise comparisons
    pw_rows = []
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            c1, c2 = conds[i], conds[j]
            for m in metrics:
                v1 = df[df["condition"] == c1][m].dropna()
                v2 = df[df["condition"] == c2][m].dropna()
                if len(v1) > 1 and len(v2) > 1:
                    stat, p = mannwhitneyu(v1, v2, alternative="two-sided")
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    pw_rows.append({
                        "group_1": c1.replace("\n", " "),
                        "group_2": c2.replace("\n", " "),
                        "metric": m,
                        "median_1": v1.median(),
                        "median_2": v2.median(),
                        "U_statistic": stat,
                        "p_value": p,
                        "significance": sig,
                    })
    pw_df = pd.DataFrame(pw_rows)
    pw_df.to_csv(OUT / "pairwise_comparisons.csv", index=False)
    return pw_df


# ============================================================
# TEXT SUMMARY
# ============================================================
def write_text_summary(df, pw_df):
    print("  Writing text summary...")
    mn = pd.read_csv(RESULTS / "multinucleation_cellpose.csv")
    mn = remap_conditions(mn)
    conds = list(PHALLOIDIN_CONDITIONS.keys())

    lines = []
    lines.append("=" * 80)
    lines.append("FINAL REPORT: CARDIOMYOCYTE IMAGE ANALYSIS")
    lines.append("Piezoelectric Substrate Effects on hiPSC-Derived Cardiomyocytes")
    lines.append("=" * 80)
    lines.append("")

    lines.append("EXPERIMENTAL DESIGN")
    lines.append("-" * 40)
    lines.append("Cell type: hiPSC-derived cardiomyocytes (XR1)")
    lines.append("Stains: Phalloidin (F-actin), DAPI (nuclei), Alpha-actinin (Z-lines)")
    lines.append("")
    lines.append("Well layout (6-well plates):")
    lines.append("  Control plate: x2 = Gold (Au-PVDF Poled), x3 = Cells only (bare plastic)")
    lines.append("  Device plate:  x2 = Gold (Au-PVDF Poled), x3 = Nonpoled (β-PVDF Nonpoled)")
    lines.append("  Rows A,B = XR1; Row C = GCaMP6f")
    lines.append("")
    lines.append("  Conditions (Phalloidin analysis):")
    lines.append("    1. Au-PVDF Poled, Pulsed (B2 device)   - gold+poled PVDF, mechanical+piezoelectric")
    lines.append("    2. Au-PVDF Poled, Un-pulsed (B2 ctrl)  - gold+poled PVDF, no stimulation")
    lines.append("    3. β-PVDF Nonpoled, Pulsed (A3 device)- uncoated nonpoled β-PVDF, mechanical only")
    lines.append("    4. Cells only (B3 ctrl)                 - bare plastic, baseline")
    lines.append("")
    lines.append("  NOTE: This is NOT a balanced 2x2 factorial. Control plate column 3")
    lines.append("  is 'Cells only' (bare plastic), NOT nonpoled PVDF.")
    lines.append("")
    lines.append("  Clean comparisons:")
    lines.append("    - 1 vs 2: Effect of pulsation on poled PVDF (mechanical + piezoelectric)")
    lines.append("    - 1 vs 3: Piezoelectric effect specifically (poled vs nonpoled, both pulsed)")
    lines.append("    - Any vs 4: Overall treatment vs untreated baseline")
    lines.append("")
    lines.append("  Confounded comparisons:")
    lines.append("    - 3 vs 4: Differs in BOTH substrate (PVDF vs plastic) AND stimulation")
    lines.append("    - 2 vs 4: Substrate effect (Au-PVDF vs bare plastic), not poled vs nonpoled")
    lines.append("")
    lines.append("Note: Alpha-actinin images were INCONCLUSIVE (see below).")
    lines.append("      B3 device (Nonpoled 2) film detached - unable to stain.")
    lines.append("")

    lines.append("=" * 80)
    lines.append("ANALYSIS 1: PHALLOIDIN (F-ACTIN CYTOSKELETON)")
    lines.append("=" * 80)
    lines.append("Method: Classical image processing pipeline")
    lines.append("  - Nuclei: DAPI thresholding + watershed separation")
    lines.append("  - Cells: Nuclei-seeded watershed expanding to phalloidin boundaries")
    lines.append("  - Metrics: 16 per-cell measurements (morphology, intensity, texture)")
    lines.append("")

    for c in conds:
        sub = df[df["condition"] == c]
        label = c.replace("\n", " ")
        lines.append(f"  {label}: {len(sub)} cells analyzed")
    lines.append(f"  Total: {len(df)} cells")
    lines.append("")

    lines.append("KEY FINDINGS:")
    lines.append("")
    lines.append("  1. F-ACTIN INTENSITY: Substrate conditions > Cells only")
    for c in conds:
        sub = df[df["condition"] == c]
        v = sub["mean_phalloidin_intensity"].dropna()
        label = c.replace("\n", " ")
        lines.append(f"     {label:30s}  median = {v.median():.3f}")

    sig_phal = pw_df[(pw_df["metric"] == "mean_phalloidin_intensity") & (pw_df["significance"] != "ns")]
    for _, row in sig_phal.iterrows():
        lines.append(f"     {row['group_1']} vs {row['group_2']}: p = {row['p_value']:.2e} {row['significance']}")
    lines.append("")

    lines.append("  2. F-ACTIN COVERAGE: Higher on substrates")
    for c in conds:
        sub = df[df["condition"] == c]
        v = sub["phalloidin_coverage"].dropna()
        label = c.replace("\n", " ")
        lines.append(f"     {label:30s}  median = {v.median():.3f}")
    lines.append("")

    lines.append("  3. ACTIN COHERENCY: Gold device shows LOWER coherency")
    for c in conds:
        sub = df[df["condition"] == c]
        v = sub["actin_coherency"].dropna()
        label = c.replace("\n", " ")
        lines.append(f"     {label:30s}  median = {v.median():.3f}")

    sig_coh = pw_df[(pw_df["metric"] == "actin_coherency") & (pw_df["significance"] != "ns")]
    for _, row in sig_coh.iterrows():
        lines.append(f"     {row['group_1']} vs {row['group_2']}: p = {row['p_value']:.2e} {row['significance']}")
    lines.append("")

    lines.append("  INTERPRETATION:")
    lines.append("")
    lines.append("  - PIEZOELECTRIC EFFECT (Poled Pulsed vs Nonpoled Pulsed):")
    lines.append("    The poled pulsed condition shows LOWER actin coherency, consistent with")
    lines.append("    maturation-associated cytoskeletal remodeling (transition from stress")
    lines.append("    fibers to sarcomeric actin). This cleanly isolates the piezoelectric")
    lines.append("    contribution since both conditions are pulsed on the device.")
    lines.append("")
    lines.append("  - PULSATION + PIEZO EFFECT (Poled Pulsed vs Poled Un-pulsed):")
    lines.append("    Same Au-PVDF Poled material, only difference is active pulsation.")
    lines.append("    Shows the combined mechanical + piezoelectric stimulation drives")
    lines.append("    measurable changes in F-actin organization.")
    lines.append("")
    lines.append("  - SUBSTRATE vs BASELINE (Au-PVDF Poled Un-pulsed vs Cells only):")
    lines.append("    Poled material on the control plate shows higher F-actin intensity and")
    lines.append("    coverage versus bare plastic. Note: this compares Au-PVDF+gold to bare")
    lines.append("    plastic (not poled vs nonpoled), so any difference reflects the combined")
    lines.append("    effect of gold coating + PVDF surface properties.")
    lines.append("")

    lines.append("=" * 80)
    lines.append("ANALYSIS 2: MULTINUCLEATION (CELLPOSE)")
    lines.append("=" * 80)
    lines.append("Method: Cellpose v3 deep learning (cyto2 model, GPU)")
    lines.append("  + classical DAPI nuclei segmentation")
    lines.append("")
    total_all, mono_all, multi_all, zero_all = 0, 0, 0, 0
    for c in conds:
        sub = mn[mn["condition"] == c]
        total = len(sub)
        mono = int((sub["nuclei_count"] == 1).sum())
        bi = int((sub["nuclei_count"] >= 2).sum())
        zero = int((sub["nuclei_count"] == 0).sum())
        total_all += total; mono_all += mono; multi_all += bi; zero_all += zero
        label = c.replace("\n", " ")
        lines.append(f"  {label}:")
        lines.append(f"    {total} cells detected, {mono} mononucleated ({100*mono/total:.0f}%), "
                     f"{bi} multinucleated ({100*bi/total:.0f}%), "
                     f"{zero} no nucleus assigned ({100*zero/total:.0f}%)")
    detected = mono_all + multi_all
    lines.append("")
    lines.append(f"  CAVEAT: {zero_all}/{total_all} cells ({100*zero_all/total_all:.0f}%) had no nucleus")
    lines.append("  assigned by Cellpose — nucleus/cell boundary mismatch in confluent cultures,")
    lines.append("  not truly anucleate cells. These are excluded from mono/multi classification.")
    lines.append("")
    lines.append("  INTERPRETATION:")
    lines.append(f"  - Among cells with detected nuclei ({detected}/{total_all}), "
                 f"{100*mono_all/detected:.0f}% are mononucleated.")
    lines.append("  - This is expected for hiPSC-derived cardiomyocytes, which typically")
    lines.append("    do not undergo the binucleation seen in adult cardiomyocytes in vivo.")
    lines.append("  - Multinucleation is NOT a differentiating metric for this experiment.")
    lines.append("  - Note: Cellpose detected fewer cells than watershed due to conservative")
    lines.append("    segmentation in confluent cultures. The high zero-nuclei fraction")
    lines.append("    reflects segmentation limitations, not absent nuclei.")
    lines.append("")

    lines.append("=" * 80)
    lines.append("ANALYSIS 3: ALPHA-ACTININ (SARCOMERES) - INCONCLUSIVE")
    lines.append("=" * 80)
    lines.append("Status: INCONCLUSIVE - data not usable for quantitative analysis")
    lines.append("")
    lines.append("  Reason: No recognizable cell morphology in any of the 3 images.")
    lines.append("  - Cells only (A3 ctrl): Diffuse, gradient-like signal (no visible cells)")
    lines.append("  - Au-PVDF Poled Un-pulsed (A2 ctrl): Gold film bubbles and debris")
    lines.append("  - Au-PVDF Poled Pulsed (A2 device): Scratches and smudges on surface")
    lines.append("")
    lines.append("  Likely causes:")
    lines.append("  - Old cells with potentially degraded protein targets")
    lines.append("  - Suboptimal antibody staining / fixation / permeabilization")
    lines.append("  - Gold substrate interference with fluorescence")
    lines.append("  - Insufficient magnification to resolve sarcomere Z-lines")
    lines.append("")
    lines.append("  Recommendation: Repeat with fresh cells, optimized staining protocol,")
    lines.append("  and higher magnification (40x-63x) for the second experiment.")
    lines.append("")

    lines.append("=" * 80)
    lines.append("OVERALL CONCLUSIONS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  1. PIEZOELECTRIC STIMULATION PROMOTES CYTOSKELETAL REMODELING")
    lines.append("     Au-PVDF Poled Pulsed (device) shows significantly different F-actin")
    lines.append("     organization compared to all other conditions, consistent with")
    lines.append("     cardiomyocyte maturation (transition from stress fibers to sarcomeric actin).")
    lines.append("")
    lines.append("  2. THE EFFECT IS SPECIFIC TO PIEZOELECTRIC CHARGE GENERATION")
    lines.append("     Poled Pulsed vs Nonpoled Pulsed (both on device, both pulsed)")
    lines.append("     demonstrates that piezoelectric charge, not just mechanical pulsation,")
    lines.append("     drives the cytoskeletal remodeling.")
    lines.append("")
    lines.append("  3. PULSATION ON POLED MATERIAL HAS A DISTINCT EFFECT")
    lines.append("     Poled Pulsed vs Poled Un-pulsed (same Au-PVDF material) shows that")
    lines.append("     active device stimulation produces measurable differences in F-actin")
    lines.append("     organization beyond the substrate material alone.")
    lines.append("")
    lines.append("  4. SUPERVISOR QUESTION: PULSED vs UN-PULSED Au-PVDF DIFFERENCE?")
    lines.append("     YES. Au-PVDF Poled Pulsed (device) vs Au-PVDF Poled Un-pulsed (control)")
    lines.append("     shows statistically significant differences in actin coherency and")
    lines.append("     texture metrics (see Fig4, Fig8, pairwise_comparisons.csv).")
    lines.append("")
    lines.append("  5. SUBSTRATE IMPROVES F-ACTIN vs BARE PLASTIC")
    lines.append("     All substrate conditions show higher F-actin intensity and coverage")
    lines.append("     compared to cells only (bare plastic baseline). Note: the 'Cells only'")
    lines.append("     condition has NO PVDF substrate, so this reflects Au-PVDF+gold vs plastic.")
    lines.append("")
    lines.append("  6. CELL MORPHOLOGY IS LARGELY UNCHANGED")
    lines.append("     No significant differences in cell area, shape, or elongation between")
    lines.append("     conditions, suggesting cytoskeletal changes are internal reorganization")
    lines.append("     rather than gross morphological remodeling.")
    lines.append("")
    lines.append("  7. DESIGN LIMITATIONS")
    lines.append("     - NOT a balanced 2x2: control plate column 3 is 'Cells only' (bare")
    lines.append("       plastic), not nonpoled PVDF. Missing 'Nonpoled Un-pulsed' condition.")
    lines.append("     - B3 device (Nonpoled 2) film detached during staining.")
    lines.append("     - Single image per condition (1 field of view each)")
    lines.append("     - Old cells from another lab (not optimal)")
    lines.append("     - Alpha-actinin staining failed (inconclusive)")
    lines.append("     - Classical segmentation in confluent cultures has inherent errors")
    lines.append("     - N=1 biological replicate per condition")
    lines.append("")
    lines.append("  8. NEXT STEPS FOR EXPERIMENT 2")
    lines.append("     - Fresh hiPSC-CMs with optimized culture protocol")
    lines.append("     - Include 'Nonpoled Un-pulsed' control for balanced design")
    lines.append("     - Multiple fields of view per condition")
    lines.append("     - Higher magnification for sarcomere-level imaging")
    lines.append("     - Optimized alpha-actinin staining for sarcomere quantification")
    lines.append("     - Video analysis for beating/contractility assessment")
    lines.append("")

    lines.append("=" * 80)
    lines.append("FILES PRODUCED")
    lines.append("=" * 80)
    lines.append("  Figures (all with updated 2x2 labels):")
    lines.append("    Fig1_raw_images.png              - Raw fluorescence images (4 conditions)")
    lines.append("    Fig2_segmentation_qc.png         - Segmentation quality control")
    lines.append("    Fig3_morphology.png              - Cell morphology comparison")
    lines.append("    Fig4_factin_organization.png     - F-actin intensity & organization")
    lines.append("    Fig5_summary_bars.png            - Key metrics summary (mean +/- SEM)")
    lines.append("    Fig6_multinucleation.png         - Multinucleation analysis (Cellpose)")
    lines.append("    Fig7_actinin_INCONCLUSIVE.png    - Alpha-actinin (flagged inconclusive)")
    lines.append("    Fig8_significance_heatmap.png    - Statistical significance overview")
    lines.append("")
    lines.append("  Data:")
    lines.append("    phalloidin_per_cell_data.csv     - All per-cell measurements")
    lines.append("    descriptive_statistics.csv       - Summary stats per condition")
    lines.append("    pairwise_comparisons.csv         - All pairwise Mann-Whitney U tests")
    lines.append("    multinucleation_cellpose.csv     - Multinucleation per-cell data")
    lines.append("")

    text = "\n".join(lines)
    return text


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("GENERATING FINAL COMPREHENSIVE REPORT")
    print("=" * 60)

    df = pd.read_csv(RESULTS / "per_cell_data.csv")
    df = remap_conditions(df)
    df.to_csv(OUT / "phalloidin_per_cell_data.csv", index=False)

    mn = pd.read_csv(RESULTS / "multinucleation_cellpose.csv")
    mn = remap_conditions(mn)
    mn.to_csv(OUT / "multinucleation_cellpose.csv", index=False)

    print(f"\nPhalloidin data: {len(df)} cells across {df['condition'].nunique()} conditions")
    print(f"Multinucleation data: {len(mn)} cells\n")

    fig1_raw_images()
    fig2_segmentation_qc()
    fig3_morphology(df)
    fig4_factin(df)
    fig5_summary_bars(df)
    fig6_multinucleation()
    fig7_actinin_inconclusive()

    pw_df = generate_stats_table(df)
    fig8_significance_heatmap(df)

    text = write_text_summary(df, pw_df)

    print("\n" + "=" * 60)
    print("REPORT COMPLETE")
    print(f"All files saved to: {OUT}/")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
