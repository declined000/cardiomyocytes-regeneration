"""
Phalloidin + DAPI Deep Analysis Pipeline v2
============================================
Cardiomyocytes on piezoelectric substrates.

Segmentation: DAPI nuclei as seeds → watershed to phalloidin cell boundaries
(CellProfiler-equivalent approach).

Conditions:
  1. Cells only (control plate, no substrate)
  2. Gold control (control plate, gold substrate, no stimulation)
  3. Gold device (on-device, gold-poled piezo + mechanical stimulation)
  4. Nonpoled device (on-device, nonpoled piezo + mechanical stimulation)

Metrics per cell:
  Morphology:    area, perimeter, aspect ratio, circularity, elongation, solidity
  Intensity:     mean, integrated, std phalloidin; coverage
  Organization:  actin coherency (structure tensor), Haralick texture energy/homogeneity
  Nuclear:       nuclei per cell, nuclear area, nuclear aspect ratio
"""

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from skimage import filters, morphology, measure, feature, segmentation, exposure
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from scipy.stats import mannwhitneyu, kruskal
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path("Cardiomyocytes")
OUTPUT_DIR = Path("results_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

CONDITIONS = {
    "Cells only\n(control)": {
        "phalloidin": DATA_DIR / "control-b3-non-phalloidin.tif",
        "dapi": DATA_DIR / "control-b3-non-dapi.tif",
    },
    "Gold\n(control plate)": {
        "phalloidin": DATA_DIR / "control-gold2-b2-phalloidin.tif",
        "dapi": DATA_DIR / "control-gold2-b2-dapi.tif",
    },
    "Gold\n(device)": {
        "phalloidin": DATA_DIR / "device-gold2-phalloidin-b2-image.tif",
        "dapi": DATA_DIR / "device-gold2-dapi-b2-image.tif",
    },
    "Nonpoled\n(device)": {
        "phalloidin": DATA_DIR / "device-non-phalloidin-a3-image.tif",
        "dapi": DATA_DIR / "device-non-dapi-a3-image.tif",
    },
}

CONDITION_COLORS = {
    "Cells only\n(control)": "#7fbfff",
    "Gold\n(control plate)": "#ffd966",
    "Gold\n(device)": "#ff9966",
    "Nonpoled\n(device)": "#99cc99",
}


def normalize_16bit(img):
    p_low, p_high = np.percentile(img, (1, 99.5))
    return np.clip((img.astype(np.float64) - p_low) / (p_high - p_low), 0, 1)


def segment_nuclei(dapi_norm):
    """Segment nuclei from DAPI with Otsu + watershed."""
    smoothed = filters.gaussian(dapi_norm, sigma=2)
    thresh = filters.threshold_otsu(smoothed)
    binary = smoothed > thresh
    binary = morphology.remove_small_objects(binary, min_size=50)
    binary = binary_fill_holes(binary)
    binary = morphology.binary_closing(binary, morphology.disk(2))

    distance = ndimage.distance_transform_edt(binary)
    coords = feature.peak_local_max(distance, min_distance=8, labels=binary)
    peak_mask = np.zeros_like(distance, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers = measure.label(peak_mask)
    nuclei_labels = segmentation.watershed(-distance, markers, mask=binary)

    return nuclei_labels


def segment_cells_watershed(phalloidin_norm, nuclei_labels):
    """
    Segment cells: use phalloidin to define cell regions, DAPI nuclei as seeds.
    Watershed expands from nuclei to the actual phalloidin-defined cell edges.
    """
    smoothed = filters.gaussian(phalloidin_norm, sigma=1.5)

    thresh = filters.threshold_otsu(smoothed)
    cell_mask = smoothed > (thresh * 0.5)
    cell_mask = morphology.binary_dilation(cell_mask, morphology.disk(3))
    cell_mask = binary_fill_holes(cell_mask)
    cell_mask = morphology.remove_small_objects(cell_mask, min_size=100)

    gradient = filters.sobel(smoothed)
    cell_labels = segmentation.watershed(gradient, nuclei_labels, mask=cell_mask)

    cell_labels = morphology.remove_small_objects(cell_labels, min_size=80)

    return cell_labels


def compute_coherency_map(phalloidin_norm, sigma=2):
    """Structure tensor coherency: 0=isotropic, 1=perfectly aligned fibers."""
    S = feature.structure_tensor(phalloidin_norm, sigma=sigma)
    eigs = feature.structure_tensor_eigenvalues(S)
    l1, l2 = eigs[0], eigs[1]
    denom = l1 + l2
    coherency = np.zeros_like(l1)
    mask = denom > 0
    coherency[mask] = (l1[mask] - l2[mask]) / denom[mask]
    return coherency


def compute_haralick_per_cell(phalloidin_norm, cell_labels):
    """Compute Haralick texture features (energy, homogeneity, contrast, correlation) per cell."""
    img_uint8 = (np.clip(phalloidin_norm, 0, 1) * 255).astype(np.uint8)
    haralick_results = {}

    for region in measure.regionprops(cell_labels):
        label = region.label
        minr, minc, maxr, maxc = region.bbox
        patch = img_uint8[minr:maxr, minc:maxc].copy()
        mask_patch = cell_labels[minr:maxr, minc:maxc] == label

        patch[~mask_patch] = 0

        if patch.shape[0] < 3 or patch.shape[1] < 3:
            haralick_results[label] = {
                "haralick_energy": np.nan,
                "haralick_homogeneity": np.nan,
                "haralick_contrast": np.nan,
                "haralick_correlation": np.nan,
            }
            continue

        try:
            distances = [1, 3]
            angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
            glcm = graycomatrix(
                patch, distances=distances, angles=angles,
                levels=256, symmetric=True, normed=True,
            )
            haralick_results[label] = {
                "haralick_energy": float(np.mean(graycoprops(glcm, "energy"))),
                "haralick_homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
                "haralick_contrast": float(np.mean(graycoprops(glcm, "contrast"))),
                "haralick_correlation": float(np.mean(graycoprops(glcm, "correlation"))),
            }
        except Exception:
            haralick_results[label] = {
                "haralick_energy": np.nan,
                "haralick_homogeneity": np.nan,
                "haralick_contrast": np.nan,
                "haralick_correlation": np.nan,
            }

    return haralick_results


def analyze_condition(condition_name, phalloidin_path, dapi_path):
    """Full analysis for one condition."""
    phalloidin = tifffile.imread(phalloidin_path)
    dapi = tifffile.imread(dapi_path)

    phalloidin_norm = normalize_16bit(phalloidin)
    dapi_norm = normalize_16bit(dapi)

    nuclei_labels = segment_nuclei(dapi_norm)
    cell_labels = segment_cells_watershed(phalloidin_norm, nuclei_labels)

    coherency_map = compute_coherency_map(phalloidin_norm)
    haralick_per_cell = compute_haralick_per_cell(phalloidin_norm, cell_labels)

    phal_bg_vals = phalloidin_norm[cell_labels == 0]
    phal_thresh = np.percentile(phal_bg_vals, 75) + 0.1 if len(phal_bg_vals) > 0 else 0.15

    rows = []
    for cell_region in measure.regionprops(cell_labels, intensity_image=phalloidin_norm):
        label = cell_region.label
        cell_coords = cell_region.coords
        phal_vals = phalloidin_norm[cell_coords[:, 0], cell_coords[:, 1]]
        coh_vals = coherency_map[cell_coords[:, 0], cell_coords[:, 1]]

        cell_mask_single = cell_labels == label
        nuc_in_cell = np.unique(nuclei_labels[cell_mask_single])
        nuc_in_cell = nuc_in_cell[nuc_in_cell > 0]
        n_nuclei = len(nuc_in_cell)

        nuc_areas = []
        nuc_ars = []
        for nuc_label in nuc_in_cell:
            nuc_mask = nuclei_labels == nuc_label
            nuc_props = measure.regionprops(nuc_mask.astype(int))
            if nuc_props:
                nuc_areas.append(nuc_props[0].area)
                if nuc_props[0].minor_axis_length > 0:
                    nuc_ars.append(nuc_props[0].major_axis_length / nuc_props[0].minor_axis_length)

        area = cell_region.area
        perimeter = cell_region.perimeter
        major = cell_region.major_axis_length
        minor = cell_region.minor_axis_length
        aspect_ratio = major / minor if minor > 0 else np.nan
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else np.nan
        elongation = 1 - (minor / major) if major > 0 else np.nan
        solidity = cell_region.solidity

        h = haralick_per_cell.get(label, {})

        rows.append({
            "condition": condition_name,
            "cell_label": label,
            "cell_area_px": area,
            "cell_perimeter_px": perimeter,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "elongation": elongation,
            "solidity": solidity,
            "mean_phalloidin_intensity": float(np.mean(phal_vals)),
            "integrated_phalloidin_intensity": float(np.sum(phal_vals)),
            "std_phalloidin_intensity": float(np.std(phal_vals)),
            "phalloidin_coverage": float(np.mean(phal_vals > phal_thresh)),
            "actin_coherency": float(np.mean(coh_vals)),
            "haralick_energy": h.get("haralick_energy", np.nan),
            "haralick_homogeneity": h.get("haralick_homogeneity", np.nan),
            "haralick_contrast": h.get("haralick_contrast", np.nan),
            "haralick_correlation": h.get("haralick_correlation", np.nan),
            "nuclei_per_cell": n_nuclei,
            "mean_nuclear_area_px": float(np.mean(nuc_areas)) if nuc_areas else np.nan,
            "mean_nuclear_aspect_ratio": float(np.mean(nuc_ars)) if nuc_ars else np.nan,
        })

    df = pd.DataFrame(rows)
    n_nuclei_total = nuclei_labels.max()
    n_cells = cell_labels.max()
    print(f"  {condition_name.replace(chr(10), ' '):25s} | nuclei: {n_nuclei_total:3d} | cells (segmented): {n_cells:3d} | cells (measured): {len(df)}")

    return df, {
        "phalloidin": phalloidin,
        "dapi": dapi,
        "phalloidin_norm": phalloidin_norm,
        "dapi_norm": dapi_norm,
        "nuclei_labels": nuclei_labels,
        "cell_labels": cell_labels,
        "coherency_map": coherency_map,
    }


# ========== PLOTTING FUNCTIONS ==========

def plot_segmentation_overview(all_data, output_path):
    n = len(all_data)
    fig, axes = plt.subplots(n, 5, figsize=(25, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (cond, data) in enumerate(all_data.items()):
        axes[i, 0].imshow(data["dapi_norm"], cmap="Blues")
        axes[i, 0].set_title(f"DAPI\n{cond}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(data["phalloidin_norm"], cmap="Greens")
        axes[i, 1].set_title(f"Phalloidin\n{cond}", fontsize=9)
        axes[i, 1].axis("off")

        overlay = np.zeros((*data["dapi_norm"].shape, 3))
        overlay[:, :, 2] = data["dapi_norm"]
        overlay[:, :, 1] = data["phalloidin_norm"]
        axes[i, 2].imshow(np.clip(overlay, 0, 1))
        axes[i, 2].set_title(f"Merge\n{cond}", fontsize=9)
        axes[i, 2].axis("off")

        seg_rgb = np.zeros((*data["phalloidin_norm"].shape, 3))
        seg_rgb[:, :, 1] = data["phalloidin_norm"] * 0.8
        boundaries = segmentation.find_boundaries(data["cell_labels"], mode="thick")
        seg_rgb[boundaries] = [1, 0, 0]
        nuc_boundaries = segmentation.find_boundaries(data["nuclei_labels"], mode="thick")
        seg_rgb[nuc_boundaries] = [0, 0.4, 1]
        axes[i, 3].imshow(np.clip(seg_rgb, 0, 1))
        axes[i, 3].set_title(f"Cell (red) + Nuclei (blue)\n{cond}", fontsize=9)
        axes[i, 3].axis("off")

        axes[i, 4].imshow(data["coherency_map"], cmap="hot", vmin=0, vmax=1)
        axes[i, 4].set_title(f"Actin Coherency\n{cond}", fontsize=9)
        axes[i, 4].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_morphology_boxplots(df, output_path):
    metrics = [
        ("cell_area_px", "Cell Area (px)"),
        ("aspect_ratio", "Aspect Ratio (L/W)"),
        ("circularity", "Circularity"),
        ("elongation", "Elongation"),
        ("solidity", "Solidity"),
        ("cell_perimeter_px", "Perimeter (px)"),
    ]
    cond_order = list(CONDITIONS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        data_list = [df[df["condition"] == c][col].dropna().values for c in cond_order]
        bp = ax.boxplot(data_list, labels=[c for c in cond_order],
                        patch_artist=True, widths=0.6, showfliers=False)
        for j, (patch, c) in enumerate(zip(bp["boxes"], cond_order)):
            patch.set_facecolor(CONDITION_COLORS[c])
            patch.set_alpha(0.7)
        for j, d in enumerate(data_list):
            x = np.random.normal(j + 1, 0.04, size=len(d))
            ax.scatter(x, d, alpha=0.4, s=12, color="black", zorder=5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)

    plt.suptitle("Cell Morphology: Gold vs Nonpoled vs Cells Only", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_intensity_boxplots(df, output_path):
    metrics = [
        ("mean_phalloidin_intensity", "Mean Phalloidin Intensity"),
        ("integrated_phalloidin_intensity", "Integrated Intensity (total F-actin)"),
        ("std_phalloidin_intensity", "Intensity Std Dev"),
        ("phalloidin_coverage", "Phalloidin Coverage"),
        ("actin_coherency", "Actin Fiber Coherency"),
        ("haralick_homogeneity", "Haralick Homogeneity"),
    ]
    cond_order = list(CONDITIONS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        data_list = [df[df["condition"] == c][col].dropna().values for c in cond_order]
        bp = ax.boxplot(data_list, labels=[c for c in cond_order],
                        patch_artist=True, widths=0.6, showfliers=False)
        for j, (patch, c) in enumerate(zip(bp["boxes"], cond_order)):
            patch.set_facecolor(CONDITION_COLORS[c])
            patch.set_alpha(0.7)
        for j, d in enumerate(data_list):
            x = np.random.normal(j + 1, 0.04, size=len(d))
            ax.scatter(x, d, alpha=0.4, s=12, color="black", zorder=5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)

    plt.suptitle("F-Actin Intensity & Organization", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_nuclear_boxplots(df, output_path):
    metrics = [
        ("nuclei_per_cell", "Nuclei per Cell"),
        ("mean_nuclear_area_px", "Nuclear Area (px)"),
        ("mean_nuclear_aspect_ratio", "Nuclear Aspect Ratio"),
    ]
    cond_order = list(CONDITIONS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        data_list = [df[df["condition"] == c][col].dropna().values for c in cond_order]
        bp = ax.boxplot(data_list, labels=[c for c in cond_order],
                        patch_artist=True, widths=0.6, showfliers=False)
        for j, (patch, c) in enumerate(zip(bp["boxes"], cond_order)):
            patch.set_facecolor(CONDITION_COLORS[c])
            patch.set_alpha(0.7)
        for j, d in enumerate(data_list):
            x = np.random.normal(j + 1, 0.04, size=len(d))
            ax.scatter(x, d, alpha=0.4, s=12, color="black", zorder=5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)

    plt.suptitle("Nuclear Metrics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_summary_heatmap(df, output_path):
    """Heatmap of z-scored means across conditions for all metrics."""
    metrics = [
        "cell_area_px", "aspect_ratio", "circularity", "elongation", "solidity",
        "mean_phalloidin_intensity", "integrated_phalloidin_intensity",
        "phalloidin_coverage", "actin_coherency",
        "haralick_energy", "haralick_homogeneity",
        "nuclei_per_cell", "mean_nuclear_area_px", "mean_nuclear_aspect_ratio",
    ]
    cond_order = list(CONDITIONS.keys())

    means = []
    for c in cond_order:
        sub = df[df["condition"] == c]
        row = [sub[m].mean() for m in metrics]
        means.append(row)

    means_arr = np.array(means)
    col_means = np.nanmean(means_arr, axis=0)
    col_stds = np.nanstd(means_arr, axis=0)
    col_stds[col_stds == 0] = 1
    z_scores = (means_arr - col_means) / col_stds

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(z_scores, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(cond_order)))
    ax.set_yticklabels([c.replace("\n", " ") for c in cond_order], fontsize=10)

    for i in range(len(cond_order)):
        for j in range(len(metrics)):
            val = z_scores[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > 1.2 else "black")

    plt.colorbar(im, ax=ax, label="Z-score (relative to mean across conditions)")
    ax.set_title("Summary Heatmap: All Metrics (Z-scored)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def compute_statistics(df):
    """Kruskal-Wallis + pairwise Mann-Whitney tests."""
    cond_order = list(CONDITIONS.keys())
    metrics = [
        "cell_area_px", "aspect_ratio", "circularity", "elongation", "solidity",
        "mean_phalloidin_intensity", "integrated_phalloidin_intensity",
        "phalloidin_coverage", "actin_coherency",
        "haralick_energy", "haralick_homogeneity", "haralick_contrast", "haralick_correlation",
        "nuclei_per_cell", "mean_nuclear_area_px", "mean_nuclear_aspect_ratio",
    ]

    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS (mean ± SD)")
    print("=" * 100)

    summary_rows = []
    for cond in cond_order:
        sub = df[df["condition"] == cond]
        row = {"Condition": cond.replace("\n", " "), "N": len(sub)}
        for m in metrics:
            vals = sub[m].dropna()
            row[m] = f"{vals.mean():.3f} ± {vals.std():.3f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("KRUSKAL-WALLIS TEST (overall difference across all 4 conditions)")
    print("=" * 100)

    kw_rows = []
    for m in metrics:
        groups = [df[df["condition"] == c][m].dropna().values for c in cond_order]
        groups = [g for g in groups if len(g) > 0]
        all_same = all(np.std(g) == 0 for g in groups)
        if len(groups) >= 2 and not all_same:
            stat, p = kruskal(*groups)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {m:45s}  H={stat:8.2f}  p={p:.6f}  {sig}")
            kw_rows.append({"metric": m, "H_statistic": stat, "p_value": p, "significance": sig})
        elif all_same:
            print(f"  {m:45s}  (constant across conditions — skipped)")
            kw_rows.append({"metric": m, "H_statistic": np.nan, "p_value": np.nan, "significance": "constant"})

    print("\n" + "=" * 100)
    print("PAIRWISE COMPARISONS (Mann-Whitney U, two-sided)")
    print("=" * 100)

    comparisons = [
        ("Gold\n(device)", "Nonpoled\n(device)", "Gold device vs Nonpoled device"),
        ("Gold\n(device)", "Gold\n(control plate)", "Gold device vs Gold control plate"),
        ("Gold\n(device)", "Cells only\n(control)", "Gold device vs Cells only"),
        ("Nonpoled\n(device)", "Cells only\n(control)", "Nonpoled device vs Cells only"),
        ("Gold\n(control plate)", "Cells only\n(control)", "Gold control plate vs Cells only"),
        ("Nonpoled\n(device)", "Gold\n(control plate)", "Nonpoled device vs Gold control plate"),
    ]

    pairwise_rows = []
    for c1, c2, label in comparisons:
        print(f"\n  {label}:")
        for m in metrics:
            v1 = df[df["condition"] == c1][m].dropna()
            v2 = df[df["condition"] == c2][m].dropna()
            if len(v1) > 0 and len(v2) > 0 and not (v1.std() == 0 and v2.std() == 0):
                stat, p = mannwhitneyu(v1, v2, alternative="two-sided")
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                if p < 0.05:
                    print(f"    {m:45s}  p={p:.6f}  {sig}  (median: {v1.median():.3f} vs {v2.median():.3f})")
                pairwise_rows.append({
                    "comparison": label, "metric": m,
                    "p_value": p, "significance": sig,
                    "median_1": v1.median(), "median_2": v2.median(),
                })

    return pd.DataFrame(kw_rows), pd.DataFrame(pairwise_rows)


def main():
    print("=" * 70)
    print("PHALLOIDIN IMAGE ANALYSIS v2 — DEEP PIPELINE")
    print("=" * 70)

    all_dfs = []
    all_data = {}

    print("\n[1/6] Segmenting and analyzing conditions...")
    for cond_name, paths in CONDITIONS.items():
        df_cond, data = analyze_condition(cond_name, paths["phalloidin"], paths["dapi"])
        all_dfs.append(df_cond)
        all_data[cond_name] = data

    df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n[2/6] Segmentation overview...")
    plot_segmentation_overview(all_data, OUTPUT_DIR / "01_segmentation_overview.png")

    print(f"\n[3/6] Morphology plots...")
    plot_morphology_boxplots(df, OUTPUT_DIR / "02_morphology_boxplots.png")

    print(f"\n[4/6] Intensity & organization plots...")
    plot_intensity_boxplots(df, OUTPUT_DIR / "03_intensity_organization_boxplots.png")

    print(f"\n[5/6] Nuclear metric plots...")
    plot_nuclear_boxplots(df, OUTPUT_DIR / "04_nuclear_boxplots.png")

    print(f"\n[6/6] Summary heatmap + statistics...")
    plot_summary_heatmap(df, OUTPUT_DIR / "05_summary_heatmap.png")
    kw_df, pairwise_df = compute_statistics(df)

    df.to_csv(OUTPUT_DIR / "per_cell_data.csv", index=False)
    kw_df.to_csv(OUTPUT_DIR / "kruskal_wallis_tests.csv", index=False)
    pairwise_df.to_csv(OUTPUT_DIR / "pairwise_comparisons.csv", index=False)

    print(f"\n  Exported: {OUTPUT_DIR / 'per_cell_data.csv'}")
    print(f"  Exported: {OUTPUT_DIR / 'kruskal_wallis_tests.csv'}")
    print(f"  Exported: {OUTPUT_DIR / 'pairwise_comparisons.csv'}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE — all results in results_v2/")
    print("=" * 70)


if __name__ == "__main__":
    main()
