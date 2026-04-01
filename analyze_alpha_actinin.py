"""
Alpha-actinin + DAPI Analysis Pipeline
=======================================
Quantifies sarcomere organization in cardiomyocytes.

Alpha-actinin marks Z-lines in sarcomeres. Well-organized sarcomeres show
periodic, parallel Z-line patterns. Key metrics:
  - Sarcomere periodicity (FFT peak detection)
  - Z-line orientation coherency (structure tensor)
  - Haralick texture features (regularity, contrast)
  - Standard morphology and intensity metrics

Conditions (control plate only for gold vs non comparison, plus gold device):
  1. Gold (control plate) — A2
  2. Cells only / Nonpoled (control plate) — A3
  3. Gold (device) — A2
"""

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import filters, morphology, measure, feature, segmentation
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from scipy.stats import mannwhitneyu
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path("Cardiomyocytes")
OUTPUT_DIR = Path("results_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

CONDITIONS = {
    "Cells only\n(control)": {
        "actinin": DATA_DIR / "control-a3-alpha-non.tif",
        "dapi": DATA_DIR / "comtrol-A3-non-dapi.tif",
    },
    "Gold\n(control plate)": {
        "actinin": DATA_DIR / "control-gold1-a2-alphaact.tif",
        "dapi": DATA_DIR / "control-gold1-a2-dapi.tif",
    },
    "Gold\n(device)": {
        "actinin": DATA_DIR / "device-gold1-alphaactinin-A2-image.tif",
        "dapi": DATA_DIR / "device-gold1-dapi-A2-image.tif",
    },
}

CONDITION_COLORS = {
    "Cells only\n(control)": "#7fbfff",
    "Gold\n(control plate)": "#ffd966",
    "Gold\n(device)": "#ff9966",
}


def normalize_16bit(img):
    p_low, p_high = np.percentile(img, (1, 99.5))
    return np.clip((img.astype(np.float64) - p_low) / (p_high - p_low), 0, 1)


def segment_nuclei(dapi_norm):
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
    return segmentation.watershed(-distance, markers, mask=binary)


def segment_cells_watershed(actinin_norm, nuclei_labels):
    smoothed = filters.gaussian(actinin_norm, sigma=1.5)
    thresh = filters.threshold_otsu(smoothed)
    cell_mask = smoothed > (thresh * 0.5)
    cell_mask = morphology.binary_dilation(cell_mask, morphology.disk(3))
    cell_mask = binary_fill_holes(cell_mask)
    cell_mask = morphology.remove_small_objects(cell_mask, min_size=100)
    gradient = filters.sobel(smoothed)
    cell_labels = segmentation.watershed(gradient, nuclei_labels, mask=cell_mask)
    return morphology.remove_small_objects(cell_labels, min_size=80)


def compute_fft_periodicity(patch, min_period=3, max_period=30):
    """
    Detect sarcomere periodicity in a cell patch using 2D FFT.
    Returns dominant period (in pixels) and FFT peak strength.
    """
    if patch.shape[0] < 8 or patch.shape[1] < 8:
        return np.nan, np.nan

    windowed = patch * np.outer(
        np.hanning(patch.shape[0]), np.hanning(patch.shape[1])
    )
    fft2 = np.fft.fft2(windowed)
    power = np.abs(np.fft.fftshift(fft2)) ** 2

    cy, cx = power.shape[0] // 2, power.shape[1] // 2
    y, x = np.ogrid[:power.shape[0], :power.shape[1]]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = min(cy, cx)
    radial_profile = np.zeros(max_r)
    for i in range(max_r):
        ring = power[r == i]
        if len(ring) > 0:
            radial_profile[i] = ring.mean()

    freq_min = max(1, int(min(patch.shape) / max_period))
    freq_max = min(max_r - 1, int(min(patch.shape) / min_period))

    if freq_max <= freq_min:
        return np.nan, np.nan

    search_range = radial_profile[freq_min:freq_max + 1]
    if len(search_range) == 0:
        return np.nan, np.nan

    peak_idx = np.argmax(search_range) + freq_min
    if peak_idx == 0:
        return np.nan, np.nan

    dominant_period = min(patch.shape) / peak_idx
    peak_strength = search_range.max() / (np.mean(search_range) + 1e-10)

    return dominant_period, peak_strength


def compute_coherency(actinin_norm, sigma=2):
    S = feature.structure_tensor(actinin_norm, sigma=sigma)
    eigs = feature.structure_tensor_eigenvalues(S)
    l1, l2 = eigs[0], eigs[1]
    denom = l1 + l2
    coherency = np.zeros_like(l1)
    mask = denom > 0
    coherency[mask] = (l1[mask] - l2[mask]) / denom[mask]
    return coherency


def compute_haralick(patch_uint8):
    if patch_uint8.shape[0] < 3 or patch_uint8.shape[1] < 3:
        return {}
    try:
        glcm = graycomatrix(
            patch_uint8, distances=[1, 3],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256, symmetric=True, normed=True,
        )
        return {
            "haralick_energy": float(np.mean(graycoprops(glcm, "energy"))),
            "haralick_homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
            "haralick_contrast": float(np.mean(graycoprops(glcm, "contrast"))),
            "haralick_correlation": float(np.mean(graycoprops(glcm, "correlation"))),
        }
    except Exception:
        return {}


def analyze_condition(cond_name, actinin_path, dapi_path):
    actinin = tifffile.imread(actinin_path)
    dapi = tifffile.imread(dapi_path)
    actinin_norm = normalize_16bit(actinin)
    dapi_norm = normalize_16bit(dapi)

    nuclei_labels = segment_nuclei(dapi_norm)
    cell_labels = segment_cells_watershed(actinin_norm, nuclei_labels)
    coherency_map = compute_coherency(actinin_norm)
    actinin_uint8 = (np.clip(actinin_norm, 0, 1) * 255).astype(np.uint8)

    rows = []
    for region in measure.regionprops(cell_labels, intensity_image=actinin_norm):
        label = region.label
        coords = region.coords
        act_vals = actinin_norm[coords[:, 0], coords[:, 1]]
        coh_vals = coherency_map[coords[:, 0], coords[:, 1]]

        minr, minc, maxr, maxc = region.bbox
        patch = actinin_norm[minr:maxr, minc:maxc].copy()
        patch_mask = cell_labels[minr:maxr, minc:maxc] == label
        patch[~patch_mask] = 0

        period, fft_strength = compute_fft_periodicity(patch)

        patch_u8 = actinin_uint8[minr:maxr, minc:maxc].copy()
        patch_u8[~patch_mask] = 0
        haralick = compute_haralick(patch_u8)

        area = region.area
        perimeter = region.perimeter
        major = region.major_axis_length
        minor = region.minor_axis_length

        rows.append({
            "condition": cond_name,
            "cell_label": label,
            "cell_area_px": area,
            "aspect_ratio": major / minor if minor > 0 else np.nan,
            "circularity": (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else np.nan,
            "elongation": 1 - (minor / major) if major > 0 else np.nan,
            "solidity": region.solidity,
            "mean_actinin_intensity": float(np.mean(act_vals)),
            "integrated_actinin_intensity": float(np.sum(act_vals)),
            "std_actinin_intensity": float(np.std(act_vals)),
            "sarcomere_coherency": float(np.mean(coh_vals)),
            "sarcomere_period_px": period,
            "fft_peak_strength": fft_strength,
            **{k: v for k, v in haralick.items()},
        })

    df = pd.DataFrame(rows)
    print(f"  {cond_name.replace(chr(10), ' '):25s} | nuclei: {nuclei_labels.max():3d} | cells: {len(df)}")

    return df, {
        "actinin": actinin,
        "dapi": dapi,
        "actinin_norm": actinin_norm,
        "dapi_norm": dapi_norm,
        "nuclei_labels": nuclei_labels,
        "cell_labels": cell_labels,
        "coherency_map": coherency_map,
    }


def plot_overview(all_data, output_path):
    n = len(all_data)
    fig, axes = plt.subplots(n, 5, figsize=(25, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (cond, data) in enumerate(all_data.items()):
        axes[i, 0].imshow(data["dapi_norm"], cmap="Blues")
        axes[i, 0].set_title(f"DAPI\n{cond}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(data["actinin_norm"], cmap="Reds")
        axes[i, 1].set_title(f"α-Actinin\n{cond}", fontsize=9)
        axes[i, 1].axis("off")

        overlay = np.zeros((*data["dapi_norm"].shape, 3))
        overlay[:, :, 2] = data["dapi_norm"]
        overlay[:, :, 0] = data["actinin_norm"]
        axes[i, 2].imshow(np.clip(overlay, 0, 1))
        axes[i, 2].set_title(f"Merge\n{cond}", fontsize=9)
        axes[i, 2].axis("off")

        seg_rgb = np.zeros((*data["actinin_norm"].shape, 3))
        seg_rgb[:, :, 0] = data["actinin_norm"] * 0.8
        bounds = segmentation.find_boundaries(data["cell_labels"], mode="thick")
        nuc_bounds = segmentation.find_boundaries(data["nuclei_labels"], mode="thick")
        seg_rgb[bounds] = [0, 1, 0]
        seg_rgb[nuc_bounds] = [0, 0.4, 1]
        axes[i, 3].imshow(np.clip(seg_rgb, 0, 1))
        axes[i, 3].set_title(f"Segmentation\n{cond}", fontsize=9)
        axes[i, 3].axis("off")

        axes[i, 4].imshow(data["coherency_map"], cmap="hot", vmin=0, vmax=1)
        axes[i, 4].set_title(f"Sarcomere Coherency\n{cond}", fontsize=9)
        axes[i, 4].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_boxplots(df, output_path):
    metrics = [
        ("mean_actinin_intensity", "Mean α-Actinin Intensity"),
        ("integrated_actinin_intensity", "Integrated Intensity"),
        ("sarcomere_coherency", "Sarcomere Coherency"),
        ("sarcomere_period_px", "Sarcomere Period (px)"),
        ("fft_peak_strength", "FFT Peak Strength\n(periodicity regularity)"),
        ("haralick_homogeneity", "Haralick Homogeneity"),
        ("cell_area_px", "Cell Area (px)"),
        ("aspect_ratio", "Aspect Ratio"),
        ("circularity", "Circularity"),
    ]
    cond_order = list(CONDITIONS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        data_list = [df[df["condition"] == c][col].dropna().values for c in cond_order]
        bp = ax.boxplot(data_list, labels=cond_order, patch_artist=True, widths=0.6, showfliers=False)
        for j, (patch, c) in enumerate(zip(bp["boxes"], cond_order)):
            patch.set_facecolor(CONDITION_COLORS[c])
            patch.set_alpha(0.7)
        for j, d in enumerate(data_list):
            x_jitter = np.random.normal(j + 1, 0.04, size=len(d))
            ax.scatter(x_jitter, d, alpha=0.4, s=12, color="black", zorder=5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)

    plt.suptitle("α-Actinin Analysis: Sarcomere Organization", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_summary_bars(df, output_path):
    metrics = [
        ("mean_actinin_intensity", "Mean α-Actinin Intensity"),
        ("sarcomere_coherency", "Sarcomere Coherency"),
        ("fft_peak_strength", "FFT Peak Strength"),
        ("haralick_correlation", "Haralick Correlation"),
    ]
    cond_order = list(CONDITIONS.keys())
    colors = [CONDITION_COLORS[c] for c in cond_order]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        means, sems = [], []
        for c in cond_order:
            vals = df[df["condition"] == c][col].dropna()
            means.append(vals.mean())
            sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        bars = ax.bar(range(len(cond_order)), means, yerr=sems, capsize=5,
                      color=colors, edgecolor="black", linewidth=0.8)
        ax.set_xticks(range(len(cond_order)))
        ax.set_xticklabels(cond_order, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        for i, (m, s) in enumerate(zip(means, sems)):
            ax.text(i, m + s + 0.01 * max(means), f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("α-Actinin Summary (Mean ± SEM)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def compute_statistics(df):
    cond_order = list(CONDITIONS.keys())
    metrics = [
        "cell_area_px", "aspect_ratio", "circularity", "elongation", "solidity",
        "mean_actinin_intensity", "integrated_actinin_intensity", "std_actinin_intensity",
        "sarcomere_coherency", "sarcomere_period_px", "fft_peak_strength",
        "haralick_energy", "haralick_homogeneity", "haralick_contrast", "haralick_correlation",
    ]

    print("\n" + "=" * 90)
    print("SUMMARY STATISTICS")
    print("=" * 90)
    for cond in cond_order:
        sub = df[df["condition"] == cond]
        print(f"\n  {cond.replace(chr(10), ' ')} (N={len(sub)}):")
        for m in metrics:
            vals = sub[m].dropna()
            if len(vals) > 0:
                print(f"    {m:40s}  {vals.mean():.4f} ± {vals.std():.4f}")

    print("\n" + "=" * 90)
    print("PAIRWISE COMPARISONS (Mann-Whitney U)")
    print("=" * 90)

    comparisons = [
        ("Gold\n(control plate)", "Cells only\n(control)", "Gold control vs Cells only"),
        ("Gold\n(device)", "Cells only\n(control)", "Gold device vs Cells only"),
        ("Gold\n(device)", "Gold\n(control plate)", "Gold device vs Gold control"),
    ]

    stat_rows = []
    for c1, c2, label in comparisons:
        print(f"\n  {label}:")
        for m in metrics:
            v1 = df[df["condition"] == c1][m].dropna()
            v2 = df[df["condition"] == c2][m].dropna()
            if len(v1) > 0 and len(v2) > 0:
                stat, p = mannwhitneyu(v1, v2, alternative="two-sided")
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                if p < 0.05:
                    print(f"    {m:40s}  p={p:.6f} {sig}  ({v1.median():.3f} vs {v2.median():.3f})")
                stat_rows.append({
                    "comparison": label, "metric": m,
                    "p_value": p, "significance": sig,
                })

    return pd.DataFrame(stat_rows)


def main():
    print("=" * 60)
    print("ALPHA-ACTININ IMAGE ANALYSIS")
    print("=" * 60)

    all_dfs = []
    all_data = {}

    print("\n[1/4] Analyzing conditions...")
    for cond_name, paths in CONDITIONS.items():
        df_cond, data = analyze_condition(cond_name, paths["actinin"], paths["dapi"])
        all_dfs.append(df_cond)
        all_data[cond_name] = data

    df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n[2/4] Segmentation overview...")
    plot_overview(all_data, OUTPUT_DIR / "10_actinin_segmentation_overview.png")

    print(f"\n[3/4] Comparison plots...")
    plot_boxplots(df, OUTPUT_DIR / "11_actinin_boxplots.png")
    plot_summary_bars(df, OUTPUT_DIR / "12_actinin_summary_bars.png")

    print(f"\n[4/4] Statistics...")
    stats_df = compute_statistics(df)

    df.to_csv(OUTPUT_DIR / "actinin_per_cell_data.csv", index=False)
    stats_df.to_csv(OUTPUT_DIR / "actinin_statistics.csv", index=False)
    print(f"\n  Exported: {OUTPUT_DIR / 'actinin_per_cell_data.csv'}")
    print(f"  Exported: {OUTPUT_DIR / 'actinin_statistics.csv'}")

    print("\n" + "=" * 60)
    print("ALPHA-ACTININ ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
