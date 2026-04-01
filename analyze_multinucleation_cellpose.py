"""
Multinucleation Analysis with Cellpose (GPU)
=============================================
Uses Cellpose deep learning for proper cell segmentation from phalloidin,
then counts DAPI nuclei per cell independently.
"""

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose import models
from skimage import filters, morphology, measure, feature, segmentation
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from scipy.stats import chi2_contingency, fisher_exact
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

CONDITION_COLORS = ["#7fbfff", "#ffd966", "#ff9966", "#99cc99"]


def normalize_16bit(img):
    p_low, p_high = np.percentile(img, (1, 99.5))
    return np.clip((img.astype(np.float64) - p_low) / (p_high - p_low), 0, 1)


def segment_nuclei_classical(dapi_norm):
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


def segment_cells_cellpose(phalloidin_norm, dapi_norm):
    """Segment cells using Cellpose cyto3 on phalloidin + DAPI."""
    phal_uint8 = (np.clip(phalloidin_norm, 0, 1) * 255).astype(np.uint8)
    dapi_uint8 = (np.clip(dapi_norm, 0, 1) * 255).astype(np.uint8)

    img_2ch = np.stack([phal_uint8, dapi_uint8], axis=0)

    model = models.Cellpose(model_type="cyto2", gpu=True)
    masks, flows, styles, diams = model.eval(
        img_2ch,
        diameter=None,
        channels=[1, 2],
        flow_threshold=0.4,
        cellprob_threshold=0.0,
    )
    return masks


def count_nuclei_per_cell(cell_labels, nuclei_labels):
    """Count independently segmented nuclei inside each Cellpose cell."""
    results = []
    for cell_region in measure.regionprops(cell_labels):
        cell_id = cell_region.label
        cell_mask = cell_labels == cell_id

        nuc_ids = np.unique(nuclei_labels[cell_mask])
        nuc_ids = nuc_ids[nuc_ids > 0]

        valid_nucs = []
        for nuc_id in nuc_ids:
            nuc_mask = nuclei_labels == nuc_id
            overlap = np.sum(nuc_mask & cell_mask)
            nuc_total = np.sum(nuc_mask)
            if overlap > 0.5 * nuc_total:
                valid_nucs.append(nuc_id)

        nuc_areas = []
        for nuc_id in valid_nucs:
            nuc_areas.append(np.sum(nuclei_labels == nuc_id))

        results.append({
            "cell_id": cell_id,
            "cell_area_px": cell_region.area,
            "nuclei_count": len(valid_nucs),
            "total_nuclear_area_px": sum(nuc_areas) if nuc_areas else 0,
        })

    return pd.DataFrame(results)


def analyze_condition(cond_name, phalloidin_path, dapi_path):
    phalloidin = tifffile.imread(phalloidin_path)
    dapi = tifffile.imread(dapi_path)

    phalloidin_norm = normalize_16bit(phalloidin)
    dapi_norm = normalize_16bit(dapi)

    print(f"    Running Cellpose on {cond_name.replace(chr(10), ' ')}...", end=" ", flush=True)
    cell_labels = segment_cells_cellpose(phalloidin_norm, dapi_norm)
    print(f"found {cell_labels.max()} cells")

    nuclei_labels = segment_nuclei_classical(dapi_norm)

    df = count_nuclei_per_cell(cell_labels, nuclei_labels)
    df["condition"] = cond_name

    n_cells = len(df)
    n_mono = (df["nuclei_count"] == 1).sum()
    n_bi = (df["nuclei_count"] == 2).sum()
    n_multi = (df["nuclei_count"] >= 3).sum()
    n_zero = (df["nuclei_count"] == 0).sum()

    print(f"    {cond_name.replace(chr(10), ' '):25s} | cells: {n_cells:3d} | "
          f"mono: {n_mono} ({n_mono/n_cells*100:.0f}%) | bi: {n_bi} ({n_bi/n_cells*100:.0f}%) | "
          f"multi(3+): {n_multi} ({n_multi/n_cells*100:.0f}%) | no nuc: {n_zero}")

    return df, {
        "phalloidin_norm": phalloidin_norm,
        "dapi_norm": dapi_norm,
        "nuclei_labels": nuclei_labels,
        "cell_labels": cell_labels,
    }


def plot_segmentation_qc(all_data, output_path):
    """Show Cellpose cell boundaries + nuclei for QC."""
    n = len(all_data)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (cond, data) in enumerate(all_data.items()):
        overlay = np.zeros((*data["dapi_norm"].shape, 3))
        overlay[:, :, 2] = data["dapi_norm"]
        overlay[:, :, 1] = data["phalloidin_norm"]
        axes[i, 0].imshow(np.clip(overlay, 0, 1))
        axes[i, 0].set_title(f"Merge — {cond}", fontsize=9)
        axes[i, 0].axis("off")

        seg_rgb = np.zeros((*data["phalloidin_norm"].shape, 3))
        seg_rgb[:, :, 1] = data["phalloidin_norm"] * 0.7
        cell_bounds = segmentation.find_boundaries(data["cell_labels"], mode="thick")
        nuc_bounds = segmentation.find_boundaries(data["nuclei_labels"], mode="thick")
        seg_rgb[cell_bounds] = [1, 1, 0]
        seg_rgb[nuc_bounds] = [1, 0, 0.5]
        axes[i, 1].imshow(np.clip(seg_rgb, 0, 1))
        axes[i, 1].set_title(f"Cellpose cells (yellow) + nuclei (magenta)\n{cond}", fontsize=9)
        axes[i, 1].axis("off")

        cell_labels = data["cell_labels"]
        nuclei_labels = data["nuclei_labels"]
        nuc_count_map = np.zeros_like(cell_labels, dtype=float)
        for cell_region in measure.regionprops(cell_labels):
            cell_mask = cell_labels == cell_region.label
            nuc_ids = np.unique(nuclei_labels[cell_mask])
            nuc_ids = nuc_ids[nuc_ids > 0]
            valid = 0
            for nid in nuc_ids:
                nm = nuclei_labels == nid
                if np.sum(nm & cell_mask) > 0.5 * np.sum(nm):
                    valid += 1
            nuc_count_map[cell_mask] = valid

        cmap_rgb = np.zeros((*cell_labels.shape, 3))
        cmap_rgb[nuc_count_map == 0] = [0.3, 0.3, 0.3]
        cmap_rgb[nuc_count_map == 1] = [0.3, 0.6, 1.0]
        cmap_rgb[nuc_count_map == 2] = [1.0, 0.85, 0.0]
        cmap_rgb[nuc_count_map >= 3] = [1.0, 0.2, 0.2]
        cmap_rgb[cell_labels == 0] = [0, 0, 0]
        axes[i, 2].imshow(cmap_rgb)
        axes[i, 2].set_title(f"Nuclei/cell: blue=1, yellow=2, red=3+\n{cond}", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_multinucleation_results(all_dfs, output_path):
    df = pd.concat(all_dfs, ignore_index=True)
    cond_order = list(CONDITIONS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Stacked bar: nucleation distribution
    nuc_colors = ["#7fbfff", "#ffd966", "#ff6666", "#aaaaaa"]
    labels_nuc = ["Mononucleated (1)", "Binucleated (2)", "Multinucleated (3+)", "No nucleus (0)"]

    pcts_list = []
    for c in cond_order:
        sub = df[df["condition"] == c]
        total = len(sub)
        mono = (sub["nuclei_count"] == 1).sum()
        bi = (sub["nuclei_count"] == 2).sum()
        multi = (sub["nuclei_count"] >= 3).sum()
        zero = (sub["nuclei_count"] == 0).sum()
        pcts_list.append([mono / total * 100, bi / total * 100,
                          multi / total * 100, zero / total * 100])

    pcts = np.array(pcts_list)
    x = np.arange(len(cond_order))
    bottom = np.zeros(len(cond_order))
    for j in range(4):
        axes[0].bar(x, pcts[:, j], 0.6, bottom=bottom, label=labels_nuc[j], color=nuc_colors[j])
        for i in range(len(cond_order)):
            if pcts[i, j] > 4:
                axes[0].text(i, bottom[i] + pcts[i, j] / 2, f"{pcts[i, j]:.0f}%",
                             ha="center", va="center", fontsize=9, fontweight="bold")
        bottom += pcts[:, j]
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cond_order, fontsize=8)
    axes[0].set_ylabel("% of cells")
    axes[0].set_title("Nucleation Distribution", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 108)

    # Bi+ rate
    bi_plus = []
    for c in cond_order:
        sub = df[(df["condition"] == c) & (df["nuclei_count"] > 0)]
        bi_plus.append((sub["nuclei_count"] >= 2).sum() / len(sub) * 100 if len(sub) > 0 else 0)

    bars = axes[1].bar(x, bi_plus, 0.6, color=CONDITION_COLORS, edgecolor="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cond_order, fontsize=8)
    axes[1].set_ylabel("% cells with ≥2 nuclei")
    axes[1].set_title("Binucleation+ Rate\n(maturation indicator)", fontsize=12, fontweight="bold")
    for i, v in enumerate(bi_plus):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # Mean nuclei per cell
    mean_nuc = []
    sem_nuc = []
    for c in cond_order:
        sub = df[(df["condition"] == c) & (df["nuclei_count"] > 0)]
        mean_nuc.append(sub["nuclei_count"].mean())
        sem_nuc.append(sub["nuclei_count"].std() / np.sqrt(len(sub)) if len(sub) > 1 else 0)

    axes[2].bar(x, mean_nuc, 0.6, yerr=sem_nuc, capsize=5, color=CONDITION_COLORS,
                edgecolor="black", linewidth=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(cond_order, fontsize=8)
    axes[2].set_ylabel("Mean nuclei per cell")
    axes[2].set_title("Mean Nuclei per Cell (±SEM)", fontsize=12, fontweight="bold")
    for i, v in enumerate(mean_nuc):
        axes[2].text(i, v + sem_nuc[i] + 0.02, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Multinucleation Analysis (Cellpose GPU)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    print("=" * 60)
    print("MULTINUCLEATION ANALYSIS — CELLPOSE GPU")
    print("=" * 60)

    all_dfs = []
    all_data = {}

    print("\n[1/3] Segmenting with Cellpose (GPU)...")
    for cond_name, paths in CONDITIONS.items():
        df_cond, data = analyze_condition(cond_name, paths["phalloidin"], paths["dapi"])
        all_dfs.append(df_cond)
        all_data[cond_name] = data

    print(f"\n[2/3] Generating visualizations...")
    plot_segmentation_qc(all_data, OUTPUT_DIR / "08_cellpose_multinucleation_qc.png")
    plot_multinucleation_results(all_dfs, OUTPUT_DIR / "09_multinucleation_cellpose_results.png")

    print(f"\n[3/3] Statistical tests...")
    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(OUTPUT_DIR / "multinucleation_cellpose.csv", index=False)

    cond_order = list(CONDITIONS.keys())
    contingency = []
    for c in cond_order:
        sub = df[(df["condition"] == c) & (df["nuclei_count"] > 0)]
        mono = (sub["nuclei_count"] == 1).sum()
        bi_plus = (sub["nuclei_count"] >= 2).sum()
        contingency.append([mono, bi_plus])
        print(f"  {c.replace(chr(10), ' '):25s}  mono={mono:3d}  bi+={bi_plus:3d}  "
              f"bi+ rate: {bi_plus/(mono+bi_plus)*100:.1f}%" if (mono+bi_plus) > 0 else "  no data")

    contingency = np.array(contingency)
    if contingency.shape[0] >= 2 and contingency.sum() > 0:
        chi2, p, dof, expected = chi2_contingency(contingency)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"\n  Chi-squared test (all 4 conditions): chi2={chi2:.2f}, p={p:.4f} {sig}")

    comparisons = [
        ("Gold\n(device)", "Nonpoled\n(device)", "Gold device vs Nonpoled device"),
        ("Gold\n(device)", "Cells only\n(control)", "Gold device vs Cells only"),
        ("Gold\n(control plate)", "Cells only\n(control)", "Gold control vs Cells only"),
        ("Gold\n(device)", "Gold\n(control plate)", "Gold device vs Gold control"),
    ]
    for c1, c2, label in comparisons:
        s1 = df[(df["condition"] == c1) & (df["nuclei_count"] > 0)]
        s2 = df[(df["condition"] == c2) & (df["nuclei_count"] > 0)]
        t = np.array([
            [(s1["nuclei_count"] == 1).sum(), (s1["nuclei_count"] >= 2).sum()],
            [(s2["nuclei_count"] == 1).sum(), (s2["nuclei_count"] >= 2).sum()],
        ])
        if t.shape == (2, 2) and t.sum() > 0:
            _, p = fisher_exact(t)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {label:40s}  Fisher p={p:.4f} {sig}")

    print(f"\n  Exported: {OUTPUT_DIR / 'multinucleation_cellpose.csv'}")
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
