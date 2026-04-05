"""
Generate fancy visualization figures for cardiomyocyte video and FL data.
FL6: Three-modality panel (staining + BF + FL)
X1-X10: Advanced visualization techniques for video_figures_extra/
"""

import cv2
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from matplotlib import cm
from pathlib import Path
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

DATA = Path("Cardiomyocytes")
REPORT = Path("final_report")
EXTRA = Path("video_figures_extra")
EXTRA.mkdir(exist_ok=True)
FL_OUT = Path("fl_results")

VID_CSV = pd.read_csv("video_results/batch_results_v3_final.csv")
FL_CSV = pd.read_csv(FL_OUT / "fl_batch_results.csv")


def norm16(img):
    img = img.astype(np.float64)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
    if hi <= lo:
        hi = lo + 1
    return np.clip((img - lo) / (hi - lo), 0, 1)


def read_frame(filepath, idx=0):
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read()
    cap.release()
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if ret else None


def read_frames_range(filepath, start=0, end=None, step=1):
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        return [], 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is None:
        end = total
    frames = []
    for i in range(start, min(end, total), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if ret:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames, fps


def read_all_frames_gray(filepath, half_res=False):
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        return None, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if half_res:
            g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2))
        frames.append(g.astype(np.float32))
    cap.release()
    if len(frames) == 0:
        return None, fps
    return np.array(frames), fps


def read_kymograph_row(filepath, row_frac=0.5):
    """Read only one row from each frame — extremely memory-efficient."""
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        return None, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows = []
    target_row = None
    while True:
        ret, f = cap.read()
        if not ret:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if target_row is None:
            target_row = int(g.shape[0] * row_frac)
        rows.append(g[target_row, :].astype(np.float32))
    cap.release()
    if len(rows) == 0:
        return None, fps, 0
    return np.array(rows), fps, target_row


# ═══════════════════════════════════════════════════════════
# FL6: THREE-MODALITY PANEL
# ═══════════════════════════════════════════════════════════
def fig_fl6_three_modality():
    print("  FL6: Three-modality panel (staining + BF + FL)...")

    STAIN_MAP = {
        "gold": {
            "phalloidin": DATA / "device-gold2-phalloidin-b2-image.tif",
            "dapi": DATA / "device-gold2-dapi-b2-image.tif",
            "label": "B2 XR1 gold (Poled Pulsed)",
        },
        "non-poled": {
            "phalloidin": DATA / "device-non-phalloidin-a3-image.tif",
            "dapi": DATA / "device-non-dapi-a3-image.tif",
            "label": "A3 XR1 non-poled (Nonpoled Pulsed)",
        },
        "plastic": {
            "phalloidin": DATA / "control-b3-non-phalloidin.tif",
            "dapi": DATA / "control-b3-non-dapi.tif",
            "label": "B3 XR1 plastic (Cells only)",
        },
    }

    COLUMNS = [
        {"bf": "C2-day2-gold.avi", "fl": "C2-day2-gold-FL.avi",
         "stain": "gold", "title": "C2 day2\ngold"},
        {"bf": "C3-day2.avi", "fl": "C3-day2-fl.avi",
         "stain": "non-poled", "title": "C3 day2\nnon-poled"},
        {"bf": "C2-day6-gold-control-bf.avi", "fl": "C2-day6-gold-FL-control.avi",
         "stain": "gold", "title": "C2 day6\ngold (ctrl)"},
        {"bf": "C2-day8-gold-Jaz.avi", "fl": "C2-day8-gold-GCaMP6f-FL.avi",
         "stain": "gold", "title": "C2 day8\ngold (dev)"},
        {"bf": "C3-day8-non-Jaz.avi", "fl": "C3-day8-non-Jaz-fl.avi",
         "stain": "non-poled", "title": "C3 day8\nnon-poled"},
    ]

    fig, axes = plt.subplots(3, 5, figsize=(28, 16))

    for col_i, col in enumerate(COLUMNS):
        stain_info = STAIN_MAP[col["stain"]]
        try:
            phal = norm16(tifffile.imread(str(stain_info["phalloidin"])))
            dapi = norm16(tifffile.imread(str(stain_info["dapi"])))
            merge = np.zeros((*phal.shape, 3))
            merge[:, :, 1] = phal
            merge[:, :, 2] = dapi
            axes[0, col_i].imshow(merge)
        except Exception:
            axes[0, col_i].text(0.5, 0.5, "Staining\nnot found", ha="center",
                                va="center", transform=axes[0, col_i].transAxes, fontsize=10)
        axes[0, col_i].set_title(col["title"], fontsize=10, fontweight="bold")
        axes[0, col_i].set_xticks([])
        axes[0, col_i].set_yticks([])

        bf = read_frame(DATA / col["bf"])
        if bf is not None:
            axes[1, col_i].imshow(bf, cmap="gray")
        else:
            axes[1, col_i].text(0.5, 0.5, "BF not found", ha="center",
                                va="center", transform=axes[1, col_i].transAxes)
        axes[1, col_i].set_xticks([])
        axes[1, col_i].set_yticks([])

        fl = read_frame(DATA / col["fl"])
        fl_row = FL_CSV[FL_CSV["filename"] == col["fl"]]
        fl_cls = fl_row["classification"].values[0] if len(fl_row) > 0 else "?"
        fl_n = int(fl_row["n_transients"].values[0]) if len(fl_row) > 0 else 0
        if fl is not None:
            axes[2, col_i].imshow(fl, cmap="inferno", vmin=0,
                                  vmax=np.percentile(fl, 99.5))
            axes[2, col_i].text(0.02, 0.98, f"{fl_cls}\n{fl_n} trans",
                                transform=axes[2, col_i].transAxes, fontsize=7,
                                va="top", color="white",
                                bbox=dict(boxstyle="round", fc="black", alpha=0.6))
        else:
            axes[2, col_i].text(0.5, 0.5, "FL not found", ha="center",
                                va="center", transform=axes[2, col_i].transAxes)
        axes[2, col_i].set_xticks([])
        axes[2, col_i].set_yticks([])

    axes[0, 0].set_ylabel("STAINING\n(XR1, phalloidin+DAPI)", fontsize=12,
                           fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("BRIGHTFIELD\n(GCaMP6f)", fontsize=12,
                           fontweight="bold", labelpad=10)
    axes[2, 0].set_ylabel("FLUORESCENCE\n(GCaMP6f, Ca2+)", fontsize=12,
                           fontweight="bold", labelpad=10)

    fig.suptitle("Figure FL6: Three-Modality Comparison\n"
                 "Row 1: F-actin staining (XR1, same substrate) | "
                 "Row 2: BF video (GCaMP6f) | Row 3: FL calcium (GCaMP6f)\n"
                 "Key finding: organized F-actin + calcium transients, but no contraction",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(REPORT / "fl_Fig_FL6_three_modality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved fl_Fig_FL6_three_modality.png")


# ═══════════════════════════════════════════════════════════
# X1: TEMPORAL COLOR-CODED PROJECTION (BF)
# ═══════════════════════════════════════════════════════════
def fig_x1_temporal_color_bf():
    print("  X1: Temporal color-coded projection (BF)...")
    videos = [
        ("B3-day8-tammy-control.avi", "B3 day8 XR1 plastic\nactive_beating"),
        ("A3-day8-non-Tammy.avi", "A3 day8 XR1 non-poled\nactive_beating"),
        ("C2-day8-gold-Jaz.avi", "C2 day8 GCaMP6f gold\nno_beating"),
        ("C3-day8-non-Jaz.avi", "C3 day8 GCaMP6f non-poled\nindividual_cells"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    for i, (fname, label) in enumerate(videos):
        frames, fps = read_all_frames_gray(DATA / fname, half_res=True)
        ax = axes[i]
        if frames is None:
            ax.text(0.5, 0.5, "Not found", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue

        n = len(frames)
        h, w = frames[0].shape
        composite = np.zeros((h, w, 3), dtype=np.float32)
        max_val = np.zeros((h, w), dtype=np.float32)

        step = max(1, n // 100)
        for j in range(0, n, step):
            t = j / max(n - 1, 1)
            color = np.array(cm.hsv(t)[:3], dtype=np.float32)
            f = frames[j] / 255.0
            mask = f > max_val
            for c in range(3):
                composite[:, :, c] = np.where(mask, f * color[c], composite[:, :, c])
            max_val = np.maximum(max_val, f)

        composite = np.clip(composite, 0, 1)
        ax.imshow(composite)
        ax.set_title(f"{label}\nFPS={fps:.1f}, {n} frames", fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        del frames

    fig.suptitle("X1: Temporal Color-Coded Projection (BF)\n"
                 "Each frame assigned a rainbow color by time (red=start, violet=end)\n"
                 "Static cells appear monochrome; moving cells show color trails",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(EXTRA / "X1_temporal_color_BF.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved X1_temporal_color_BF.png")


# ═══════════════════════════════════════════════════════════
# X2: TEMPORAL COLOR-CODED PROJECTION (FL)
# ═══════════════════════════════════════════════════════════
def fig_x2_temporal_color_fl():
    print("  X2: Temporal color-coded projection (FL)...")
    videos = [
        ("Jaz-day2-fluorescence-gold.avi", "C2 day2 gold 10FPS\n5 transients"),
        ("C2-day6-gold-FL-control.avi", "C2 day6 gold 10FPS\n10 transients"),
        ("C2-day8-gold-GCaMP6f-FL.avi", "C2 day8 gold 12.5FPS\n1 transient"),
        ("C3-day8-non-Jaz-fl.avi", "C3 day8 non-poled 11.1FPS\n7 transients"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    for i, (fname, label) in enumerate(videos):
        frames, fps = read_all_frames_gray(DATA / fname, half_res=True)
        ax = axes[i]
        if frames is None:
            ax.text(0.5, 0.5, "Not found", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue

        n = len(frames)
        h, w = frames[0].shape
        composite = np.zeros((h, w, 3), dtype=np.float32)
        max_val = np.zeros((h, w), dtype=np.float32)
        fmax = frames.max()
        if fmax < 1:
            fmax = 1

        step = max(1, n // 120)
        for j in range(0, n, step):
            t = j / max(n - 1, 1)
            color = np.array(cm.hsv(t)[:3], dtype=np.float32)
            f = frames[j] / fmax
            mask = f > max_val
            for c in range(3):
                composite[:, :, c] = np.where(mask, f * color[c], composite[:, :, c])
            max_val = np.maximum(max_val, f)

        composite = np.clip(composite / (composite.max() + 1e-10), 0, 1)
        ax.imshow(composite)
        ax.set_title(f"{label}", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        del frames

    fig.suptitle("X2: Temporal Color-Coded Projection (Fluorescence)\n"
                 "GCaMP6f calcium transients color-coded by time (red=start, violet=end)\n"
                 "Regions fluorescing at different times appear in different colors",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(EXTRA / "X2_temporal_color_FL.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved X2_temporal_color_FL.png")


# ═══════════════════════════════════════════════════════════
# X3: BF MOTION VECTOR FIELDS
# ═══════════════════════════════════════════════════════════
def fig_x3_quiver():
    print("  X3: BF motion vector fields...")
    videos = [
        ("B3-day8-tammy-control.avi", "XR1 active beating\nB3 day8 plastic"),
        ("C2-day8-gold-Jaz.avi", "GCaMP6f no beating\nC2 day8 gold"),
        ("C3-day8-non-Jaz.avi", "GCaMP6f individual cells\nC3 day8 non-poled"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(16, 22))

    for row, (fname, label) in enumerate(videos):
        fpath = DATA / fname
        cap = cv2.VideoCapture(str(fpath))
        if not cap.isOpened():
            for c in range(2):
                axes[row, c].text(0.5, 0.5, "Not found", ha="center",
                                  va="center", transform=axes[row, c].transAxes)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, f1 = cap.read()
        f1_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY) if ret else None

        best_mag = 0
        best_flow = None
        best_frame = None
        prev = f1_gray

        sample_indices = np.linspace(1, total - 1, min(80, total - 1), dtype=int)
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret:
                continue
            curr = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            small_prev = cv2.resize(prev, (prev.shape[1] // 2, prev.shape[0] // 2))
            small_curr = cv2.resize(curr, (curr.shape[1] // 2, curr.shape[0] // 2))
            flow = cv2.calcOpticalFlowFarneback(small_prev, small_curr, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2).mean()
            if mag > best_mag:
                best_mag = mag
                best_flow = flow
                best_frame = curr
            prev = curr
        cap.release()

        ax_d = axes[row, 0]
        ax_d.imshow(f1_gray, cmap="gray")
        ax_d.set_title(f"{label}\nDiastole (frame 0)", fontsize=9, fontweight="bold")
        ax_d.set_xticks([])
        ax_d.set_yticks([])

        ax_s = axes[row, 1]
        if best_frame is not None and best_flow is not None:
            h, w = best_flow.shape[:2]
            step = 12
            Y, X = np.mgrid[0:h:step, 0:w:step]
            U = best_flow[::step, ::step, 0]
            V = best_flow[::step, ::step, 1]
            M = np.sqrt(U**2 + V**2)

            small_frame = cv2.resize(best_frame, (w, h))
            ax_s.imshow(small_frame, cmap="gray")
            q = ax_s.quiver(X, Y, U, -V, M, cmap="inferno", scale=50,
                            width=0.003, headwidth=4, alpha=0.85)
            ax_s.set_title(f"Systole (peak motion)\nMean flow={best_mag:.2f} px/frame",
                           fontsize=9, fontweight="bold")
        else:
            ax_s.text(0.5, 0.5, "No flow", ha="center", va="center",
                      transform=ax_s.transAxes)
        ax_s.set_xticks([])
        ax_s.set_yticks([])

    fig.suptitle("X3: BF Motion Vector Fields (Optical Flow)\n"
                 "Left: diastole (resting) | Right: systole (peak contraction)\n"
                 "Arrows show displacement direction and magnitude (inferno colormap)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(EXTRA / "X3_motion_vectors_BF.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved X3_motion_vectors_BF.png")


# ═══════════════════════════════════════════════════════════
# X4: BF KYMOGRAPHS
# ═══════════════════════════════════════════════════════════
def fig_x4_kymo_bf():
    print("  X4: BF kymographs...")
    videos = [
        ("B3-day8-tammy-control.avi", "XR1 active beating\nB3 day8 plastic"),
        ("A3-day2-Tammy-NEW.avi", "XR1 baseline beating\nA3 day2 plastic"),
        ("C3-day8-non-Jaz.avi", "GCaMP6f individual cells\nC3 day8 non-poled"),
        ("C2-day8-gold-Jaz.avi", "GCaMP6f no beating\nC2 day8 gold"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for i, (fname, label) in enumerate(videos):
        kymo, fps, mid_row = read_kymograph_row(DATA / fname)
        ax = axes[i]
        if kymo is None:
            ax.text(0.5, 0.5, "Not found", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue

        n, w = kymo.shape
        time_s = np.arange(n) / fps

        ax.imshow(kymo, aspect="auto", cmap="gray",
                  extent=[0, w, time_s[-1], time_s[0]])
        ax.set_xlabel("Position (px)")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"{label}\nFPS={fps:.1f} | Horizontal line at row {mid_row}",
                     fontsize=9, fontweight="bold")

    fig.suptitle("X4: BF Kymographs (Space-Time Plots)\n"
                 "Horizontal line scan through center of each video\n"
                 "Beating appears as periodic undulations; non-beating is flat/noisy",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(EXTRA / "X4_kymograph_BF.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved X4_kymograph_BF.png")


# ═══════════════════════════════════════════════════════════
# X5: FL KYMOGRAPHS
# ═══════════════════════════════════════════════════════════
def fig_x5_kymo_fl():
    print("  X5: FL kymographs...")
    videos = [
        ("Jaz-day2-fluorescence-gold.avi", "C2 day2 gold 10FPS\n5 transients"),
        ("C2-day6-gold-FL-control.avi", "C2 day6 gold 10FPS\n10 transients"),
        ("C2-day8-gold-GCaMP6f-FL.avi", "C2 day8 gold 12.5FPS\n1 transient"),
        ("C3-day8-non-Jaz-fl.avi", "C3 day8 non-poled 11.1FPS\n7 transients"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for i, (fname, label) in enumerate(videos):
        kymo, fps, mid_row = read_kymograph_row(DATA / fname)
        ax = axes[i]
        if kymo is None:
            ax.text(0.5, 0.5, "Not found", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue

        n, w = kymo.shape
        time_s = np.arange(n) / fps

        ax.imshow(kymo, aspect="auto", cmap="inferno",
                  extent=[0, w, time_s[-1], time_s[0]],
                  vmin=np.percentile(kymo, 5), vmax=np.percentile(kymo, 99))
        ax.set_xlabel("Position (px)")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"{label}\nKymograph at row {mid_row}",
                     fontsize=9, fontweight="bold")

    fig.suptitle("X5: FL Kymographs — Calcium Wave Propagation\n"
                 "Horizontal line scan through center of each FL video\n"
                 "Ca2+ transients appear as bright horizontal bands; "
                 "diagonal streaks indicate wave propagation",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(EXTRA / "X5_kymograph_FL.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved X5_kymograph_FL.png")


# ═══════════════════════════════════════════════════════════
# X6: COMPLETE WELL TEMPORAL PROGRESSION GRID
# ═══════════════════════════════════════════════════════════
def fig_x6_grid_all():
    print("  X6: Complete well temporal grid...")

    wells = ["A2", "A3", "B2", "B3", "C2", "C3"]
    days = ["day1", "day2", "day6", "day8"]
    well_info = {
        "A2": ("XR1", "gold"),
        "A3": ("XR1", "non-poled/plastic"),
        "B2": ("XR1", "gold"),
        "B3": ("XR1", "non-poled/plastic"),
        "C2": ("GCaMP6f", "gold"),
        "C3": ("GCaMP6f", "non-poled/plastic"),
    }
    cls_colors = {
        "active_beating": "#2ca02c",
        "individual_cells_beating": "#ff7f0e",
        "non_contractile": "#d62728",
        "no_beating": "#7f7f7f",
    }

    fig, axes = plt.subplots(6, 4, figsize=(22, 30))

    for row_i, well in enumerate(wells):
        cell_line, substrate = well_info[well]
        for col_i, day in enumerate(days):
            ax = axes[row_i, col_i]
            vids = VID_CSV[(VID_CSV["well"] == well) & (VID_CSV["day"] == day)]

            if len(vids) == 0:
                ax.set_facecolor("#1a1a2e")
                ax.text(0.5, 0.5, "No\nrecording", ha="center", va="center",
                        fontsize=11, color="#555555", transform=ax.transAxes,
                        fontweight="bold")
            else:
                v = vids.iloc[0]
                frame = read_frame(DATA / v["filename"])
                if frame is not None:
                    ax.imshow(frame, cmap="gray")
                cls = v["manual_classification"]
                fps_val = v["fps"]
                amp = v.get("roi_amplitude", 0)
                color = cls_colors.get(cls, "#999999")
                badge = cls.replace("_", "\n")
                ax.text(0.02, 0.98, f"{fps_val:.1f} FPS",
                        transform=ax.transAxes, fontsize=7, va="top",
                        color="white", bbox=dict(boxstyle="round", fc="black", alpha=0.7))
                ax.text(0.98, 0.98, badge, transform=ax.transAxes, fontsize=6,
                        va="top", ha="right", color="white",
                        bbox=dict(boxstyle="round", fc=color, alpha=0.8))
                if pd.notna(amp) and amp > 0.1:
                    ax.text(0.5, 0.02, f"amp={amp:.1f}px", transform=ax.transAxes,
                            fontsize=7, ha="center", va="bottom", color="white",
                            bbox=dict(boxstyle="round", fc="black", alpha=0.6))
                if len(vids) > 1:
                    ax.text(0.02, 0.02, f"+{len(vids)-1} more", transform=ax.transAxes,
                            fontsize=6, va="bottom", color="yellow")

            ax.set_xticks([])
            ax.set_yticks([])
            if row_i == 0:
                ax.set_title(day.replace("day", "Day "), fontsize=12, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(f"{well}\n{cell_line}\n{substrate}",
                              fontsize=10, fontweight="bold", labelpad=10)

    fig.suptitle("X6: Complete Well Temporal Progression (BF)\n"
                 "All 6 wells x 4 timepoints | Badge = classification | "
                 "FPS annotated\nGreen=beating, Orange=individual, Red=non-contractile, "
                 "Gray=no beating",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(EXTRA / "X6_well_grid_complete.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved X6_well_grid_complete.png")


# ═══════════════════════════════════════════════════════════
# X7: FL CALCIUM ACTIVATION MAP (ISOCHRONE)
# ═══════════════════════════════════════════════════════════
def fig_x7_isochrone():
    print("  X7: FL calcium activation maps...")
    videos = [
        ("Jaz-day2-fluorescence-gold.avi", "C2 day2 gold\n10 FPS"),
        ("C2-day6-gold-FL-control.avi", "C2 day6 gold\n10 FPS"),
        ("C2-day8-gold-GCaMP6f-FL.avi", "C2 day8 gold\n12.5 FPS"),
        ("C3-day8-non-Jaz-fl.avi", "C3 day8 non-poled\n11.1 FPS"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    for col_i, (fname, label) in enumerate(videos):
        frames, fps = read_all_frames_gray(DATA / fname, half_res=True)
        if frames is None:
            for r in range(2):
                axes[r, col_i].text(0.5, 0.5, "Not found", ha="center",
                                    va="center", transform=axes[r, col_i].transAxes)
            continue

        mean_trace = np.array([f.mean() for f in frames])
        peak_idx = np.argmax(mean_trace[5:]) + 5

        window = max(1, int(fps * 0.5))
        start = max(0, peak_idx - window)
        end = min(len(frames), peak_idx + window)
        segment = frames[start:end]

        if len(segment) < 3:
            for r in range(2):
                axes[r, col_i].text(0.5, 0.5, "Too short", ha="center",
                                    va="center", transform=axes[r, col_i].transAxes)
            continue

        block = 4
        h, w = segment[0].shape
        h_b, w_b = h // block, w // block
        ttp_map = np.zeros((h_b, w_b))
        for by in range(h_b):
            for bx in range(w_b):
                roi = segment[:, by*block:(by+1)*block, bx*block:(bx+1)*block].mean(axis=(1, 2))
                ttp_map[by, bx] = np.argmax(roi)

        ttp_ms = ttp_map * (1000.0 / fps)

        ax = axes[0, col_i]
        ax.imshow(frames[0], cmap="gray", vmin=np.percentile(frames[0], 2),
                  vmax=np.percentile(frames[0], 98))
        im = ax.imshow(ttp_ms, cmap="jet", alpha=0.65,
                       extent=[0, w, h, 0])
        ax.set_title(f"{label}\nFirst frame + activation overlay", fontsize=9,
                     fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        ax2 = axes[1, col_i]
        im2 = ax2.imshow(ttp_ms, cmap="jet", extent=[0, w, h, 0])
        ax2.set_title(f"Activation map\n(ms to peak, {block}x{block} blocks)",
                      fontsize=9, fontweight="bold")
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, label="Time to peak (ms)", shrink=0.8)

        del frames

    axes[0, 0].set_ylabel("Frame +\nactivation overlay", fontsize=11,
                           fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("Activation map\n(isochrone)", fontsize=11,
                           fontweight="bold", labelpad=10)

    fig.suptitle("X7: Calcium Activation Maps (Isochrone)\n"
                 "Color = time-to-peak per pixel block during one Ca2+ transient\n"
                 "Red=early activation, blue=late → reveals wave propagation direction",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(EXTRA / "X7_activation_map_FL.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved X7_activation_map_FL.png")


# ═══════════════════════════════════════════════════════════
# X8: BF CONTRACTION CYCLE MONTAGE
# ═══════════════════════════════════════════════════════════
def fig_x8_beat_cycle():
    print("  X8: BF contraction cycle montage...")
    videos = [
        ("B3-day8-tammy-control.avi", "B3 day8 XR1 plastic (active beating, 18.5 BPM)"),
        ("A3-day6-NON1-TAMMY.avi", "A3 day6 XR1 non-poled (active beating, 10.5 BPM)"),
    ]

    fig, all_axes = plt.subplots(4, 1, figsize=(24, 18),
                                  gridspec_kw={"height_ratios": [3, 1.5, 3, 1.5]})

    for vid_i, (fname, label) in enumerate(videos):
        ax_strip = all_axes[vid_i * 2]
        ax_trace = all_axes[vid_i * 2 + 1]

        frames, fps = read_all_frames_gray(DATA / fname, half_res=True)
        if frames is None:
            ax_strip.text(0.5, 0.5, "Not found", ha="center", va="center",
                          transform=ax_strip.transAxes)
            continue

        n = len(frames)
        mean_trace = np.array([f.mean() for f in frames])
        time_s = np.arange(n) / fps

        peaks, _ = find_peaks(-mean_trace, distance=int(fps * 1.5), prominence=0.5)
        if len(peaks) < 2:
            peaks, _ = find_peaks(-mean_trace, distance=int(fps * 0.8))
        if len(peaks) < 2:
            peaks = np.array([n // 4, 3 * n // 4])

        p1, p2 = peaks[0], peaks[min(1, len(peaks) - 1)]
        if p2 <= p1:
            p2 = min(p1 + int(fps * 3), n - 1)

        n_panels = 10
        indices = np.linspace(p1, p2, n_panels, dtype=int)
        indices = np.clip(indices, 0, n - 1)

        h, w = frames[0].shape
        strip = np.zeros((h, w * n_panels + (n_panels - 1) * 4), dtype=np.float64)
        for j, idx in enumerate(indices):
            x_start = j * (w + 4)
            strip[:, x_start:x_start + w] = frames[idx]

        ax_strip.imshow(strip, cmap="gray", aspect="auto",
                        vmin=np.percentile(strip[strip > 0], 1),
                        vmax=np.percentile(strip[strip > 0], 99))
        for j, idx in enumerate(indices):
            x_center = j * (w + 4) + w // 2
            t_ms = idx / fps * 1000
            ax_strip.text(x_center, -5, f"{t_ms:.0f}ms", ha="center",
                          fontsize=7, fontweight="bold", color="red")
        ax_strip.set_title(f"{label} — one beat cycle ({n_panels} frames)",
                           fontsize=10, fontweight="bold")
        ax_strip.set_xticks([])
        ax_strip.set_yticks([])

        ax_trace.plot(time_s, mean_trace, "k-", linewidth=0.8)
        for idx in indices:
            ax_trace.axvline(time_s[idx], color="red", alpha=0.5, linewidth=0.8)
        ax_trace.set_xlabel("Time (s)")
        ax_trace.set_ylabel("Mean intensity")
        ax_trace.set_title("Mean intensity trace — red lines mark extracted frames",
                           fontsize=9)

        del frames

    fig.suptitle("X8: BF Contraction Cycle Filmstrip\n"
                 "Top: 10 frames spanning one beat cycle | "
                 "Bottom: intensity trace with frame positions marked",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(EXTRA / "X8_beat_cycle_BF.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved X8_beat_cycle_BF.png")


# ═══════════════════════════════════════════════════════════
# X9: FL PEAK-DIASTOLE ANIMATION STRIP
# ═══════════════════════════════════════════════════════════
def fig_x9_fl_strip():
    print("  X9: FL peak-diastole strip...")
    fname = "C2-day6-gold-FL-control.avi"
    label = "C2 day6 gold (ctrl) — 10 FPS, 10 transients"

    frames, fps = read_all_frames_gray(DATA / fname, half_res=True)
    if frames is None:
        print("    SKIPPED: file not found")
        return

    n = len(frames)
    h, w = frames[0].shape
    mean_trace = np.array([f.mean() for f in frames])
    time_s = np.arange(n) / fps

    bg = np.percentile(frames[:, :int(h * 0.15), :int(w * 0.15)], axis=(1, 2), q=50)
    corrected = mean_trace - bg
    from scipy.signal import savgol_filter
    try:
        smooth = savgol_filter(corrected, min(15, len(corrected) // 2 * 2 - 1), 3)
    except Exception:
        smooth = corrected
    f0 = np.percentile(smooth[:max(5, int(fps * 2))], 5)
    if f0 < 1:
        f0 = 1
    df_f0 = (smooth - f0) / f0

    peaks, props = find_peaks(df_f0, distance=int(fps * 1.2),
                              prominence=0.01, height=0.01)
    if len(peaks) < 2:
        peaks = np.array([np.argmax(df_f0)])

    n_show = min(4, len(peaks))
    peak_indices = peaks[:n_show]

    n_cols = n_show * 3
    fig = plt.figure(figsize=(5 * n_cols, 16))
    gs = GridSpec(2, n_cols, height_ratios=[3, 1.2], hspace=0.3)

    vmin = np.percentile(frames, 2)
    vmax = np.percentile(frames, 99.5)

    for j, pidx in enumerate(peak_indices):
        trough_search = max(0, pidx - int(fps * 0.4))
        diastole_idx = trough_search + np.argmin(mean_trace[trough_search:pidx]) if pidx > trough_search else max(0, pidx - 3)

        col_d = j * 3
        col_p = j * 3 + 1
        col_diff = j * 3 + 2

        ax_d = fig.add_subplot(gs[0, col_d])
        ax_d.imshow(frames[diastole_idx], cmap="inferno", vmin=vmin, vmax=vmax)
        ax_d.set_title(f"Diastole\n{time_s[diastole_idx]:.1f}s", fontsize=9, fontweight="bold")
        ax_d.set_xticks([])
        ax_d.set_yticks([])

        ax_p = fig.add_subplot(gs[0, col_p])
        ax_p.imshow(frames[pidx], cmap="inferno", vmin=vmin, vmax=vmax)
        ax_p.set_title(f"Peak #{j+1}\n{time_s[pidx]:.1f}s", fontsize=9, fontweight="bold")
        ax_p.set_xticks([])
        ax_p.set_yticks([])

        diff = frames[pidx].astype(np.float64) - frames[diastole_idx].astype(np.float64)
        abs_max = max(np.abs(np.percentile(diff, 2)), np.abs(np.percentile(diff, 98)))
        if abs_max < 1:
            abs_max = 1
        ax_diff = fig.add_subplot(gs[0, col_diff])
        ax_diff.imshow(diff, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
        ax_diff.set_title(f"Difference\n(peak-diastole)", fontsize=9, fontweight="bold")
        ax_diff.set_xticks([])
        ax_diff.set_yticks([])

    ax_trace = fig.add_subplot(gs[1, :])
    ax_trace.plot(time_s, df_f0, "g-", linewidth=1)
    ax_trace.set_xlabel("Time (s)", fontsize=11)
    ax_trace.set_ylabel("dF/F0", fontsize=11)
    for j, pidx in enumerate(peak_indices):
        ax_trace.axvline(time_s[pidx], color="red", alpha=0.7, linewidth=1.5)
        ax_trace.text(time_s[pidx], df_f0.max() * 0.95, f"#{j+1}",
                      ha="center", fontsize=8, color="red", fontweight="bold")
    ax_trace.set_title(f"{label} — dF/F0 trace with extracted transient positions",
                       fontsize=10, fontweight="bold")

    fig.suptitle("X9: FL Peak vs Diastole Animation Strip\n"
                 "Each transient: diastole (left) | peak (center) | "
                 "difference map (right, red=Ca2+ increase)\n"
                 "Bottom: dF/F0 trace with red lines marking extracted peaks",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(EXTRA / "X9_peak_diastole_FL.png", dpi=150, bbox_inches="tight")
    plt.close()
    del frames
    print("    Saved X9_peak_diastole_FL.png")


# ═══════════════════════════════════════════════════════════
# X10: COMBINED DASHBOARD
# ═══════════════════════════════════════════════════════════
def fig_x10_dashboard():
    print("  X10: Combined dashboard...")
    fig = plt.figure(figsize=(28, 20))
    gs = GridSpec(3, 4, hspace=0.35, wspace=0.25)

    # Panel A: 3-modality strip (one condition: gold, day8)
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    try:
        phal = norm16(tifffile.imread(str(DATA / "device-gold2-phalloidin-b2-image.tif")))
        dapi = norm16(tifffile.imread(str(DATA / "device-gold2-dapi-b2-image.tif")))
        merge = np.zeros((*phal.shape, 3))
        merge[:, :, 1] = phal
        merge[:, :, 2] = dapi
        ax_a1.imshow(merge)
    except Exception:
        ax_a1.text(0.5, 0.5, "Staining", ha="center", va="center",
                   transform=ax_a1.transAxes)
    ax_a1.set_title("A: Staining\n(XR1, gold, phalloidin+DAPI)", fontsize=9,
                     fontweight="bold")
    ax_a1.set_xticks([])
    ax_a1.set_yticks([])

    bf = read_frame(DATA / "C2-day8-gold-Jaz.avi")
    fl = read_frame(DATA / "C2-day8-gold-GCaMP6f-FL.avi")
    if bf is not None:
        ax_a2.imshow(bf, cmap="gray")
    ax_a2.set_title("A: BF + FL (C2 day8 gold)\nGCaMP6f: 0 beating, 1 transient",
                     fontsize=9, fontweight="bold")
    ax_a2.set_xticks([])
    ax_a2.set_yticks([])
    if fl is not None:
        ax_fl_inset = ax_a2.inset_axes([0.55, 0.02, 0.43, 0.43])
        ax_fl_inset.imshow(fl, cmap="inferno")
        ax_fl_inset.set_xticks([])
        ax_fl_inset.set_yticks([])
        ax_fl_inset.set_title("FL", fontsize=7, color="orange")

    # Panel B: Temporal color projection
    ax_b1 = fig.add_subplot(gs[0, 2])
    ax_b2 = fig.add_subplot(gs[0, 3])
    for ax_b, vname, ttl in [
        (ax_b1, "B3-day8-tammy-control.avi", "B: BF temporal color\nB3 XR1 beating"),
        (ax_b2, "C2-day6-gold-FL-control.avi", "B: FL temporal color\nC2 day6 gold"),
    ]:
        frames, fps = read_all_frames_gray(DATA / vname, half_res=True)
        if frames is not None:
            n = len(frames)
            h, w = frames[0].shape
            comp = np.zeros((h, w, 3), dtype=np.float32)
            mx = np.zeros((h, w), dtype=np.float32)
            step = max(1, n // 60)
            for j in range(0, n, step):
                t = j / max(n - 1, 1)
                color = np.array(cm.hsv(t)[:3])
                f = frames[j] / max(frames.max(), 1)
                mask = f > mx
                for c in range(3):
                    comp[:, :, c] = np.where(mask, f * color[c], comp[:, :, c])
                mx = np.maximum(mx, f)
            comp = np.clip(comp / (comp.max() + 1e-10), 0, 1)
            ax_b.imshow(comp)
            del frames
        ax_b.set_title(ttl, fontsize=9, fontweight="bold")
        ax_b.set_xticks([])
        ax_b.set_yticks([])

    # Panel C: Kymograph BF vs FL
    ax_c1 = fig.add_subplot(gs[1, 0])
    ax_c2 = fig.add_subplot(gs[1, 1])
    for ax_c, vname, cmp, ttl in [
        (ax_c1, "B3-day8-tammy-control.avi", "gray", "C: BF kymograph\nB3 XR1 beating"),
        (ax_c2, "C2-day6-gold-FL-control.avi", "inferno", "C: FL kymograph\nC2 day6 Ca2+"),
    ]:
        kymo, fps_k, _ = read_kymograph_row(DATA / vname)
        if kymo is not None:
            n_k, w = kymo.shape
            time_s = np.arange(n_k) / fps_k
            ax_c.imshow(kymo, aspect="auto", cmap=cmp,
                        extent=[0, w, time_s[-1], time_s[0]],
                        vmin=np.percentile(kymo, 3), vmax=np.percentile(kymo, 97))
        ax_c.set_title(ttl, fontsize=9, fontweight="bold")
        ax_c.set_xlabel("Position")
        ax_c.set_ylabel("Time (s)")

    # Panel D: Vector field
    ax_d = fig.add_subplot(gs[1, 2])
    cap = cv2.VideoCapture(str(DATA / "B3-day8-tammy-control.avi"))
    if cap.isOpened():
        fps_d = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, f0_raw = cap.read()
        prev = cv2.cvtColor(f0_raw, cv2.COLOR_BGR2GRAY) if ret else None
        best_mag, best_flow, best_fr = 0, None, None
        for idx in np.linspace(1, total - 1, 40, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret:
                continue
            curr = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            sp = cv2.resize(prev, (prev.shape[1] // 2, prev.shape[0] // 2))
            sc = cv2.resize(curr, (curr.shape[1] // 2, curr.shape[0] // 2))
            flow = cv2.calcOpticalFlowFarneback(sp, sc, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2).mean()
            if mag > best_mag:
                best_mag, best_flow, best_fr = mag, flow, curr
            prev = curr
        cap.release()
        if best_flow is not None:
            hf, wf = best_flow.shape[:2]
            step = 14
            Y, X = np.mgrid[0:hf:step, 0:wf:step]
            U = best_flow[::step, ::step, 0]
            V = best_flow[::step, ::step, 1]
            M = np.sqrt(U**2 + V**2)
            ax_d.imshow(cv2.resize(best_fr, (wf, hf)), cmap="gray")
            ax_d.quiver(X, Y, U, -V, M, cmap="inferno", scale=50,
                        width=0.004, headwidth=4, alpha=0.85)
    ax_d.set_title("D: Flow vectors at systole\nB3 XR1 beating", fontsize=9,
                    fontweight="bold")
    ax_d.set_xticks([])
    ax_d.set_yticks([])

    # Panel E: Activation map
    ax_e = fig.add_subplot(gs[1, 3])
    frames_e, fps_e = read_all_frames_gray(DATA / "C2-day6-gold-FL-control.avi",
                                            half_res=True)
    if frames_e is not None:
        mt = np.array([f.mean() for f in frames_e])
        pidx = np.argmax(mt[5:]) + 5
        win = max(1, int(fps_e * 0.5))
        seg = frames_e[max(0, pidx - win):min(len(frames_e), pidx + win)]
        if len(seg) > 2:
            block = 4
            h, w = seg[0].shape
            hb, wb = h // block, w // block
            ttp = np.zeros((hb, wb))
            for by in range(hb):
                for bx in range(wb):
                    roi = seg[:, by*block:(by+1)*block, bx*block:(bx+1)*block].mean(axis=(1, 2))
                    ttp[by, bx] = np.argmax(roi) * (1000.0 / fps_e)
            ax_e.imshow(ttp, cmap="jet", extent=[0, w, h, 0])
        del frames_e
    ax_e.set_title("E: Ca2+ activation map\nC2 day6 gold (isochrone)", fontsize=9,
                    fontweight="bold")
    ax_e.set_xticks([])
    ax_e.set_yticks([])

    # Panel F: Key findings text
    ax_f = fig.add_subplot(gs[2, :])
    ax_f.axis("off")
    findings = (
        "KEY FINDINGS\n\n"
        "1. ELECTROMECHANICAL DISSOCIATION: GCaMP6f cells show Ca2+ transients (7/8 FL videos) "
        "but 0/7 active BF beating\n"
        "2. XR1 cells beat spontaneously: 11/16 active beating, increasing amplitude day2→day8\n"
        "3. Substrate effect: Gold-PVDF shows ~3x higher Ca2+ amplitude vs non-poled (dF/F0 0.117 vs 0.041)\n"
        "4. Maturation hierarchy: F-actin organization → Ca2+ cycling → mechanical contraction\n"
        "5. CaV1 perturbation (Yang 2018): GCaMP6f CaM moiety partially impairs L-type Ca2+ channels\n"
        "6. Piezoelectric stimulation: significantly lower actin coherency (p<0.001), cytoskeletal remodeling"
    )
    ax_f.text(0.5, 0.5, findings, ha="center", va="center", fontsize=12,
              fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=1", fc="#1a1a2e", ec="#4a90d9",
                        linewidth=2, alpha=0.95),
              color="white", transform=ax_f.transAxes)

    fig.suptitle("X10: Combined Dashboard — Piezoelectric Substrate Effects on hiPSC-CMs\n"
                 "Three-modality analysis: Staining + Brightfield Video + Fluorescence Calcium Imaging",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(EXTRA / "X10_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved X10_dashboard.png")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("GENERATING FANCY VISUALIZATION FIGURES")
    print("=" * 60)
    print()

    print("Three-modality panel:")
    fig_fl6_three_modality()
    print()

    print("Extra figures (video_figures_extra/):")
    fig_x1_temporal_color_bf()
    fig_x2_temporal_color_fl()
    fig_x3_quiver()
    fig_x4_kymo_bf()
    fig_x5_kymo_fl()
    fig_x6_grid_all()
    fig_x7_isochrone()
    fig_x8_beat_cycle()
    fig_x9_fl_strip()
    fig_x10_dashboard()

    print()
    print("=" * 60)
    print("ALL FANCY FIGURES COMPLETE")
    print(f"  FL6 -> {REPORT}/")
    print(f"  X1-X10 -> {EXTRA}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
