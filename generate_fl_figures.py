"""
Generate visual figures for fluorescence calcium imaging analysis.

FL1: Overview - first frames from all 8 FL videos
FL2: Temporal progression - C2 and C3 across days (day2→day6→day8)
FL3: Calcium heatmaps - spatial activity maps (std projection + peak regions)
FL4: Transient montage - peak vs diastole frames for usable videos
FL5: BF vs FL comparison - same well, same day, side by side
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

DATA = Path("Cardiomyocytes")
OUT = Path("final_report")
FL_OUT = Path("fl_results")


def read_frame(filepath, frame_idx=0):
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def read_all_frames(filepath):
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        return None, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64))
    cap.release()
    return np.array(frames), fps


FL_VIDEOS = {
    "Jaz-day2-fluorescence-gold.avi": ("day2", "C2", "gold", "10 FPS"),
    "C2-day2-gold-FL.avi": ("day2", "C2", "gold", "2 FPS"),
    "C3-day2-fl.avi": ("day2", "C3", "non-poled", "1.1 FPS"),
    "C2-day6-gold-FL-control.avi": ("day6", "C2", "gold (ctrl)", "10 FPS"),
    "C2-day6-non-FL.avi": ("day6", "C2", "gold", "2.6 FPS"),
    "C3-day6-gold-fl.avi": ("day6", "C3", "non-poled", "3.2 FPS"),
    "C2-day8-gold-GCaMP6f-FL.avi": ("day8", "C2", "gold (dev)", "12.5 FPS"),
    "C3-day8-non-Jaz-fl.avi": ("day8", "C3", "non-poled", "11.1 FPS"),
}

BF_FL_PAIRS = [
    ("C2-day2-gold.avi", "C2-day2-gold-FL.avi", "C2, day2, gold"),
    ("C3-day2.avi", "C3-day2-fl.avi", "C3, day2, non-poled"),
    ("C2-day6-gold-control-bf.avi", "C2-day6-gold-FL-control.avi", "C2, day6, gold (ctrl)"),
    ("C2-day8-gold-Jaz.avi", "C2-day8-gold-GCaMP6f-FL.avi", "C2, day8, gold (dev)"),
    ("C3-day8-non-Jaz.avi", "C3-day8-non-Jaz-fl.avi", "C3, day8, non-poled"),
]


def fig_fl1_overview():
    """First frame from each FL video in a grid."""
    print("  Fig FL1: FL overview frames...")

    n = len(FL_VIDEOS)
    fig, axes = plt.subplots(2, 4, figsize=(20, 11))
    axes = axes.flatten()

    fl_results = pd.read_csv(FL_OUT / "fl_batch_results.csv")

    for i, (fname, (day, well, substrate, fps_str)) in enumerate(FL_VIDEOS.items()):
        ax = axes[i]
        fpath = DATA / fname
        frame = read_frame(fpath)

        row = fl_results[fl_results["filename"] == fname]
        classification = row["classification"].values[0] if len(row) > 0 else "?"
        n_trans = int(row["n_transients"].values[0]) if len(row) > 0 else 0

        if frame is not None:
            ax.imshow(frame, cmap="inferno", vmin=0, vmax=np.percentile(frame, 99.5))
            ax.set_title(f"{well} {day} {substrate}\n{fps_str} | {classification}\n"
                         f"{n_trans} transients", fontsize=9, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "File not found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(f"{well} {day} {substrate}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Figure FL1: Fluorescence Overview — First Frame from Each FL Video\n"
                 "GCaMP6f calcium indicator (inferno colormap: dark=low Ca2+, bright=high Ca2+)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "fl_Fig_FL1_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved fl_Fig_FL1_overview.png")


def fig_fl2_temporal():
    """Temporal progression for C2 and C3 across days."""
    print("  Fig FL2: Temporal progression...")

    wells = {
        "C2": [
            ("day2", "C2-day2-gold-FL.avi", "gold, 2 FPS"),
            ("day6", "C2-day6-gold-FL-control.avi", "gold ctrl, 10 FPS"),
            ("day8", "C2-day8-gold-GCaMP6f-FL.avi", "gold dev, 12.5 FPS"),
        ],
        "C3": [
            ("day2", "C3-day2-fl.avi", "non-poled, 1.1 FPS"),
            ("day6", "C3-day6-gold-fl.avi", "non-poled, 3.2 FPS"),
            ("day8", "C3-day8-non-Jaz-fl.avi", "non-poled, 11.1 FPS"),
        ],
    }

    fl_results = pd.read_csv(FL_OUT / "fl_batch_results.csv")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for row_i, (well, videos) in enumerate(wells.items()):
        for col_i, (day, fname, desc) in enumerate(videos):
            ax = axes[row_i, col_i]
            fpath = DATA / fname
            frame = read_frame(fpath)

            r = fl_results[fl_results["filename"] == fname]
            cls = r["classification"].values[0] if len(r) > 0 else "?"
            n_t = int(r["n_transients"].values[0]) if len(r) > 0 else 0
            amp = r["mean_amplitude_dff0"].values[0] if len(r) > 0 else 0

            if frame is not None:
                ax.imshow(frame, cmap="inferno", vmin=0,
                          vmax=np.percentile(frame, 99.5))
            else:
                ax.text(0.5, 0.5, "Not found", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="white")
                ax.set_facecolor("black")

            ax.set_title(f"{day} — {desc}\n{cls}: {n_t} trans, dF/F0={amp:.3f}",
                         fontsize=9, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

            if col_i == 0:
                ax.set_ylabel(f"{well}\n(GCaMP6f)", fontsize=12, fontweight="bold",
                              labelpad=10)

    fig.suptitle("Figure FL2: Fluorescence Temporal Progression\n"
                 "GCaMP6f calcium imaging across days (day2 → day6 → day8)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "fl_Fig_FL2_temporal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved fl_Fig_FL2_temporal.png")


def fig_fl3_heatmaps():
    """Spatial calcium activity maps using temporal std and max-min projection."""
    print("  Fig FL3: Calcium heatmaps...")

    usable = [
        ("Jaz-day2-fluorescence-gold.avi", "C2 day2 gold"),
        ("C2-day6-gold-FL-control.avi", "C2 day6 gold (ctrl)"),
        ("C2-day8-gold-GCaMP6f-FL.avi", "C2 day8 gold (dev)"),
        ("C3-day8-non-Jaz-fl.avi", "C3 day8 non-poled"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))

    for col_i, (fname, label) in enumerate(usable):
        fpath = DATA / fname
        frames, fps = read_all_frames(fpath)

        if frames is None:
            for r in range(2):
                axes[r, col_i].text(0.5, 0.5, "Not found", ha="center",
                                    va="center", transform=axes[r, col_i].transAxes)
                axes[r, col_i].set_title(label)
            continue

        first_frame = frames[0]
        std_proj = np.std(frames, axis=0)
        max_proj = np.max(frames, axis=0)
        min_proj = np.min(frames, axis=0)
        delta_proj = max_proj - min_proj

        ax = axes[0, col_i]
        ax.imshow(first_frame, cmap="gray", vmin=0, vmax=np.percentile(first_frame, 99))
        im = ax.imshow(std_proj, cmap="hot", alpha=0.6,
                       vmin=np.percentile(std_proj, 10),
                       vmax=np.percentile(std_proj, 99))
        ax.set_title(f"{label}\nTemporal SD overlay (hot=active)", fontsize=9,
                     fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1, col_i]
        im2 = ax.imshow(delta_proj, cmap="inferno",
                        vmin=np.percentile(delta_proj, 5),
                        vmax=np.percentile(delta_proj, 99))
        ax.set_title(f"Max-Min projection\n(calcium range per pixel)", fontsize=9,
                     fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        del frames

    axes[0, 0].set_ylabel("Temporal SD\n(overlay on frame)", fontsize=11,
                          fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("Max-Min\n(calcium range)", fontsize=11,
                          fontweight="bold", labelpad=10)

    fig.suptitle("Figure FL3: Spatial Calcium Activity Maps (usable FPS videos only)\n"
                 "Top: temporal standard deviation highlights regions with fluctuating Ca2+\n"
                 "Bottom: max-min projection shows total calcium range per pixel",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(OUT / "fl_Fig_FL3_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved fl_Fig_FL3_heatmaps.png")


def fig_fl4_transient_montage():
    """Show peak vs diastole frames for usable videos."""
    print("  Fig FL4: Transient montage (peak vs diastole)...")

    usable = [
        ("Jaz-day2-fluorescence-gold.avi", "C2 day2 gold"),
        ("C2-day6-gold-FL-control.avi", "C2 day6 gold (ctrl)"),
        ("C2-day8-gold-GCaMP6f-FL.avi", "C2 day8 gold (dev)"),
        ("C3-day8-non-Jaz-fl.avi", "C3 day8 non-poled"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(22, 15))

    for col_i, (fname, label) in enumerate(usable):
        fpath = DATA / fname
        frames, fps = read_all_frames(fpath)

        if frames is None or len(frames) < 10:
            for r in range(3):
                axes[r, col_i].text(0.5, 0.5, "Not found", ha="center",
                                    va="center", transform=axes[r, col_i].transAxes)
            continue

        mean_trace = np.array([f.mean() for f in frames])
        peak_idx = np.argmax(mean_trace[5:]) + 5
        diastole_idx = np.argmin(mean_trace[5:]) + 5

        vmin = np.percentile(frames, 2)
        vmax = np.percentile(frames, 99.5)

        ax = axes[0, col_i]
        ax.imshow(frames[diastole_idx], cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"{label}\nDiastole (frame {diastole_idx})\nMean={mean_trace[diastole_idx]:.1f}",
                     fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1, col_i]
        ax.imshow(frames[peak_idx], cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"Peak (frame {peak_idx})\nMean={mean_trace[peak_idx]:.1f}",
                     fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[2, col_i]
        diff = frames[peak_idx].astype(np.float64) - frames[diastole_idx].astype(np.float64)
        abs_max = max(np.abs(np.percentile(diff, 2)), np.abs(np.percentile(diff, 98)))
        if abs_max < 1:
            abs_max = 1
        ax.imshow(diff, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
        ax.set_title(f"Difference (peak - diastole)\nRange: [{diff.min():.1f}, {diff.max():.1f}]",
                     fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        del frames

    axes[0, 0].set_ylabel("DIASTOLE\n(min intensity)", fontsize=11,
                          fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("PEAK\n(max intensity)", fontsize=11,
                          fontweight="bold", labelpad=10)
    axes[2, 0].set_ylabel("DIFFERENCE\n(peak - diastole)", fontsize=11,
                          fontweight="bold", labelpad=10)

    fig.suptitle("Figure FL4: Calcium Transient Montage — Peak vs Diastole\n"
                 "Top: diastolic (resting) frame | Middle: peak (systolic) frame | "
                 "Bottom: difference map (red=Ca2+ increase, blue=decrease)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(OUT / "fl_Fig_FL4_transient_montage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved fl_Fig_FL4_transient_montage.png")


def fig_fl5_bf_vs_fl():
    """BF vs FL side-by-side for matched recordings."""
    print("  Fig FL5: BF vs FL comparison...")

    n_pairs = len(BF_FL_PAIRS)
    fig, axes = plt.subplots(2, n_pairs, figsize=(5 * n_pairs, 10))

    fl_results = pd.read_csv(FL_OUT / "fl_batch_results.csv")

    for col_i, (bf_file, fl_file, label) in enumerate(BF_FL_PAIRS):
        bf_path = DATA / bf_file
        fl_path = DATA / fl_file

        bf_frame = read_frame(bf_path)
        fl_frame = read_frame(fl_path)

        fl_row = fl_results[fl_results["filename"] == fl_file]
        fl_cls = fl_row["classification"].values[0] if len(fl_row) > 0 else "?"
        fl_n = int(fl_row["n_transients"].values[0]) if len(fl_row) > 0 else 0

        ax = axes[0, col_i]
        if bf_frame is not None:
            ax.imshow(bf_frame, cmap="gray")
            ax.set_title(f"BF: {label}\n{bf_frame.shape[1]}x{bf_frame.shape[0]}",
                         fontsize=9, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "BF not found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(f"BF: {label}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1, col_i]
        if fl_frame is not None:
            ax.imshow(fl_frame, cmap="inferno", vmin=0,
                      vmax=np.percentile(fl_frame, 99.5))
            ax.set_title(f"FL: {fl_cls}, {fl_n} trans\n{fl_frame.shape[1]}x{fl_frame.shape[0]}",
                         fontsize=9, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "FL not found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(f"FL: {label}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel("BRIGHTFIELD\n(1392x1040)", fontsize=12,
                          fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("FLUORESCENCE\n(348x260, GCaMP6f)", fontsize=12,
                          fontweight="bold", labelpad=10)

    fig.suptitle("Figure FL5: Brightfield vs Fluorescence — Same Well, Same Day\n"
                 "BF: 0/7 active beating | FL: calcium transients detected in most\n"
                 "Key question: calcium present but mechanically uncoupled?",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(OUT / "fl_Fig_FL5_bf_vs_fl.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved fl_Fig_FL5_bf_vs_fl.png")


def main():
    print("=" * 60)
    print("GENERATING FLUORESCENCE VISUAL FIGURES")
    print("=" * 60)
    print()

    fig_fl1_overview()
    fig_fl2_temporal()
    fig_fl3_heatmaps()
    fig_fl4_transient_montage()
    fig_fl5_bf_vs_fl()

    print()
    print("=" * 60)
    print("FL FIGURES COMPLETE")
    print(f"All saved to {OUT}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
