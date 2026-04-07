"""
Fluorescence calcium transient analysis for GCaMP6f hiPSC-CMs.

Intensity-based pipeline (NOT optical flow):
  1. Read frames as grayscale
  2. Background subtraction (border region)
  3. Extract mean fluorescence trace F(t)
  4. Photobleach correction (mono-exponential fit)
  5. Compute dF/F0 (rolling 5th percentile baseline)
  6. Detect calcium transients (scipy find_peaks)
  7. Per-transient kinetics (amplitude, TTP, CaTD50, CaTD90, tau, dF/dt)

References:
  Psaras et al. Circ Res 2021 (CalTrack)
  Bedut et al. Front Physiol 2022
  Yang et al. Nat Commun 2018 (GCaMP-X / CaV1 perturbation)
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

DATA = Path("Cardiomyocytes")
OUT = Path("fl_results")
OUT.mkdir(exist_ok=True)

FL_VIDEOS = [
    "Jaz-day2-fluorescence-gold.avi",
    "C2-day2-gold-FL.avi",
    "C3-day2-fl.avi",
    "C2-day6-gold-FL-control.avi",
    "C2-day6-non-FL.avi",
    "C3-day6-gold-fl.avi",
    "C2-day8-gold-GCaMP6f-FL.avi",
    "C3-day8-non-Jaz-fl.avi",
]

WELL_MAP = {
    "Jaz-day2-fluorescence-gold.avi": ("day2", "C2", "gold", "GCaMP6f", "baseline"),
    "C2-day2-gold-FL.avi": ("day2", "C2", "gold", "GCaMP6f", "baseline"),
    "C3-day2-fl.avi": ("day2", "C3", "non-poled", "GCaMP6f", "baseline"),
    "C2-day6-gold-FL-control.avi": ("day6", "C2", "gold", "GCaMP6f", "control"),
    "C2-day6-non-FL.avi": ("day6", "C2", "gold", "GCaMP6f", "on-device"),
    "C3-day6-gold-fl.avi": ("day6", "C3", "non-poled", "GCaMP6f", "on-device"),
    "C2-day8-gold-GCaMP6f-FL.avi": ("day8", "C2", "gold", "GCaMP6f", "on-device"),
    "C3-day8-non-Jaz-fl.avi": ("day8", "C3", "non-poled", "GCaMP6f", "on-device"),
}


def load_fl_video(filepath):
    """Load FL AVI as grayscale frames + fps."""
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        raise IOError(f"Cannot open {filepath}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        frames.append(gray)
    cap.release()
    return np.array(frames), fps


def extract_traces(frames):
    """Extract whole-field and background traces.
    Background = 5-pixel border strip (typically cell-free in FL).
    """
    h, w = frames[0].shape
    border = 5
    bg_mask = np.zeros((h, w), dtype=bool)
    bg_mask[:border, :] = True
    bg_mask[-border:, :] = True
    bg_mask[:, :border] = True
    bg_mask[:, -border:] = True

    cell_mask = ~bg_mask

    raw_trace = np.array([f[cell_mask].mean() for f in frames])
    bg_trace = np.array([f[bg_mask].mean() for f in frames])
    corrected = raw_trace - bg_trace

    return raw_trace, bg_trace, corrected


def mono_exp(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def correct_photobleach(trace, time_s):
    """Fit mono-exponential to diastolic envelope and divide out."""
    n = len(trace)
    if n < 10:
        return trace, np.ones(n)

    window = max(5, n // 10)
    baseline_env = np.array([
        np.percentile(trace[max(0, i - window):i + window + 1], 10)
        for i in range(n)
    ])

    try:
        p0 = [trace[0] - trace[-1], max(time_s[-1] / 3, 1.0), trace[-1]]
        bounds = ([0, 0.1, -np.inf], [np.inf, time_s[-1] * 10, np.inf])
        popt, _ = curve_fit(mono_exp, time_s, baseline_env, p0=p0,
                            bounds=bounds, maxfev=5000)
        bleach_curve = mono_exp(time_s, *popt)
        bleach_curve = np.maximum(bleach_curve, 1.0)
        corrected = trace / bleach_curve * bleach_curve[0]
    except (RuntimeError, ValueError):
        corrected = trace
        bleach_curve = np.ones(n) * np.mean(trace)

    return corrected, bleach_curve


def compute_df_f0(trace, fps, window_s=2.0):
    """Compute dF/F0 using rolling 5th percentile as F0."""
    window = max(3, int(window_s * fps))
    f0 = np.array([
        np.percentile(trace[max(0, i - window):i + window + 1], 5)
        for i in range(len(trace))
    ])
    f0 = np.maximum(f0, 1.0)
    df_f0 = (trace - f0) / f0
    return df_f0, f0


def detect_transients(df_f0, fps):
    """Detect calcium transients using peak detection."""
    if len(df_f0) < 5:
        return np.array([]), {}

    if fps >= 5:
        smooth = savgol_filter(df_f0, window_length=min(7, len(df_f0) // 2 * 2 + 1),
                               polyorder=2)
    else:
        smooth = df_f0.copy()

    noise = np.std(df_f0[:max(5, int(fps))])
    prominence = max(0.02, noise * 3)
    min_distance = max(2, int(0.3 * fps))

    peaks, props = find_peaks(smooth, prominence=prominence,
                              distance=min_distance, height=0.01)

    return peaks, props


def extract_kinetics(df_f0, peaks, fps):
    """Extract per-transient kinetics."""
    dt = 1.0 / fps
    results = []

    for pk in peaks:
        amplitude = df_f0[pk]

        onset = pk
        for j in range(pk - 1, max(pk - int(2 * fps), -1), -1):
            if j < 0:
                break
            if df_f0[j] <= amplitude * 0.1:
                onset = j
                break

        ttp_ms = (pk - onset) * dt * 1000

        half_amp = amplitude * 0.5
        amp_10 = amplitude * 0.1

        catd50_ms = np.nan
        catd90_ms = np.nan
        for j in range(pk + 1, min(pk + int(3 * fps), len(df_f0))):
            if df_f0[j] <= half_amp and np.isnan(catd50_ms):
                catd50_ms = (j - pk) * dt * 1000
            if df_f0[j] <= amp_10 and np.isnan(catd90_ms):
                catd90_ms = (j - pk) * dt * 1000
                break

        tau_ms = np.nan
        decay_start = pk + 1
        decay_end = min(pk + int(2 * fps), len(df_f0))
        if decay_end - decay_start > 3:
            decay_seg = df_f0[decay_start:decay_end]
            t_seg = np.arange(len(decay_seg)) * dt
            try:
                popt, _ = curve_fit(
                    lambda t, a, tau: a * np.exp(-t / tau),
                    t_seg, decay_seg,
                    p0=[amplitude, 0.3],
                    bounds=([0, 0.01], [amplitude * 2, 5.0]),
                    maxfev=2000
                )
                tau_ms = popt[1] * 1000
            except (RuntimeError, ValueError):
                pass

        deriv = np.gradient(df_f0, dt)
        rise_window = slice(max(onset, 0), pk + 1)
        decay_window = slice(pk, min(pk + int(2 * fps), len(df_f0)))
        upstroke_vel = np.max(deriv[rise_window]) if rise_window.stop > rise_window.start else np.nan
        decay_vel = np.min(deriv[decay_window]) if decay_window.stop > decay_window.start else np.nan

        results.append({
            "peak_idx": pk,
            "peak_time_s": pk * dt,
            "amplitude_dff0": amplitude,
            "ttp_ms": ttp_ms,
            "catd50_ms": catd50_ms,
            "catd90_ms": catd90_ms,
            "decay_tau_ms": tau_ms,
            "upstroke_vel": upstroke_vel,
            "decay_vel": decay_vel,
        })

    return results


def classify_fl(n_transients, mean_amplitude, fps):
    """Classify FL signal quality."""
    if fps < 5:
        fps_flag = "low_fps"
    else:
        fps_flag = "ok"

    if n_transients == 0:
        classification = "no_transients"
    elif mean_amplitude < 0.03:
        classification = "weak_transients"
    elif n_transients >= 2:
        classification = "active_transients"
    else:
        classification = "single_transient"

    return classification, fps_flag


def generate_fl_figure(filename, time_s, raw_trace, bg_trace, corrected,
                       bleach_curve, df_f0, f0, peaks, kinetics,
                       classification, fps, fps_flag, meta):
    """Generate 4-panel analysis figure."""
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 1, hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_s, raw_trace, "b-", alpha=0.7, linewidth=0.8, label="Raw F(t)")
    ax1.plot(time_s, bg_trace, "gray", alpha=0.5, linewidth=0.8, label="Background")
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.set_title(f"A) Raw Fluorescence Trace — {filename}", fontweight="bold")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time_s, corrected, "g-", alpha=0.7, linewidth=0.8, label="BG-subtracted")
    ax2.plot(time_s, bleach_curve, "r--", linewidth=1.5, label="Bleach fit")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.set_title("B) Background-Subtracted + Photobleach Fit", fontweight="bold")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(time_s, df_f0, "k-", linewidth=0.8)
    if len(peaks) > 0:
        ax3.plot(time_s[peaks], df_f0[peaks], "rv", markersize=8,
                 label=f"{len(peaks)} transients detected")
    ax3.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax3.set_ylabel("dF/F0")
    ax3.set_xlabel("Time (s)")
    ax3.set_title(f"C) dF/F0 — Classification: {classification.upper()}"
                  f" (FPS: {fps_flag})", fontweight="bold")
    if len(peaks) > 0:
        ax3.legend(fontsize=9)

    ax4 = fig.add_subplot(gs[3])
    ax4.axis("off")

    day, well, substrate, cell_type, location = meta
    n_trans = len(kinetics)

    summary_lines = [
        f"File: {filename}",
        f"Well: {well}  |  Day: {day}  |  Substrate: {substrate}  |  Cell line: {cell_type}  |  Location: {location}",
        f"FPS: {fps:.1f}  |  Duration: {time_s[-1]:.1f}s  |  Frames: {len(time_s)}",
        f"Classification: {classification}  |  FPS flag: {fps_flag}",
        f"Transients detected: {n_trans}",
    ]

    if n_trans > 0:
        amps = [k["amplitude_dff0"] for k in kinetics]
        ttps = [k["ttp_ms"] for k in kinetics if not np.isnan(k["ttp_ms"])]
        catd50s = [k["catd50_ms"] for k in kinetics if not np.isnan(k["catd50_ms"])]
        catd90s = [k["catd90_ms"] for k in kinetics if not np.isnan(k["catd90_ms"])]
        taus = [k["decay_tau_ms"] for k in kinetics if not np.isnan(k["decay_tau_ms"])]

        summary_lines.append(f"Mean amplitude (dF/F0): {np.mean(amps):.4f}")
        if ttps:
            summary_lines.append(f"Mean TTP: {np.mean(ttps):.1f} ms")
        if catd50s:
            summary_lines.append(f"Mean CaTD50: {np.mean(catd50s):.1f} ms")
        if catd90s:
            summary_lines.append(f"Mean CaTD90: {np.mean(catd90s):.1f} ms")
        if taus:
            summary_lines.append(f"Mean decay tau: {np.mean(taus):.1f} ms")

        if n_trans >= 2:
            ibis = np.diff([k["peak_time_s"] for k in kinetics])
            bpm = 60.0 / np.mean(ibis) if np.mean(ibis) > 0 else 0
            ibi_cv = np.std(ibis) / np.mean(ibis) * 100 if np.mean(ibis) > 0 else 0
            summary_lines.append(f"Beat rate: {bpm:.1f} BPM  |  IBI CV: {ibi_cv:.1f}%")

    for i, line in enumerate(summary_lines):
        ax4.text(0.02, 0.95 - i * 0.11, line, transform=ax4.transAxes,
                 fontsize=10, verticalalignment="top", fontfamily="monospace")

    ax4.set_title("D) Summary", fontweight="bold")

    fig.suptitle(
        f"Fluorescence Calcium Transient Analysis — {filename}\n"
        f"GCaMP6f intensity-based pipeline (NOT optical flow)",
        fontsize=13, fontweight="bold", y=1.01)

    stem = filename.replace(".avi", "")
    out_path = OUT / f"{stem}_fl_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved {out_path}")
    return out_path


def process_fl_video(filename):
    """Full pipeline for one FL video."""
    filepath = DATA / filename
    if not filepath.exists():
        print(f"  [SKIP] {filename} -- file not found")
        return None

    print(f"  Processing: {filename}")
    meta = WELL_MAP.get(filename, ("unknown", "unknown", "unknown", "GCaMP6f", "unknown"))
    day, well, substrate, cell_type, location = meta

    frames, fps = load_fl_video(filepath)
    n_frames = len(frames)
    duration = n_frames / fps if fps > 0 else 0
    time_s = np.arange(n_frames) / fps

    print(f"    {n_frames} frames, {fps:.1f} FPS, {duration:.1f}s")

    raw_trace, bg_trace, corrected = extract_traces(frames)
    resolution_str = f"{frames[0].shape[1]}x{frames[0].shape[0]}"
    del frames

    corrected_debleach, bleach_curve = correct_photobleach(corrected, time_s)
    df_f0, f0 = compute_df_f0(corrected_debleach, fps)
    peaks, props = detect_transients(df_f0, fps)
    kinetics = extract_kinetics(df_f0, peaks, fps)

    n_trans = len(kinetics)
    mean_amp = np.mean([k["amplitude_dff0"] for k in kinetics]) if n_trans > 0 else 0
    classification, fps_flag = classify_fl(n_trans, mean_amp, fps)

    print(f"    Classification: {classification}, {n_trans} transients, "
          f"mean amp={mean_amp:.4f}")

    generate_fl_figure(filename, time_s, raw_trace, bg_trace, corrected,
                       bleach_curve, df_f0, f0, peaks, kinetics,
                       classification, fps, fps_flag, meta)

    row = {
        "filename": filename,
        "day": day,
        "well": well,
        "substrate": substrate,
        "cell_type": cell_type,
        "location": location,
        "fps": fps,
        "n_frames": n_frames,
        "duration_s": round(duration, 2),
        "resolution": resolution_str,
        "classification": classification,
        "fps_flag": fps_flag,
        "n_transients": n_trans,
        "mean_amplitude_dff0": round(mean_amp, 5) if n_trans > 0 else 0,
    }

    if n_trans > 0:
        ttps = [k["ttp_ms"] for k in kinetics if not np.isnan(k["ttp_ms"])]
        catd50s = [k["catd50_ms"] for k in kinetics if not np.isnan(k["catd50_ms"])]
        catd90s = [k["catd90_ms"] for k in kinetics if not np.isnan(k["catd90_ms"])]
        taus = [k["decay_tau_ms"] for k in kinetics if not np.isnan(k["decay_tau_ms"])]
        row["mean_ttp_ms"] = round(np.mean(ttps), 1) if ttps else np.nan
        row["mean_catd50_ms"] = round(np.mean(catd50s), 1) if catd50s else np.nan
        row["mean_catd90_ms"] = round(np.mean(catd90s), 1) if catd90s else np.nan
        row["mean_decay_tau_ms"] = round(np.mean(taus), 1) if taus else np.nan

        upstrokes = [k["upstroke_vel"] for k in kinetics if not np.isnan(k["upstroke_vel"])]
        decays = [k["decay_vel"] for k in kinetics if not np.isnan(k["decay_vel"])]
        row["mean_upstroke_vel"] = round(np.mean(upstrokes), 5) if upstrokes else np.nan
        row["mean_decay_vel"] = round(np.mean(decays), 5) if decays else np.nan

        if n_trans >= 2:
            ibis = np.diff([k["peak_time_s"] for k in kinetics])
            row["bpm"] = round(60.0 / np.mean(ibis), 1) if np.mean(ibis) > 0 else 0
            row["ibi_cv_pct"] = round(np.std(ibis) / np.mean(ibis) * 100, 1) if np.mean(ibis) > 0 else 0
        else:
            row["bpm"] = np.nan
            row["ibi_cv_pct"] = np.nan
    else:
        for col in ["mean_ttp_ms", "mean_catd50_ms", "mean_catd90_ms",
                     "mean_decay_tau_ms", "mean_upstroke_vel", "mean_decay_vel",
                     "bpm", "ibi_cv_pct"]:
            row[col] = np.nan

    return row


def main():
    print("=" * 60)
    print("FLUORESCENCE CALCIUM TRANSIENT ANALYSIS")
    print("GCaMP6f intensity-based pipeline")
    print("=" * 60)
    print()

    results = []
    for fname in FL_VIDEOS:
        row = process_fl_video(fname)
        if row is not None:
            results.append(row)
        print()

    if results:
        df = pd.DataFrame(results)
        csv_path = OUT / "fl_batch_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        print(f"\nSummary: {len(results)} videos processed")
        for _, r in df.iterrows():
            print(f"  {r['filename']}: {r['classification']} "
                  f"({r['n_transients']} transients, "
                  f"amp={r['mean_amplitude_dff0']:.4f})")
    else:
        print("No videos processed.")

    print("\n" + "=" * 60)
    print("FLUORESCENCE ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
