"""
Video analysis pipeline for cardiomyocyte beating (BF).
v3: Per-center contractile foci detection via convergence thresholding.
    Builds on v2 (convergence, detrending, per-beat kinetics, classification).

Key additions in v3:
  - detect_contractile_foci(): threshold convergence map → label connected
    components → beat detection per foci → n_foci, coverage, per-foci metrics
  - "individual_cells_beating" classification for videos where whole-field
    signal is sub-threshold but individual contractile centers are detected
  - Foci contours (cyan) overlaid on convergence map panel in figures

Reference: Sci Rep 7, 10094 (2017) — optical-flow based cardiomyocyte analysis.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from mps_motion import MPSData, OpticalFlow, FLOW_ALGORITHMS
from scipy.signal import find_peaks, detrend
from scipy.ndimage import median_filter, label as ndimage_label
import gc
import warnings
warnings.filterwarnings("ignore")

DATA = Path("Cardiomyocytes")
OUT = Path("video_results")
OUT.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def load_avi_as_mpsdata(filepath, grayscale=True, spatial_downsample=2):
    """Load AVI → MPSData. spatial_downsample=2 halves each dim (~4x RAM savings)."""
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        raise IOError(f"Cannot open {filepath}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        if spatial_downsample > 1:
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (w // spatial_downsample, h // spatial_downsample),
                              interpolation=cv2.INTER_AREA)
        frames_list.append(gray)
    cap.release()

    frames = np.stack(frames_list, axis=-1).astype(np.float32)
    del frames_list
    time_stamps = np.arange(frames.shape[-1]) / fps * 1000  # ms
    info = {"num_frames": frames.shape[-1], "fps": fps, "dt": 1000.0 / fps}
    return MPSData(frames=frames, time_stamps=time_stamps, info=info), fps


def parse_video_metadata(filename):
    """Extract experimental metadata from video filename."""
    name = filename.lower().replace(".avi", "")

    day = "unknown"
    for d in ["day1", "day2", "day3", "day6", "day8"]:
        if d in name:
            day = d
            break

    well = "unknown"
    for w in ["a2", "a3", "b2", "b3", "c2", "c3"]:
        if w in name:
            well = w.upper()
            break

    SUBSTRATE_OVERRIDES = {
        "c2-day6-non-fl.avi": "gold",      # filename says "non" but C2 = gold
        "c3-day6-gold-fl.avi": "non",       # filename says "gold" but C3 = non-poled
    }
    substrate = SUBSTRATE_OVERRIDES.get(name, "gold" if "gold" in name else "non")

    if day in ("day1", "day2"):
        recording_type = "baseline"
    elif "control" in name or "comtrol" in name:
        recording_type = "control"
    else:
        recording_type = "on-device"

    cell_type = "XR1" if "tammy" in name else ("GCaMP6f" if "jaz" in name else "Unknown")

    phase = "baseline" if day in ("day1", "day2") else "treatment"

    return {
        "day": day, "well": well, "substrate": substrate,
        "recording_type": recording_type, "cell_type": cell_type,
        "phase": phase,
    }


# ═══════════════════════════════════════════════════════════════
# Convergence analysis
# ═══════════════════════════════════════════════════════════════

def compute_convergence(displacements):
    """Compute convergence (= -divergence) of the displacement field.

    Positive convergence → active contraction (flow converging inward).
    Zero / negative → passive deformation, drift, or noise.

    Memory-efficient: accumulates per-frame without storing full 3D array.
    Returns (convergence_map, convergence_trace, peak_convergence_map).
    """
    disp_x = np.array(displacements.x.compute())  # (H, W, T)
    disp_y = np.array(displacements.y.compute())
    T = disp_x.shape[-1]
    H, W = disp_x.shape[:2]

    conv_pos_accum = np.zeros((H, W), dtype=np.float64)
    peak_conv_map = np.zeros((H, W), dtype=np.float32)
    conv_trace = np.zeros(T, dtype=np.float32)

    for t in range(T):
        du_x_dx = np.gradient(disp_x[:, :, t], axis=1)
        du_y_dy = np.gradient(disp_y[:, :, t], axis=0)
        conv_t = -(du_x_dx + du_y_dy)
        conv_pos = np.clip(conv_t, 0, None)
        conv_pos_accum += conv_pos
        conv_trace[t] = np.mean(conv_pos)
        np.maximum(peak_conv_map, conv_pos, out=peak_conv_map)

    del disp_x, disp_y

    convergence_map = (conv_pos_accum / T).astype(np.float32)
    return convergence_map, conv_trace, peak_conv_map


# ═══════════════════════════════════════════════════════════════
# Baseline drift removal
# ═══════════════════════════════════════════════════════════════

def detrend_trace(trace, fps):
    """Remove baseline drift using rolling-median subtraction.

    A 10-second rolling median preserves beats ≥6 BPM while removing
    slow drift (stage motion, focus drift, media convection).
    Falls back to linear detrend for very short traces.
    """
    n = len(trace)
    if n < 20:
        return trace - np.median(trace)

    window = max(3, int(fps * 10))
    if window % 2 == 0:
        window += 1
    if window >= n:
        return detrend(trace, type="linear")

    baseline = median_filter(trace, size=window)
    return trace - baseline


# ═══════════════════════════════════════════════════════════════
# Beat detection with per-beat kinetics
# ═══════════════════════════════════════════════════════════════

def _empty_beat_result():
    return {
        "num_beats": 0, "bpm": 0, "regularity": 0, "ibi_cv": 0,
        "mean_amplitude": 0, "peak_indices": None, "inverted": False,
        "mean_contraction_time_s": 0, "mean_relaxation_time_s": 0,
        "mean_contraction_vel": 0, "mean_relaxation_vel": 0,
        "relax_contract_ratio": 0,
    }


def detect_beats(trace, fps, min_prominence_frac=0.25):
    """Detect beats and extract per-beat kinetics.

    Auto-detects whether beats appear as positive peaks or negative dips
    (contraction direction varies by video) and uses whichever is more
    prominent.  Returns dict with BPM, regularity, IBI CV, per-beat
    contraction/relaxation times and velocities, relaxation/contraction
    time ratio, and an `inverted` flag (True when beats are negative dips).
    """
    trace_range = float(np.max(trace) - np.min(trace))
    if trace_range < 1e-6:
        return _empty_beat_result()

    prominence = trace_range * min_prominence_frac
    min_distance = max(1, int(fps * 0.4))

    peaks_pos, props_pos = find_peaks(trace, prominence=prominence, distance=min_distance)
    peaks_neg, props_neg = find_peaks(-trace, prominence=prominence, distance=min_distance)

    prom_pos = float(np.sum(props_pos.get("prominences", [0])))
    prom_neg = float(np.sum(props_neg.get("prominences", [0])))

    if prom_neg > prom_pos and len(peaks_neg) > 0:
        work_trace = -trace
        peaks = peaks_neg
        inverted = True
    else:
        work_trace = trace
        peaks = peaks_pos
        inverted = False

    if len(peaks) == 0:
        return _empty_beat_result()

    # BPM and regularity
    ibi_cv = 0.0
    if len(peaks) >= 2:
        intervals_s = np.diff(peaks) / fps
        bpm = 60.0 / np.mean(intervals_s)
        ibi_cv = float(np.std(intervals_s) / np.mean(intervals_s)) if np.mean(intervals_s) > 0 else 1.0
        regularity = max(0.0, 1.0 - ibi_cv)
    else:
        bpm = 0.0
        regularity = 0.0

    # Per-beat kinetics (on work_trace so contraction = rise-to-peak)
    ct_list, rt_list, cv_list, rv_list, amp_list = [], [], [], [], []

    for i, pk in enumerate(peaks):
        left_bound = peaks[i - 1] if i > 0 else 0
        right_bound = peaks[i + 1] if i < len(peaks) - 1 else len(work_trace) - 1

        left_trough = left_bound + int(np.argmin(work_trace[left_bound:pk]))
        right_trough = pk + int(np.argmin(work_trace[pk:right_bound + 1]))

        amp = float(work_trace[pk] - work_trace[left_trough])
        if amp <= 0:
            continue

        ct = (pk - left_trough) / fps
        rt = (right_trough - pk) / fps

        amp_list.append(amp)
        ct_list.append(ct)
        rt_list.append(rt)
        if ct > 0:
            cv_list.append(amp / ct)
        if rt > 0:
            rv_list.append(amp / rt)

    def _safe_mean(lst):
        return round(float(np.mean(lst)), 4) if lst else 0

    rc_ratio = 0
    if ct_list and rt_list and np.mean(ct_list) > 0:
        rc_ratio = round(float(np.mean(rt_list)) / float(np.mean(ct_list)), 2)

    return {
        "num_beats": len(peaks),
        "bpm": round(bpm, 2),
        "regularity": round(regularity, 4),
        "ibi_cv": round(ibi_cv, 4),
        "mean_amplitude": _safe_mean(amp_list),
        "peak_indices": peaks,
        "inverted": inverted,
        "mean_contraction_time_s": _safe_mean(ct_list),
        "mean_relaxation_time_s": _safe_mean(rt_list),
        "mean_contraction_vel": _safe_mean(cv_list),
        "mean_relaxation_vel": _safe_mean(rv_list),
        "relax_contract_ratio": rc_ratio,
    }


# ═══════════════════════════════════════════════════════════════
# Motion classification
# ═══════════════════════════════════════════════════════════════

def classify_motion(beats_disp, beats_conv, conv_disp_ratio):
    """Classify motion source using convergence-displacement agreement.

    active_beating:      convergence + displacement agree, meaningful amplitude
    likely_beating:      some convergence signal, moderate amplitude
    individual_cells:    convergence hotspots visible but no synchronized trace
    noise:               many tiny peaks detected (high BPM, tiny amplitude)
    no_beating:          no displacement peaks detected
    """
    has_disp = beats_disp["num_beats"] >= 2
    has_conv = beats_conv["num_beats"] >= 2
    amp = beats_disp["mean_amplitude"]
    bpm = beats_disp["bpm"]

    # Sub-pixel noise floor: nothing real below 0.05 px
    if amp < 0.05:
        return "no_beating"

    # Tiny-amplitude high-frequency "beats" are noise, not real contraction
    if has_disp and amp < 0.1 and bpm > 40:
        return "noise"

    if has_disp and has_conv and conv_disp_ratio > 0.015 and amp >= 0.1:
        return "active_beating"
    if has_disp and (has_conv or conv_disp_ratio > 0.01) and amp >= 0.1:
        return "likely_beating"
    if has_disp and amp < 0.1:
        return "noise"
    if has_disp:
        return "likely_beating" if amp >= 0.1 else "no_beating"
    return "no_beating"


# ═══════════════════════════════════════════════════════════════
# Contractile foci detection
# ═══════════════════════════════════════════════════════════════

def detect_contractile_foci(convergence_map, disp_norm_array, fps,
                            settle_frames, min_area_px=80):
    """Detect individual contractile foci from the convergence map.

    Thresholds the convergence map (mean + 1 std of positive values),
    labels connected components, filters by minimum area, and runs
    beat detection on the local displacement trace of each foci.

    Based on Huebsch/Bhatt et al. Sci Rep 2017: convergence localises
    active contraction centers even when whole-field signal is weak.

    Returns (foci_mask, foci_results) where foci_mask is an integer
    array (0 = background, 1..N = beating foci) and foci_results is
    a dict with aggregate metrics.
    """
    empty = {
        "n_foci": 0, "foci_coverage_pct": 0.0,
        "foci_mean_bpm": 0.0, "foci_mean_amplitude": 0.0,
        "max_foci_amplitude": 0.0,
    }

    pos_vals = convergence_map[convergence_map > 0]
    if len(pos_vals) < 10:
        return np.zeros_like(convergence_map, dtype=np.int32), empty

    threshold = float(np.mean(pos_vals) + np.std(pos_vals))
    binary = convergence_map > threshold
    labeled, n_components = ndimage_label(binary)

    if n_components == 0:
        return np.zeros_like(convergence_map, dtype=np.int32), empty

    flow_h, flow_w = disp_norm_array.shape[:2]
    map_h, map_w = convergence_map.shape

    beating_foci = []
    foci_mask = np.zeros_like(convergence_map, dtype=np.int32)
    foci_id = 0

    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        area = int(comp_mask.sum())
        if area < min_area_px:
            continue

        if (map_h, map_w) != (flow_h, flow_w):
            comp_resized = cv2.resize(
                comp_mask.astype(np.float32), (flow_w, flow_h),
                interpolation=cv2.INTER_NEAREST
            ) > 0.5
        else:
            comp_resized = comp_mask

        if not comp_resized.any():
            continue

        local_trace = np.array([
            float(disp_norm_array[:, :, t][comp_resized].mean())
            for t in range(disp_norm_array.shape[-1])
        ])

        local_detrended = detrend_trace(local_trace, fps)
        beats = detect_beats(local_detrended[settle_frames:], fps)

        if beats["num_beats"] >= 2 and beats["mean_amplitude"] >= 0.05:
            foci_id += 1
            foci_mask[comp_mask] = foci_id
            beating_foci.append({
                "foci_id": foci_id,
                "area_px": area,
                "bpm": beats["bpm"],
                "amplitude": beats["mean_amplitude"],
                "num_beats": beats["num_beats"],
            })

    n_foci = len(beating_foci)
    total_coverage = float((foci_mask > 0).sum()) / (map_h * map_w) * 100

    foci_results = {
        "n_foci": n_foci,
        "foci_coverage_pct": round(total_coverage, 2),
        "foci_mean_bpm": round(float(np.mean([f["bpm"] for f in beating_foci])), 2) if beating_foci else 0.0,
        "foci_mean_amplitude": round(float(np.mean([f["amplitude"] for f in beating_foci])), 4) if beating_foci else 0.0,
        "max_foci_amplitude": round(float(max(f["amplitude"] for f in beating_foci)), 4) if beating_foci else 0.0,
    }

    return foci_mask, foci_results


# ═══════════════════════════════════════════════════════════════
# Main analysis per video
# ═══════════════════════════════════════════════════════════════

def analyze_single_video(filepath, flow_algorithm=FLOW_ALGORITHMS.farneback,
                         spatial_downsample=2, flow_scale=0.5):
    """Full analysis of one BF video: optical flow → convergence → beat kinetics.

    Produces:
      - Convergence-based ROI (active contraction regions)
      - Detrended displacement traces (whole-frame and conv-ROI)
      - Beat detection with per-beat kinetics
      - Motion classification (active vs passive vs noise)
      - 6-panel verification figure
    """
    name = filepath.stem

    print(f"\n{'='*60}")
    print(f"Analyzing: {filepath.name}")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────
    print("  Loading video...", end="", flush=True)
    mps_data, fps = load_avi_as_mpsdata(filepath, spatial_downsample=spatial_downsample)
    n_frames = mps_data.frames.shape[-1]
    h, w = mps_data.frames.shape[:2]
    duration = n_frames / fps
    first_frame = mps_data.frames[:, :, 0].copy()
    print(f" {w}x{h}, {n_frames} frames, {fps:.1f} fps, {duration:.1f}s")

    # ── Optical flow ──────────────────────────────────────────
    print(f"  Computing optical flow ({flow_algorithm.value}, scale={flow_scale})...", end="", flush=True)
    of = OpticalFlow(mps_data, flow_algorithm=flow_algorithm, data_scale=flow_scale)
    print(" done")

    print("  Extracting displacements...", end="", flush=True)
    displacements = of.get_displacements()
    print(" done")

    print("  Extracting velocities...", end="", flush=True)
    velocities = of.get_velocities()
    print(" done")

    # ── Whole-frame displacement + velocity traces ────────────
    print("  Computing whole-frame traces...", end="", flush=True)
    disp_norm = displacements.norm()
    mean_disp_trace = np.array(disp_norm.mean().compute())
    vel_norm = velocities.norm()
    mean_vel_trace = np.array(vel_norm.mean().compute())
    t_s = mps_data.time_stamps / 1000.0
    print(" done")

    # ── Convergence analysis ──────────────────────────────────
    print("  Computing convergence...", end="", flush=True)
    convergence_map, conv_trace, peak_conv_map = compute_convergence(displacements)
    print(" done")

    # ── Build convergence-based ROI ───────────────────────────
    print("  Building convergence ROI...", end="", flush=True)
    positive_vals = convergence_map[convergence_map > 0]
    if len(positive_vals) > 0:
        conv_roi_thresh = np.percentile(convergence_map, 70)
        conv_roi_mask = convergence_map > conv_roi_thresh
    else:
        conv_roi_mask = np.zeros_like(convergence_map, dtype=bool)

    conv_roi_coverage = conv_roi_mask.mean() * 100
    print(f" done ({conv_roi_coverage:.0f}% of FOV)")

    # ── Displacement within convergence ROI ───────────────────
    print("  Computing conv-ROI displacement traces...", end="", flush=True)
    disp_norm_array = np.array(disp_norm.compute())  # (H_flow, W_flow, T)

    if conv_roi_mask.any():
        flow_h, flow_w = disp_norm_array.shape[:2]
        mask_h, mask_w = conv_roi_mask.shape
        if (mask_h, mask_w) != (flow_h, flow_w):
            conv_roi_resized = cv2.resize(
                conv_roi_mask.astype(np.float32), (flow_w, flow_h),
                interpolation=cv2.INTER_NEAREST
            ) > 0.5
        else:
            conv_roi_resized = conv_roi_mask

        conv_roi_disp_trace = np.array([
            disp_norm_array[:, :, t][conv_roi_resized].mean()
            for t in range(disp_norm_array.shape[-1])
        ])
    else:
        conv_roi_disp_trace = mean_disp_trace.copy()
        conv_roi_resized = np.zeros(disp_norm_array.shape[:2], dtype=bool)

    motion_heatmap = np.mean(disp_norm_array, axis=-1)

    # ── Flow detection (spatial uniformity of displacement) ───
    disp_p50 = float(np.percentile(motion_heatmap, 50))
    disp_p90 = float(np.percentile(motion_heatmap, 90))
    flow_uniformity = round(disp_p50 / disp_p90, 3) if disp_p90 > 1e-9 else 0

    print(" done")

    # ── Settling exclusion: skip first 1.5s for beat detection ──
    settle_frames = min(int(fps * 1.5), n_frames // 4)

    # ── Contractile foci detection ────────────────────────────
    print("  Detecting contractile foci...", end="", flush=True)
    foci_mask, foci_results = detect_contractile_foci(
        convergence_map, disp_norm_array, fps, settle_frames
    )
    print(f" {foci_results['n_foci']} beating foci ({foci_results['foci_coverage_pct']:.1f}% FOV)")

    del disp_norm_array

    # ── Free heavy objects ────────────────────────────────────
    del mps_data, of, displacements, velocities, disp_norm, vel_norm
    gc.collect()

    # ── Detrend traces ────────────────────────────────────────
    print("  Detrending traces...", end="", flush=True)
    wf_detrended = detrend_trace(mean_disp_trace, fps)
    roi_detrended = detrend_trace(conv_roi_disp_trace, fps)
    conv_detrended = detrend_trace(conv_trace, fps)
        print(" done")

    # ── Beat detection ────────────────────────────────────────
    print("  Detecting beats (WF displacement)...", end="", flush=True)
    beats_wf = detect_beats(wf_detrended[settle_frames:], fps)
    print(f" {beats_wf['num_beats']} beats, {beats_wf['bpm']:.1f} BPM")

    print("  Detecting beats (conv-ROI displacement)...", end="", flush=True)
    beats_roi = detect_beats(roi_detrended[settle_frames:], fps)
    print(f" {beats_roi['num_beats']} beats, {beats_roi['bpm']:.1f} BPM")

    print("  Detecting beats (convergence trace)...", end="", flush=True)
    beats_conv = detect_beats(conv_detrended[settle_frames:], fps)
    print(f" {beats_conv['num_beats']} beats, {beats_conv['bpm']:.1f} BPM")

    # Offset peak indices back to full-trace coordinates for plotting
    for b in [beats_wf, beats_roi, beats_conv]:
        if b["peak_indices"] is not None:
            b["peak_indices"] = b["peak_indices"] + settle_frames

    # ── Motion classification ─────────────────────────────────
    mean_conv_signal = float(np.mean(conv_trace))
    mean_disp_signal = float(np.mean(mean_disp_trace))
    conv_disp_ratio = mean_conv_signal / max(mean_disp_signal, 1e-9)

    classification = classify_motion(beats_wf, beats_conv, conv_disp_ratio)

    # Upgrade classification if foci detected but whole-field missed them
    if classification in ("noise", "no_beating") and foci_results["n_foci"] > 0:
        classification = "individual_cells_beating"

    # Finalize flow flag: uniform displacement + weak beating = passive flow
    has_flow = flow_uniformity > 0.4 and beats_wf["mean_amplitude"] < 0.5
    flow_msg = ""
    if has_flow:
        flow_msg = f"  ** Flow/drift detected (uniformity={flow_uniformity:.2f})\n"

    foci_msg = ""
    if foci_results["n_foci"] > 0:
        foci_msg = f"  Contractile foci: {foci_results['n_foci']} (max amp={foci_results['max_foci_amplitude']:.3f} px)\n"

    print(f"{flow_msg}{foci_msg}  Classification: {classification} (conv/disp ratio={conv_disp_ratio:.3f})")

    # ── Metadata ──────────────────────────────────────────────
    meta = parse_video_metadata(filepath.name)

    # ── Results dict ──────────────────────────────────────────
    results = {
        "filename": filepath.name,
        **meta,
        "fps": round(fps, 2),
        "n_frames": n_frames,
        "duration_s": round(duration, 2),
        "original_resolution": f"{w * spatial_downsample}x{h * spatial_downsample}",
        "analyzed_resolution": f"{w}x{h}",
        "classification": classification,
        "conv_disp_ratio": round(conv_disp_ratio, 4),
        "flow_uniformity": flow_uniformity,
        "has_flow": has_flow,
        # Whole-frame (detrended)
        "wf_peak_disp": round(float(np.max(wf_detrended)), 4),
        "wf_mean_disp": round(float(np.mean(np.abs(wf_detrended))), 4),
        "wf_peak_velocity": round(float(np.max(mean_vel_trace)), 4),
        "wf_num_beats": beats_wf["num_beats"],
        "wf_bpm": beats_wf["bpm"],
        "wf_regularity": beats_wf["regularity"],
        "wf_ibi_cv": beats_wf["ibi_cv"],
        "wf_amplitude": beats_wf["mean_amplitude"],
        "wf_contraction_time_s": beats_wf["mean_contraction_time_s"],
        "wf_relaxation_time_s": beats_wf["mean_relaxation_time_s"],
        "wf_contraction_vel": beats_wf["mean_contraction_vel"],
        "wf_relaxation_vel": beats_wf["mean_relaxation_vel"],
        "wf_relax_contract_ratio": beats_wf["relax_contract_ratio"],
        # Convergence-ROI (detrended)
        "conv_roi_coverage_pct": round(conv_roi_coverage, 1),
        "roi_peak_disp": round(float(np.max(roi_detrended)), 4),
        "roi_mean_disp": round(float(np.mean(np.abs(roi_detrended))), 4),
        "roi_num_beats": beats_roi["num_beats"],
        "roi_bpm": beats_roi["bpm"],
        "roi_regularity": beats_roi["regularity"],
        "roi_ibi_cv": beats_roi["ibi_cv"],
        "roi_amplitude": beats_roi["mean_amplitude"],
        "roi_contraction_time_s": beats_roi["mean_contraction_time_s"],
        "roi_relaxation_time_s": beats_roi["mean_relaxation_time_s"],
        "roi_contraction_vel": beats_roi["mean_contraction_vel"],
        "roi_relaxation_vel": beats_roi["mean_relaxation_vel"],
        "roi_relax_contract_ratio": beats_roi["relax_contract_ratio"],
        # Convergence signal
        "conv_num_beats": beats_conv["num_beats"],
        "conv_bpm": beats_conv["bpm"],
        "conv_peak_signal": round(float(np.max(conv_trace)), 6),
        "conv_mean_signal": round(mean_conv_signal, 6),
        # Contractile foci
        "n_foci": foci_results["n_foci"],
        "foci_coverage_pct": foci_results["foci_coverage_pct"],
        "foci_mean_bpm": foci_results["foci_mean_bpm"],
        "foci_mean_amplitude": foci_results["foci_mean_amplitude"],
        "max_foci_amplitude": foci_results["max_foci_amplitude"],
    }

    # ── 6-panel verification figure ───────────────────────────
    _make_figure(name, filepath.name, first_frame, w, h,
                 convergence_map, conv_roi_mask, motion_heatmap,
                 wf_detrended, roi_detrended, conv_detrended,
                 t_s, beats_wf, beats_roi, beats_conv,
                 classification, results, foci_mask)

    gc.collect()
    return results


def _make_figure(name, filename, first_frame, w, h,
                 convergence_map, conv_roi_mask, motion_heatmap,
                 wf_detrended, roi_detrended, conv_detrended,
                 t_s, beats_wf, beats_roi, beats_conv,
                 classification, results, foci_mask=None):
    """Generate 6-panel verification figure."""

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    # ── Row 0: Images ─────────────────────────────────────────
    # (0,0) Raw first frame
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(first_frame, cmap="gray", aspect="equal")
    ax00.set_title("Raw First Frame", fontsize=11)
    ax00.axis("off")

    # (0,1) Convergence map overlaid on first frame
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(first_frame, cmap="gray", aspect="equal")
    conv_resized = cv2.resize(convergence_map, (w, h), interpolation=cv2.INTER_LINEAR)
    im = ax01.imshow(conv_resized, cmap="hot", alpha=0.55, aspect="equal")
    n_foci = results.get("n_foci", 0)
    foci_title = "Convergence Map (active contraction = bright)"
    if foci_mask is not None and n_foci > 0:
        foci_resized = cv2.resize(foci_mask.astype(np.float32), (w, h),
                                  interpolation=cv2.INTER_NEAREST)
        ax01.contour(foci_resized, levels=[0.5], colors=["cyan"], linewidths=1.5)
        foci_title = f"Convergence Map  |  {n_foci} beating foci (cyan)"
    ax01.set_title(foci_title, fontsize=11)
    ax01.axis("off")
    plt.colorbar(im, ax=ax01, fraction=0.046, pad=0.04, label="convergence")

    # ── Row 1: ROI masks ──────────────────────────────────────
    # (1,0) Convergence-based ROI
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(first_frame, cmap="gray", aspect="equal")
    mask_resized = cv2.resize(conv_roi_mask.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_NEAREST)
    ax10.imshow(mask_resized, cmap="Greens", alpha=0.4, aspect="equal")
    ax10.set_title(f"Convergence ROI ({results['conv_roi_coverage_pct']:.0f}% of FOV)", fontsize=11)
    ax10.axis("off")

    # (1,1) Displacement heatmap (for comparison)
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.imshow(first_frame, cmap="gray", aspect="equal")
    hm_resized = cv2.resize(motion_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    ax11.imshow(hm_resized, cmap="hot", alpha=0.5, aspect="equal")
    ax11.set_title("Displacement Heatmap (motion, incl. passive)", fontsize=11)
    ax11.axis("off")

    # ── Row 2: Traces ─────────────────────────────────────────
    # (2,0) WF displacement trace (detrended) with beat peaks
    ax20 = fig.add_subplot(gs[2, 0])
    t_wf = t_s[:len(wf_detrended)]
    ax20.plot(t_wf, wf_detrended, color="steelblue", linewidth=0.8)
    ax20.set_ylabel("Displacement (px, detrended)")
    title_wf = f"Whole-Frame: {beats_wf['num_beats']} beats, {beats_wf['bpm']:.1f} BPM"
    if beats_wf["mean_contraction_time_s"] > 0:
        title_wf += f"\nCT={beats_wf['mean_contraction_time_s']:.2f}s  RT={beats_wf['mean_relaxation_time_s']:.2f}s"
    ax20.set_title(title_wf, fontsize=10)
    if beats_wf["peak_indices"] is not None:
        pk = beats_wf["peak_indices"]
        marker = "rv" if beats_wf.get("inverted") else "r^"
        ax20.plot(t_wf[pk], wf_detrended[pk], marker, markersize=7, label="beat peaks")
    ax20.set_xlabel("Time (s)")
    ax20.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # (2,1) Conv-ROI displacement + convergence overlay
    ax21 = fig.add_subplot(gs[2, 1])
    t_roi = t_s[:len(roi_detrended)]
    ax21.plot(t_roi, roi_detrended, color="firebrick", linewidth=0.8, label="Conv-ROI disp")
    ax21.set_ylabel("Displacement (px, detrended)", color="firebrick")

    title_roi = f"Conv-ROI: {beats_roi['num_beats']} beats, {beats_roi['bpm']:.1f} BPM"
    if beats_roi["mean_contraction_time_s"] > 0:
        title_roi += f"\nCT={beats_roi['mean_contraction_time_s']:.2f}s  RT={beats_roi['mean_relaxation_time_s']:.2f}s"
    ax21.set_title(title_roi, fontsize=10)

    if beats_roi["peak_indices"] is not None:
        pk = beats_roi["peak_indices"]
        marker = "rv" if beats_roi.get("inverted") else "r^"
        ax21.plot(t_roi[pk], roi_detrended[pk], marker, markersize=7)
    ax21.set_xlabel("Time (s)")
    ax21.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    ax21b = ax21.twinx()
    t_conv = t_s[:len(conv_detrended)]
    ax21b.plot(t_conv, conv_detrended, color="goldenrod", linewidth=0.6, alpha=0.7)
    ax21b.set_ylabel("Convergence (detrended)", color="goldenrod", fontsize=9)
    ax21b.tick_params(axis="y", labelcolor="goldenrod", labelsize=8)

    # ── Title with classification ─────────────────────────────
    cls_colors = {
        "active_beating": "green", "likely_beating": "orange",
        "individual_cells_beating": "dodgerblue",
        "individual_cells": "dodgerblue", "noise": "red",
        "no_beating": "gray", "non_contractile": "darkorange",
    }
    flow_tag = "   |   FLOW/DRIFT" if results.get("has_flow") else ""
    foci_tag = f"   |   {n_foci} foci" if n_foci > 0 else ""
    fig.suptitle(
        f"{filename}   |   {classification.upper().replace('_', ' ')}   |   "
        f"conv/disp = {results['conv_disp_ratio']:.3f}{foci_tag}{flow_tag}",
        fontsize=14, fontweight="bold",
        color=cls_colors.get(classification, "black"),
    )

    fig.savefig(OUT / f"{name}_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Batch: all usable BF videos
# ═══════════════════════════════════════════════════════════════

USABLE_VIDEOS = [
    # Baseline day 1
    "A3-day1-fixed.avi",
    # A3-day1-phase3.avi excluded: duplicate recording of A3-day1-fixed
    "B3-day1-fixed.avi",
    "C3-day1-fixed.avi",
    # Baseline day 2
    "A3-day2-Tammy-NEW.avi",
    "B3-day2.avi",
    "C3-day2.avi",
    # Day 6 on-device + control
    "A3-day6-NON1-TAMMY.avi",
    "A3-day6-control-NON1-TAMMY.avi",
    "B3-day6-NON2-tammy.avi",
    "day6-B3-control-non.avi",
    "C3-day6-control-bf-new.avi",
    # Day 8 on-device + control (all 6 wells)
    "A2-day8-gold-tAMMY-new.avi",
    "A2-day8-gold-tAMMY-CONTROL.avi",
    "A3-day8-non-Tammy.avi",
    "A3-day8-Tammy-CONTROL.avi",
    "B2-day8-gold-tAMMY.avi",
    "B2-day8-gold-tAMMY-control.avi",
    "B3-day8-non-tammy.avi",
    "B3-day8-tammy-control.avi",
    "C2-day8-gold-Jaz.avi",
    "C2-day8-gold-Jaz-CONTROL-bf.avi",
    "C3-day8-non-Jaz.avi",
    "C3-day8-control-Jaz-bf.avi",
]

if __name__ == "__main__":
    import time

    all_results = []
    failed = []
    total = len(USABLE_VIDEOS)

    print(f"{'='*60}")
    print(f"BATCH ANALYSIS v3: {total} videos")
    print(f"v3: + contractile foci detection (per-center analysis)")
    print(f"{'='*60}")
    t_start = time.time()

    for i, vname in enumerate(USABLE_VIDEOS):
        vpath = DATA / vname
        if not vpath.exists():
            print(f"\n[{i+1}/{total}] MISSING: {vname}")
            failed.append(vname)
            continue

        print(f"\n[{i+1}/{total}] {vname}")
        try:
            results = analyze_single_video(vpath)
            all_results.append(results)
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed.append(vname)

    elapsed = time.time() - t_start

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUT / "batch_results_v3.csv", index=False)

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    n_done = len(all_results)
    print(f"  Processed: {n_done}/{total}")
    print(f"  Failed:    {len(failed)}")
    if n_done > 0:
        print(f"  Time:      {elapsed/60:.1f} min ({elapsed/n_done:.0f}s per video)")
    print(f"  Results:   {OUT / 'batch_results_v3.csv'}")
    if failed:
        print(f"  Failed:    {failed}")

    # Quick summary
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n--- Classification Summary ---")
        print(df["classification"].value_counts().to_string())
        print(f"\n--- Videos with active beating ---")
        active = df[df["classification"].isin(["active_beating", "likely_beating"])]
        if len(active) > 0:
            cols = ["filename", "classification", "roi_bpm", "roi_amplitude",
                    "roi_contraction_time_s", "roi_relaxation_time_s", "conv_disp_ratio"]
            print(active[cols].to_string(index=False))
    print(f"{'='*60}")
