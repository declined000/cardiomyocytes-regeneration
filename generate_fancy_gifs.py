"""
Generate 10 press-release quality animated GIFs for cardiomyocyte data.
Each GIF tells a specific scientific story using cross-referenced data.
Output: fancy_gifs/G01-G10 GIF files.

Usage: python generate_fancy_gifs.py
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import imageio
import gc
import tifffile
import traceback
import warnings
warnings.filterwarnings("ignore")

DATA = Path("Cardiomyocytes")
OUT = Path("fancy_gifs")
SETTLE = 18

DARK_BG = "#0e1117"
plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_BG,
    "savefig.facecolor": DARK_BG,
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ================================================================
#  UTILITIES
# ================================================================

def read_video(path, start=0, end=None, step=1, gray=True, half_res=False):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is None:
        end = total
    frames = []
    for i in range(start, min(end, total), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if not ret:
            break
        if gray:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if half_res:
            h, w = f.shape[:2]
            f = cv2.resize(f, (w // 2, h // 2))
        frames.append(f)
    cap.release()
    return frames, fps


def compute_flow(prev, curr):
    return cv2.calcOpticalFlowFarneback(
        prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )


def flow_mag(flow):
    return np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)


def flow_conv(flow):
    dudx = np.gradient(flow[..., 0], axis=1)
    dvdy = np.gradient(flow[..., 1], axis=0)
    return -(dudx + dvdy)


def disp_trace(frames):
    trace = [0.0]
    for i in range(1, len(frames)):
        fl = compute_flow(frames[i - 1], frames[i])
        trace.append(float(np.mean(flow_mag(fl))))
    return np.array(trace)


def disp_maps(frames):
    maps = [np.zeros_like(frames[0], dtype=np.float32)]
    for i in range(1, len(frames)):
        fl = compute_flow(frames[i - 1], frames[i])
        maps.append(flow_mag(fl).astype(np.float32))
    return maps


def conv_maps(frames):
    maps = [np.zeros_like(frames[0], dtype=np.float32)]
    for i in range(1, len(frames)):
        fl = compute_flow(frames[i - 1], frames[i])
        maps.append(flow_conv(fl).astype(np.float32))
    return maps


def flow_fields(frames):
    fields = [np.zeros((*frames[0].shape, 2), dtype=np.float32)]
    for i in range(1, len(frames)):
        fields.append(compute_flow(frames[i - 1], frames[i]))
    return fields


def blend_heatmap(gray, values, cmap_name="inferno", vmax=None,
                  alpha=0.5, thresh_frac=0.05):
    if vmax is None or vmax < 1e-6:
        vmax = max(float(np.percentile(values, 99.5)), 0.01)
    norm = np.clip(values / vmax, 0, 1)
    cmap = plt.colormaps[cmap_name]
    heat = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    base = np.stack([gray] * 3, axis=-1) if gray.ndim == 2 else gray.copy()
    mask = values > (vmax * thresh_frac)
    out = base.copy()
    out[mask] = (base[mask].astype(np.float32) * (1 - alpha)
                 + heat[mask].astype(np.float32) * alpha).astype(np.uint8)
    return out


def fig_to_rgb(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def save_gif(frames, path, fps=10):
    dur = int(1000 / fps)
    imageio.mimsave(str(path), frames, duration=dur, loop=0)
    mb = Path(path).stat().st_size / 1e6
    print(f"    -> {path.name}  ({mb:.1f} MB, {len(frames)} frames, {fps} FPS)")


def find_beat_window(trace, fps, n_cycles=1):
    sm = gaussian_filter(trace, sigma=2)
    thresh = np.mean(sm) + 0.3 * np.std(sm)
    min_d = max(int(fps * 1.5), 5)
    peaks, _ = find_peaks(sm, height=thresh, distance=min_d)
    if len(peaks) < 2:
        return 0, len(trace)
    best = int(np.argmax(sm[peaks]))
    start = peaks[max(0, best - 1)]
    end_idx = min(best + n_cycles, len(peaks) - 1)
    end = peaks[end_idx]
    pad = int(fps * 0.3)
    return max(0, start - pad), min(len(trace), end + pad)


def norm16(img):
    img = img.astype(np.float64)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
    if hi <= lo:
        hi = lo + 1
    return np.clip((img - lo) / (hi - lo), 0, 1)


def resize_match_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h))


def add_annotation_bar(frame, text, bar_height=26, font_scale=0.45):
    h, w = frame.shape[:2]
    bar = np.full((bar_height, w, 3), (14, 17, 23), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
    tx = (w - tw) // 2
    ty = (bar_height + th) // 2
    cv2.putText(bar, text, (tx, ty), font, font_scale,
                (180, 185, 195), 1, cv2.LINE_AA)
    return np.vstack([frame, bar])


# ================================================================
#  GIF 1: "The Heartbeat" — hero contraction heatmap loop
# ================================================================

def gif01_heartbeat():
    print("  Loading A3-day8-non-Tammy.avi ...")
    frames, fps = read_video(DATA / "A3-day8-non-Tammy.avi",
                             start=SETTLE, gray=True, half_res=True)
    print(f"    {len(frames)} frames @ {fps:.1f} FPS")

    print("  Computing displacement maps ...")
    dmaps = disp_maps(frames)
    trace = np.array([float(np.mean(d)) for d in dmaps])
    vmax = float(np.percentile(np.concatenate([d.ravel() for d in dmaps[1:]]), 99))

    s, e = find_beat_window(trace, fps, n_cycles=2)
    print(f"    Beat window: frames {s}-{e} ({(e-s)/fps:.1f}s)")
    cyc_frames = frames[s:e]
    cyc_dmaps = dmaps[s:e]
    cyc_trace = trace[s:e]

    fig, (ax_img, ax_tr) = plt.subplots(
        2, 1, figsize=(8, 7), dpi=100,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08})
    ax_img.set_xticks([]); ax_img.set_yticks([])

    composite = blend_heatmap(cyc_frames[0], cyc_dmaps[0],
                              vmax=vmax, alpha=0.55)
    im = ax_img.imshow(composite)
    title = ax_img.set_title("The Heartbeat — Contraction Displacement",
                             fontsize=14, fontweight="bold", pad=10)
    time_txt = ax_img.text(0.02, 0.95, "", transform=ax_img.transAxes,
                           fontsize=12, va="top",
                           bbox=dict(boxstyle="round,pad=0.3",
                                     fc="#00000088", ec="none"))

    sm = plt.cm.ScalarMappable(cmap="inferno",
                               norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=ax_img, fraction=0.03, pad=0.01,
                      label="Displacement (px)")

    t_arr = np.arange(len(cyc_trace)) / fps
    ax_tr.plot(t_arr, cyc_trace, color="#ff6b6b", lw=1.5, alpha=0.6)
    cursor = ax_tr.axvline(0, color="#ffd93d", lw=2)
    ax_tr.set_xlim(t_arr[0], t_arr[-1])
    ax_tr.set_ylim(0, max(cyc_trace) * 1.15)
    ax_tr.set_xlabel("Time (s)")
    ax_tr.set_ylabel("Mean disp.")
    fig.tight_layout()

    gif_frames = []
    for i in range(len(cyc_frames)):
        comp = blend_heatmap(cyc_frames[i], cyc_dmaps[i],
                             vmax=vmax, alpha=0.55)
        im.set_data(comp)
        time_txt.set_text(f"t = {i/fps:.2f} s")
        cursor.set_xdata([t_arr[i], t_arr[i]])
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f, "Well A3 | XR1 | Non-poled PVDF | Device (Pulsed) | Day 8 | Peak: 7.68 px") for f in gif_frames]
    save_gif(gif_frames, OUT / "G01_heartbeat.gif", fps=int(min(fps, 12)))
    del frames, dmaps, gif_frames, cyc_frames, cyc_dmaps
    gc.collect()


# ================================================================
#  GIF 2: "Hidden Calcium" — EMD split-screen BF vs FL
# ================================================================

def gif02_hidden_calcium():
    print("  Loading C3-day8 BF + FL ...")
    bf, bf_fps = read_video(DATA / "C3-day8-non-Jaz.avi",
                            start=SETTLE, gray=True, half_res=True)
    fl_raw, fl_fps = read_video(DATA / "C3-day8-non-Jaz-fl.avi",
                                start=0, gray=True, half_res=False)
    print(f"    BF: {len(bf)} @ {bf_fps:.1f}  FL: {len(fl_raw)} @ {fl_fps:.1f}")

    n_use = min(len(bf), int(len(fl_raw) * bf_fps / fl_fps), 120)
    bf = bf[:n_use]

    fl_arr = np.array(fl_raw, dtype=np.float64)
    f0 = np.maximum(np.mean(fl_arr[:10], axis=0), 1.0)
    dff0 = (fl_arr - f0) / f0
    dff0_smooth = gaussian_filter(dff0, sigma=(1, 1, 1))
    del fl_arr
    fl_vmax = max(float(np.percentile(dff0_smooth, 99.5)), 0.01)

    bf_h, bf_w = bf[0].shape
    fl_h, fl_w = dff0_smooth.shape[1], dff0_smooth.shape[2]

    fig = plt.figure(figsize=(12, 5.5), dpi=100)
    gs = GridSpec(2, 2, height_ratios=[5, 1], width_ratios=[1, 1],
                  hspace=0.12, wspace=0.08)
    ax_bf = fig.add_subplot(gs[0, 0])
    ax_fl = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1, :])

    ax_bf.set_xticks([]); ax_bf.set_yticks([])
    ax_fl.set_xticks([]); ax_fl.set_yticks([])
    ax_txt.set_xticks([]); ax_txt.set_yticks([])
    for sp in ax_txt.spines.values():
        sp.set_visible(False)

    bf_rgb = np.stack([bf[0]] * 3, axis=-1)
    im_bf = ax_bf.imshow(bf_rgb)
    ax_bf.set_title("Brightfield (BF)", fontsize=13, fontweight="bold")

    fl_frame = dff0_smooth[0]
    im_fl = ax_fl.imshow(fl_frame, cmap="hot", vmin=0, vmax=fl_vmax)
    ax_fl.set_title("Calcium Fluorescence (FL)", fontsize=13, fontweight="bold")
    fig.colorbar(im_fl, ax=ax_fl, fraction=0.04, pad=0.01,
                 label="ΔF/F₀")

    finding_txt = ax_txt.text(
        0.5, 0.55,
        "Calcium transients PRESENT  ·  Mechanical contraction ABSENT\n"
        "→ Electromechanical Dissociation (EMD)",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#ffd93d", transform=ax_txt.transAxes)
    time_txt = ax_txt.text(0.5, 0.05, "", ha="center", va="bottom",
                           fontsize=11, transform=ax_txt.transAxes,
                           color="#aaaaaa")
    fig.suptitle("Hidden Calcium — Same Well, Same Day (C3 day8 GCaMP6f)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    gif_frames = []
    out_fps = 10
    for i in range(n_use):
        bf_rgb = np.stack([bf[i]] * 3, axis=-1)
        im_bf.set_data(bf_rgb)

        fl_idx = min(int(i * fl_fps / bf_fps), len(dff0_smooth) - 1)
        im_fl.set_data(dff0_smooth[fl_idx])

        t = i / bf_fps
        time_txt.set_text(f"t = {t:.1f} s")
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f, "Well C3 | GCaMP6f | Non-poled PVDF | Device | Day 8 | EMD: Ca2+ without contraction") for f in gif_frames]
    save_gif(gif_frames, OUT / "G02_hidden_calcium.gif", fps=out_fps)
    del bf, fl_raw, dff0, dff0_smooth, gif_frames
    gc.collect()


# ================================================================
#  GIF 3: "The Device Effect" — paired device vs control
# ================================================================

def gif03_device_effect():
    print("  Loading A3-day8 device + control ...")
    dev, dev_fps = read_video(DATA / "A3-day8-non-Tammy.avi",
                              start=SETTLE, gray=True, half_res=True)
    ctrl, ctrl_fps = read_video(DATA / "A3-day8-Tammy-CONTROL.avi",
                                start=SETTLE, gray=True, half_res=True)
    n = min(len(dev), len(ctrl), 100)
    dev = dev[:n]; ctrl = ctrl[:n]
    fps = dev_fps
    print(f"    {n} frames each")

    print("  Computing displacement ...")
    dev_dm = disp_maps(dev)
    ctrl_dm = disp_maps(ctrl)
    dev_tr = np.array([float(np.mean(d)) for d in dev_dm])
    ctrl_tr = np.array([float(np.mean(d)) for d in ctrl_dm])
    vmax = max(float(np.percentile(np.concatenate(
        [d.ravel() for d in dev_dm[1:]]), 99)), 0.5)

    fig = plt.figure(figsize=(12, 7), dpi=100)
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.15, wspace=0.08)
    ax_dev = fig.add_subplot(gs[0, 0])
    ax_ctrl = fig.add_subplot(gs[0, 1])
    ax_tr = fig.add_subplot(gs[1, :])

    for ax in (ax_dev, ax_ctrl):
        ax.set_xticks([]); ax.set_yticks([])

    comp_dev = blend_heatmap(dev[0], dev_dm[0], vmax=vmax, alpha=0.55)
    comp_ctrl = blend_heatmap(ctrl[0], ctrl_dm[0], vmax=vmax, alpha=0.55)
    im_dev = ax_dev.imshow(comp_dev)
    im_ctrl = ax_ctrl.imshow(comp_ctrl)
    ax_dev.set_title("On Device (pulsed)", fontsize=13, fontweight="bold",
                     color="#ff6b6b")
    ax_ctrl.set_title("Control (same well, unpulsed)", fontsize=13,
                      fontweight="bold", color="#74b9ff")

    sm = plt.cm.ScalarMappable(cmap="inferno",
                               norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=[ax_dev, ax_ctrl], fraction=0.02, pad=0.01,
                 label="Displacement (px)")

    t_arr = np.arange(n) / fps
    ax_tr.plot(t_arr, dev_tr, color="#ff6b6b", lw=2, label="Device (7.3 px peak)")
    ax_tr.plot(t_arr, ctrl_tr, color="#74b9ff", lw=2, label="Control (1.9 px peak)")
    cursor_tr = ax_tr.axvline(0, color="#ffd93d", lw=2, alpha=0.7)
    ax_tr.legend(loc="upper right", fontsize=10)
    ax_tr.set_xlim(0, t_arr[-1])
    y_hi = max(max(dev_tr), max(ctrl_tr)) * 1.2
    ax_tr.set_ylim(0, y_hi)
    ax_tr.set_xlabel("Time (s)")
    ax_tr.set_ylabel("Mean disp. (px)")

    fig.suptitle("The Device Effect — 4× Contraction Amplification",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    gif_frames = []
    for i in range(n):
        im_dev.set_data(blend_heatmap(dev[i], dev_dm[i], vmax=vmax, alpha=0.55))
        im_ctrl.set_data(blend_heatmap(ctrl[i], ctrl_dm[i], vmax=vmax, alpha=0.55))
        cursor_tr.set_xdata([t_arr[i], t_arr[i]])
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f, "Well A3 | XR1 | Non-poled PVDF | Pulsation Effect (not piezoelectric) | Day 8 | Device 7.68 px vs Control 2.02 px") for f in gif_frames]
    save_gif(gif_frames, OUT / "G03_device_effect.gif", fps=int(min(fps, 12)))
    del dev, ctrl, dev_dm, ctrl_dm, gif_frames
    gc.collect()


# ================================================================
#  GIF 4: "8 Days of Beating" — maturation timeline
# ================================================================

def gif04_maturation():
    vids = [
        ("A3-day1-fixed.avi", "Day 1", "Individual cells", 0.02),
        ("A3-day2-Tammy-NEW.avi", "Day 2", "Active beating begins", 0.75),
        ("A3-day6-NON1-TAMMY.avi", "Day 6", "On device — growing", 3.16),
        ("A3-day8-non-Tammy.avi", "Day 8", "Peak contraction", 7.26),
    ]
    FRAMES_PER = 30
    FADE = 8

    all_segments = []
    for fname, day_label, desc, amp in vids:
        print(f"  Loading {fname} ...")
        fr, fps_v = read_video(DATA / fname, start=SETTLE,
                               end=SETTLE + FRAMES_PER + 10,
                               gray=True, half_res=True)
        fr = fr[:FRAMES_PER]
        fr_rgb = [np.stack([f] * 3, axis=-1) for f in fr]
        all_segments.append((fr_rgb, day_label, desc, amp, fps_v))

    target_h, target_w = all_segments[0][0][0].shape[:2]
    for seg in all_segments:
        for j in range(len(seg[0])):
            f = seg[0][j]
            if f.shape[:2] != (target_h, target_w):
                seg[0][j] = cv2.resize(f, (target_w, target_h))

    max_amp = max(s[3] for s in all_segments)
    fig = plt.figure(figsize=(11, 7), dpi=100)
    gs = GridSpec(1, 2, width_ratios=[3.5, 1.2], wspace=0.20)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_img.set_xticks([]); ax_img.set_yticks([])

    im = ax_img.imshow(all_segments[0][0][0])
    day_txt = ax_img.text(0.03, 0.97, "", transform=ax_img.transAxes,
                          fontsize=16, fontweight="bold", va="top",
                          color="#ffd93d",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    fc="#00000099", ec="none"))
    desc_txt = ax_img.text(0.03, 0.03, "", transform=ax_img.transAxes,
                           fontsize=10, va="bottom", color="white",
                           bbox=dict(boxstyle="round,pad=0.2",
                                     fc="#00000077", ec="none"))
    fig.suptitle("8 Days of Beating — Well A3 (XR1)",
                 fontsize=14, fontweight="bold", y=0.97)

    amp_labels = [s[1] for s in all_segments]
    amp_vals = [s[3] for s in all_segments]
    bar_colors = ["#636e72", "#74b9ff", "#fd79a8", "#ff6b6b"]

    gif_frames = []
    for seg_idx, (seg_frames, day_label, desc, amp, _) in enumerate(all_segments):
        ax_bar.clear()
        current_vals = [amp_vals[j] if j <= seg_idx else 0
                        for j in range(len(amp_vals))]
        colors = [bar_colors[j] if j <= seg_idx else "#333333"
                  for j in range(len(bar_colors))]
        ax_bar.barh(amp_labels, current_vals, color=colors, height=0.6)
        ax_bar.set_xlim(0, max_amp * 1.2)
        ax_bar.set_xlabel("Peak disp. (px)", fontsize=10)
        ax_bar.set_title("Amplitude", fontsize=12, fontweight="bold")
        for j, v in enumerate(current_vals):
            if v > 0:
                ax_bar.text(v + 0.1, j, f"{v:.1f}", va="center",
                            fontsize=10, color="white")
        ax_bar.invert_yaxis()

        day_txt.set_text(day_label)
        desc_txt.set_text(desc)

        for i in range(len(seg_frames)):
            im.set_data(seg_frames[i])
            gif_frames.append(fig_to_rgb(fig))

        if seg_idx < len(all_segments) - 1:
            next_frames = all_segments[seg_idx + 1][0]
            for f_i in range(FADE):
                alpha = (f_i + 1) / FADE
                blended = (seg_frames[-1].astype(np.float32) * (1 - alpha)
                           + next_frames[0].astype(np.float32) * alpha
                           ).astype(np.uint8)
                im.set_data(blended)
                gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f, "Well A3 | XR1 | Non-poled PVDF | Device | Days 1-8 Maturation Timeline") for f in gif_frames]
    save_gif(gif_frames, OUT / "G04_maturation.gif", fps=10)
    del all_segments, gif_frames
    gc.collect()


# ================================================================
#  GIF 5: "Calcium Waves" — FL dF/F0 heatmap animation
# ================================================================

def gif05_calcium_waves():
    fl_vids = [
        ("Jaz-day2-fluorescence-gold.avi", "Day 2 (Baseline)", 0.086),
        ("C2-day6-gold-FL-control.avi",    "Day 6 (Control plate)", 0.216),
        ("C2-day8-gold-GCaMP6f-FL.avi",    "Day 8 (Device plate)", 0.050),
    ]

    processed = []
    for fname, label, amp in fl_vids:
        print(f"  Loading {fname} ...")
        raw, fps_v = read_video(DATA / fname, start=0, gray=True,
                                half_res=False)
        arr = np.array(raw, dtype=np.float64)
        f0 = np.maximum(np.percentile(arr[:15], 5, axis=0), 1.0)
        dff0 = (arr - f0) / f0
        sm = gaussian_filter(dff0, sigma=(0.5, 1.5, 1.5))
        del arr, dff0, raw
        processed.append((sm, fps_v, label, amp))

    shared_vmax = max(
        float(np.percentile(p[0], 99.8)) for p in processed)
    shared_vmax = max(shared_vmax, 0.02)

    target_h, target_w = processed[1][0][0].shape
    for idx, (sm, fp, lb, am) in enumerate(processed):
        if sm[0].shape != (target_h, target_w):
            resized = np.array([cv2.resize(f, (target_w, target_h))
                                for f in sm])
            processed[idx] = (resized, fp, lb, am)

    n_use = min(min(len(p[0]) for p in processed), 100)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=100)
    ims = []
    time_txts = []
    for col, (sm, fps_v, label, amp) in enumerate(processed):
        ax = axes[col]
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(sm[0], cmap="magma", vmin=-0.02, vmax=shared_vmax,
                       interpolation="bilinear")
        ims.append(im)
        ax.set_title(f"{label}\ndF/F0 = {amp:.3f}", fontsize=11,
                     fontweight="bold", pad=6)
        tt = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                     fontsize=10, va="top", color="white",
                     bbox=dict(boxstyle="round,pad=0.2",
                               fc="#00000088", ec="none"))
        time_txts.append(tt)

    fig.colorbar(ims[1], ax=axes.tolist(), fraction=0.02, pad=0.05,
                 label="dF/F0")
    fig.suptitle("Calcium Activity on Gold PVDF -- Three Timepoints (Well C2, GCaMP6f)",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 0.87, 0.93])

    gif_frames = []
    for i in range(n_use):
        for col, (sm, fps_v, _, _) in enumerate(processed):
            fi = min(i, len(sm) - 1)
            ims[col].set_data(sm[fi])
            time_txts[col].set_text(f"t = {fi/fps_v:.1f} s")
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f,
        "Well C2 | GCaMP6f | Gold PVDF | DIFFERENT PLATES -- not a temporal progression | "
        "Day 2: baseline | Day 6: control (un-pulsed) | Day 8: device (pulsed)"
    ) for f in gif_frames]
    save_gif(gif_frames, OUT / "G05_calcium_waves.gif", fps=8)
    del processed, gif_frames
    gc.collect()


# ================================================================
#  GIF 6: "The Adhesion Problem" — gold vs non-poled side-by-side
# ================================================================

def gif06_adhesion():
    print("  Loading gold (C2-day8) + non-poled (A3-day8) ...")
    gold, g_fps = read_video(DATA / "C2-day8-gold-Jaz.avi",
                             start=SETTLE, gray=True, half_res=True)
    nonp, n_fps = read_video(DATA / "A3-day8-non-Tammy.avi",
                             start=SETTLE, gray=True, half_res=True)
    n = min(len(gold), len(nonp), 80)
    gold = gold[:n]; nonp = nonp[:n]
    fps = g_fps

    print("  Computing displacement ...")
    nonp_dm = disp_maps(nonp)
    gold_dm = disp_maps(gold)
    vmax = max(float(np.percentile(np.concatenate(
        [d.ravel() for d in nonp_dm[1:]]), 99)), 0.5)

    fig, (ax_g, ax_n) = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
    for ax in (ax_g, ax_n):
        ax.set_xticks([]); ax.set_yticks([])

    g_rgb = np.stack([gold[0]] * 3, axis=-1)
    n_comp = blend_heatmap(nonp[0], nonp_dm[0], vmax=vmax, alpha=0.55)
    im_g = ax_g.imshow(g_rgb)
    im_n = ax_n.imshow(n_comp)

    ax_g.set_title("Gold-PVDF — Poor Adhesion", fontsize=13,
                   fontweight="bold", color="#ff7675")
    ax_n.set_title("Non-poled PVDF — Strong Contraction", fontsize=13,
                   fontweight="bold", color="#55efc4")

    ax_g.text(0.5, 0.05, "Floating cells / debris / no beating",
              transform=ax_g.transAxes, ha="center", fontsize=10,
              color="#ff7675",
              bbox=dict(boxstyle="round,pad=0.3", fc="#00000088", ec="none"))
    ax_n.text(0.5, 0.05, "Attached cells / active beating",
              transform=ax_n.transAxes, ha="center", fontsize=10,
              color="#55efc4",
              bbox=dict(boxstyle="round,pad=0.3", fc="#00000088", ec="none"))

    sm = plt.cm.ScalarMappable(cmap="inferno",
                               norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=ax_n, fraction=0.03, pad=0.01,
                 label="Displacement (px)")

    fig.suptitle("The Adhesion Problem — Substrate Surface Matters",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    gif_frames = []
    for i in range(n):
        im_g.set_data(np.stack([gold[i]] * 3, axis=-1))
        im_n.set_data(blend_heatmap(nonp[i], nonp_dm[i], vmax=vmax, alpha=0.55))
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f, "Left: C2 GCaMP6f Gold PVDF | Right: A3 XR1 Non-poled PVDF | Day 8 Device") for f in gif_frames]
    save_gif(gif_frames, OUT / "G06_adhesion.gif", fps=int(min(fps, 12)))
    del gold, nonp, nonp_dm, gold_dm, gif_frames
    gc.collect()


# ================================================================
#  GIF 7: "Structure Meets Function" — staining cross-dissolve
# ================================================================

def gif07_structure_function():
    print("  Loading A3 staining TIFFs ...")
    phall = tifffile.imread(str(DATA / "device-non-phalloidin-a3-image.tif"))
    dapi = tifffile.imread(str(DATA / "device-non-dapi-a3-image.tif"))

    phall_n = norm16(phall)
    dapi_n = norm16(dapi)
    stain_rgb = np.zeros((*phall_n.shape, 3))
    stain_rgb[..., 1] = phall_n
    stain_rgb[..., 2] = dapi_n
    stain_rgb = (np.clip(stain_rgb, 0, 1) * 255).astype(np.uint8)

    print("  Loading A3-day8-non-Tammy.avi ...")
    bf, fps = read_video(DATA / "A3-day8-non-Tammy.avi",
                         start=SETTLE, gray=True, half_res=True)
    bf = bf[:80]
    bf_dm = disp_maps(bf)
    vmax = max(float(np.percentile(np.concatenate(
        [d.ravel() for d in bf_dm[1:]]), 99)), 0.5)

    bf_h, bf_w = bf[0].shape
    stain_resized = cv2.resize(stain_rgb, (bf_w, bf_h))

    HOLD_STAIN = 20
    FADE = 12
    VIDEO_FRAMES = len(bf)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(stain_resized)
    phase_txt = ax.text(0.5, 0.05, "Cytoskeletal Structure", ha="center",
                        transform=ax.transAxes, fontsize=14,
                        fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.4",
                                  fc="#00000099", ec="none"))
    ax.set_title("Structure Meets Function — Well A3 (XR1, Non-poled)",
                 fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()

    gif_frames = []

    for _ in range(HOLD_STAIN):
        gif_frames.append(fig_to_rgb(fig))

    for f_i in range(FADE):
        alpha = (f_i + 1) / FADE
        bf_comp = blend_heatmap(bf[0], bf_dm[0], vmax=vmax, alpha=0.55)
        blended = (stain_resized.astype(np.float32) * (1 - alpha)
                   + bf_comp.astype(np.float32) * alpha).astype(np.uint8)
        im.set_data(blended)
        phase_txt.set_text("Structure → Function" if alpha < 0.5
                           else "Contractile Function")
        gif_frames.append(fig_to_rgb(fig))

    phase_txt.set_text("Contractile Function")
    for i in range(VIDEO_FRAMES):
        comp = blend_heatmap(bf[i], bf_dm[i], vmax=vmax, alpha=0.55)
        im.set_data(comp)
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f, "Well A3 | XR1 | Non-poled PVDF | Device | Phalloidin/DAPI + BF Day 8") for f in gif_frames]
    save_gif(gif_frames, OUT / "G07_structure_function.gif",
             fps=int(min(fps, 12)))
    del bf, bf_dm, gif_frames, stain_rgb, stain_resized
    gc.collect()


# ================================================================
#  GIF 8: "Contraction Vectors" — animated optical flow arrows
# ================================================================

def gif08_contraction_vectors():
    print("  Loading A3-day6-NON1-TAMMY.avi ...")
    frames, fps = read_video(DATA / "A3-day6-NON1-TAMMY.avi",
                             start=SETTLE, gray=True, half_res=True)
    n = min(len(frames), 90)
    frames = frames[:n]
    print(f"    {n} frames @ {fps:.1f} FPS")

    print("  Computing flow fields ...")
    fields = flow_fields(frames)

    h, w = frames[0].shape
    step = 20
    gy, gx = np.mgrid[step//2:h:step, step//2:w:step]

    max_mag = 0
    for fl in fields[1:]:
        m = flow_mag(fl)
        p99 = np.percentile(m, 99)
        if p99 > max_mag:
            max_mag = p99
    if max_mag < 0.1:
        max_mag = 1.0

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, w); ax.set_ylim(h, 0)

    bg = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=255)

    u0 = fields[0][step//2::step, step//2::step, 0]
    v0 = fields[0][step//2::step, step//2::step, 1]
    m0 = np.sqrt(u0**2 + v0**2)
    Q = ax.quiver(gx, gy, u0, v0, m0, cmap="inferno",
                  clim=(0, max_mag), scale=max_mag * 15,
                  width=0.003, headwidth=4, headlength=5, alpha=0.85)

    sm = plt.cm.ScalarMappable(cmap="inferno",
                               norm=plt.Normalize(0, max_mag))
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01,
                 label="Flow magnitude (px)")

    time_txt = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                       fontsize=12, va="top", color="white",
                       bbox=dict(boxstyle="round,pad=0.3",
                                 fc="#00000088", ec="none"))
    ax.set_title("Contraction Vectors — Optical Flow Field",
                 fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()

    gif_frames = []
    for i in range(n):
        bg.set_data(frames[i])
        u = fields[i][step//2::step, step//2::step, 0]
        v = fields[i][step//2::step, step//2::step, 1]
        m = np.sqrt(u**2 + v**2)
        Q.set_UVC(u, v, m)
        time_txt.set_text(f"t = {i/fps:.2f} s")
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f, "Well A3 | XR1 | Non-poled PVDF | Device | Day 6 | Peak: 3.16 px") for f in gif_frames]
    save_gif(gif_frames, OUT / "G08_contraction_vectors.gif",
             fps=int(min(fps, 10)))
    del frames, fields, gif_frames
    gc.collect()


# ================================================================
#  GIF 9: "Two Cell Lines" — XR1 vs GCaMP6f with FL inset
# ================================================================

def gif09_cell_lines():
    print("  Loading XR1 (B3-day8) + GCaMP6f (C3-day8) BF + FL ...")
    xr1, xr1_fps = read_video(DATA / "B3-day8-tammy-control.avi",
                               start=SETTLE, gray=True, half_res=True)
    gcf, gcf_fps = read_video(DATA / "C3-day8-non-Jaz.avi",
                               start=SETTLE, gray=True, half_res=True)
    fl_raw, fl_fps = read_video(DATA / "C3-day8-non-Jaz-fl.avi",
                                 start=0, gray=True, half_res=False)

    n = min(len(xr1), len(gcf), 80)
    xr1 = xr1[:n]; gcf = gcf[:n]

    fl_arr = np.array(fl_raw, dtype=np.float64)
    fl_f0 = np.maximum(np.mean(fl_arr[:10], axis=0), 1.0)
    fl_dff0 = (fl_arr - fl_f0) / fl_f0
    fl_dff0_sm = gaussian_filter(fl_dff0, sigma=(0.5, 1, 1))
    fl_vmax = max(float(np.percentile(fl_dff0_sm, 99.5)), 0.01)
    del fl_arr, fl_dff0

    print("  Computing XR1 displacement ...")
    xr1_dm = disp_maps(xr1)
    vmax = max(float(np.percentile(np.concatenate(
        [d.ravel() for d in xr1_dm[1:]]), 99)), 0.5)

    fig, (ax_x, ax_g) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=100)
    for ax in (ax_x, ax_g):
        ax.set_xticks([]); ax.set_yticks([])

    comp_x = blend_heatmap(xr1[0], xr1_dm[0], vmax=vmax, alpha=0.55)
    im_x = ax_x.imshow(comp_x)
    ax_x.set_title("XR1 — Active Beating", fontsize=13,
                   fontweight="bold", color="#55efc4")
    ax_x.text(0.03, 0.06, "3.9 px peak · 18.5 BPM",
              transform=ax_x.transAxes, fontsize=10, color="#55efc4",
              bbox=dict(boxstyle="round,pad=0.3", fc="#00000088", ec="none"))

    g_rgb = np.stack([gcf[0]] * 3, axis=-1)
    im_g = ax_g.imshow(g_rgb)
    ax_g.set_title("GCaMP6f — No Contraction", fontsize=13,
                   fontweight="bold", color="#ff7675")

    inset_h = int(gcf[0].shape[0] * 0.35)
    inset_w = int(gcf[0].shape[1] * 0.35)
    fl_frame_resized = cv2.resize(fl_dff0_sm[0], (inset_w, inset_h))
    ax_inset = ax_g.inset_axes([0.6, 0.02, 0.38, 0.38])
    ax_inset.set_xticks([]); ax_inset.set_yticks([])
    im_inset = ax_inset.imshow(fl_frame_resized, cmap="hot",
                                vmin=0, vmax=fl_vmax)
    ax_inset.set_title("Ca²⁺ (FL)", fontsize=9, color="#ffd93d",
                       fontweight="bold")
    for sp in ax_inset.spines.values():
        sp.set_edgecolor("#ffd93d")
        sp.set_linewidth(2)

    ax_g.text(0.03, 0.06,
              "But calcium transients ARE present →",
              transform=ax_g.transAxes, fontsize=10, color="#ffd93d",
              bbox=dict(boxstyle="round,pad=0.3", fc="#00000088", ec="none"))

    sm = plt.cm.ScalarMappable(cmap="inferno",
                               norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=ax_x, fraction=0.03, pad=0.01,
                 label="Displacement (px)")

    fig.suptitle("Two Cell Lines, Same Experiment — Genetics Determines Function",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    gif_frames = []
    for i in range(n):
        im_x.set_data(blend_heatmap(xr1[i], xr1_dm[i], vmax=vmax, alpha=0.55))
        im_g.set_data(np.stack([gcf[i]] * 3, axis=-1))

        fl_idx = min(int(i * fl_fps / xr1_fps), len(fl_dff0_sm) - 1)
        fl_fr = cv2.resize(fl_dff0_sm[fl_idx], (inset_w, inset_h))
        im_inset.set_data(fl_fr)

        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f, "Left: B3 XR1 Plastic Control | Right: C3 GCaMP6f Non-poled Device | Day 8") for f in gif_frames]
    save_gif(gif_frames, OUT / "G09_cell_lines.gif", fps=int(min(xr1_fps, 12)))
    del xr1, gcf, fl_raw, fl_dff0_sm, xr1_dm, gif_frames
    gc.collect()


# ================================================================
#  GIF 10: "The Full Dashboard" — multi-panel animated figure
# ================================================================

def gif10_dashboard():
    print("  Loading A3-day8-non-Tammy.avi ...")
    frames, fps = read_video(DATA / "A3-day8-non-Tammy.avi",
                             start=SETTLE, gray=True, half_res=True)
    n = min(len(frames), 90)
    frames = frames[:n]

    print("  Computing displacement + convergence ...")
    dmaps_list = disp_maps(frames)
    cmaps_list = conv_maps(frames)
    trace = np.array([float(np.mean(d)) for d in dmaps_list])
    d_vmax = max(float(np.percentile(np.concatenate(
        [d.ravel() for d in dmaps_list[1:]]), 99)), 0.5)
    c_abs = [np.abs(c) for c in cmaps_list[1:]]
    c_vmax = max(float(np.percentile(np.concatenate(
        [c.ravel() for c in c_abs]), 99)), 0.001)

    print("  Loading staining ...")
    phall = tifffile.imread(str(DATA / "device-non-phalloidin-a3-image.tif"))
    dapi = tifffile.imread(str(DATA / "device-non-dapi-a3-image.tif"))
    phall_n = norm16(phall); dapi_n = norm16(dapi)
    stain = np.zeros((*phall_n.shape, 3))
    stain[..., 1] = phall_n; stain[..., 2] = dapi_n
    stain = (np.clip(stain, 0, 1) * 255).astype(np.uint8)
    bf_h, bf_w = frames[0].shape
    stain_sm = cv2.resize(stain, (bf_w // 2, bf_h // 2))

    fig = plt.figure(figsize=(12, 8), dpi=100)
    gs = GridSpec(2, 3, height_ratios=[3, 1.5],
                  width_ratios=[2, 2, 1],
                  hspace=0.2, wspace=0.15)

    ax_bf = fig.add_subplot(gs[0, 0])
    ax_conv = fig.add_subplot(gs[0, 1])
    ax_stain = fig.add_subplot(gs[0, 2])
    ax_trace = fig.add_subplot(gs[1, :])

    for ax in (ax_bf, ax_conv, ax_stain):
        ax.set_xticks([]); ax.set_yticks([])

    comp0 = blend_heatmap(frames[0], dmaps_list[0], vmax=d_vmax, alpha=0.55)
    im_bf = ax_bf.imshow(comp0)
    ax_bf.set_title("BF + Displacement", fontsize=12, fontweight="bold")
    sm_d = plt.cm.ScalarMappable(cmap="inferno",
                                  norm=plt.Normalize(0, d_vmax))
    fig.colorbar(sm_d, ax=ax_bf, fraction=0.04, pad=0.02, label="px")

    im_conv = ax_conv.imshow(cmaps_list[0], cmap="RdBu_r",
                              vmin=-c_vmax, vmax=c_vmax)
    ax_conv.set_title("Convergence Map", fontsize=12, fontweight="bold")
    sm_c = plt.cm.ScalarMappable(cmap="RdBu_r",
                                  norm=plt.Normalize(-c_vmax, c_vmax))
    fig.colorbar(sm_c, ax=ax_conv, fraction=0.04, pad=0.02,
                 label="conv")

    ax_stain.imshow(stain_sm)
    ax_stain.set_title("Phalloidin / DAPI", fontsize=11, fontweight="bold")

    t_arr = np.arange(n) / fps
    ax_trace.plot(t_arr, trace, color="#ff6b6b", lw=1.5, alpha=0.6)
    cursor = ax_trace.axvline(0, color="#ffd93d", lw=2)
    ax_trace.set_xlim(0, t_arr[-1])
    ax_trace.set_ylim(0, max(trace) * 1.2)
    ax_trace.set_xlabel("Time (s)")
    ax_trace.set_ylabel("Mean displacement (px)")
    ax_trace.set_title("Displacement Trace", fontsize=12, fontweight="bold")

    metrics_txt = ax_stain.text(
        0.5, -0.15, "8.8 BPM\n103 foci\n5.8% coverage",
        transform=ax_stain.transAxes, ha="center", va="top",
        fontsize=9, color="#aaaaaa",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", ec="#333333"))

    fig.suptitle("Well A3 · Day 8 · XR1 on Non-poled PVDF · Pulsed Device",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    gif_frames = []
    for i in range(n):
        im_bf.set_data(blend_heatmap(frames[i], dmaps_list[i],
                                     vmax=d_vmax, alpha=0.55))
        im_conv.set_data(cmaps_list[i])
        cursor.set_xdata([t_arr[i], t_arr[i]])
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f, "Well A3 | XR1 | Non-poled PVDF | Device (Pulsed) | Day 8 | 8.8 BPM") for f in gif_frames]
    save_gif(gif_frames, OUT / "G10_dashboard.gif", fps=int(min(fps, 12)))
    del frames, dmaps_list, cmaps_list, gif_frames
    gc.collect()


# ================================================================
#  GIF 11: "All Wells Day 8" -- 2x3 grid of all device wells
# ================================================================

def gif11_all_wells():
    wells = [
        ("A2-day8-gold-tAMMY-new.avi",  "A2 Gold", "Non-contractile", 0.01),
        ("B2-day8-gold-tAMMY.avi",       "B2 Gold", "Active beating",  3.05),
        ("C2-day8-gold-Jaz.avi",         "C2 Gold", "Floaters only",   0.22),
        ("A3-day8-non-Tammy.avi",        "A3 Non-poled", "Strong beating", 7.68),
        ("B3-day8-non-tammy.avi",        "B3 Non-poled", "Film detached",  0.03),
        ("C3-day8-non-Jaz.avi",          "C3 Non-poled", "Individual cells", 0.08),
    ]

    n_frames = 70
    composites = []
    for fname, label, status, amp in wells:
        print(f"  Loading {fname} ...")
        fr, fps = read_video(DATA / fname, start=SETTLE,
                             end=SETTLE + n_frames + 5,
                             gray=True, half_res=True)
        fr = fr[:n_frames]
        if amp > 0.5:
            dm = disp_maps(fr)
            vmax = max(float(np.percentile(
                np.concatenate([d.ravel() for d in dm[1:]]), 99)), 0.5)
            comp = [blend_heatmap(fr[i], dm[i], vmax=vmax, alpha=0.55)
                    for i in range(len(fr))]
            del dm
        else:
            comp = [np.stack([f] * 3, axis=-1) for f in fr]
        while len(comp) < n_frames:
            comp.append(comp[-1])
        composites.append((comp, label, status, amp))
        del fr
        gc.collect()

    target_h, target_w = composites[0][0][0].shape[:2]
    for comp_list, _, _, _ in composites:
        for j in range(len(comp_list)):
            if comp_list[j].shape[:2] != (target_h, target_w):
                comp_list[j] = cv2.resize(comp_list[j], (target_w, target_h))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=100)
    fig.subplots_adjust(hspace=0.45, wspace=0.10)
    ims = []
    for idx, (comp, label, status, amp) in enumerate(composites):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(comp[0])
        ims.append(im)
        color = "#55efc4" if amp > 1.0 else "#ff7675"
        ax.set_title(f"{label}\n{status} ({amp:.2f} px)", fontsize=9,
                     fontweight="bold", color=color, pad=4)

    axes[0, 0].set_ylabel("Gold-PVDF\n(Poled)", fontsize=9, fontweight="bold")
    axes[1, 0].set_ylabel("Non-poled\nPVDF", fontsize=9, fontweight="bold")
    for c_i, lbl in enumerate(["Row A (XR1)", "Row B (XR1)", "Row C (GCaMP6f)"]):
        axes[0, c_i].text(0.5, 1.22, lbl, transform=axes[0, c_i].transAxes,
                          ha="center", fontsize=8, color="#aaaaaa")

    fig.suptitle("All Device Wells -- Day 8 Overview",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    gif_frames = []
    for i in range(n_frames):
        for idx, (comp, _, _, _) in enumerate(composites):
            ims[idx].set_data(comp[i])
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f,
        "Device Plate Day 8 | XR1 (A,B rows) + GCaMP6f (C row) | Gold vs Non-poled PVDF"
    ) for f in gif_frames]
    save_gif(gif_frames, OUT / "G11_all_wells.gif", fps=10)
    del composites, gif_frames
    gc.collect()


# ================================================================
#  GIF 12: "Piezoelectric Isolation" -- B2 gold device vs control
# ================================================================

def gif12_piezo_isolation():
    print("  Loading B2-day8 device + control ...")
    dev, dev_fps = read_video(DATA / "B2-day8-gold-tAMMY.avi",
                              start=SETTLE, gray=True, half_res=True)
    ctrl, ctrl_fps = read_video(DATA / "B2-day8-gold-tAMMY-control.avi",
                                start=SETTLE, gray=True, half_res=True)
    n = min(len(dev), len(ctrl), 90)
    dev = dev[:n]; ctrl = ctrl[:n]
    fps = dev_fps

    print("  Computing displacement ...")
    dev_dm = disp_maps(dev)
    ctrl_dm = disp_maps(ctrl)
    dev_tr = np.array([float(np.mean(d)) for d in dev_dm])
    ctrl_tr = np.array([float(np.mean(d)) for d in ctrl_dm])
    vmax = max(
        float(np.percentile(np.concatenate([d.ravel() for d in dev_dm[1:]]), 99)),
        float(np.percentile(np.concatenate([d.ravel() for d in ctrl_dm[1:]]), 99)),
        0.5)

    fig = plt.figure(figsize=(12, 7), dpi=100)
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.15, wspace=0.08)
    ax_dev = fig.add_subplot(gs[0, 0])
    ax_ctrl = fig.add_subplot(gs[0, 1])
    ax_tr = fig.add_subplot(gs[1, :])
    for ax in (ax_dev, ax_ctrl):
        ax.set_xticks([]); ax.set_yticks([])

    im_dev = ax_dev.imshow(blend_heatmap(dev[0], dev_dm[0], vmax=vmax, alpha=0.55))
    im_ctrl = ax_ctrl.imshow(blend_heatmap(ctrl[0], ctrl_dm[0], vmax=vmax, alpha=0.55))
    ax_dev.set_title("B2 Device (Pulsed)\n3.05 px, 3.9 BPM", fontsize=12,
                     fontweight="bold", color="#ff6b6b")
    ax_ctrl.set_title("B2 Control (Un-pulsed)\n1.15 px, 10.2 BPM", fontsize=12,
                      fontweight="bold", color="#74b9ff")

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=[ax_dev, ax_ctrl], fraction=0.02, pad=0.01,
                 label="Displacement (px)")

    t_arr = np.arange(n) / fps
    ax_tr.plot(t_arr, dev_tr, color="#ff6b6b", lw=2, label="Pulsed (3.05 px)")
    ax_tr.plot(t_arr, ctrl_tr, color="#74b9ff", lw=2, label="Un-pulsed (1.15 px)")
    cursor = ax_tr.axvline(0, color="#ffd93d", lw=2, alpha=0.7)
    ax_tr.legend(loc="upper right", fontsize=10)
    ax_tr.set_xlim(0, t_arr[-1])
    ax_tr.set_ylim(0, max(max(dev_tr), max(ctrl_tr)) * 1.2)
    ax_tr.set_xlabel("Time (s)")
    ax_tr.set_ylabel("Mean disp. (px)")

    fig.suptitle("Piezoelectric Isolation -- Gold PVDF Pulsed vs Un-pulsed (B2)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    gif_frames = []
    for i in range(n):
        im_dev.set_data(blend_heatmap(dev[i], dev_dm[i], vmax=vmax, alpha=0.55))
        im_ctrl.set_data(blend_heatmap(ctrl[i], ctrl_dm[i], vmax=vmax, alpha=0.55))
        cursor.set_xdata([t_arr[i], t_arr[i]])
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f,
        "Well B2 | XR1 | Gold PVDF (Poled) | PIEZOELECTRIC Comparison | Day 8 | Device 3.05 px vs Control 1.15 px | 2.7x amplification"
    ) for f in gif_frames]
    save_gif(gif_frames, OUT / "G12_piezo_isolation.gif", fps=int(min(fps, 12)))
    del dev, ctrl, dev_dm, ctrl_dm, gif_frames
    gc.collect()


# ================================================================
#  GIF 13: "Substrate Comparison" -- B2 gold vs A3 non-poled
# ================================================================

def gif13_substrate_compare():
    print("  Loading B2-day8 gold + A3-day8 non-poled ...")
    gold, g_fps = read_video(DATA / "B2-day8-gold-tAMMY.avi",
                             start=SETTLE, gray=True, half_res=True)
    nonp, n_fps = read_video(DATA / "A3-day8-non-Tammy.avi",
                             start=SETTLE, gray=True, half_res=True)
    n = min(len(gold), len(nonp), 80)
    gold = gold[:n]; nonp = nonp[:n]
    fps = g_fps

    print("  Computing displacement ...")
    gold_dm = disp_maps(gold)
    nonp_dm = disp_maps(nonp)
    gold_tr = np.array([float(np.mean(d)) for d in gold_dm])
    nonp_tr = np.array([float(np.mean(d)) for d in nonp_dm])
    vmax = max(
        float(np.percentile(np.concatenate([d.ravel() for d in nonp_dm[1:]]), 99)),
        0.5)

    fig = plt.figure(figsize=(12, 7), dpi=100)
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.15, wspace=0.08)
    ax_g = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[0, 1])
    ax_tr = fig.add_subplot(gs[1, :])
    for ax in (ax_g, ax_n):
        ax.set_xticks([]); ax.set_yticks([])

    im_g = ax_g.imshow(blend_heatmap(gold[0], gold_dm[0], vmax=vmax, alpha=0.55))
    im_n = ax_n.imshow(blend_heatmap(nonp[0], nonp_dm[0], vmax=vmax, alpha=0.55))
    ax_g.set_title("B2 Gold PVDF (Poled)\n3.05 px peak", fontsize=12,
                   fontweight="bold", color="#fdcb6e")
    ax_n.set_title("A3 Non-poled PVDF\n7.68 px peak", fontsize=12,
                   fontweight="bold", color="#55efc4")

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=[ax_g, ax_n], fraction=0.02, pad=0.01,
                 label="Displacement (px)")

    t_arr = np.arange(n) / fps
    ax_tr.plot(t_arr, gold_tr, color="#fdcb6e", lw=2, label="B2 Gold (3.05 px)")
    ax_tr.plot(t_arr, nonp_tr, color="#55efc4", lw=2, label="A3 Non-poled (7.68 px)")
    cursor = ax_tr.axvline(0, color="#ffd93d", lw=2, alpha=0.7)
    ax_tr.legend(loc="upper right", fontsize=10)
    ax_tr.set_xlim(0, t_arr[-1])
    ax_tr.set_ylim(0, max(max(gold_tr), max(nonp_tr)) * 1.2)
    ax_tr.set_xlabel("Time (s)")
    ax_tr.set_ylabel("Mean disp. (px)")

    fig.suptitle("Substrate Comparison -- Gold vs Non-poled (Same Cell Line XR1)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    gif_frames = []
    for i in range(n):
        im_g.set_data(blend_heatmap(gold[i], gold_dm[i], vmax=vmax, alpha=0.55))
        im_n.set_data(blend_heatmap(nonp[i], nonp_dm[i], vmax=vmax, alpha=0.55))
        cursor.set_xdata([t_arr[i], t_arr[i]])
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f,
        "XR1 | Device (Pulsed) | Day 8 | Non-poled beats 2.5x harder than gold"
    ) for f in gif_frames]
    save_gif(gif_frames, OUT / "G13_substrate_compare.gif", fps=int(min(fps, 12)))
    del gold, nonp, gold_dm, nonp_dm, gif_frames
    gc.collect()


# ================================================================
#  GIF 14: "Gold Adhesion Challenge" -- A2 vs B2 vs A3
# ================================================================

def gif14_adhesion_challenge():
    vids = [
        ("A2-day8-gold-tAMMY-new.avi", "A2 Gold PVDF", "Adhesion failure", 0.01, "#ff7675"),
        ("B2-day8-gold-tAMMY.avi",     "B2 Gold PVDF", "Beating (3.05 px)", 3.05, "#fdcb6e"),
        ("A3-day8-non-Tammy.avi",      "A3 Non-poled",  "Strong (7.68 px)",  7.68, "#55efc4"),
    ]
    n_frames = 70

    composites = []
    for fname, label, status, amp, color in vids:
        print(f"  Loading {fname} ...")
        fr, fps = read_video(DATA / fname, start=SETTLE,
                             end=SETTLE + n_frames + 5,
                             gray=True, half_res=True)
        fr = fr[:n_frames]
        if amp > 0.5:
            dm = disp_maps(fr)
            vmax_v = max(float(np.percentile(
                np.concatenate([d.ravel() for d in dm[1:]]), 99)), 0.5)
            comp = [blend_heatmap(fr[i], dm[i], vmax=vmax_v, alpha=0.55)
                    for i in range(len(fr))]
            del dm
        else:
            comp = [np.stack([f] * 3, axis=-1) for f in fr]
        while len(comp) < n_frames:
            comp.append(comp[-1])
        composites.append((comp, label, status, amp, color))
        del fr
        gc.collect()

    target_h, target_w = composites[0][0][0].shape[:2]
    for comp_list, _, _, _, _ in composites:
        for j in range(len(comp_list)):
            if comp_list[j].shape[:2] != (target_h, target_w):
                comp_list[j] = cv2.resize(comp_list[j], (target_w, target_h))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=100)
    ims = []
    for idx, (comp, label, status, amp, color) in enumerate(composites):
        ax = axes[idx]
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(comp[0])
        ims.append(im)
        ax.set_title(f"{label}\n{status}", fontsize=12,
                     fontweight="bold", color=color)

    fig.suptitle("Gold Adhesion Challenge -- Variable Outcomes on Gold Substrate",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    gif_frames = []
    for i in range(n_frames):
        for idx, (comp, _, _, _, _) in enumerate(composites):
            ims[idx].set_data(comp[i])
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f,
        "XR1 | Device Day 8 | Gold: adhesion-dependent (A2 fail, B2 ok) | Non-poled: consistent"
    ) for f in gif_frames]
    save_gif(gif_frames, OUT / "G14_adhesion_challenge.gif", fps=10)
    del composites, gif_frames
    gc.collect()


# ================================================================
#  GIF 15: "Biological Replicates" -- A3 vs B3 day 6
# ================================================================

def gif15_bio_replicates():
    print("  Loading A3-day6 + B3-day6 non-poled device ...")
    a3, a3_fps = read_video(DATA / "A3-day6-NON1-TAMMY.avi",
                            start=SETTLE, gray=True, half_res=True)
    b3, b3_fps = read_video(DATA / "B3-day6-NON2-tammy.avi",
                            start=SETTLE, gray=True, half_res=True)
    n = min(len(a3), len(b3), 80)
    a3 = a3[:n]; b3 = b3[:n]
    fps = a3_fps

    print("  Computing displacement ...")
    a3_dm = disp_maps(a3)
    b3_dm = disp_maps(b3)
    a3_tr = np.array([float(np.mean(d)) for d in a3_dm])
    b3_tr = np.array([float(np.mean(d)) for d in b3_dm])
    vmax = max(
        float(np.percentile(np.concatenate([d.ravel() for d in a3_dm[1:]]), 99)),
        float(np.percentile(np.concatenate([d.ravel() for d in b3_dm[1:]]), 99)),
        0.5)

    fig = plt.figure(figsize=(12, 7), dpi=100)
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.15, wspace=0.08)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_tr = fig.add_subplot(gs[1, :])
    for ax in (ax_a, ax_b):
        ax.set_xticks([]); ax.set_yticks([])

    im_a = ax_a.imshow(blend_heatmap(a3[0], a3_dm[0], vmax=vmax, alpha=0.55))
    im_b = ax_b.imshow(blend_heatmap(b3[0], b3_dm[0], vmax=vmax, alpha=0.55))
    ax_a.set_title("A3 Non-poled (Replicate 1)\n3.16 px peak", fontsize=12,
                   fontweight="bold", color="#55efc4")
    ax_b.set_title("B3 Non-poled (Replicate 2)\n3.63 px peak", fontsize=12,
                   fontweight="bold", color="#81ecec")

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=[ax_a, ax_b], fraction=0.02, pad=0.01,
                 label="Displacement (px)")

    t_arr = np.arange(n) / fps
    ax_tr.plot(t_arr, a3_tr, color="#55efc4", lw=2, label="A3 (3.16 px)")
    ax_tr.plot(t_arr, b3_tr, color="#81ecec", lw=2, label="B3 (3.63 px)")
    cursor = ax_tr.axvline(0, color="#ffd93d", lw=2, alpha=0.7)
    ax_tr.legend(loc="upper right", fontsize=10)
    ax_tr.set_xlim(0, t_arr[-1])
    ax_tr.set_ylim(0, max(max(a3_tr), max(b3_tr)) * 1.2)
    ax_tr.set_xlabel("Time (s)")
    ax_tr.set_ylabel("Mean disp. (px)")

    fig.suptitle("Biological Replicates (N=2) -- Consistent Beating on Non-poled PVDF",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    gif_frames = []
    for i in range(n):
        im_a.set_data(blend_heatmap(a3[i], a3_dm[i], vmax=vmax, alpha=0.55))
        im_b.set_data(blend_heatmap(b3[i], b3_dm[i], vmax=vmax, alpha=0.55))
        cursor.set_xdata([t_arr[i], t_arr[i]])
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f,
        "XR1 | Non-poled PVDF | Device | Day 6 | N=2: 3.16 vs 3.63 px (consistent)"
    ) for f in gif_frames]
    save_gif(gif_frames, OUT / "G15_bio_replicates.gif", fps=int(min(fps, 10)))
    del a3, b3, a3_dm, b3_dm, gif_frames
    gc.collect()


# ================================================================
#  GIF 16: "Calcium Substrate Effect" -- C2 gold vs C3 non-poled FL
# ================================================================

def gif16_calcium_substrate():
    print("  Loading C2-day8-gold + C3-day8-non-poled (same device plate) ...")
    c2_raw, c2_fps = read_video(DATA / "C2-day8-gold-GCaMP6f-FL.avi",
                                start=0, gray=True, half_res=False)
    c3_raw, c3_fps = read_video(DATA / "C3-day8-non-Jaz-fl.avi",
                                start=0, gray=True, half_res=False)

    c2_arr = np.array(c2_raw, dtype=np.float64)
    c2_f0 = np.maximum(np.percentile(c2_arr[:15], 5, axis=0), 1.0)
    c2_dff0 = (c2_arr - c2_f0) / c2_f0
    c2_dff0 = gaussian_filter(c2_dff0, sigma=(0.5, 1.5, 1.5))
    del c2_arr

    c3_arr = np.array(c3_raw, dtype=np.float64)
    c3_f0 = np.maximum(np.percentile(c3_arr[:15], 5, axis=0), 1.0)
    c3_dff0 = (c3_arr - c3_f0) / c3_f0
    c3_dff0 = gaussian_filter(c3_dff0, sigma=(0.5, 1.5, 1.5))
    del c3_arr

    c2_vmax = max(float(np.percentile(c2_dff0, 99.5)), 0.02)
    c3_vmax = max(float(np.percentile(c3_dff0, 99.5)), 0.02)
    shared_vmax = max(c2_vmax, c3_vmax)

    c2_trace = np.array([float(np.mean(c2_dff0[i])) for i in range(len(c2_dff0))])
    c3_trace = np.array([float(np.mean(c3_dff0[i])) for i in range(len(c3_dff0))])

    n_use = min(len(c2_dff0), len(c3_dff0), 100)

    fig = plt.figure(figsize=(12, 7), dpi=100)
    gs = GridSpec(2, 2, height_ratios=[3, 1.2], hspace=0.2, wspace=0.1)
    ax_c2 = fig.add_subplot(gs[0, 0])
    ax_c3 = fig.add_subplot(gs[0, 1])
    ax_tr = fig.add_subplot(gs[1, :])
    for ax in (ax_c2, ax_c3):
        ax.set_xticks([]); ax.set_yticks([])

    im_c2 = ax_c2.imshow(c2_dff0[0], cmap="magma", vmin=-0.02, vmax=shared_vmax)
    im_c3 = ax_c3.imshow(c3_dff0[0], cmap="magma", vmin=-0.02, vmax=shared_vmax)
    ax_c2.set_title("C2 Gold PVDF (Poled) Device Day 8\n"
                    "dF/F0 = 0.050, 1 transient", fontsize=11,
                    fontweight="bold", color="#fdcb6e")
    ax_c3.set_title("C3 Non-poled PVDF Device Day 8\n"
                    "dF/F0 = 0.041, 7 transients", fontsize=11,
                    fontweight="bold", color="#74b9ff")
    fig.colorbar(im_c2, ax=[ax_c2, ax_c3], fraction=0.02, pad=0.01,
                 label="dF/F0")

    c2_t = np.arange(len(c2_trace)) / c2_fps
    c3_t = np.arange(len(c3_trace)) / c3_fps
    ax_tr.plot(c2_t[:n_use], c2_trace[:n_use], color="#fdcb6e", lw=2,
               label=f"C2 Gold (dF/F0={np.max(c2_trace[:n_use]):.3f})")
    ax_tr.plot(c3_t[:n_use], c3_trace[:n_use], color="#74b9ff", lw=2,
               label=f"C3 Non-poled (dF/F0={np.max(c3_trace[:n_use]):.3f})")
    cursor = ax_tr.axvline(0, color="#ffd93d", lw=2, alpha=0.7)
    ax_tr.legend(loc="upper right", fontsize=9)
    ax_tr.set_xlabel("Time (s)")
    ax_tr.set_ylabel("Mean dF/F0")

    fig.suptitle("Calcium Substrate Effect -- Gold vs Non-poled (Same Plate, Day 8)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    gif_frames = []
    for i in range(n_use):
        im_c2.set_data(c2_dff0[min(i, len(c2_dff0)-1)])
        im_c3.set_data(c3_dff0[min(i, len(c3_dff0)-1)])
        t_now = i / max(c2_fps, c3_fps)
        cursor.set_xdata([t_now, t_now])
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f,
        "GCaMP6f | Device Plate Day 8 | Gold 1 transient vs Non-poled 7 transients | Same plate comparison"
    ) for f in gif_frames]
    save_gif(gif_frames, OUT / "G16_calcium_substrate.gif", fps=8)
    del c2_raw, c3_raw, c2_dff0, c3_dff0, gif_frames
    gc.collect()


# ================================================================
#  GIF 17: "EMD Evidence" -- BF + FL for C3 showing dissociation
# ================================================================

def gif17_emd_evidence():
    print("  Loading C3-day8 BF + FL ...")
    bf, bf_fps = read_video(DATA / "C3-day8-non-Jaz.avi",
                            start=SETTLE, gray=True, half_res=True)
    fl_raw, fl_fps = read_video(DATA / "C3-day8-non-Jaz-fl.avi",
                                start=0, gray=True, half_res=False)

    print("  Loading C2-day8 BF + FL ...")
    bf2, bf2_fps = read_video(DATA / "C2-day8-gold-Jaz.avi",
                              start=SETTLE, gray=True, half_res=True)
    fl2_raw, fl2_fps = read_video(DATA / "C2-day8-gold-GCaMP6f-FL.avi",
                                  start=0, gray=True, half_res=False)

    fl_arr = np.array(fl_raw, dtype=np.float64)
    fl_f0 = np.maximum(np.mean(fl_arr[:10], axis=0), 1.0)
    fl_dff0 = gaussian_filter((fl_arr - fl_f0) / fl_f0, sigma=(0.5, 1, 1))
    del fl_arr
    fl_vmax = max(float(np.percentile(fl_dff0, 99.5)), 0.01)

    fl2_arr = np.array(fl2_raw, dtype=np.float64)
    fl2_f0 = np.maximum(np.mean(fl2_arr[:10], axis=0), 1.0)
    fl2_dff0 = gaussian_filter((fl2_arr - fl2_f0) / fl2_f0, sigma=(0.5, 1, 1))
    del fl2_arr
    fl2_vmax = max(float(np.percentile(fl2_dff0, 99.5)), 0.01)
    shared_fl_vmax = max(fl_vmax, fl2_vmax)

    fl_trace = np.array([float(np.mean(fl_dff0[i])) for i in range(len(fl_dff0))])
    fl2_trace = np.array([float(np.mean(fl2_dff0[i])) for i in range(len(fl2_dff0))])

    n_use = min(len(bf), int(len(fl_raw) * bf_fps / fl_fps),
                len(bf2), int(len(fl2_raw) * bf2_fps / fl2_fps), 100)
    bf = bf[:n_use]; bf2 = bf2[:n_use]

    fig = plt.figure(figsize=(12, 8), dpi=100)
    gs = GridSpec(2, 4, height_ratios=[1, 1], hspace=0.25, wspace=0.1)
    ax_bf3 = fig.add_subplot(gs[0, 0])
    ax_fl3 = fig.add_subplot(gs[0, 1])
    ax_bf2 = fig.add_subplot(gs[1, 0])
    ax_fl2 = fig.add_subplot(gs[1, 1])
    ax_tr3 = fig.add_subplot(gs[0, 2:])
    ax_tr2 = fig.add_subplot(gs[1, 2:])

    for ax in (ax_bf3, ax_fl3, ax_bf2, ax_fl2):
        ax.set_xticks([]); ax.set_yticks([])

    im_bf3 = ax_bf3.imshow(np.stack([bf[0]]*3, axis=-1))
    ax_bf3.set_title("C3 BF (Non-poled)\nNo contraction", fontsize=10,
                     fontweight="bold", color="#ff7675")
    im_fl3 = ax_fl3.imshow(fl_dff0[0], cmap="hot", vmin=0, vmax=shared_fl_vmax)
    ax_fl3.set_title("C3 FL (Non-poled)\n7 transients", fontsize=10,
                     fontweight="bold", color="#55efc4")

    im_bf2 = ax_bf2.imshow(np.stack([bf2[0]]*3, axis=-1))
    ax_bf2.set_title("C2 BF (Gold)\nNo contraction", fontsize=10,
                     fontweight="bold", color="#ff7675")
    im_fl2 = ax_fl2.imshow(fl2_dff0[0], cmap="hot", vmin=0, vmax=shared_fl_vmax)
    ax_fl2.set_title("C2 FL (Gold)\n1 transient", fontsize=10,
                     fontweight="bold", color="#fdcb6e")

    fl_t = np.arange(len(fl_trace)) / fl_fps
    ax_tr3.plot(fl_t, fl_trace, color="#55efc4", lw=1.5)
    ax_tr3.set_ylabel("dF/F0")
    ax_tr3.set_title("C3 Ca2+ trace (12.7 BPM)", fontsize=10, fontweight="bold")

    fl2_t = np.arange(len(fl2_trace)) / fl2_fps
    ax_tr2.plot(fl2_t, fl2_trace, color="#fdcb6e", lw=1.5)
    ax_tr2.set_xlabel("Time (s)")
    ax_tr2.set_ylabel("dF/F0")
    ax_tr2.set_title("C2 Ca2+ trace (single transient)", fontsize=10, fontweight="bold")

    fig.suptitle("Electromechanical Dissociation -- Ca2+ Active, Contraction Absent",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    gif_frames = []
    for i in range(n_use):
        im_bf3.set_data(np.stack([bf[i]]*3, axis=-1))
        fl_idx = min(int(i * fl_fps / bf_fps), len(fl_dff0) - 1)
        im_fl3.set_data(fl_dff0[fl_idx])

        im_bf2.set_data(np.stack([bf2[i]]*3, axis=-1))
        fl2_idx = min(int(i * fl2_fps / bf2_fps), len(fl2_dff0) - 1)
        im_fl2.set_data(fl2_dff0[fl2_idx])

        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f,
        "GCaMP6f | Day 8 | EMD: immature excitation-contraction coupling in both wells"
    ) for f in gif_frames]
    save_gif(gif_frames, OUT / "G17_emd_evidence.gif", fps=8)
    del bf, bf2, fl_raw, fl2_raw, fl_dff0, fl2_dff0, gif_frames
    gc.collect()


# ================================================================
#  GIF 18: "Calcium Microdomains" -- spatial heterogeneity map
# ================================================================

def gif18_microdomains():
    print("  Loading C2-day6-gold-FL-control.avi (control) ...")
    ctrl_raw, ctrl_fps = read_video(DATA / "C2-day6-gold-FL-control.avi",
                                    start=0, gray=True, half_res=False)
    print(f"    Control: {len(ctrl_raw)} frames @ {ctrl_fps:.1f} FPS")

    print("  Loading C2-day8-gold-GCaMP6f-FL.avi (device) ...")
    dev_raw, dev_fps = read_video(DATA / "C2-day8-gold-GCaMP6f-FL.avi",
                                  start=0, gray=True, half_res=False)
    print(f"    Device:  {len(dev_raw)} frames @ {dev_fps:.1f} FPS")

    def _compute_maps(raw_frames, fps_val):
        arr = np.array(raw_frames, dtype=np.float64)
        f0 = np.maximum(np.percentile(arr[:15], 5, axis=0), 1.0)
        dff0 = (arr - f0) / f0
        sm = gaussian_filter(dff0, sigma=(0.5, 1.5, 1.5))
        del arr, dff0
        window = 20
        act = []
        for s in range(0, len(sm) - window, 2):
            act.append(np.std(sm[s:s+window], axis=0))
        g_std = np.std(sm, axis=0)
        cv = gaussian_filter(g_std / np.maximum(np.mean(np.abs(sm), axis=0), 1e-6),
                             sigma=2)
        return sm, act, cv

    ctrl_sm, ctrl_act, ctrl_cv = _compute_maps(ctrl_raw, ctrl_fps)
    dev_sm, dev_act, dev_cv = _compute_maps(dev_raw, dev_fps)
    del ctrl_raw, dev_raw

    act_vmax = max(
        float(np.percentile(np.concatenate([a.ravel() for a in ctrl_act]), 99.5)),
        float(np.percentile(np.concatenate([a.ravel() for a in dev_act]), 99.5)),
        0.005)
    live_vmax = act_vmax * 3
    cv_vmax = max(float(np.percentile(ctrl_cv, 99)),
                  float(np.percentile(dev_cv, 99)), 0.1)

    fig = plt.figure(figsize=(14, 8), dpi=100)
    gs = GridSpec(2, 3, width_ratios=[2, 2, 1.2], wspace=0.12, hspace=0.25)

    ax_cl = fig.add_subplot(gs[0, 0])
    ax_ca = fig.add_subplot(gs[0, 1])
    ax_cc = fig.add_subplot(gs[0, 2])
    ax_dl = fig.add_subplot(gs[1, 0])
    ax_da = fig.add_subplot(gs[1, 1])
    ax_dc = fig.add_subplot(gs[1, 2])

    for ax in (ax_cl, ax_ca, ax_cc, ax_dl, ax_da, ax_dc):
        ax.set_xticks([]); ax.set_yticks([])

    ax_cl.set_ylabel("Control (Day 6)\ndF/F0 = 0.216", fontsize=10,
                     fontweight="bold", color="#74b9ff")
    ax_dl.set_ylabel("Device (Day 8)\ndF/F0 = 0.050", fontsize=10,
                     fontweight="bold", color="#ff6b6b")

    im_cl = ax_cl.imshow(ctrl_sm[0], cmap="magma", vmin=-0.02, vmax=live_vmax)
    ax_cl.set_title("Live dF/F0", fontsize=11, fontweight="bold")
    im_ca = ax_ca.imshow(ctrl_act[0], cmap="inferno", vmin=0, vmax=act_vmax)
    ax_ca.set_title("Temporal Std (Activity)", fontsize=11, fontweight="bold")
    ax_cc.imshow(ctrl_cv, cmap="plasma", vmin=0, vmax=cv_vmax)
    ax_cc.set_title("Spatial CV", fontsize=11, fontweight="bold")

    im_dl = ax_dl.imshow(dev_sm[0], cmap="magma", vmin=-0.02, vmax=live_vmax)
    im_da = ax_da.imshow(dev_act[0], cmap="inferno", vmin=0, vmax=act_vmax)
    ax_dc.imshow(dev_cv, cmap="plasma", vmin=0, vmax=cv_vmax)

    fig.colorbar(im_cl, ax=[ax_cl, ax_dl], fraction=0.04, pad=0.02,
                 label="dF/F0", shrink=0.8)
    fig.colorbar(im_ca, ax=[ax_ca, ax_da], fraction=0.04, pad=0.02,
                 label="Std(dF/F0)", shrink=0.8)

    ctrl_t = ax_cl.text(0.02, 0.93, "", transform=ax_cl.transAxes,
                        fontsize=10, va="top", color="white",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="#00000088", ec="none"))
    dev_t = ax_dl.text(0.02, 0.93, "", transform=ax_dl.transAxes,
                       fontsize=10, va="top", color="white",
                       bbox=dict(boxstyle="round,pad=0.3",
                                 fc="#00000088", ec="none"))

    fig.suptitle("Calcium Microdomains -- Control vs Device (Gold PVDF, C2)",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    n_act = min(len(ctrl_act), len(dev_act))
    gif_frames = []
    for i in range(n_act):
        ci = min(i * 2, len(ctrl_sm) - 1)
        di = min(i * 2, len(dev_sm) - 1)
        im_cl.set_data(ctrl_sm[ci])
        im_ca.set_data(ctrl_act[i])
        im_dl.set_data(dev_sm[di])
        im_da.set_data(dev_act[i])
        ctrl_t.set_text(f"t = {ci / ctrl_fps:.1f} s")
        dev_t.set_text(f"t = {di / dev_fps:.1f} s")
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f,
        "Well C2 | GCaMP6f | Gold PVDF | Control Day 6 vs Device Day 8 | "
        "Different plates -- spatial heterogeneity in both conditions"
    ) for f in gif_frames]
    save_gif(gif_frames, OUT / "G18_microdomains.gif", fps=8)
    del ctrl_sm, dev_sm, ctrl_act, dev_act, gif_frames
    gc.collect()


# ================================================================
#  GIF 19: "Gold Heartbeat" -- hero contraction on gold PVDF (B2)
# ================================================================

def gif19_gold_heartbeat():
    print("  Loading B2-day8-gold-tAMMY.avi ...")
    frames, fps = read_video(DATA / "B2-day8-gold-tAMMY.avi",
                             start=SETTLE, gray=True, half_res=True)
    print(f"    {len(frames)} frames @ {fps:.1f} FPS")

    print("  Computing displacement maps ...")
    dmaps = disp_maps(frames)
    trace = np.array([float(np.mean(d)) for d in dmaps])
    vmax = float(np.percentile(np.concatenate([d.ravel() for d in dmaps[1:]]), 99))

    s, e = find_beat_window(trace, fps, n_cycles=2)
    print(f"    Beat window: frames {s}-{e} ({(e-s)/fps:.1f}s)")
    cyc_frames = frames[s:e]
    cyc_dmaps = dmaps[s:e]
    cyc_trace = trace[s:e]

    fig, (ax_img, ax_tr) = plt.subplots(
        2, 1, figsize=(8, 7), dpi=100,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08})
    ax_img.set_xticks([]); ax_img.set_yticks([])

    composite = blend_heatmap(cyc_frames[0], cyc_dmaps[0],
                              vmax=vmax, alpha=0.55)
    im = ax_img.imshow(composite)
    title = ax_img.set_title(
        "Beating on Gold PVDF -- Piezoelectric Substrate",
        fontsize=14, fontweight="bold", pad=10)
    time_txt = ax_img.text(0.02, 0.95, "", transform=ax_img.transAxes,
                           fontsize=12, va="top", color="white",
                           bbox=dict(boxstyle="round,pad=0.3",
                                     fc="#00000088", ec="none"))

    sm = plt.cm.ScalarMappable(cmap="inferno",
                               norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=ax_img, fraction=0.03, pad=0.01,
                 label="Displacement (px)")

    t_arr = np.arange(len(cyc_trace)) / fps
    ax_tr.plot(t_arr, cyc_trace, color="#fdcb6e", lw=2)
    cursor = ax_tr.axvline(0, color="#ffd93d", lw=2, alpha=0.7)
    ax_tr.set_xlim(0, t_arr[-1])
    ax_tr.set_ylim(0, max(cyc_trace) * 1.3)
    ax_tr.set_xlabel("Time (s)")
    ax_tr.set_ylabel("Mean disp.")

    fig.tight_layout()

    gif_frames = []
    for i in range(len(cyc_frames)):
        comp = blend_heatmap(cyc_frames[i], cyc_dmaps[i],
                             vmax=vmax, alpha=0.55)
        im.set_data(comp)
        time_txt.set_text(f"t = {i/fps:.2f} s")
        cursor.set_xdata([t_arr[i], t_arr[i]])
        gif_frames.append(fig_to_rgb(fig))

    plt.close(fig)
    gif_frames = [add_annotation_bar(f,
        "Well B2 | XR1 | Gold PVDF (Poled) | Device (Pulsed) | Day 8 | "
        "Peak: 3.05 px | Cells beating on piezoelectric substrate"
    ) for f in gif_frames]
    save_gif(gif_frames, OUT / "G19_gold_heartbeat.gif",
             fps=int(min(fps, 12)))
    del frames, dmaps, gif_frames
    gc.collect()


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    print("=" * 60)
    print("  GENERATING FANCY GIFS (19 total)")
    print("  Output: fancy_gifs/")
    print("=" * 60)

    tasks = [
        ("G01 The Heartbeat",          gif01_heartbeat),
        ("G02 Hidden Calcium (EMD)",   gif02_hidden_calcium),
        ("G03 Device Effect",          gif03_device_effect),
        ("G04 Maturation Timeline",    gif04_maturation),
        ("G05 Calcium Waves",          gif05_calcium_waves),
        ("G06 Adhesion Problem",       gif06_adhesion),
        ("G07 Structure-to-Function",  gif07_structure_function),
        ("G08 Contraction Vectors",    gif08_contraction_vectors),
        ("G09 Two Cell Lines",         gif09_cell_lines),
        ("G10 Full Dashboard",         gif10_dashboard),
        ("G11 All Wells Day 8",        gif11_all_wells),
        ("G12 Piezo Isolation (B2)",   gif12_piezo_isolation),
        ("G13 Substrate Comparison",   gif13_substrate_compare),
        ("G14 Gold Adhesion Challenge", gif14_adhesion_challenge),
        ("G15 Biological Replicates",  gif15_bio_replicates),
        ("G16 Calcium Substrate",      gif16_calcium_substrate),
        ("G17 EMD Evidence",           gif17_emd_evidence),
        ("G18 Calcium Microdomains",   gif18_microdomains),
        ("G19 Gold Heartbeat",         gif19_gold_heartbeat),
    ]

    results = []
    for name, func in tasks:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            func()
            results.append((name, "OK"))
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results.append((name, f"FAILED: {e}"))
        gc.collect()

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, status in results:
        icon = "[OK]" if status == "OK" else "[FAIL]"
        print(f"  {icon} {name}: {status}")
    print(f"{'='*60}")
    print("  Done!")
