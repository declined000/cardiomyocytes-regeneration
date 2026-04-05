"""
Comprehensive video quality inspection for cardiomyocyte beating analysis.
Extracts metadata, quality metrics, and identifies duplicates.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA = Path("Cardiomyocytes")
OUT = Path("video_inspection")
OUT.mkdir(exist_ok=True)

SAMPLE_FRAMES = 10  # frames to sample for quality metrics


def laplacian_variance(gray_frame):
    """Sharpness metric via Laplacian variance (higher = sharper)."""
    return cv2.Laplacian(gray_frame, cv2.CV_64F).var()


def estimate_motion(cap, n_pairs=5):
    """Estimate motion by computing mean frame-to-frame pixel difference."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        return 0.0, 0.0

    step = max(1, total_frames // (n_pairs + 1))
    diffs = []

    ret, prev = cap.read()
    if not ret:
        return 0.0, 0.0
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(float)

    for i in range(n_pairs):
        target = (i + 1) * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        diff = np.mean(np.abs(curr_gray - prev_gray))
        diffs.append(diff)
        prev_gray = curr_gray

    if len(diffs) == 0:
        return 0.0, 0.0
    return np.mean(diffs), np.max(diffs)


def classify_video(filename):
    """Parse filename to extract condition, cell type, day, modality."""
    fn = filename.lower()

    # Day
    day = "unknown"
    for d in range(1, 10):
        if f"day{d}" in fn:
            day = f"day{d}"
            break

    # Modality
    if "fl" in fn or "fluorescence" in fn:
        modality = "FL"
    elif "bf" in fn:
        modality = "BF"
    elif "dapi" in fn:
        modality = "DAPI"
    elif "phase" in fn:
        modality = "Phase"
    else:
        modality = "BF"  # default for unlabeled

    # Cell type
    if "jaz" in fn:
        cell_type = "GCaMP6f"
    elif "tammy" in fn:
        cell_type = "XR1"
    else:
        cell_type = "Unknown"

    # Condition
    if "control" in fn or "comtrol" in fn:
        location = "control"
    elif "device" in fn:
        location = "device"
    else:
        location = "unknown"

    # Substrate
    if "gold" in fn:
        substrate = "gold"
    elif "non" in fn:
        substrate = "non"
    else:
        substrate = "unknown"

    # Well
    well = "unknown"
    for w in ["A2", "A3", "B2", "B3", "C2", "C3"]:
        if w.lower() in fn or w.upper() in fn:
            well = w.upper()
            break
    if well == "unknown" and "middlegold" in fn:
        well = "mid"

    # Duplicate marker
    is_variant = "new" in fn or "fixed" in fn or "beating" in fn

    return day, modality, cell_type, location, substrate, well, is_variant


def inspect_video(filepath):
    """Extract all metadata and quality metrics from a single video."""
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    file_size_mb = filepath.stat().st_size / (1024 * 1024)

    # Sample frames for quality metrics
    step = max(1, total_frames // SAMPLE_FRAMES)
    brightnesses = []
    contrasts = []
    sharpnesses = []

    for i in range(min(SAMPLE_FRAMES, total_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightnesses.append(np.mean(gray))
        contrasts.append(np.std(gray))
        sharpnesses.append(laplacian_variance(gray))

    mean_brightness = np.mean(brightnesses) if brightnesses else 0
    mean_contrast = np.mean(contrasts) if contrasts else 0
    mean_sharpness = np.mean(sharpnesses) if sharpnesses else 0
    brightness_stability = np.std(brightnesses) if len(brightnesses) > 1 else 0

    # Motion estimation
    mean_motion, max_motion = estimate_motion(cap)

    cap.release()

    day, modality, cell_type, location, substrate, well, is_variant = classify_video(filepath.name)

    return {
        "filename": filepath.name,
        "day": day,
        "well": well,
        "substrate": substrate,
        "location": location,
        "cell_type": cell_type,
        "modality": modality,
        "is_variant": is_variant,
        "fps": round(fps, 2),
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_s": round(duration, 2),
        "file_size_mb": round(file_size_mb, 1),
        "mean_brightness": round(mean_brightness, 1),
        "mean_contrast": round(mean_contrast, 1),
        "mean_sharpness": round(mean_sharpness, 1),
        "brightness_stability": round(brightness_stability, 2),
        "mean_motion": round(mean_motion, 2),
        "max_motion": round(max_motion, 2),
    }


def quality_flag(row):
    """Flag potential quality issues."""
    flags = []
    if row["fps"] < 10:
        flags.append("LOW_FPS")
    if row["total_frames"] < 50:
        flags.append("TOO_SHORT")
    if row["duration_s"] < 2:
        flags.append("VERY_SHORT")
    if row["mean_brightness"] < 20:
        flags.append("TOO_DARK")
    if row["mean_brightness"] > 240:
        flags.append("OVEREXPOSED")
    if row["mean_contrast"] < 5:
        flags.append("LOW_CONTRAST")
    if row["mean_sharpness"] < 10:
        flags.append("BLURRY")
    if row["mean_motion"] < 0.5 and row["modality"] == "BF":
        flags.append("NO_MOTION")
    if row["brightness_stability"] > 15:
        flags.append("UNSTABLE_LIGHT")
    return "; ".join(flags) if flags else "OK"


def main():
    print("=" * 70)
    print("VIDEO QUALITY INSPECTION")
    print("=" * 70)

    videos = sorted(DATA.glob("*.avi"), key=lambda p: p.stat().st_mtime)
    print(f"\nFound {len(videos)} video files\n")

    rows = []
    for i, v in enumerate(videos):
        print(f"  [{i+1:2d}/{len(videos)}] {v.name}...", end="", flush=True)
        info = inspect_video(v)
        if info:
            rows.append(info)
            print(f" {info['fps']} fps, {info['total_frames']} frames, {info['duration_s']}s")
        else:
            print(" FAILED TO OPEN")

    df = pd.DataFrame(rows)
    df["quality_flag"] = df.apply(quality_flag, axis=1)

    # Sort by day then well
    day_order = {"day1": 1, "day2": 2, "day3": 3, "day6": 6, "day8": 8, "unknown": 99}
    df["day_num"] = df["day"].map(day_order).fillna(99)
    df = df.sort_values(["day_num", "well", "substrate", "modality", "is_variant"])
    df = df.drop(columns=["day_num"])

    # Save full table
    df.to_csv(OUT / "video_inspection_full.csv", index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total videos: {len(df)}")
    print(f"Days: {sorted(df['day'].unique())}")
    print(f"Modalities: BF={len(df[df['modality']=='BF'])}, FL={len(df[df['modality']=='FL'])}, Phase={len(df[df['modality']=='Phase'])}, DAPI={len(df[df['modality']=='DAPI'])}")
    print(f"Cell types: XR1={len(df[df['cell_type']=='XR1'])}, GCaMP6f={len(df[df['cell_type']=='GCaMP6f'])}, Unknown={len(df[df['cell_type']=='Unknown'])}")
    print(f"Variants (retakes): {len(df[df['is_variant']==True])}")

    flagged = df[df["quality_flag"] != "OK"]
    print(f"\nQuality issues: {len(flagged)} videos flagged")
    if len(flagged) > 0:
        for _, r in flagged.iterrows():
            print(f"  {r['filename']:45s}  {r['quality_flag']}")

    # Identify duplicate groups (same well+day+substrate+modality)
    print("\n" + "=" * 70)
    print("DUPLICATE GROUPS (same well/day/substrate/modality)")
    print("=" * 70)
    group_cols = ["day", "well", "substrate", "modality"]
    grouped = df.groupby(group_cols).filter(lambda x: len(x) > 1)
    if len(grouped) > 0:
        for name, group in grouped.groupby(group_cols):
            print(f"\n  Group: {name}")
            for _, r in group.iterrows():
                variant_tag = " [VARIANT]" if r["is_variant"] else ""
                print(f"    {r['filename']:45s}  {r['fps']:6.1f}fps  {r['total_frames']:5d}fr  "
                      f"sharp={r['mean_sharpness']:7.1f}  motion={r['mean_motion']:5.2f}  "
                      f"bright={r['mean_brightness']:5.1f}  {r['quality_flag']}{variant_tag}")

    # Compact table for display
    display_cols = [
        "filename", "day", "well", "substrate", "location", "cell_type",
        "modality", "fps", "resolution", "total_frames", "duration_s",
        "mean_brightness", "mean_contrast", "mean_sharpness", "mean_motion",
        "quality_flag"
    ]
    compact = df[display_cols].copy()
    compact.to_csv(OUT / "video_inspection_compact.csv", index=False)

    print("\n" + "=" * 70)
    print(f"Full results:    {OUT / 'video_inspection_full.csv'}")
    print(f"Compact results: {OUT / 'video_inspection_compact.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
