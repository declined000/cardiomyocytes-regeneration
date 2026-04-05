"""
Comprehensive video quality audit for cardiomyocyte beating analysis.
Reads the inspection CSV, classifies usability, resolves duplicates,
and produces actionable tables + recommendations for next experiment.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INSP = Path("video_inspection/video_inspection_full.csv")
OUT = Path("video_inspection")

MIN_FPS_BEATING = 8.0       # minimum fps for reliable beat detection
MIN_SHARPNESS = 10.0        # laplacian variance threshold
MIN_CONTRAST = 8.0          # std of pixel intensity
MIN_DURATION = 10.0         # seconds
MIN_FRAMES_BEATING = 100    # frames
MAX_BRIGHTNESS = 230        # overexposure
MIN_BRIGHTNESS = 25         # underexposure


def usability_verdict(row):
    """Classify each video for beating/motion analysis usability."""
    issues = []

    if row["fps"] < MIN_FPS_BEATING:
        issues.append(f"FPS too low ({row['fps']:.1f} < {MIN_FPS_BEATING})")
    if row["mean_sharpness"] < MIN_SHARPNESS:
        issues.append(f"Blurry (sharpness {row['mean_sharpness']:.1f})")
    if row["mean_contrast"] < MIN_CONTRAST:
        issues.append(f"Low contrast ({row['mean_contrast']:.1f})")
    if row["duration_s"] < MIN_DURATION:
        issues.append(f"Too short ({row['duration_s']:.1f}s)")
    if row["total_frames"] < MIN_FRAMES_BEATING:
        issues.append(f"Too few frames ({row['total_frames']})")
    if row["mean_brightness"] > MAX_BRIGHTNESS:
        issues.append(f"Overexposed ({row['mean_brightness']:.0f})")
    if row["mean_brightness"] < MIN_BRIGHTNESS:
        issues.append(f"Underexposed ({row['mean_brightness']:.0f})")
    if row["brightness_stability"] > 10:
        issues.append(f"Unstable lighting (σ={row['brightness_stability']:.1f})")

    if row["filename"] == "test frames.avi":
        return "SKIP", "Test file, not experimental data"

    if len(issues) == 0:
        return "USABLE", ""
    elif len(issues) == 1 and "FPS" in issues[0]:
        return "UNUSABLE_FPS", issues[0]
    elif len(issues) == 1:
        return "MARGINAL", issues[0]
    else:
        return "UNUSABLE", "; ".join(issues)


def assign_recording_type(row):
    """
    Clarify recording type based on day and filename metadata.
    Days 1-2: baseline only (no device placed yet).
    Days 6, 8: paired on-device vs off-device control recordings.
    """
    if row["day"] in ("day1", "day2"):
        return "baseline"
    if row["location"] == "control":
        return "control"
    if row["location"] == "device" or row["day"] in ("day3", "day6", "day8"):
        return "on-device"
    return "unknown"


def pick_best_variant(group):
    """Among duplicate videos (same well/day/substrate/modality/recording_type), pick the best."""
    if len(group) == 1:
        return group.index[0], "only version"

    usable = group[group["usability"] == "USABLE"]
    if len(usable) == 1:
        return usable.index[0], "only usable version"
    if len(usable) > 1:
        best_idx = usable["mean_sharpness"].idxmax()
        return best_idx, "sharpest among usable"

    marginal = group[group["usability"] == "MARGINAL"]
    if len(marginal) >= 1:
        best_idx = marginal["mean_sharpness"].idxmax()
        return best_idx, "best marginal"

    best_idx = group["mean_sharpness"].idxmax()
    return best_idx, "sharpest (all unusable)"


def main():
    df = pd.read_csv(INSP)
    print(f"Loaded {len(df)} videos from inspection data\n")

    # --- Assign recording type ---
    df["recording_type"] = df.apply(assign_recording_type, axis=1)

    # --- Usability classification ---
    verdicts = df.apply(usability_verdict, axis=1, result_type="expand")
    df["usability"] = verdicts[0]
    df["usability_reason"] = verdicts[1]

    # --- Resolve duplicates (now includes recording_type) ---
    group_cols = ["day", "well", "substrate", "modality", "recording_type"]
    df["is_best_version"] = False
    df["selection_reason"] = ""

    for _, group in df.groupby(group_cols):
        best_idx, reason = pick_best_variant(group)
        df.loc[best_idx, "is_best_version"] = True
        df.loc[best_idx, "selection_reason"] = reason

    for idx, row in df.iterrows():
        if row["selection_reason"] == "only version":
            df.loc[idx, "is_best_version"] = True

    # --- Build the audit table ---
    audit_cols = [
        "filename", "day", "well", "substrate", "cell_type", "modality",
        "recording_type",
        "fps", "resolution", "total_frames", "duration_s",
        "mean_brightness", "mean_contrast", "mean_sharpness",
        "brightness_stability", "mean_motion", "max_motion",
        "usability", "usability_reason", "is_best_version", "selection_reason",
        "is_variant", "quality_flag",
    ]
    audit = df[audit_cols].copy()

    day_order = {"day1": 1, "day2": 2, "day3": 3, "day6": 6, "day8": 8, "unknown": 99}
    audit["_day_num"] = audit["day"].map(day_order).fillna(99)
    audit = audit.sort_values(["_day_num", "well", "substrate", "modality", "is_variant"])
    audit = audit.drop(columns=["_day_num"])

    audit.to_csv(OUT / "video_quality_audit.csv", index=False)

    # --- Console report ---
    print("=" * 90)
    print("VIDEO QUALITY AUDIT — CARDIOMYOCYTE BEATING ANALYSIS")
    print("=" * 90)

    # Usability summary
    print("\n1. USABILITY CLASSIFICATION")
    print("-" * 50)
    for status in ["USABLE", "MARGINAL", "UNUSABLE_FPS", "UNUSABLE", "SKIP"]:
        count = len(audit[audit["usability"] == status])
        if count > 0:
            print(f"   {status:15s}: {count:3d} videos")
    print(f"   {'TOTAL':15s}: {len(audit):3d} videos")

    # Best versions
    best = audit[audit["is_best_version"]]
    print(f"\n2. BEST VERSION PER CONDITION (after resolving {len(audit) - len(best)} duplicates)")
    print("-" * 105)
    print(f"   {'Filename':<45s} {'Day':>5s} {'Well':>5s} {'Sub':>5s} {'Mod':>5s} "
          f"{'RecType':>10s} {'FPS':>6s} {'Frames':>6s} {'Dur':>6s} {'Sharp':>7s} {'Usable':>14s}")
    print("   " + "-" * 102)

    for _, r in best.iterrows():
        print(f"   {r['filename']:<45s} {r['day']:>5s} {r['well']:>5s} {r['substrate']:>5s} "
              f"{r['modality']:>5s} {r['recording_type']:>10s} {r['fps']:>6.1f} "
              f"{r['total_frames']:>6d} {r['duration_s']:>6.1f} {r['mean_sharpness']:>7.1f} "
              f"{r['usability']:>14s}")

    # Usable for beating analysis
    beating_ready = best[(best["usability"] == "USABLE") & (best["modality"].isin(["BF", "Phase"]))]
    print(f"\n3. VIDEOS READY FOR BEATING ANALYSIS (BF/Phase, usable quality)")
    print("-" * 105)
    print(f"   {len(beating_ready)} videos ready")
    print(f"   {'Filename':<45s} {'Day':>5s} {'Well':>5s} {'Sub':>5s} {'RecType':>10s} "
          f"{'Cell':>6s} {'FPS':>6s} {'Dur':>6s} {'Motion':>7s} {'Sharp':>7s}")
    print("   " + "-" * 102)
    for _, r in beating_ready.iterrows():
        print(f"   {r['filename']:<45s} {r['day']:>5s} {r['well']:>5s} {r['substrate']:>5s} "
              f"{r['recording_type']:>10s} {r['cell_type']:>6s} {r['fps']:>6.1f} "
              f"{r['duration_s']:>6.1f} {r['mean_motion']:>7.2f} {r['mean_sharpness']:>7.1f}")

    # FPS problem breakdown
    low_fps = audit[audit["usability"] == "UNUSABLE_FPS"]
    print(f"\n4. LOW-FPS VIDEOS (unusable for beating, {len(low_fps)} total)")
    print("-" * 90)
    print(f"   {'Filename':<45s} {'Day':>5s} {'Well':>5s} {'Sub':>5s} "
          f"{'FPS':>6s} {'Dur':>6s}  Note")
    print("   " + "-" * 87)
    for _, r in low_fps.iterrows():
        note = f"gold BF — camera was likely in timelapse mode"
        if r["modality"] == "FL":
            note = "fluorescence — long exposure per frame"
        print(f"   {r['filename']:<45s} {r['day']:>5s} {r['well']:>5s} {r['substrate']:>5s} "
              f"{r['fps']:>6.1f} {r['duration_s']:>6.1f}  {note}")

    # Brightness comparison
    print(f"\n5. BRIGHTNESS & CONTRAST COMPARISON (by condition)")
    print("-" * 70)
    bf_best = best[best["modality"].isin(["BF", "Phase"])]
    for sub in ["gold", "non"]:
        subset = bf_best[bf_best["substrate"] == sub]
        if len(subset) == 0:
            continue
        print(f"\n   Substrate: {sub.upper()}")
        print(f"   {'Day':<6s} {'Well':<5s} {'Brightness':>11s} {'Contrast':>10s} {'Sharpness':>10s} {'Motion':>8s} {'FPS':>6s}")
        for _, r in subset.sort_values(["day", "well"]).iterrows():
            bri_flag = " ** BRIGHT" if r["mean_brightness"] > 200 else ""
            print(f"   {r['day']:<6s} {r['well']:<5s} {r['mean_brightness']:>11.1f} "
              f"{r['mean_contrast']:>10.1f} {r['mean_sharpness']:>10.1f} "
              f"{r['mean_motion']:>8.2f} {r['fps']:>6.1f}{bri_flag}")

    # Day coverage matrix with device vs control
    print(f"\n6. COVERAGE MATRIX -- BF/Phase videos per well x day")
    print(f"   (Days 1-2 = baseline only, Days 6+8 = on-device + control pairs)")
    print("-" * 95)

    wells = ["A2", "A3", "B2", "B3", "C2", "C3"]
    days_baseline = ["day1", "day2"]
    days_paired = ["day6", "day8"]
    all_bf = best[best["modality"].isin(["BF", "Phase"])]

    def cell_label(subset):
        if len(subset) == 0:
            return "-"
        r = subset.iloc[0]
        if r["usability"] in ("UNUSABLE_FPS", "UNUSABLE"):
            return f"lowFPS({r['fps']:.1f})"
        return f"{r['fps']:.0f}fps OK"

    # Baseline days (no device/control split)
    print(f"\n   BASELINE (Days 1-2, pre-device)")
    header = f"   {'Well':<6s} {'Sub':<6s}" + "".join(f"{'day1':>14s}{'day2':>14s}")
    print(header)
    print("   " + "-" * 38)
    for w in wells:
        sub = all_bf[all_bf["well"] == w]
        if len(sub) == 0 and w not in ["A2", "B2", "C2"]:
            continue
        substrate = "gold" if w in ["A2", "B2", "C2"] else "non"
        row_str = f"   {w:<6s} {substrate:<6s}"
        for d in days_baseline:
            match = sub[(sub["day"] == d) & (sub["recording_type"] == "baseline")]
            row_str += f"{cell_label(match):>14s}"
        print(row_str)

    # Paired days (device + control)
    print(f"\n   ON-DEVICE vs CONTROL (Days 6 & 8)")
    header = (f"   {'Well':<6s} {'Sub':<6s}"
              + f"{'day6-DEV':>14s}{'day6-CTL':>14s}"
              + f"{'day8-DEV':>14s}{'day8-CTL':>14s}")
    print(header)
    print("   " + "-" * 66)
    for w in wells:
        sub = all_bf[all_bf["well"] == w]
        substrate = "gold" if w in ["A2", "B2", "C2"] else "non"
        row_str = f"   {w:<6s} {substrate:<6s}"
        for d in days_paired:
            dev = sub[(sub["day"] == d) & (sub["recording_type"] == "on-device")]
            ctl = sub[(sub["day"] == d) & (sub["recording_type"] == "control")]
            row_str += f"{cell_label(dev):>14s}{cell_label(ctl):>14s}"
        print(row_str)

    print(f"\n   Summary of usable pairs (both device + control at >=8 fps):")
    for d in days_paired:
        for w in wells:
            sub = all_bf[all_bf["well"] == w]
            dev = sub[(sub["day"] == d) & (sub["recording_type"] == "on-device") & (sub["usability"] == "USABLE")]
            ctl = sub[(sub["day"] == d) & (sub["recording_type"] == "control") & (sub["usability"] == "USABLE")]
            if len(dev) > 0 and len(ctl) > 0:
                substrate = "gold" if w in ["A2", "B2", "C2"] else "non"
                print(f"     {d} {w} ({substrate}): PAIRED -- "
                      f"device={dev.iloc[0]['filename']}, control={ctl.iloc[0]['filename']}")
            elif len(dev) > 0 and len(ctl) == 0:
                substrate = "gold" if w in ["A2", "B2", "C2"] else "non"
                print(f"     {d} {w} ({substrate}): device ONLY (no usable control)")
            elif len(dev) == 0 and len(ctl) > 0:
                substrate = "gold" if w in ["A2", "B2", "C2"] else "non"
                print(f"     {d} {w} ({substrate}): control ONLY (no usable device)")

    # Recommendations
    print(f"\n{'=' * 90}")
    print("7. RECOMMENDATIONS FOR NEXT RECORDING SESSION")
    print("=" * 90)

    gold_fps = audit[(audit["substrate"] == "gold") & (audit["modality"] == "BF")]["fps"]
    non_fps = audit[(audit["substrate"] == "non") & (audit["modality"] == "BF")]["fps"]

    recs = [
        ("FPS SETTINGS",
         f"Gold BF videos averaged {gold_fps.mean():.1f} fps (range {gold_fps.min():.1f}–{gold_fps.max():.1f}). "
         f"Non-gold BF averaged {non_fps.mean():.1f} fps. "
         f"The gold recordings were likely in timelapse/slow-capture mode. "
         f"Set ALL recordings to >= 12 fps (ideally 15-30 fps for beat detection). "
         f"Check camera software is in 'video' not 'timelapse' mode before each recording."),

        ("RECORDING DURATION",
         f"Target 25-40 seconds per video (300-500 frames at 12 fps). "
         f"Some gold videos ran 2-3 minutes at <2 fps — that's timelapse, not video."),

        ("BRIGHTNESS",
         f"Several videos >190 brightness (near saturation): A3-day6 at 192.8, B3-day6 at 216.6, "
         f"B2-day8 at 198.6, A3-day8 at 183.5. Reduce lamp intensity or exposure. "
         f"Target: 80-160 mean brightness for good dynamic range."),

        ("FOCUS / SHARPNESS",
         f"Day1 originals (A3-day1, B3-day1) had sharpness ~3 (very blurry). "
         f"The '-fixed' retakes improved to ~200+. Always check focus before recording. "
         f"Minimum acceptable sharpness: ~15."),

        ("FLUORESCENCE VIDEOS",
         f"FL videos are low-resolution (348x260 vs 1392x1040 for BF) and some have low FPS. "
         f"These are fine for intensity tracking but not ideal for motion analysis. "
         f"For next experiment: consider recording FL at full resolution if motion matters."),

        ("NAMING CONVENTION",
         f"Standardize filenames: WELL-DAYn-SUBSTRATE-CELLTYPE-MODALITY.avi "
         f"(e.g., A2-day6-gold-cellline-BF.avi). Current names are inconsistent "
         f"(typos like 'comtrol', mixed case, varying field order)."),

        ("CONTROLS",
         f"Device vs control recordings exist for day6 and day8. Ensure control "
         f"recordings use identical camera settings (same FPS, exposure, resolution) "
         f"as device recordings for fair comparison."),
    ]

    for title, text in recs:
        print(f"\n   {title}")
        # Word wrap at ~80 chars
        words = text.split()
        line = "     "
        for w in words:
            if len(line) + len(w) + 1 > 85:
                print(line)
                line = "     " + w
            else:
                line += " " + w if line.strip() else "     " + w
        if line.strip():
            print(line)

    print(f"\n{'=' * 90}")
    print(f"Audit table saved to: {OUT / 'video_quality_audit.csv'}")
    print(f"{'=' * 90}")

    return audit


if __name__ == "__main__":
    audit = main()
