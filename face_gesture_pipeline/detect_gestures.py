"""
OpenFace Gesture Detection Pipeline
====================================
Detects facial gestures from OpenFace CSV output.

Supports:
- Static expressions (single image): smile, surprise, frown, etc.
- Temporal gestures (video): nodding, head shaking, blinking rate, etc.

Usage:
    python detect_gestures.py <csv_file> [--output <output_file>] [--verbose]

Examples:
    python detect_gestures.py ../output/sample1.csv
    python detect_gestures.py ../output/2015-10-15-15-14.csv --output results.json --verbose
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ─── Action Unit Thresholds ───────────────────────────────────────────────────
# AU intensity (_r) ranges from 0 to 5. These thresholds define "active".
AU_THRESHOLDS = {
    "smile": {"AU06_r": 1.0, "AU12_r": 1.5},
    "laugh": {"AU06_r": 2.0, "AU12_r": 3.0, "AU25_r": 2.0},
    "surprise": {"AU01_r": 1.5, "AU02_r": 1.0, "AU05_r": 1.0},
    "frown": {"AU04_r": 1.5},
    "anger": {"AU04_r": 1.5, "AU07_r": 1.5, "AU23_r": 1.0},
    "disgust": {"AU09_r": 1.5, "AU10_r": 1.0},
    "fear": {"AU01_r": 1.0, "AU04_r": 1.0, "AU20_r": 1.0},
    "contempt": {"AU14_r": 1.5},
    "blink": {"AU45_r": 1.0},
    "mouth_open": {"AU25_r": 1.5, "AU26_r": 1.5},
    "brow_raise": {"AU01_r": 1.0, "AU02_r": 1.0},
    "lip_tighten": {"AU23_r": 1.5},
}

# ─── Head Pose Thresholds (radians) ──────────────────────────────────────────
HEAD_POSE = {
    "nod_threshold": 0.15,        # pitch change for nod detection
    "shake_threshold": 0.15,      # yaw change for shake detection
    "tilt_threshold": 0.20,       # roll threshold
    "nod_min_cycles": 2,          # minimum oscillations for a nod
    "shake_min_cycles": 2,
}

# ─── Gaze Thresholds (radians) ───────────────────────────────────────────────
GAZE = {
    "looking_away_threshold": 0.35,
    "looking_down_threshold": 0.30,
}


@dataclass
class FrameGestures:
    """Gestures detected in a single frame."""
    frame: int = 0
    confidence: float = 0.0
    expressions: list = field(default_factory=list)
    head_pose: dict = field(default_factory=dict)
    gaze: dict = field(default_factory=dict)
    au_values: dict = field(default_factory=dict)


@dataclass
class TemporalGesture:
    """A gesture detected over multiple frames."""
    gesture: str = ""
    start_frame: int = 0
    end_frame: int = 0
    confidence: float = 0.0
    details: str = ""


@dataclass
class GestureReport:
    """Full analysis report."""
    input_file: str = ""
    total_frames: int = 0
    is_video: bool = False
    frame_gestures: list = field(default_factory=list)
    temporal_gestures: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def parse_csv(csv_path: str) -> list[dict]:
    """Parse OpenFace CSV into list of frame dicts."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys (OpenFace adds spaces)
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            rows.append(cleaned)
    return rows


def get_float(row: dict, key: str, default: float = 0.0) -> float:
    """Safely get a float value from a row."""
    try:
        return float(row.get(key, default))
    except (ValueError, TypeError):
        return default


def detect_expressions(row: dict) -> tuple[list[str], dict]:
    """Detect facial expressions from Action Units in a single frame."""
    expressions = []
    au_values = {}

    # Collect all AU values
    for key, val in row.items():
        if key.startswith("AU") and key.endswith("_r"):
            au_values[key] = get_float(row, key)

    # Check each expression against thresholds
    for expression, required_aus in AU_THRESHOLDS.items():
        match = True
        for au, threshold in required_aus.items():
            if get_float(row, au) < threshold:
                match = False
                break
        if match:
            expressions.append(expression)

    return expressions, au_values


def analyze_head_pose(row: dict) -> dict:
    """Analyze head pose for a single frame."""
    rx = get_float(row, "pose_Rx")  # pitch (nod)
    ry = get_float(row, "pose_Ry")  # yaw (shake)
    rz = get_float(row, "pose_Rz")  # roll (tilt)

    pose = {
        "pitch_rad": round(rx, 3),
        "yaw_rad": round(ry, 3),
        "roll_rad": round(rz, 3),
    }

    # Classify static head orientation
    if abs(rx) > 0.15:
        pose["vertical"] = "looking_down" if rx > 0 else "looking_up"
    else:
        pose["vertical"] = "neutral"

    if abs(ry) > 0.15:
        pose["horizontal"] = "turned_right" if ry > 0 else "turned_left"
    else:
        pose["horizontal"] = "facing_forward"

    if abs(rz) > 0.15:
        pose["tilt"] = "tilted_right" if rz > 0 else "tilted_left"
    else:
        pose["tilt"] = "upright"

    return pose


def analyze_gaze(row: dict) -> dict:
    """Analyze gaze direction for a single frame."""
    gx = get_float(row, "gaze_angle_x")
    gy = get_float(row, "gaze_angle_y")

    gaze = {
        "angle_x_rad": round(gx, 3),
        "angle_y_rad": round(gy, 3),
    }

    # Horizontal gaze
    if abs(gx) > GAZE["looking_away_threshold"]:
        gaze["horizontal"] = "looking_right" if gx > 0 else "looking_left"
    else:
        gaze["horizontal"] = "center"

    # Vertical gaze
    if abs(gy) > GAZE["looking_down_threshold"]:
        gaze["vertical"] = "looking_down" if gy > 0 else "looking_up"
    else:
        gaze["vertical"] = "center"

    # Overall engagement
    total_deviation = abs(gx) + abs(gy)
    if total_deviation < 0.3:
        gaze["engagement"] = "engaged"
    elif total_deviation < 0.6:
        gaze["engagement"] = "partially_engaged"
    else:
        gaze["engagement"] = "disengaged"

    return gaze


def detect_temporal_gestures(frames: list[FrameGestures], rows: list[dict]) -> list[TemporalGesture]:
    """Detect gestures that span multiple frames (video only)."""
    if len(rows) < 5:
        return []

    temporal = []

    # ── Nodding Detection (pitch oscillation) ────────────────────────────
    pitches = [get_float(r, "pose_Rx") for r in rows]
    nods = detect_oscillation(
        pitches,
        threshold=HEAD_POSE["nod_threshold"],
        min_cycles=HEAD_POSE["nod_min_cycles"],
    )
    for start, end, cycles in nods:
        temporal.append(TemporalGesture(
            gesture="nodding",
            start_frame=start,
            end_frame=end,
            confidence=min(1.0, cycles / 3.0),
            details=f"{cycles} nod cycles detected",
        ))

    # ── Head Shaking Detection (yaw oscillation) ─────────────────────────
    yaws = [get_float(r, "pose_Ry") for r in rows]
    shakes = detect_oscillation(
        yaws,
        threshold=HEAD_POSE["shake_threshold"],
        min_cycles=HEAD_POSE["shake_min_cycles"],
    )
    for start, end, cycles in shakes:
        temporal.append(TemporalGesture(
            gesture="head_shaking",
            start_frame=start,
            end_frame=end,
            confidence=min(1.0, cycles / 3.0),
            details=f"{cycles} shake cycles detected",
        ))

    # ── Blink Detection ──────────────────────────────────────────────────
    blink_regions = detect_blink_events(rows)
    for start, end in blink_regions:
        temporal.append(TemporalGesture(
            gesture="blink",
            start_frame=start,
            end_frame=end,
            confidence=0.9,
            details="eye closure detected",
        ))

    # ── Smile Duration ───────────────────────────────────────────────────
    smile_regions = detect_sustained_expression(frames, "smile", min_frames=5)
    for start, end in smile_regions:
        temporal.append(TemporalGesture(
            gesture="sustained_smile",
            start_frame=start,
            end_frame=end,
            confidence=0.85,
            details=f"smile held for {end - start + 1} frames",
        ))

    # ── Surprise Reaction ────────────────────────────────────────────────
    surprise_regions = detect_sustained_expression(frames, "surprise", min_frames=3)
    for start, end in surprise_regions:
        temporal.append(TemporalGesture(
            gesture="surprise_reaction",
            start_frame=start,
            end_frame=end,
            confidence=0.80,
            details=f"surprise for {end - start + 1} frames",
        ))

    # ── Gaze Aversion ────────────────────────────────────────────────────
    gaze_away_regions = detect_gaze_aversion(rows, min_frames=10)
    for start, end in gaze_away_regions:
        temporal.append(TemporalGesture(
            gesture="gaze_aversion",
            start_frame=start,
            end_frame=end,
            confidence=0.75,
            details=f"looked away for {end - start + 1} frames",
        ))

    temporal.sort(key=lambda g: g.start_frame)
    return temporal


def detect_oscillation(
    values: list[float],
    threshold: float,
    min_cycles: int,
    window: int = 30,
) -> list[tuple[int, int, int]]:
    """Detect oscillation patterns (for nod/shake detection)."""
    results = []
    if len(values) < window:
        return results

    for start in range(0, len(values) - window, window // 2):
        end = min(start + window, len(values))
        segment = values[start:end]

        # Count direction changes that exceed threshold
        mean_val = sum(segment) / len(segment)
        crossings = 0
        above = segment[0] > mean_val

        for v in segment[1:]:
            now_above = v > mean_val
            if now_above != above and abs(v - mean_val) > threshold * 0.5:
                crossings += 1
                above = now_above

        cycles = crossings // 2
        if cycles >= min_cycles:
            results.append((start, end - 1, cycles))

    # Merge overlapping detections
    return merge_regions(results)


def detect_blink_events(rows: list[dict]) -> list[tuple[int, int]]:
    """Detect individual blink events from AU45."""
    regions = []
    in_blink = False
    start = 0

    for i, row in enumerate(rows):
        au45 = get_float(row, "AU45_r")
        if au45 >= 1.0 and not in_blink:
            in_blink = True
            start = i
        elif au45 < 0.5 and in_blink:
            in_blink = False
            # Blinks are typically 3-15 frames (~100-500ms at 30fps)
            duration = i - start
            if 1 <= duration <= 20:
                regions.append((start, i - 1))

    return regions


def detect_sustained_expression(
    frames: list[FrameGestures],
    expression: str,
    min_frames: int = 5,
) -> list[tuple[int, int]]:
    """Find regions where an expression is sustained."""
    regions = []
    in_region = False
    start = 0

    for i, frame in enumerate(frames):
        if expression in frame.expressions:
            if not in_region:
                in_region = True
                start = i
        else:
            if in_region:
                in_region = False
                if i - start >= min_frames:
                    regions.append((start, i - 1))

    if in_region and len(frames) - start >= min_frames:
        regions.append((start, len(frames) - 1))

    return regions


def detect_gaze_aversion(rows: list[dict], min_frames: int = 10) -> list[tuple[int, int]]:
    """Detect sustained gaze aversion."""
    regions = []
    in_aversion = False
    start = 0

    for i, row in enumerate(rows):
        gx = abs(get_float(row, "gaze_angle_x"))
        gy = abs(get_float(row, "gaze_angle_y"))
        total = gx + gy

        if total > 0.6:
            if not in_aversion:
                in_aversion = True
                start = i
        else:
            if in_aversion:
                in_aversion = False
                if i - start >= min_frames:
                    regions.append((start, i - 1))

    if in_aversion and len(rows) - start >= min_frames:
        regions.append((start, len(rows) - 1))

    return regions


def merge_regions(regions: list[tuple]) -> list[tuple]:
    """Merge overlapping (start, end, ...) regions."""
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda r: r[0])
    merged = [sorted_regions[0]]

    for region in sorted_regions[1:]:
        if region[0] <= merged[-1][1]:
            # Merge: keep max end, sum cycles
            prev = merged[-1]
            extra = tuple(a + b for a, b in zip(prev[2:], region[2:])) if len(prev) > 2 else ()
            merged[-1] = (prev[0], max(prev[1], region[1])) + extra
        else:
            merged.append(region)

    return merged


def generate_summary(
    frames: list[FrameGestures],
    temporal: list[TemporalGesture],
    is_video: bool,
) -> dict:
    """Generate a human-readable summary."""
    summary = {
        "total_frames": len(frames),
        "avg_confidence": round(
            sum(f.confidence for f in frames) / max(len(frames), 1), 3
        ),
    }

    # Expression frequency
    expr_count = {}
    for frame in frames:
        for expr in frame.expressions:
            expr_count[expr] = expr_count.get(expr, 0) + 1

    summary["expression_frequency"] = {
        expr: {
            "frame_count": count,
            "percentage": round(count / max(len(frames), 1) * 100, 1),
        }
        for expr, count in sorted(expr_count.items(), key=lambda x: -x[1])
    }

    # Dominant expression
    if expr_count:
        dominant = max(expr_count, key=expr_count.get)
        summary["dominant_expression"] = dominant
    else:
        summary["dominant_expression"] = "neutral"

    if is_video:
        # Blink rate
        blink_count = sum(1 for t in temporal if t.gesture == "blink")
        summary["blink_count"] = blink_count

        # Gesture counts
        gesture_counts = {}
        for t in temporal:
            gesture_counts[t.gesture] = gesture_counts.get(t.gesture, 0) + 1
        summary["temporal_gesture_counts"] = gesture_counts

    return summary


def run_pipeline(csv_path: str, verbose: bool = False) -> GestureReport:
    """Main pipeline: parse CSV → detect gestures → generate report."""
    rows = parse_csv(csv_path)

    if not rows:
        print(f"Error: No data found in {csv_path}", file=sys.stderr)
        sys.exit(1)

    is_video = len(rows) > 1
    report = GestureReport(
        input_file=csv_path,
        total_frames=len(rows),
        is_video=is_video,
    )

    # ── Per-frame analysis ────────────────────────────────────────────────
    frame_gestures = []
    for i, row in enumerate(rows):
        confidence = get_float(row, "confidence")
        expressions, au_values = detect_expressions(row)
        head_pose = analyze_head_pose(row)
        gaze = analyze_gaze(row)

        fg = FrameGestures(
            frame=i,
            confidence=confidence,
            expressions=expressions,
            head_pose=head_pose,
            gaze=gaze,
            au_values=au_values,
        )
        frame_gestures.append(fg)

        if verbose:
            expr_str = ", ".join(expressions) if expressions else "neutral"
            print(f"  Frame {i:4d} | conf={confidence:.2f} | {expr_str} | "
                  f"gaze={gaze.get('engagement', '?')}")

    report.frame_gestures = frame_gestures

    # ── Temporal analysis (video only) ────────────────────────────────────
    if is_video:
        temporal = detect_temporal_gestures(frame_gestures, rows)
        report.temporal_gestures = temporal

        if verbose:
            print(f"\n  Temporal gestures found: {len(temporal)}")
            for t in temporal:
                print(f"    {t.gesture}: frames {t.start_frame}-{t.end_frame} "
                      f"({t.details})")

    # ── Summary ──────────────────────────────────────────────────────────
    report.summary = generate_summary(
        frame_gestures,
        report.temporal_gestures,
        is_video,
    )

    return report


def print_report(report: GestureReport):
    """Pretty-print the gesture report to console."""
    print("\n" + "=" * 60)
    print("  OPENFACE GESTURE DETECTION REPORT")
    print("=" * 60)
    print(f"  Input:  {report.input_file}")
    print(f"  Frames: {report.total_frames}")
    print(f"  Type:   {'Video' if report.is_video else 'Image'}")
    print(f"  Avg Confidence: {report.summary.get('avg_confidence', 0)}")
    print()

    # ── Expressions ──────────────────────────────────────────────────────
    print("  EXPRESSIONS DETECTED")
    print("  " + "-" * 40)
    dominant = report.summary.get("dominant_expression", "neutral")
    print(f"  Dominant: {dominant}")
    print()

    freq = report.summary.get("expression_frequency", {})
    if freq:
        for expr, info in freq.items():
            bar = "█" * int(info["percentage"] / 5)
            print(f"    {expr:<20} {info['percentage']:5.1f}% {bar}")
    else:
        print("    No expressions detected (neutral face)")
    print()

    # ── Head Pose (first/only frame) ─────────────────────────────────────
    if report.frame_gestures:
        f0 = report.frame_gestures[0]
        pose = f0.head_pose
        print("  HEAD POSE (frame 0)")
        print("  " + "-" * 40)
        print(f"    Pitch: {pose.get('pitch_rad', 0):+.3f} rad  ({pose.get('vertical', '')})")
        print(f"    Yaw:   {pose.get('yaw_rad', 0):+.3f} rad  ({pose.get('horizontal', '')})")
        print(f"    Roll:  {pose.get('roll_rad', 0):+.3f} rad  ({pose.get('tilt', '')})")
        print()

    # ── Gaze ─────────────────────────────────────────────────────────────
    if report.frame_gestures:
        f0 = report.frame_gestures[0]
        gaze = f0.gaze
        print("  GAZE (frame 0)")
        print("  " + "-" * 40)
        print(f"    Horizontal: {gaze.get('horizontal', '')}")
        print(f"    Vertical:   {gaze.get('vertical', '')}")
        print(f"    Engagement: {gaze.get('engagement', '')}")
        print()

    # ── Temporal Gestures (video) ─────────────────────────────────────────
    if report.is_video and report.temporal_gestures:
        print("  TEMPORAL GESTURES")
        print("  " + "-" * 40)
        for t in report.temporal_gestures:
            print(f"    {t.gesture:<20} frames {t.start_frame:4d}-{t.end_frame:4d}  "
                  f"conf={t.confidence:.2f}  {t.details}")
        print()

        counts = report.summary.get("temporal_gesture_counts", {})
        if "blink" in counts:
            print(f"  Blink count: {counts['blink']}")
        print()

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Detect facial gestures from OpenFace CSV output"
    )
    parser.add_argument("csv_file", help="Path to OpenFace CSV file")
    parser.add_argument(
        "--output", "-o",
        help="Save report as JSON to this path",
        default=None,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-frame details",
    )
    args = parser.parse_args()

    csv_path = args.csv_file
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nAnalyzing: {csv_path}")
    report = run_pipeline(csv_path, verbose=args.verbose)
    print_report(report)

    # Save JSON if requested
    if args.output:
        output_data = asdict(report)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Report saved to: {args.output}\n")


if __name__ == "__main__":
    main()
