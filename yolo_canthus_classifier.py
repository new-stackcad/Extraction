"""
Thermal Health Monitor — YOLOv8 Canthus Edition (multi-scale)
=============================================================
Drop-in replacement for tfake_classifier.py that uses the
soibkhon/thermal-canthus-detector model instead of T-FAKE.

Distance improvement over the plain version
--------------------------------------------
Detection is attempted at three progressively larger upscale factors
(1×, 2×, 3×) so that at longer distances — where the face occupies fewer
sensor pixels — the YOLO model still receives enough resolution to find
the inner canthus keypoints.  A lower confidence threshold (0.10 vs the
default 0.25) is also used so weak-but-correct detections at range are
not suppressed.

Setup
-----
# 1. Clone the detector repo alongside this script
#    git clone https://github.com/soibkhon/thermal-canthus-detector
#
# 2. Install dependencies
#    pip install "numpy<2" ultralytics opencv-python
#
# 3. Run
#    PYTHONPATH=build/python python3 yolo_canthus_classifier.py
#
# CLI flags
#    --weights  /path/to/thermal_canthus_yolo.pt
#    --port     /dev/ttyACM0
#    --conf     0.10          (YOLO keypoint confidence threshold)
#    --scales   1 2 3         (upscale factors tried in order)
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, 'build/python')

import guideusb2 as g
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Locate detect_canthus.py
# ---------------------------------------------------------------------------
_REPO_CANDIDATES = [
    Path(__file__).parent / "thermal-canthus-detector",
    Path(__file__).parent,
    Path.home() / "thermal-canthus-detector",
]
for _p in _REPO_CANDIDATES:
    if (_p / "detect_canthus.py").exists():
        sys.path.insert(0, str(_p))
        break

from detect_canthus import load_model, _preprocess  # noqa: E402

# ---------------------------------------------------------------------------
# Thresholds (SPIRIT 2025, inner canthus)
# ---------------------------------------------------------------------------
THRESH_NO_FACE    = 28.0
THRESH_NORMAL     = 37.1
THRESH_LOW_FEVER  = 37.5
THRESH_HIGH_FEVER = 39.0

MAX_NO_DETECT = 3   # frames to hold last valid reading when detection drops


def classify(temp_c: float) -> tuple:
    if temp_c < THRESH_NO_FACE:
        return "NO FACE",    (128, 128, 128)
    elif temp_c < THRESH_NORMAL:
        return "NORMAL",     (0, 255, 0)
    elif temp_c < THRESH_LOW_FEVER:
        return "LOW FEVER",  (0, 165, 255)
    elif temp_c < THRESH_HIGH_FEVER:
        return "HIGH FEVER", (0, 0, 255)
    else:
        return "EMERGENCY",  (0, 0, 180)


def get_region_temp(temps: np.ndarray, px: int, py: int, radius: int = 3) -> float:
    h, w = temps.shape
    x1, x2 = max(0, px - radius), min(w, px + radius)
    y1, y2 = max(0, py - radius), min(h, py + radius)
    region = temps[y1:y2, x1:x2]
    return float(np.max(region)) if region.size > 0 else 0.0


# ---------------------------------------------------------------------------
# Multi-scale canthus detector
# ---------------------------------------------------------------------------
def detect_multiscale(
    model,
    bgr: np.ndarray,
    scales: list,
    conf: float,
) -> tuple:
    """
    Try YOLO detection at each upscale factor in `scales` (e.g. [1, 2, 3]).
    Returns (left_pt, right_pt, scale_used) where each point is (x, y) in
    the *original* image's pixel space, or None if not found.
    Stops at the first scale that gives at least one keypoint.
    """
    orig_h, orig_w = bgr.shape[:2]

    for scale in scales:
        if scale == 1:
            inp = bgr
        else:
            inp = cv2.resize(bgr, (orig_w * scale, orig_h * scale),
                             interpolation=cv2.INTER_LINEAR)

        preprocessed = _preprocess(inp)
        results = model(preprocessed, verbose=False, conf=conf)[0]

        if results.keypoints is None or len(results.keypoints) == 0:
            continue

        best = int(results.boxes.conf.cpu().numpy().argmax())
        kpts = results.keypoints.xy[best].cpu().numpy()  # (2, 2)

        if kpts.shape[0] < 2:
            continue

        lx, ly = int(kpts[0, 0]), int(kpts[0, 1])
        rx, ry = int(kpts[1, 0]), int(kpts[1, 1])

        left_valid  = (lx, ly) if (lx > 0 or ly > 0) else None
        right_valid = (rx, ry) if (rx > 0 or ry > 0) else None

        if left_valid is None and right_valid is None:
            continue

        # Map coordinates back to original pixel space
        if scale > 1:
            left_valid  = (left_valid[0]  // scale, left_valid[1]  // scale) \
                          if left_valid  else None
            right_valid = (right_valid[0] // scale, right_valid[1] // scale) \
                          if right_valid else None

        return left_valid, right_valid, scale

    return None, None, None


# ---------------------------------------------------------------------------
# Camera state
# ---------------------------------------------------------------------------
linear_cal  = g.CameraLinearCal()
last_status = [None]
last_frame  = [None]


def on_frame(frame: g.Frame) -> None:
    last_frame[0] = frame
    if last_status[0] is not None:
        try:
            linear_cal.fit(frame.y16, last_status[0])
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Thermal Health Monitor — YOLO multi-scale canthus")
    ap.add_argument("--weights", default=None,
                    help="Path to thermal_canthus_yolo.pt (auto-detected if omitted)")
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="Serial port for camera (default: /dev/ttyACM0)")
    ap.add_argument("--conf", type=float, default=0.10,
                    help="YOLO confidence threshold (default: 0.10, lower = more sensitive)")
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 2, 3],
                    help="Upscale factors to try in order (default: 1 2 3)")
    args = ap.parse_args()

    # --- Weights ---
    if args.weights:
        weights_path = Path(args.weights)
    else:
        candidates = [
            Path(__file__).parent / "thermal-canthus-detector" / "thermal_canthus_yolo.pt",
            Path(__file__).parent / "thermal_canthus_yolo.pt",
            Path.home() / "thermal-canthus-detector" / "thermal_canthus_yolo.pt",
        ]
        weights_path = next((p for p in candidates if p.exists()), candidates[0])

    print(f"Loading YOLO canthus model from: {weights_path}")
    model = load_model(weights_path)
    print(f"YOLO canthus model ready  |  conf={args.conf}  scales={args.scales}")

    # --- Camera ---
    cam = g.Camera(g.DeviceInfo(serial_port=args.port))
    cam.configure_thermography(g.ThermographyConfig(emissivity=98))
    cam.start(on_frame)
    print("Warming up... 5 seconds")
    time.sleep(5)

    # --- Calibration ---
    print("Calibrating...")
    for _ in range(20):
        try:
            s = cam.serial().query_temp_status(retries=2, wait_seconds=1.5)
            if s:
                last_status[0] = s
                if last_frame[0] is not None:
                    linear_cal.fit(last_frame[0].y16, s)
                print(f"Calibrated — min:{s.min_c:.1f}  max:{s.max_c:.1f}")
                break
        except Exception:
            pass
        time.sleep(1)

    if not linear_cal.ready():
        print("Calibration failed — exiting.")
        cam.stop()
        sys.exit(1)

    last_valid      = [None]
    no_detect_count = [0]

    print("Live view started. Press Q or Escape to quit.")
    cv2.namedWindow("Thermal Health Monitor", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Refresh calibration
            try:
                s = cam.serial().query_temp_status(retries=1, wait_seconds=1.0)
                if s and last_frame[0] is not None:
                    last_status[0] = s
                    linear_cal.fit(last_frame[0].y16, s)
            except Exception:
                pass

            if last_frame[0] is None or not linear_cal.ready():
                print("Waiting for calibration...")
                time.sleep(1)
                continue

            # --- Average 10 frames ---
            all_temps    = []
            latest_frame = None
            for _ in range(10):
                if last_frame[0] is not None and linear_cal.ready():
                    all_temps.append(linear_cal.decode(last_frame[0].y16))
                    latest_frame = last_frame[0]
                time.sleep(0.1)

            if not all_temps or latest_frame is None:
                continue

            temps_avg = np.mean(all_temps, axis=0)
            h, w      = temps_avg.shape

            # --- Build BGR image (iron-red palette, same as training data) ---
            palette = g.Palette.IronRed
            opts    = g.PaletteOptions(auto_range=True)
            rgb     = g.apply_palette(latest_frame.y16, palette, opts)
            bgr     = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # --- Multi-scale detection ---
            try:
                left_pt, right_pt, scale_used = detect_multiscale(
                    model, bgr, scales=args.scales, conf=args.conf
                )
                face_detected = (left_pt is not None) or (right_pt is not None)
            except Exception as exc:
                print(f"Detection error: {exc}")
                face_detected = False
                left_pt = right_pt = scale_used = None

            # --- Display (2× for readability) ---
            display = cv2.resize(bgr, (w * 2, h * 2))

            # --- Temporal smoothing ---
            if face_detected:
                lc_x, lc_y = left_pt  if left_pt  is not None else (0, 0)
                rc_x, rc_y = right_pt if right_pt is not None else (0, 0)

                lc_temp = get_region_temp(temps_avg, lc_x, lc_y, radius=3) \
                          if left_pt  is not None else 0.0
                rc_temp = get_region_temp(temps_avg, rc_x, rc_y, radius=3) \
                          if right_pt is not None else 0.0

                valid_temps  = [t for t in (lc_temp, rc_temp) if t > THRESH_NO_FACE]
                canthus_temp = max(valid_temps) if valid_temps else max(lc_temp, rc_temp)
                combined     = canthus_temp
                label, color = classify(combined)

                last_valid[0] = dict(
                    lc_temp=lc_temp, rc_temp=rc_temp,
                    combined=combined, label=label, color=color,
                    left_pt=left_pt, right_pt=right_pt,
                    scale_used=scale_used,
                )
                no_detect_count[0] = 0

            elif last_valid[0] is not None and no_detect_count[0] < MAX_NO_DETECT:
                no_detect_count[0] += 1
                r             = last_valid[0]
                lc_temp       = r["lc_temp"]
                rc_temp       = r["rc_temp"]
                combined      = r["combined"]
                label         = r["label"]
                color         = r["color"]
                left_pt       = r["left_pt"]
                right_pt      = r["right_pt"]
                scale_used    = r["scale_used"]
                face_detected = True
                print(f"Holding last reading ({no_detect_count[0]}/{MAX_NO_DETECT})")

            else:
                last_valid[0]      = None
                no_detect_count[0] = 0

            # --- Draw ---
            if face_detected:
                if left_pt is not None:
                    cv2.circle(display, (left_pt[0]*2, left_pt[1]*2), 8, (0, 255, 255), 2)
                    cv2.putText(display, f"{lc_temp:.1f}C",
                                (left_pt[0]*2 + 9, left_pt[1]*2 - 9),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                if right_pt is not None:
                    cv2.circle(display, (right_pt[0]*2, right_pt[1]*2), 8, (0, 255, 255), 2)
                    cv2.putText(display, f"{rc_temp:.1f}C",
                                (right_pt[0]*2 + 9, right_pt[1]*2 - 9),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                # Scale indicator (useful for tuning)
                scale_label = f"scale:{scale_used}x" if scale_used else ""
                cv2.putText(display, label,
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(display,
                            f"LC:{lc_temp:.1f}  RC:{rc_temp:.1f}  "
                            f"Canthus:{combined:.1f}C  {scale_label}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

                print(f"L:{lc_temp:.1f}C  R:{rc_temp:.1f}C  "
                      f"Canthus:{combined:.1f}C  → {label}  [{scale_label}]")

            else:
                cv2.putText(display, "NO FACE DETECTED",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
                print("No face detected")

            if last_status[0] is not None:
                cv2.putText(display,
                            f"min:{last_status[0].min_c:.1f}  max:{last_status[0].max_c:.1f}",
                            (20, h*2 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1)

            cv2.imshow("Thermal Health Monitor", display)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("Quitting...")
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
