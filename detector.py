#!/usr/bin/env python3
"""
Mac + Raspberry Pi webcam marker detector (ORB + optional ArUco)
-----------------------------------------------------------------

Purpose
  • Detect either (A) a custom reference image (e.g., a tattoo/photo/logo) using ORB feature matching
    OR (B) an ArUco marker if OpenCV-contrib is available.
  • When a target is confidently detected, overlay and print: "manual control on".
  • Defaults to using your built‑in webcam on macOS, but runs on Raspberry Pi 4 (USB webcam or Pi Camera via v4l2) with the same code.

Tested Env Targets
  • Python ≥ 3.8
  • OpenCV 4.8.1.78 (works with opencv-python). ArUco requires opencv-contrib-python of the SAME version.

Usage examples
  # 1) Custom image matching (no ArUco required):
  python3 detector.py --ref ./my_reference.jpg

  # 2) ArUco detection (requires opencv-contrib-python==4.8.1.78):
  python3 detector.py --aruco 4X4_50

  # 3) Webcam selection and performance options:
  python3 detector.py --ref ./tattoo.png --camera 0 --width 640 --height 480 --display

Notes
  • Tattoos deform with skin movement; feature matching is most reliable on (mostly) planar, high‑contrast images.
  • For maximum reliability, consider printing a small ArUco/AprilTag sticker next to the tattoo and detect that.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# ----------------------------
# Utility: optional ArUco
# ----------------------------
ARUCO_AVAILABLE = False
ARUCO_DICTIONARIES = {}
try:
    aruco = cv2.aruco  # type: ignore[attr-defined]
    # Map simple names to OpenCV constants (only if contrib available)
    ARUCO_DICTIONARIES = {
        "4X4_50": aruco.DICT_4X4_50,
        "4X4_100": aruco.DICT_4X4_100,
        "5X5_50": aruco.DICT_5X5_50,
        "5X5_100": aruco.DICT_5X5_100,
        "6X6_50": aruco.DICT_6X6_50,
        "6X6_100": aruco.DICT_6X6_100,
        "7X7_50": aruco.DICT_7X7_50,
        "7X7_100": aruco.DICT_7X7_100,
        "APRILTAG_16h5": aruco.DICT_APRILTAG_16h5 if hasattr(aruco, 'DICT_APRILTAG_16h5') else None,
    }
    # Remove None entries (in case the build lacks AprilTag)
    ARUCO_DICTIONARIES = {k: v for k, v in ARUCO_DICTIONARIES.items() if v is not None}
    ARUCO_AVAILABLE = True
except Exception:
    ARUCO_AVAILABLE = False


@dataclass
class Debounce:
    required_hits: int = 5  # how many consecutive frames to confirm ON
    grace_frames: int = 15  # frames to keep ON after last seen
    hits: int = 0
    miss: int = 0
    on: bool = False

    def update(self, seen: bool) -> bool:
        if seen:
            self.hits += 1
            self.miss = 0
            if self.hits >= self.required_hits:
                self.on = True
        else:
            self.miss += 1
            self.hits = 0
            if self.miss >= self.grace_frames:
                self.on = False
        return self.on


@dataclass
class OrthoTarget:
    """Planar target for ORB matching."""
    image: np.ndarray
    keypoints: list
    descriptors: np.ndarray
    size: Tuple[int, int]


class ORBMatcher:
    def __init__(self, ref_img_path: str, max_features: int = 1500):
        img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read reference image: {ref_img_path}")
        h, w = img.shape[:2]
        self.orb = cv2.ORB_create(nfeatures=max_features)
        kps, des = self.orb.detectAndCompute(img, None)
        if des is None or len(kps) < 10:
            raise ValueError("Not enough features in reference image. Use higher-contrast/texture image.")
        self.target = OrthoTarget(img, kps, des, (w, h))
        # HAMMING norm for ORB
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect(self, frame_bgr, draw=True):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kps, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kps) < 8:
            return False, frame_bgr

        # KNN match + Lowe ratio
        matches = self.matcher.knnMatch(self.target.descriptors, des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 12:
            return False, frame_bgr

        src_pts = np.float32([self.target.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return False, frame_bgr

        h, w = self.target.image.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H)

        if draw:
            cv2.polylines(frame_bgr, [np.int32(projected)], True, (0, 255, 0), 2)
        return True, frame_bgr


class ArUcoDetector:
    def __init__(self, dict_name: str = "4X4_50"):
        if not ARUCO_AVAILABLE:
            raise RuntimeError("cv2.aruco is not available. Install opencv-contrib-python to use ArUco.")
        if dict_name not in ARUCO_DICTIONARIES:
            raise ValueError(f"Unknown dictionary '{dict_name}'. Available: {list(ARUCO_DICTIONARIES.keys())}")
        self.dictionary = aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dict_name])

        # Handle both legacy and new ArUco APIs across OpenCV versions
        # Legacy (<=4.6): DetectorParameters_create(), aruco.detectMarkers(...)
        # New (>=4.7):   DetectorParameters(), ArucoDetector(dictionary, parameters).detectMarkers(gray)
        self.legacy_api = hasattr(aruco, 'DetectorParameters_create') and callable(getattr(aruco, 'DetectorParameters_create'))

        if self.legacy_api:
            self.parameters = aruco.DetectorParameters_create()
            self.detector = None
        else:
            # New-style API
            # Some builds expose DetectorParameters but not DetectorParameters_create
            if hasattr(aruco, 'DetectorParameters'):
                self.parameters = aruco.DetectorParameters()
            else:
                # Fallback: try to use default parameters via ArucoDetector without explicit params
                self.parameters = None
            if hasattr(aruco, 'ArucoDetector'):
                self.detector = aruco.ArucoDetector(self.dictionary, self.parameters) if self.parameters is not None else aruco.ArucoDetector(self.dictionary)
            else:
                # If neither legacy nor new detector exists, raise a helpful error
                raise RuntimeError("Your cv2.aruco build lacks both legacy and new detector APIs. Install opencv-contrib-python (same version as cv2) to enable ArUco.")

    def detect(self, frame_bgr, draw=True):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.legacy_api:
            corners, ids, _ = aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        else:
            corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return False, frame_bgr
        if draw:
            aruco.drawDetectedMarkers(frame_bgr, corners, ids)
        return True, frame_bgr

def put_banner(img, text: str, on: bool):
    h, w = img.shape[:2]
    bg = (0, 180, 0) if on else (0, 0, 0)
    cv2.rectangle(img, (0, 0), (w, 40), bg, -1)
    cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='Webcam index (0=default)')
    parser.add_argument('--width', type=int, default=640, help='Capture width')
    parser.add_argument('--height', type=int, default=480, help='Capture height')
    parser.add_argument('--display', action='store_true', help='Show preview window')
    parser.add_argument('--ref', type=str, default=None, help='Path to custom reference image (for ORB matching)')
    parser.add_argument('--aruco', type=str, default=None, help='ArUco dictionary name, e.g., 4X4_50 (requires contrib)')
    parser.add_argument('--hits', type=int, default=5, help='Consecutive frames to confirm ON state')
    parser.add_argument('--grace', type=int, default=15, help='Frames to keep ON after last detection')
    args = parser.parse_args()

    # Initialize detectors
    orb_matcher: Optional[ORBMatcher] = None
    aruco_detector: Optional[ArUcoDetector] = None

    if args.ref:
        try:
            orb_matcher = ORBMatcher(args.ref)
            print(f"[INFO] ORB ready with reference: {args.ref}")
        except Exception as e:
            print(f"[WARN] ORB init failed: {e}")

    if args.aruco:
        try:
            aruco_detector = ArUcoDetector(args.aruco)
            print(f"[INFO] ArUco ready with dict: {args.aruco}")
        except Exception as e:
            print(f"[WARN] ArUco init failed: {e}")

    if orb_matcher is None and aruco_detector is None:
        print("[ERROR] No detector configured. Provide --ref or --aruco.")
        sys.exit(2)

    # Video capture
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        # On Raspberry Pi with v4l2, you may need: cv2.CAP_V4L2 | index
        cap.release()
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(3)

    debounce = Debounce(required_hits=args.hits, grace_frames=args.grace)
    last_state = None

    print("[INFO] Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame grab failed; retrying...")
            time.sleep(0.02)
            continue

        seen = False
        # Try ArUco first if configured, else ORB
        if aruco_detector is not None:
            seen, frame = aruco_detector.detect(frame, draw=True)
        if not seen and orb_matcher is not None:
            seen, frame = orb_matcher.detect(frame, draw=True)

        on = debounce.update(seen)
        put_banner(frame, "manual control on" if on else "manual control off", on)

        if on != last_state:
            last_state = on
            print("manual control on" if on else "manual control off")

        if args.display:
            cv2.imshow('Detector', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            # Headless mode: small sleep to limit CPU
            time.sleep(0.01)

    cap.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
