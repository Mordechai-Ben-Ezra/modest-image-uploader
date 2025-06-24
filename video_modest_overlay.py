# video_modest_overlay.py
"""Apply a modest long‑sleeve overlay on every frame of a video.

* Works on CPU only – uses MediaPipe Pose to align a PNG overlay.
* Save this file in the project root, commit, then run:

    python video_modest_overlay.py input.mp4 output.mp4

Requirements (already in requirements.txt):
    mediapipe
    opencv-python-headless
    pillow
    numpy

You also need the overlay image at static/overlays/modest_overlay.png
"""

from __future__ import annotations

import sys
from pathlib import Path
import math

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# ---------- Globals ----------------------------------------------------------
OVERLAY_PATH = Path("static/overlays/modest_overlay.png")

# load overlay once (RGBA)
_overlay_png = Image.open(OVERLAY_PATH).convert("RGBA")
_overlay_ratio = _overlay_png.height / _overlay_png.width

# initialise MediaPipe Pose once (static mode = True ⇒ faster on single images)
_mp_pose = mp.solutions.pose
_pose = _mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False)

# ---------- Core function ----------------------------------------------------

def dress_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Return a *new* BGR frame with the modest overlay pasted on the torso.

    *frame_bgr* must be a NumPy array in BGR (as provided by OpenCV).
    The function keeps the original untouched and returns a copy.
    """
    h, w = frame_bgr.shape[:2]

    # convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = _pose.process(rgb)
    if not res.pose_landmarks:
        # nothing detected – return original frame
        return frame_bgr.copy()

    lm = res.pose_landmarks.landmark
    left = _mp_pose.PoseLandmark.LEFT_SHOULDER
    right = _mp_pose.PoseLandmark.RIGHT_SHOULDER

    l_sh = lm[left]
    r_sh = lm[right]

    # coordinates in image space
    sx, sy = l_sh.x * w, l_sh.y * h
    ex, ey = r_sh.x * w, r_sh.y * h

    # shoulder width & angle (degrees, negative for clockwise rotation in PIL)
    shoulder_w = math.hypot(ex - sx, ey - sy)
    angle = -math.degrees(math.atan2(ey - sy, ex - sx))

    # scale overlay: a bit wider than shoulders (factor 1.15)
    scale = shoulder_w / _overlay_png.width * 1.15
    new_w = int(_overlay_png.width * scale)
    new_h = int(new_w * _overlay_ratio)

    # resize & rotate overlay
    overlay = _overlay_png.resize((new_w, new_h), Image.LANCZOS).rotate(angle, expand=True)

    # anchor point: centre between shoulders, shift downward 15% of shoulder width
    cx, cy = (sx + ex) / 2, (sy + ey) / 2 + shoulder_w * 0.15
    ox = int(cx - overlay.width / 2)
    oy = int(cy - overlay.height / 4)

    # paste on PIL image for alpha support
    base_rgba = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA))
    base_rgba.paste(overlay, (ox, oy), overlay)

    # convert back to BGR for OpenCV
    return cv2.cvtColor(np.array(base_rgba), cv2.COLOR_RGBA2BGR)

# ---------- CLI utility ------------------------------------------------------

def process_video(input_path: str | Path, output_path: str | Path):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dressed = dress_frame(frame)
        out.write(dressed)

    cap.release()
    out.release()
    print(f"✅ Saved modest video to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python video_modest_overlay.py <input.mp4> <output.mp4>")
        sys.exit(1)
    process_video(sys.argv[1], sys.argv[2])
