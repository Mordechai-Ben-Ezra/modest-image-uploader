"""
make_image_modest_pose.py
כולל פונקציה dress_frame(frame) שמדביקה חולצה PNG
על כל פריים בעזרת MediaPipe Pose.
"""
import cv2, mediapipe as mp, numpy as np
from pathlib import Path
from PIL import Image

# טעינת Pose ומעטפת
pose = mp.solutions.pose.Pose(static_image_mode=True)

OVERLAY_PATH = Path("static/overlays/modest_overlay.png")
if not OVERLAY_PATH.exists():
    raise FileNotFoundError("חסר static/overlays/modest_overlay.png")
overlay_png = Image.open(OVERLAY_PATH).convert("RGBA")

def dress_frame(frame_bgr):
    h, w = frame_bgr.shape[:2]
    res = pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:        # אם לא זוהה אדם
        return frame_bgr

    lm = res.pose_landmarks.landmark
    L, R = mp.solutions.pose.PoseLandmark, mp.solutions.pose.PoseLandmark
    sx, sy = int(lm[L.LEFT_SHOULDER].x * w), int(lm[L.LEFT_SHOULDER].y * h)
    ex, ey = int(lm[R.RIGHT_SHOULDER].x * w), int(lm[R.RIGHT_SHOULDER].y * h)

    shoulder_w = ((ex - sx)**2 + (ey - sy)**2) ** 0.5
    angle = -np.degrees(np.arctan2(ey - sy, ex - sx))

    scale = shoulder_w / overlay_png.width * 1.2
    new_size = (int(overlay_png.width * scale), int(overlay_png.height * scale))
    overlay = overlay_png.resize(new_size, Image.LANCZOS).rotate(angle, expand=True)

    cx, cy = (sx + ex) // 2, int((sy + ey)/2 + shoulder_w*0.15)
    ox, oy = int(cx - overlay.width/2), int(cy - overlay.height/4)

    base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA))
    base.paste(overlay, (ox, oy), overlay)
    return cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)
