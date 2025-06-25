# video_modest_overlay.py – גרסה תקינה (Codec mp4v, גודל נכון)
import sys, cv2
from make_image_modest_pose import dress_frame   # פונקציה שמלבישה כל פריים

def process_video(input_path, output_path):
    # --- פתיחת וידאו לקריאה ---
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    if not ret:
        print("❌ cannot read video"); return

    h, w = frame.shape[:2]
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25        # אם אין FPS בקובץ

    # --- פתיחת VideoWriter עם codec mp4v (נתמך ב-Render) ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    idx = 0
    while True:
        dressed = dress_frame(frame)              # מדביק חולצה
        out.write(dressed)                        # כותב לפריים

        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % 30 == 0:
            print(f"Processed {idx} frames")

    cap.release(); out.release()
    print(f"✅ Saved modest video to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python video_modest_overlay.py <input.mp4> <output.mp4>")
        sys.exit(1)
    process_video(sys.argv[1], sys.argv[2])
