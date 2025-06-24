from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os
import uuid
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import mediapipe as mp

app = FastAPI()

# תבניות וסטטיים
templates = Jinja2Templates(directory="templates")
app.mount("/static",  StaticFiles(directory="static"),  name="static")
app.mount("/results", StaticFiles(directory="static/results"), name="results")

# וודא שהתיקיות קיימות
os.makedirs("static/uploads",  exist_ok=True)
os.makedirs("static/results",  exist_ok=True)
os.makedirs("static/overlays", exist_ok=True)

# אתחול MediaPipe Selfie Segmentation
mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

def make_image_modest(input_path: str, output_path: str):
    """Overlay חולצה על תמונת סטילס אחת."""
    pil_img = Image.open(input_path).convert("RGB")
    np_rgb  = np.array(pil_img)

    results = mp_seg.process(cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR))
    mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).convert("L").resize(pil_img.size)

    overlay = (Image.open("static/overlays/modest_overlay.png")
               .convert("RGBA")
               .resize(pil_img.size))

    composed = Image.composite(overlay, pil_img.convert("RGBA"), mask_img)
    composed.convert("RGB").save(output_path)

# ---------- עמוד הראשי --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# ---------- העלאת תמונה -------------------------------------------------
@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    ext      = os.path.splitext(file.filename)[1]
    file_id  = f"{uuid.uuid4()}{ext}"
    in_path  = f"static/uploads/{file_id}"
    out_path = f"static/results/{file_id}"

    with open(in_path, "wb") as buf:
        buf.write(await file.read())

    make_image_modest(in_path, out_path)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "output_url": f"/static/results/{file_id}"
    })

# ---------- חשיפת תוצאות תמונה ------------------------------------------
@app.get("/static/results/{filename}")
def results(filename: str):
    return FileResponse(f"static/results/{filename}", media_type="image/jpeg")

# ---------- העלאת וידאו -------------------------------------------------
@app.post("/process_video")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    # ‎1) שמירת קובץ הווידאו שהועלה
    vid_id  = uuid.uuid4().hex
    in_path  = f"static/uploads/{vid_id}.mp4"
    out_path = f"static/results/{vid_id}_modest.mp4"

    with open(in_path, "wb") as f:
        f.write(await file.read())

    # ‎2) הרצת העיבוד ברקע (לא חוסם את הבקשה)
    cmd = ["python", "video_modest_overlay.py", in_path, out_path]
    background_tasks.add_task(subprocess.run, cmd)

    # ‎3) מחזירים URL; הלקוח יוכל להוריד כשהעיבוד יסתיים
    dl_url = f"/results/{Path(out_path).name}"
    return JSONResponse({"download_url": dl_url})
