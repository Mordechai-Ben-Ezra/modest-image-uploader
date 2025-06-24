from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os
import uuid

import numpy as np
from PIL import Image
import cv2
import mediapipe as mp

app = FastAPI()

# תבניות וסטטיים
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# וודא שהתיקיות קיימות
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)
os.makedirs("static/overlays", exist_ok=True)

# אתחול MediaPipe Selfie Segmentation
mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

def make_image_modest(input_path: str, output_path: str):
    # טען תמונה
    pil_img = Image.open(input_path).convert("RGB")
    np_rgb = np.array(pil_img)

    # הפעל סגמנטציה
    results = mp_seg.process(cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR))

    # בנה מסכת בינארית (255 = אזור האדם)
    mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).convert("L").resize(pil_img.size)

    # טען את ה-overlay של הבגד הצנוע
    overlay = (
        Image.open("static/overlays/modest_overlay.png")
        .convert("RGBA")
        .resize(pil_img.size)
    )

    # מיזוג: overlay איפה שה-mask לבן, אחרת התמונה המקורית
    composed = Image.composite(overlay, pil_img.convert("RGBA"), mask_img)

    # שמור חזרה
    composed.convert("RGB").save(output_path)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    # הכנת שמות קבצים ייחודיים
    ext = os.path.splitext(file.filename)[1]
    file_id = f"{uuid.uuid4()}{ext}"
    in_path = os.path.join("static/uploads", file_id)
    out_path = os.path.join("static/results", file_id)

    # שמירת הקובץ שהועלה
    with open(in_path, "wb") as buf:
        buf.write(await file.read())

    # עיבוד התמונה עם overlay
    make_image_modest(in_path, out_path)

    # הצגת דף התוצאות
    return templates.TemplateResponse("result.html", {
        "request": request,
        "output_url": f"/static/results/{file_id}"
    })

# (אופציונלי) אם תרצה לשלוף ישירות עם FileResponse
@app.get("/static/results/{filename}")
def results(filename: str):
    return FileResponse(os.path.join("static/results", filename), media_type="image/jpeg")
