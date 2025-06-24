from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, uuid, shutil
import numpy as np
from PIL import Image
import mediapipe as mp

app = FastAPI()

# הגדרות לתיקיות סטטיות ותבניות
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def make_image_modest(input_path: str, output_path: str) -> None:
    """
    טוען תמונה -> מזהה פנים -> מצייר כיסוי שחור-שקוף על אזור החזה הרחבה כלפי מטה
    """
    # 1) טען תמונה
    img = Image.open(input_path).convert("RGB")
    w, h = img.size

    # 2) המרה ל-BGR numpy (דרוש ל-Mediapipe)
    np_img = np.array(img)[:, :, ::-1]

    # 3) זיהוי פנים עם Mediapipe FaceDetection
    mp_fd = mp.solutions.face_detection
    with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        results = detector.process(np_img)

    # 4) אם לא זוהו פנים, שמור כמו שהייתי
    if not results.detections:
        img.save(output_path)
        return

    # 5) בחר תיבת הפנים הראשונה
    box = results.detections[0].location_data.relative_bounding_box
    x0, y0 = int(box.xmin * w), int(box.ymin * h)
    bw, bh = int(box.width * w), int(box.height * h)
    # תחתון התיבה (y1) ואז הרחבה כלפי מטה ל-y2
    y1 = y0 + bh
    y2 = min(h, y1 + int(bh * 1.5))

    # 6) צור כיסוי שחור-שקוף בגובה (y2-y1) וברוחב bw
    overlay = Image.new("RGBA", (bw, y2 - y1), (0, 0, 0, 180))
    img_rgba = img.convert("RGBA")
    img_rgba.paste(overlay, (x0, y1), overlay)

    # 7) שמור את התוצאה
    img_rgba.convert("RGB").save(output_path)

@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[-1]
    img_id = f"{uuid.uuid4()}{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, img_id)

    # שמירת הקובץ שהתקבל
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # עיבוד צניעות
    result_filename = f"modest_{img_id}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    make_image_modest(upload_path, result_path)

    # הצגת התוצאה בדף
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "result_url": f"/static/results/{result_filename}"}
    )

@app.get("/static/results/{filename}")
def get_result(filename: str):
    return FileResponse(os.path.join(RESULT_FOLDER, filename), media_type="image/jpeg")
