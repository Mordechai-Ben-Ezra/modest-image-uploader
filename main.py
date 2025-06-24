from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, uuid, shutil
from PIL import Image, ImageDraw
import mediapipe as mp

app = FastAPI()

# --- קבצים סטטיים ותבניות ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------- פונקציית העיבוד החדשה ----------
def make_image_modest(input_path: str, output_path: str) -> None:
    """
    טוענת תמונה -> מזהה אדם -> מציירת כיסוי שחור-שקוף על אזור החזה
    """
    # 1) טען תמונה
    img = Image.open(input_path).convert("RGB")
    w, h = img.size

    # 2) מדיה־פייפ: גילוי אובייקט Person
    mp_obj = mp.solutions.object_detection
    detector = mp_obj.ObjectDetection(model_name="Person")  # מודל קטן
    # MediaPipe מקבלת numpy, נשתמש בהמרה מהירה
    results = detector.process(mp_obj.python_image_from_pil(img))
    detector.close()

    # אם אין זיהוי – שמור כמו שהוא
    if not results.detections:
        img.save(output_path)
        return

    # 3) תיבת האדם הראשונה
    box = results.detections[0].location_data.relative_bounding_box
    x0 = int(box.xmin * w)
    y0 = int(box.ymin * h)
    bw = int(box.width * w)
    bh = int(box.height * h)

    # 4) חישוב אזור החזה (40 % מגובה התיבה העליונה)
    y1 = y0
    y2 = y0 + int(bh * 0.4)

    # 5) יצירת כיסוי שחור-שקוף
    overlay = Image.new("RGBA", (bw, y2 - y1), (0, 0, 0, 180))
    img = img.convert("RGBA")
    img.paste(overlay, (x0, y1), overlay)

    # 6) שמירה
    img.convert("RGB").save(output_path)

# ---------- ראוטים ----------
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

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "result_url": f"/static/results/{result_filename}"}
    )

@app.get("/static/results/{filename}")
def get_result(filename: str):
    return FileResponse(os.path.join(RESULT_FOLDER, filename), media_type="image/jpeg")
