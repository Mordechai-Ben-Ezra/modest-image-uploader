from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, uuid, shutil
import numpy as np
from PIL import Image
import mediapipe as mp

app = FastAPI()

# סטטיים ותבניות
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def make_image_modest(input_path: str, output_path: str) -> None:
    # 1) טען תמונה
    img = Image.open(input_path).convert("RGB")
    w, h = img.size

    # 2) המרה ל-BGR numpy (ל-Mediapipe)
    np_img = np.array(img)[:, :, ::-1]

    # 3) זיהוי פנים
    mp_fd = mp.solutions.face_detection
    with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
        face_results = face_detector.process(np_img)

    # 4) זיהוי ידיים
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=0.5) as hands_detector:
        hands_results = hands_detector.process(np_img)

    # 5) אם לא זוהו פנים ולא ידיים, העתק ושמור
    if not face_results.detections and not hands_results.multi_hand_landmarks:
        img.save(output_path)
        return

    # נפעיל צנזורה על אזורי הפנים, החזה, והידיים
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 6) לכסות את החזה מתיבת הפנים
    if face_results.detections:
        box = face_results.detections[0].location_data.relative_bounding_box
        x0 = int(box.xmin * w)
        y0 = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)
        y1 = y0 + bh
        y2 = min(h, y1 + int(bh * 1.5))
        draw.rectangle([ (x0, y1), (x0 + bw, y2) ], fill=(0, 0, 0, 180))

    # 7) לכסות את הידיים
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # חשב גבולות של כל ה־21 נקודות
            xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
            ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            # הרחבה קלה סביב היד
            pad = 20
            draw.rectangle(
                [ (x_min - pad, y_min - pad), (x_max + pad, y_max + pad) ],
                fill=(0, 0, 0, 180)
            )

    # 8) הדבק את ה־overlay על התמונה ושמור
    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, overlay)
    img_rgba.convert("RGB").save(output_path)


@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[-1]
    img_id = f"{uuid.uuid4()}{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, img_id)

    # שמירת הקובץ שהגיע
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
