from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os
import uuid
import numpy as np
from PIL import Image, ImageDraw
import mediapipe as mp

app = FastAPI()
# נתיבים ל־HTML ולסטטיים
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# תיקיות לאחסון הקבצים
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

def make_image_modest(input_path: str, output_path: str):
    # פותחים את התמונה ב־PIL
    image = Image.open(input_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # ממירים ל־numpy RGB ול־BGR
    np_rgb = np.array(image)
    np_bgr = np_rgb[..., ::-1]

    # מאתחלים את המודלים
    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )
    hand_detector = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    # מריצים גילוי פנים
    faces = face_detector.process(np_bgr).detections or []
    for face in faces:
        bbox = face.location_data.relative_bounding_box
        x1 = int(bbox.xmin * image.width)
        y1 = int(bbox.ymin * image.height)
        x2 = x1 + int(bbox.width * image.width)
        y2 = y1 + int(bbox.height * image.height)
        # מצניעים בריבוע שחור
        draw.rectangle([x1, y1, x2, y2], fill="black")

    # מריצים גילוי ידיים
    hands = hand_detector.process(np_bgr).multi_hand_landmarks or []
    for hand_landmarks in hands:
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        x1 = int(min(xs) * image.width)
        y1 = int(min(ys) * image.height)
        x2 = int(max(xs) * image.width)
        y2 = int(max(ys) * image.height)
        draw.rectangle([x1, y1, x2, y2], fill="black")

    image.save(output_path)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    file_id = f"{uuid.uuid4()}{ext}"
    in_path = os.path.join("static/uploads", file_id)
    out_path = os.path.join("static/results", file_id)

    # שומרים את הקובץ
    with open(in_path, "wb") as buf:
        buf.write(await file.read())

    # מצניעים
    make_image_modest(in_path, out_path)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "output_url": f"/static/results/{file_id}"
    })

@app.get("/static/results/{filename}")
def results(filename: str):
    return FileResponse(os.path.join("static/results", filename), media_type="image/jpeg")
