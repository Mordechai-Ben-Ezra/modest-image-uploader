from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import uuid

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[-1]
    img_id = f"{uuid.uuid4()}{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, img_id)

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # כאן במקום לעבד באמת, נעתיק את הקובץ (בהמשך נוסיף עיבוד צנוע)
    result_path = os.path.join(RESULT_FOLDER, img_id)
    shutil.copyfile(upload_path, result_path)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "result_url": f"/static/results/{img_id}"
    })
