"""OnionEye API.  Run:  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"""
import base64
import hashlib
import json
import uuid

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import db
from app.config import (CLASSIFIER_CLASSES, CLASSIFIER_PATH, CLASSIFIER_URL, DATA_DIR, DEFAULT_COIN_MM,
                        FINDER_CLASSES, FINDER_PATH, FINDER_URL, INLINE_MEDIA, MEDIA_DIR, RULES_PATH)
from app.cv.measure import CalibrationError
from app.ml.detector import load_models
from app.services.grading import load_rules
from app.services.advisor import LANGS, advise
from app.services.pipeline import inspect

app = FastAPI(title="OnionEye API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

db.init_db()
FINDER, CLASSIFIER = load_models(FINDER_PATH, FINDER_CLASSES, CLASSIFIER_PATH, CLASSIFIER_CLASSES,
                                 FINDER_URL, CLASSIFIER_URL, DATA_DIR / "models")
RULES = load_rules(RULES_PATH)


@app.get("/api/health")
def health():
    return {"ok": True,
            "finder_loaded": FINDER is not None, "classifier_loaded": CLASSIFIER is not None,
            "finder_classes": FINDER.classes if FINDER else None,
            "classifier_classes": CLASSIFIER.classes if CLASSIFIER else None,
            "rules_version": RULES["version"], "inline_media": INLINE_MEDIA}


@app.get("/api/rules")
def rules():
    return RULES


@app.post("/api/inspections")
async def create_inspection(
    image: UploadFile = File(...),
    farmer: str = Form(""),
    lot_ref: str = Form(""),
    centre: str = Form(""),
    coin_mm: float = Form(DEFAULT_COIN_MM),
):
    raw = await image.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read the image.")
    # phones send huge photos; 2000 px on the long side is plenty and keeps it fast
    scale = 2000 / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    try:
        result, annotated = inspect(img, FINDER, CLASSIFIER, RULES, coin_mm)
    except CalibrationError as e:
        raise HTTPException(422, str(e))

    iid = uuid.uuid4().hex[:12]
    if INLINE_MEDIA:     # serverless: send the graded image back inside the JSON
        ok, buf = cv2.imencode(".jpg", _fit(annotated, 1400), [cv2.IMWRITE_JPEG_QUALITY, 82])
        image_url, annotated_url = "", "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    else:
        img_name, ann_name = f"{iid}.jpg", f"{iid}_graded.jpg"
        cv2.imwrite(str(MEDIA_DIR / img_name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(MEDIA_DIR / ann_name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_url, annotated_url = f"/media/{img_name}", f"/media/{ann_name}"
    rec = {
        "id": iid, "created_at": db.now_iso(), "farmer": farmer, "lot_ref": lot_ref, "centre": centre,
        "image": image_url, "annotated": annotated_url,
        "image_sha256": hashlib.sha256(raw).hexdigest(),
        "result_sha256": hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(),
        "result": result,
    }
    db.save(rec)
    return rec


def _fit(img, max_side):
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


@app.get("/api/inspections")
def list_inspections(limit: int = 50):
    return db.list_recent(limit)


@app.get("/api/inspections/{iid}")
def get_inspection(iid: str):
    rec = db.get(iid)
    if not rec:
        raise HTTPException(404, "Inspection not found")
    return rec


@app.get("/api/languages")
def languages():
    return LANGS


@app.post("/api/advice")
def advice(payload: dict = Body(...)):
    """payload: {result: <inspection result>, meta: {farmer, lot_ref, centre}, lang: 'hi', use_llm: true}"""
    result = payload.get("result")
    if not result or "summary" not in result:
        raise HTTPException(400, "Send the inspection result.")
    return advise(result, payload.get("meta") or {}, payload.get("lang", "en"), payload.get("use_llm", True))
