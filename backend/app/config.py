from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent           # backend/


def _load_env_file():
    """Read KEY=value lines from onioneye/.env or backend/.env (no extra dependency). Real env vars win."""
    for f in (BASE_DIR.parent / ".env", BASE_DIR / ".env"):
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if v.strip():
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file()
ON_VERCEL = bool(os.getenv("VERCEL"))                       # serverless: only /tmp is writable
DATA_DIR = Path(os.getenv("ONIONEYE_DATA", "/tmp/onioneye" if ON_VERCEL else BASE_DIR / "data"))
# return images inside the JSON (data: URLs) instead of files under /media; needed on serverless hosts
INLINE_MEDIA = os.getenv("ONIONEYE_INLINE_MEDIA", "1" if ON_VERCEL else "0") == "1"
MEDIA_DIR = DATA_DIR / "media"
DB_PATH = DATA_DIR / "onioneye.db"
FINDER_PATH = Path(os.getenv("ONIONEYE_FINDER", BASE_DIR / "models" / "finder.onnx"))            # onion + coin (YOLO11n-seg)
CLASSIFIER_PATH = Path(os.getenv("ONIONEYE_CLASSIFIER", BASE_DIR / "models" / "classifier.onnx"))  # defect per onion (YOLO11n-cls)
RULES_PATH = Path(__file__).resolve().parent / "rules" / "grading_rules.json"
# optional: download the models at startup if the files aren't bundled (e.g. a public Google Drive / HF link)
FINDER_URL = os.getenv("ONIONEYE_FINDER_URL")
CLASSIFIER_URL = os.getenv("ONIONEYE_CLASSIFIER_URL")

# Fallback class names if an ONNX file has no metadata (Ultralytics exports normally include them)
FINDER_CLASSES = ["onion", "coin"]
CLASSIFIER_CLASSES = ["black_mould", "damaged", "good", "rotten", "sprouted"]   # Ultralytics sorts folder names

DEFAULT_COIN_MM = 27.0      # Rs 10 coin

for d in (DATA_DIR, MEDIA_DIR):
    d.mkdir(parents=True, exist_ok=True)
