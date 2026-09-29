"""Generates the 4 Colab notebooks in ml/notebooks/. Each one embeds onioneye_data.py so it runs on its own.
Run:  python build_notebooks.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = open(os.path.join(HERE, "onioneye_data.py")).read()


def nb(cells, gpu):
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
            "colab": {"provenance": []}}
    if gpu:
        meta["accelerator"] = "GPU"
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


SETUP = code(r'''
# ---- setup: works on Google Colab (default) and Kaggle
import os, sys, glob, shutil, zipfile, random
IN_COLAB = 'google.colab' in sys.modules
IN_KAGGLE = os.path.exists('/kaggle')
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT, WORK = '/content/drive/MyDrive/onioneye', '/content'
elif IN_KAGGLE:
    ROOT, WORK = '/kaggle/working/onioneye', '/kaggle/temp'
else:
    ROOT, WORK = os.path.abspath('onioneye_data'), os.path.abspath('work')
for d in (ROOT, WORK, f'{ROOT}/models', f'{ROOT}/reports'):
    os.makedirs(d, exist_ok=True)
print('saving to', ROOT)
''')

HELPER = code(r'''
# the shared data code lives in your Drive: MyDrive/onioneye/onioneye_data.py (copy it from the repo's ml/ folder)
shutil.copy(f'{ROOT}/onioneye_data.py', 'onioneye_data.py')
import importlib, onioneye_data as od
importlib.reload(od)
''')


def get_key_cell():
    return code(r'''
# Roboflow key from Secrets (Colab: key icon on the left, name ROBOFLOW_API_KEY, notebook access ON)
def get_roboflow_key():
    if IN_COLAB:
        from google.colab import userdata
        return userdata.get('ROBOFLOW_API_KEY')
    if IN_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret('ROBOFLOW_API_KEY')
    return os.environ['ROBOFLOW_API_KEY']
ROBOFLOW_API_KEY = get_roboflow_key()
assert ROBOFLOW_API_KEY, 'Add ROBOFLOW_API_KEY in Secrets first'
''')


def zip_dir(src, dst_zip):
    return f"shutil.make_archive({dst_zip!r}[:-4], 'zip', {src!r})"


# ============================================================================ 01
nb1 = nb([
    md(r"""
# OnionEye · 01 · Get the data
Downloads all 8 public datasets and builds **Dataset 1 (finder: onion + coin)**. No GPU needed, about 20 min.

Before running: Roboflow account → Settings → API Keys → copy the private key → Colab **Secrets** (key icon, left bar) → name `ROBOFLOW_API_KEY`, turn on notebook access. Then **Runtime → Run all**.

Output in `MyDrive/onioneye/`: `finder_dataset.zip`, `raw_defect_sources.zip`, `mendeley_onion.zip`, `reports/finder_samples.png`.
"""),
    code("!pip -q install roboflow pyyaml opencv-python-headless"),
    SETUP, HELPER, get_key_cell(),
    md("## 1 · Download from Roboflow (latest version of each)"),
    code(r'''
import onioneye_data as od
from roboflow import Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
RAW = f'{WORK}/raw'
failed = []
for src in od.FINDER_SOURCES + od.FINDER_TEST_SOURCES + od.DEFECT_BOX_SOURCES:
    try: od.rf_download(rf, src, RAW, 'yolov8')
    except Exception as e: failed.append(src['name']); print('  FAILED', src['name'], e)
for src in od.DEFECT_FOLDER_SOURCES:
    try: od.rf_download(rf, src, RAW, 'folder')
    except Exception as e: failed.append(src['name']); print('  FAILED', src['name'], e)
print('failed:', failed or 'none')
'''),
    md("## 2 · Build Dataset 1 (finder)"),
    code(r'''
import pandas as pd
FINDER = f'{WORK}/finder'
stats, unmapped, dropped = od.build_finder_dataset(od.FINDER_SOURCES, od.FINDER_TEST_SOURCES, RAW, FINDER)
df = pd.Series(stats).unstack(0).fillna(0).astype(int)
print('images per split:', {sp: len(os.listdir(f'{FINDER}/images/{sp}')) for sp in ['train', 'val', 'test', 'test_unseen']})
display(df)
print('dropped images with no usable label:', dropped)
if unmapped: print('⚠ source classes not in the map (tell Claude):', dict(unmapped))
'''),
    md("## 3 · Look at 24 random labelled images: green = onion, white = coin"),
    code(r'''
import cv2, numpy as np
def draw(im_path):
    img = cv2.imread(im_path); H, W = img.shape[:2]
    for ln in open(od.label_path(im_path)).read().splitlines():
        c, *v = ln.split(); pts = (np.array(v, float).reshape(-1, 2) * [W, H]).astype(np.int32)
        cv2.polylines(img, [pts], True, (60, 200, 60) if c == '0' else (255, 255, 255), 3)
    return img
picks = random.sample(glob.glob(f'{FINDER}/images/train/*'), 24)
tmp = [f'{WORK}/_s{i}.jpg' for i in range(24)]
for p, t in zip(picks, tmp): cv2.imwrite(t, draw(p))
sheet = od.contact_sheet(tmp, f'{ROOT}/reports/finder_samples.png', cols=6, cell=260, title='finder dataset')
from IPython.display import Image; display(Image(sheet))
'''),
    md("## 4 · Save to Drive"),
    code(r'''
shutil.make_archive(f'{ROOT}/finder_dataset', 'zip', FINDER)
defect_raw = f'{WORK}/raw_defects'
os.makedirs(defect_raw, exist_ok=True)
for s in od.DEFECT_BOX_SOURCES + od.DEFECT_FOLDER_SOURCES:
    if os.path.exists(f'{RAW}/{s["name"]}'): shutil.copytree(f'{RAW}/{s["name"]}', f'{defect_raw}/{s["name"]}', dirs_exist_ok=True)
shutil.make_archive(f'{ROOT}/raw_defect_sources', 'zip', defect_raw)
print(os.listdir(ROOT))
'''),
    md("## 5 · Mendeley bulbs (1.6 GB, straight to your Drive, 5–15 min). Only needed for notebook 03."),
    code(r'''
try:
    MZIP = od.mendeley_download(f'{ROOT}/mendeley_onion.zip')
    print('OK', MZIP, round(os.path.getsize(MZIP) / 1e9, 2), 'GB')
except Exception as e:
    print('⚠ Mendeley download failed, the rest is fine. Fix before notebook 03:', e)
'''),
    md("Done ✅ Next: **02_train_finder** (switch the runtime to a T4 GPU)."),
], gpu=False)

# ============================================================================ 02
nb2 = nb([
    md(r"""
# OnionEye · 02 · Train the finder (onion + coin)
**Runtime → Change runtime type → T4 GPU**, then Run all. About 2–3 h. If Colab disconnects, just Run all again: it resumes from the last epoch (saved in your Drive).

Output: `MyDrive/onioneye/models/finder.onnx` and `finder_best.pt`, plus scores in `reports/`.
"""),
    code("!pip -q install ultralytics onnx onnxruntime onnxslim"),
    SETUP,
    code(r'''
FINDER = f'{WORK}/finder'
if not os.path.exists(f'{FINDER}/data.yaml'):
    zipfile.ZipFile(f'{ROOT}/finder_dataset.zip').extractall(FINDER)
import yaml
for name in ['data.yaml', 'data_unseen.yaml']:
    cfg = yaml.safe_load(open(f'{FINDER}/{name}')); cfg['path'] = FINDER
    yaml.safe_dump(cfg, open(f'{FINDER}/{name}', 'w'), sort_keys=False)
!nvidia-smi --query-gpu=name,memory.total --format=csv
'''),
    md("## 1 · Train YOLO11n-seg"),
    code(r'''
from ultralytics import YOLO
RUNS = f'{ROOT}/runs'
last = f'{RUNS}/finder/weights/last.pt'
if os.path.exists(last):
    YOLO(last).train(resume=True)
else:
    YOLO('yolo11n-seg.pt').train(
        data=f'{FINDER}/data.yaml', imgsz=640, epochs=60, patience=15, batch=16, workers=4,
        project=RUNS, name='finder', exist_ok=True, seed=42, cos_lr=True,
        fliplr=0.5, flipud=0.5, degrees=15, hsv_h=0.01, close_mosaic=10)
'''),
    md("## 2 · Scores: own test split, and a different author's photos (onionthesis1)"),
    code(r'''
import pandas as pd
best = YOLO(f'{RUNS}/finder/weights/best.pt')
rows = []
for label, cfg in [('test (same sources)', 'data.yaml'), ('unseen onionthesis1', 'data_unseen.yaml')]:
    m = best.val(data=f'{FINDER}/{cfg}', split='test', imgsz=640, plots=False)
    rows.append({'set': label, 'box mAP50': m.box.map50, 'mask mAP50': m.seg.map50,
                 'mask mAP50-95': m.seg.map, 'precision': m.box.mp, 'recall': m.box.mr})
df = pd.DataFrame(rows).round(3); display(df)
df.to_csv(f'{ROOT}/reports/finder_scores.csv', index=False)
'''),
    md("## 3 · Predictions on 8 unseen photos"),
    code(r'''
import matplotlib.pyplot as plt
imgs = random.sample(glob.glob(f'{FINDER}/images/test_unseen/*') or glob.glob(f'{FINDER}/images/test/*'), 8)
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
for ax, r in zip(axs.flat, best.predict(imgs, imgsz=640, conf=0.4, verbose=False)):
    ax.imshow(r.plot()[:, :, ::-1]); ax.axis('off')
plt.tight_layout(); plt.savefig(f'{ROOT}/reports/finder_predictions.png', dpi=70); plt.show()
'''),
    md("## 4 · Export for the backend"),
    code(r'''
shutil.copy(f'{RUNS}/finder/weights/best.pt', f'{ROOT}/models/finder_best.pt')
onnx_path = best.export(format='onnx', imgsz=640, opset=12, simplify=True)
shutil.copy(onnx_path, f'{ROOT}/models/finder.onnx')
import onnxruntime as ort
s = ort.InferenceSession(onnx_path)
print([(o.name, o.shape) for o in s.get_outputs()])          # expect (1, 38, 8400) and mask protos
print('classes:', s.get_modelmeta().custom_metadata_map.get('names'))
'''),
    md("Done ✅ Download `models/finder.onnx` → put it in `backend/models/`. Next: **03_build_defect_data**."),
], gpu=True)

# ============================================================================ 03
nb3 = nb([
    md(r"""
# OnionEye · 03 · Build Dataset 2 (defect crops)
Uses the trained finder to cut every onion out of the defect photos and label it from the damage boxes inside it. GPU helps (T4), ~20 min.

Output: `MyDrive/onioneye/defect_dataset.zip` + one review sheet per class in `reports/`. **Look at the sheets** and list any wrong crops in step 4.
"""),
    code("!pip -q install ultralytics opencv-python-headless"),
    SETUP, HELPER,
    code(r'''
import onioneye_data as od
RAW = f'{WORK}/raw_defects'; CLS = f'{WORK}/defects'; HOLD = f'{WORK}/mendeley_unhealthy_holdout'
if not os.path.exists(RAW): zipfile.ZipFile(f'{ROOT}/raw_defect_sources.zip').extractall(RAW)
shutil.rmtree(CLS, ignore_errors=True)
from ultralytics import YOLO
finder = YOLO(f'{ROOT}/models/finder_best.pt')
ONION = [k for k, v in finder.names.items() if v == 'onion'][0]
def find_onions(img, conf=0.4):
    r = finder.predict(img, imgsz=640, conf=conf, verbose=False)[0]
    return [tuple(b) for b, c in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.cls.cpu().numpy()) if int(c) == ONION]
'''),
    md("## 1 · Crops from the box datasets (vcsiy, ciqkj, urad) and the folder dataset"),
    code(r'''
import pandas as pd
from collections import Counter
total = Counter()
for src in od.DEFECT_BOX_SOURCES:
    if not os.path.exists(f'{RAW}/{src["name"]}'): print('missing', src['name']); continue
    st, skipped = od.build_defect_from_boxes(src, RAW, CLS, find_onions)
    print(src['name'], dict(sum((Counter({k[1]: v}) for k, v in st.items()), Counter())), 'skipped:', dict(skipped))
    total.update(st)
for src in od.DEFECT_FOLDER_SOURCES:
    if os.path.exists(f'{RAW}/{src["name"]}'):
        st = od.build_defect_from_folders(src, RAW, CLS); total.update(st); print(src['name'], dict(st))
'''),
    md("## 2 · Mendeley healthy bulbs → `good` (capped), unhealthy → held out for a check in notebook 04"),
    code(r'''
MX = f'{WORK}/mendeley'
if not os.path.exists(MX):
    zipfile.ZipFile(f'{ROOT}/mendeley_onion.zip').extractall(MX)
    for inner in glob.glob(f'{MX}/**/*.zip', recursive=True):
        zipfile.ZipFile(inner).extractall(os.path.dirname(inner))
st, n_bad = od.build_defect_from_mendeley(MX, CLS, holdout_dir=HOLD); total.update(st)
print('mendeley good:', sum(st.values()), '| unhealthy held out:', len(os.listdir(HOLD)), 'of', n_bad)
df = pd.Series(total).unstack(0).reindex(od.DEFECT_CLASSES).fillna(0).astype(int)
df['total'] = df.sum(axis=1); display(df)
'''),
    md("## 3 · Review sheets: 48 random crops per class (saved to `reports/`)"),
    code(r'''
from IPython.display import Image
for c in od.DEFECT_CLASSES:
    files = sorted(glob.glob(f'{CLS}/*/{c}/*.jpg'))
    pick = random.Random(0).sample(files, min(48, len(files)))
    p = od.contact_sheet(pick, f'{ROOT}/reports/defect_{c}.png', cols=8, cell=150, title=f'{c}: {len(files)} crops')
    print(c); display(Image(p))
    open(f'{ROOT}/reports/defect_{c}_files.txt', 'w').write('\n'.join(os.path.basename(x) for x in pick))
'''),
    md("## 4 · Remove wrong crops (optional)\nFind a bad crop's position on a sheet (row by row, starting at 0), look up its name in `reports/defect_<class>_files.txt`, paste names below, run."),
    code(r'''
BAD = [
    # 'vcsiy__IMG-20250310-WA0381_jpg__0.jpg',
]
for name in BAD:
    for p in glob.glob(f'{CLS}/*/*/{name}'): os.remove(p); print('removed', p)
'''),
    md("## 5 · Balance and save"),
    code(r'''
# keep 'good' at most 3x the biggest defect class, so the model doesn't just say 'good'
counts = {c: len(glob.glob(f'{CLS}/train/{c}/*')) for c in od.DEFECT_CLASSES}
cap = 3 * max(v for k, v in counts.items() if k != 'good')
goods = sorted(glob.glob(f'{CLS}/train/good/*')); random.Random(1).shuffle(goods)
for p in goods[cap:]: os.remove(p)
for sp in ['train', 'val', 'test']:                     # every class folder must exist in every split
    for c in od.DEFECT_CLASSES: os.makedirs(f'{CLS}/{sp}/{c}', exist_ok=True)
print({sp: {c: len(os.listdir(f'{CLS}/{sp}/{c}')) for c in od.DEFECT_CLASSES} for sp in ['train', 'val', 'test']})
shutil.make_archive(f'{ROOT}/defect_dataset', 'zip', CLS)
shutil.make_archive(f'{ROOT}/mendeley_unhealthy_holdout', 'zip', HOLD)
'''),
    md("Done ✅ Next: **04_train_classifier**."),
], gpu=True)

# ============================================================================ 04
nb4 = nb([
    md(r"""
# OnionEye · 04 · Train the defect classifier
T4 GPU, about 30 min. Output: `MyDrive/onioneye/models/classifier.onnx` + confusion matrix and scores in `reports/`.
"""),
    code("!pip -q install ultralytics onnx onnxruntime onnxslim scikit-learn"),
    SETUP,
    code(r'''
CLS = f'{WORK}/defects'; HOLD = f'{WORK}/holdout'
if not os.path.exists(CLS): zipfile.ZipFile(f'{ROOT}/defect_dataset.zip').extractall(CLS)
if not os.path.exists(HOLD) and os.path.exists(f'{ROOT}/mendeley_unhealthy_holdout.zip'):
    zipfile.ZipFile(f'{ROOT}/mendeley_unhealthy_holdout.zip').extractall(HOLD)
from ultralytics import YOLO
RUNS = f'{ROOT}/runs'
last = f'{RUNS}/classifier/weights/last.pt'
if os.path.exists(last):
    YOLO(last).train(resume=True)
else:
    YOLO('yolo11n-cls.pt').train(data=CLS, imgsz=224, epochs=40, patience=10, batch=64,
                                 project=RUNS, name='classifier', exist_ok=True, seed=42,
                                 fliplr=0.5, flipud=0.5, degrees=20, hsv_h=0.01)
'''),
    md("## 1 · Test-set scores (per class) and confusion matrix"),
    code(r'''
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
best = YOLO(f'{RUNS}/classifier/weights/best.pt')
names = [best.names[i] for i in range(len(best.names))]
y_true, y_pred = [], []
for i, c in enumerate(names):
    files = glob.glob(f'{CLS}/test/{c}/*')
    for r in best.predict(files, imgsz=224, verbose=False, stream=True):
        y_true.append(i); y_pred.append(int(r.probs.top1))
rep = classification_report(y_true, y_pred, labels=list(range(len(names))), target_names=names, digits=3, output_dict=True, zero_division=0)
df = pd.DataFrame(rep).T.round(3); display(df); df.to_csv(f'{ROOT}/reports/classifier_scores.csv')
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=list(range(len(names)))), display_labels=names).plot(xticks_rotation=45)
plt.title('Defect classifier, test set'); plt.tight_layout(); plt.savefig(f'{ROOT}/reports/classifier_confusion.png', dpi=110); plt.show()
'''),
    md("## 2 · Sanity check on 1,500 held-out Mendeley *unhealthy* bulbs (never trained on)"),
    code(r'''
files = glob.glob(f'{HOLD}/*.jpg')
if files:
    preds = [best.names[int(r.probs.top1)] for r in best.predict(files, imgsz=224, verbose=False, stream=True)]
    s = pd.Series(preds).value_counts(normalize=True).mul(100).round(1)
    print(f"called NOT good: {100 - s.get('good', 0):.1f}% (higher is better)"); display(s)
'''),
    md("## 3 · Export for the backend"),
    code(r'''
shutil.copy(f'{RUNS}/classifier/weights/best.pt', f'{ROOT}/models/classifier_best.pt')
onnx_path = best.export(format='onnx', imgsz=224, opset=12, simplify=True)
shutil.copy(onnx_path, f'{ROOT}/models/classifier.onnx')
import onnxruntime as ort
s = ort.InferenceSession(onnx_path)
print(s.get_outputs()[0].shape, s.get_modelmeta().custom_metadata_map.get('names'))
'''),
    md("Done ✅ Download `models/classifier.onnx` → `backend/models/`, restart the backend."),
], gpu=True)

out = os.path.join(HERE, "notebooks")
os.makedirs(out, exist_ok=True)
for f in glob_old if (glob_old := [p for p in os.listdir(out) if p.endswith(".ipynb")]) else []:
    os.remove(os.path.join(out, f))
for name, n in [("01_get_data", nb1), ("02_train_finder", nb2), ("03_build_defect_data", nb3), ("04_train_classifier", nb4)]:
    json.dump(n, open(os.path.join(out, f"{name}.ipynb"), "w"), indent=1)
print("wrote", sorted(os.listdir(out)))
