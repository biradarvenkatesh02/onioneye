"""Writes ml/notebooks/05_hard_negatives.ipynb: teach both models what is NOT an onion.

Problem seen in the live app: the finder calls ginger, tomatoes, garlic, potatoes "onion" (~0.75 conf),
because every training photo contained onions only. Fix, two layers:
  A) defect classifier gets a 6th class `not_onion` (crops of other produce + the finder's own false alarms);
     the backend drops any detection the classifier calls not_onion.
  B) finder fine-tuned with ~2-3k background photos of other vegetables (empty label files = "nothing here").
Before/after false-alarm rates are measured on held-out negative photos and saved to reports/.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from build_notebooks import SETUP, HELPER, get_key_cell, code, md, nb  # noqa: E402

cells = [
    md(r"""
# OnionEye · 05 · Hard negatives: stop calling ginger / tomato / garlic "onion"
**Runtime → T4 GPU → Run all.** ~1.5 h. Everything saves to Drive, safe to re-run (it skips finished steps).

1. Download photos of other produce (Roboflow Universe), drop any photo that contains an onion.
2. Measure how often the current finder raises a false alarm on them.
3. Classifier v2 = 5 onion classes + `not_onion` → `models/classifier_v2.onnx`
4. Finder v2 = fine-tuned with those photos as background → `models/finder_v2.onnx`
5. Re-measure false alarms and onion accuracy → `reports/hard_negatives.txt`
"""),
    code("!pip -q install ultralytics roboflow onnx onnxruntime onnxslim"),
    SETUP, HELPER, get_key_cell(),
    md("## 1 · Other-produce photos (onions removed)"),
    code(r'''
from roboflow import Roboflow
import cv2, json
RAW = f'{WORK}/neg_raw'; os.makedirs(RAW, exist_ok=True)
NEG_SOURCES = [
    dict(name='veg',     ws='vegetables',          proj='vegetables-el4g6'),     # 8k photos, many vegetables
    dict(name='veg2',    ws='vegetables-19jw7',    proj='vegetables-2a164'),
    dict(name='ginger1', ws='rashad-wwx8w',        proj='ginger-root'),
    dict(name='ginger2', ws='ginger-pejzv',        proj='ginger-fnxoy'),
    dict(name='ginger3', ws='curry',               proj='ginger-o3ixd'),
    dict(name='pottom',  ws='new-workspace-c1vsu', proj='potato-and-tomato'),
    dict(name='garlic',  ws='garlic-ccsom',        proj='garlic-sc8yu'),
]
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
for s in NEG_SOURCES:
    try: od.rf_download(rf, s, RAW, 'yolov8')
    except Exception as e: print('  skip', s['name'], '->', str(e)[:120])

ONIONY = ('onion', 'shallot', 'bawang', 'scallion', 'leek', 'spring')
neg = []            # (image path, [boxes of produce in pixels])
for s in NEG_SOURCES:
    d = f"{RAW}/{s['name']}"
    if not os.path.exists(f'{d}/data.yaml'): continue
    names = od.read_names(d); bad = {i for i, n in enumerate(names) if any(k in n for k in ONIONY)}
    kept = 0; per = []
    for im in od.list_images(d):
        lp = od.label_path(im); rows = [od.parse_label_line(l) for l in open(lp)] if os.path.exists(lp) else []
        rows = [r for r in rows if r]
        if any(r[0] in bad for r in rows): continue
        per.append((im, rows)); kept += 1
    random.Random(1).shuffle(per); neg += per[:1200]   # cap big generic sets, keep all ginger/garlic/potato
    print(f"{s['name']:8s} kept {kept} (classes: {', '.join(n for i, n in enumerate(names) if i not in bad)[:90]})")
random.Random(0).shuffle(neg)
n_test = min(600, max(150, len(neg) * 15 // 100))
NEG_TEST, NEG_TRAIN = neg[:n_test], neg[n_test:]
print('negative photos: train', len(NEG_TRAIN), '| held-out test', len(NEG_TEST))
'''),
    md("## 2 · How often does the current finder cry 'onion' on them?"),
    code(r'''
from ultralytics import YOLO
F1 = YOLO(f'{ROOT}/models/finder_best.pt'); ONION = [k for k, v in F1.names.items() if v == 'onion'][0]
def false_alarm_rate(model, items, conf=0.5):
    hit = 0
    for p, _ in items:                      # one at a time: a list would load every image into GPU memory
        r = model.predict(p, imgsz=640, conf=conf, verbose=False)[0]
        hit += int(any(int(c) == ONION for c in r.boxes.cls.cpu().numpy()))
    return hit / max(1, len(items))
FA_BEFORE = {c: false_alarm_rate(F1, NEG_TEST, c) for c in (0.5, 0.75)}
print('finder v1 false alarms on held-out other-produce photos:', {k: f'{v:.1%}' for k, v in FA_BEFORE.items()})
'''),
    md("## 3 · Classifier v2 with a `not_onion` class"),
    code(r'''
CLS1 = f'{WORK}/defects'
if not os.path.exists(CLS1): zipfile.ZipFile(f'{ROOT}/defect_dataset.zip').extractall(CLS1)
CLS2 = f'{WORK}/defects_v2'
if not os.path.exists(CLS2): shutil.copytree(CLS1, CLS2)
def crops_for(items, split, limit):
    n = 0; out = f'{CLS2}/{split}/not_onion'; os.makedirs(out, exist_ok=True)
    for im_path, rows in items:
        img = cv2.imread(im_path)
        if img is None: continue
        H, W = img.shape[:2]
        boxes = [(min(v[0::2]) * W, min(v[1::2]) * H, max(v[0::2]) * W, max(v[1::2]) * H) for _, v in rows]
        r = F1.predict(img, imgsz=640, conf=0.25, verbose=False)[0]          # the finder's own false alarms
        boxes += [tuple(b) for b, c in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.cls.cpu().numpy()) if int(c) == ONION]
        for k, b in enumerate(boxes[:6]):
            if (b[2] - b[0]) < 32 or (b[3] - b[1]) < 32: continue
            cv2.imwrite(f'{out}/{od.orig_stem(im_path)[:40]}_{k}.jpg', od.crop_with_margin(img, b)); n += 1
            if n >= limit: return n
    return n
if not os.path.exists(f'{CLS2}/train/not_onion'):
    ntr = crops_for(NEG_TRAIN[:int(len(NEG_TRAIN) * .85)], 'train', 2500)
    nva = crops_for(NEG_TRAIN[int(len(NEG_TRAIN) * .85):], 'val', 400)
    nte = crops_for(NEG_TEST, 'test', 400)
    print('not_onion crops', ntr, nva, nte)
print({sp: {c: len(os.listdir(f'{CLS2}/{sp}/{c}')) for c in sorted(os.listdir(f'{CLS2}/{sp}'))} for sp in ['train', 'val', 'test']})
RUNS = f'{ROOT}/runs'
last = f'{RUNS}/classifier_v2/weights/last.pt'
if os.path.exists(f'{RUNS}/classifier_v2/weights/best.pt') and not os.path.exists(last): pass
elif os.path.exists(last):
    try: YOLO(last).train(resume=True)
    except Exception as e: print('resume skipped:', e)
else:
    YOLO('yolo11n-cls.pt').train(data=CLS2, imgsz=224, epochs=30, patience=8, batch=64, project=RUNS, name='classifier_v2',
                                 exist_ok=True, seed=42, fliplr=0.5, flipud=0.5, degrees=20, hsv_h=0.01)
C2 = YOLO(f'{RUNS}/classifier_v2/weights/best.pt')
from sklearn.metrics import classification_report
names = [C2.names[i] for i in range(len(C2.names))]; yt, yp = [], []
for i, c in enumerate(names):
    for f in glob.glob(f'{CLS2}/test/{c}/*'): yt.append(i); yp.append(int(C2.predict(f, imgsz=224, verbose=False)[0].probs.top1))
REP_C2 = classification_report(yt, yp, labels=list(range(len(names))), target_names=names, digits=3, zero_division=0); print(REP_C2)
shutil.copy(f'{RUNS}/classifier_v2/weights/best.pt', f'{ROOT}/models/classifier_v2_best.pt')
shutil.copy(C2.export(format='onnx', imgsz=224, opset=12, simplify=True), f'{ROOT}/models/classifier_v2.onnx'); print('saved models/classifier_v2.onnx')
'''),
    md("## 4 · Finder v2: fine-tune with other produce as background"),
    code(r'''
import yaml
FINDER = f'{WORK}/finder'
if not os.path.exists(f'{FINDER}/data.yaml'): zipfile.ZipFile(f'{ROOT}/finder_dataset.zip').extractall(FINDER)
for name in ['data.yaml', 'data_unseen.yaml']:
    cfg = yaml.safe_load(open(f'{FINDER}/{name}')); cfg['path'] = FINDER
    yaml.safe_dump(cfg, open(f'{FINDER}/{name}', 'w'), sort_keys=False)
marker = f'{FINDER}/.negatives_added'
if not os.path.exists(marker):
    for k, (im, _) in enumerate(NEG_TRAIN[:3000]):
        sp = 'val' if k % 8 == 0 else 'train'
        stem = 'neg__' + od.orig_stem(im)[:50] + f'_{k}'
        shutil.copy(im, f'{FINDER}/images/{sp}/{stem}.jpg'); open(f'{FINDER}/labels/{sp}/{stem}.txt', 'w').close()
    open(marker, 'w').close()
for c in glob.glob(f'{FINDER}/labels/*.cache'): os.remove(c)
last = f'{RUNS}/finder_v2/weights/last.pt'
if os.path.exists(last):
    try: YOLO(last).train(resume=True)
    except Exception as e: print('resume skipped:', e)
else:
    YOLO(f'{ROOT}/models/finder_best.pt').train(
        data=f'{FINDER}/data.yaml', imgsz=640, epochs=15, patience=6, batch=16, workers=4, lr0=0.003, warmup_epochs=1,
        project=RUNS, name='finder_v2', exist_ok=True, seed=42, cos_lr=True, fliplr=0.5, flipud=0.5, degrees=15,
        hsv_h=0.01, close_mosaic=5)
F2 = YOLO(f'{RUNS}/finder_v2/weights/best.pt')
FA_AFTER = {c: false_alarm_rate(F2, NEG_TEST, c) for c in (0.5, 0.75)}
rows = []
for label, cfgname in [('test', 'data.yaml'), ('unseen onionthesis1', 'data_unseen.yaml')]:
    m = F2.val(data=f'{FINDER}/{cfgname}', split='test', imgsz=640, plots=False)
    rows.append(f'finder v2 {label}: box mAP50 {m.box.map50:.3f} | mask mAP50 {m.seg.map50:.3f} | P {m.box.mp:.3f} R {m.box.mr:.3f}')
report = ['False alarms on held-out other-produce photos (share of photos with >=1 "onion"):',
          *[f'  conf>={c}: finder v1 {FA_BEFORE[c]:.1%}  ->  finder v2 {FA_AFTER[c]:.1%}' for c in (0.5, 0.75)],
          *rows, '', 'Classifier v2 (with not_onion):', REP_C2]
open(f'{ROOT}/reports/hard_negatives.txt', 'w').write('\n'.join(report)); print('\n'.join(report))
shutil.copy(f'{RUNS}/finder_v2/weights/best.pt', f'{ROOT}/models/finder_v2_best.pt')
shutil.copy(F2.export(format='onnx', imgsz=640, opset=12, simplify=True), f'{ROOT}/models/finder_v2.onnx'); print('saved models/finder_v2.onnx')
'''),
    md("Done ✅ New files in Drive `models/`: `classifier_v2.onnx`, `finder_v2.onnx`; numbers in `reports/hard_negatives.txt`."),
]
out = os.path.join(os.path.dirname(__file__), 'notebooks', '05_hard_negatives.ipynb')
json.dump(nb(cells, gpu=True), open(out, 'w'), indent=1)
print('wrote', out)
