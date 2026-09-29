"""The two models, both Ultralytics YOLO11 exported to ONNX and run with onnxruntime (no PyTorch on the server).

  finder.onnx      YOLO11n-seg, classes onion + coin   (ml/notebooks/02_train_finder.ipynb)
  classifier.onnx  YOLO11n-cls, one label per onion    (ml/notebooks/04_train_classifier.ipynb)

Class names are read from the ONNX metadata Ultralytics writes, so the order can never get mixed up.
"""
import ast
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    cls: str
    conf: float
    box: tuple  # x1, y1, x2, y2 in original image pixels


def _onnx_names(session, fallback):
    try:
        names = ast.literal_eval(session.get_modelmeta().custom_metadata_map["names"])
        return [names[i] for i in sorted(names)]
    except Exception:
        return list(fallback)


def letterbox(img, size=640):
    """Resize keeping aspect ratio and pad to size x size (same as Ultralytics)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, left, top


def parse_yolo_output(out, classes, r, pad_x, pad_y, img_w, img_h, conf_thres=0.25, iou_thres=0.5):
    """out: (1, 4 + nc [+ 32 mask coefficients], N) from a YOLOv8/11 detect or segment head.
    Returns boxes only (sizing uses the photo itself, so the mask protos aren't needed)."""
    nc = len(classes)
    pred = np.squeeze(out, 0)
    valid_rows = (4 + nc, 4 + nc + 32)         # detect head, or segment head with 32 mask coefficients
    if pred.shape[0] not in valid_rows and pred.shape[1] in valid_rows:
        pred = pred.T                           # (N, C) -> (C, N)
    if pred.shape[0] not in valid_rows:
        raise ValueError(f"Model output {out.shape} doesn't match {nc} classes {classes}")
    boxes_cxcywh, scores = pred[:4].T, pred[4:4 + nc].T
    cls_id = scores.argmax(1)
    conf = scores.max(1)
    keep = conf >= conf_thres
    boxes_cxcywh, cls_id, conf = boxes_cxcywh[keep], cls_id[keep], conf[keep]
    if len(conf) == 0:
        return []
    cx, cy, w, h = boxes_cxcywh.T
    xyxy = np.stack([(cx - w / 2 - pad_x) / r, (cy - h / 2 - pad_y) / r,
                     (cx + w / 2 - pad_x) / r, (cy + h / 2 - pad_y) / r], 1)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, img_w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, img_h)
    dets = []
    for c in np.unique(cls_id):                # NMS per class
        idx = np.where(cls_id == c)[0]
        b = xyxy[idx]
        xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1).tolist()
        kept = cv2.dnn.NMSBoxes(xywh, conf[idx].tolist(), conf_thres, iou_thres)
        for k in np.array(kept).flatten():
            i = idx[int(k)]
            dets.append(Detection(classes[int(c)], float(conf[i]), tuple(float(v) for v in xyxy[i])))
    return sorted(dets, key=lambda d: -d.conf)


class Finder:
    def __init__(self, model_path, fallback_classes, imgsz=640):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.classes = _onnx_names(self.session, fallback_classes)
        self.imgsz = imgsz
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, img_bgr, conf_thres=0.25):
        lb, r, px, py = letterbox(img_bgr, self.imgsz)
        x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: x})[0]      # output0; output1 (mask protos) ignored
        return parse_yolo_output(out, self.classes, r, px, py, img_bgr.shape[1], img_bgr.shape[0], conf_thres)


def classify_preprocess(crop_bgr, size=224):
    """Ultralytics classify transform: shortest side -> size, centre crop, RGB, 0..1."""
    h, w = crop_bgr.shape[:2]
    s = size / min(h, w)
    img = cv2.resize(crop_bgr, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    img = img[y0:y0 + size, x0:x0 + size]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0


class Classifier:
    def __init__(self, model_path, fallback_classes, imgsz=224):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.classes = _onnx_names(self.session, fallback_classes)
        self.imgsz = imgsz
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, crop_bgr):
        """-> (label, confidence, {label: prob})"""
        p = self.session.run(None, {self.input_name: classify_preprocess(crop_bgr, self.imgsz)})[0][0]
        if abs(float(p.sum()) - 1) > 1e-3:      # not softmaxed yet
            p = np.exp(p - p.max()); p = p / p.sum()
        i = int(p.argmax())
        return self.classes[i], float(p[i]), {c: round(float(v), 3) for c, v in zip(self.classes, p)}


def _fetch(url, dest):
    """Download a model file once (used when the host can't bundle it). Handles Google Drive share links."""
    import re
    import urllib.request
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1e5:
        return dest
    m = re.search(r"/d/([\w-]+)|[?&]id=([\w-]+)", url)
    if "drive.google.com" in url and m:
        fid = m.group(1) or m.group(2)
        url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def _resolve(path, url, cache_dir):
    if Path(path).exists():
        return Path(path)
    parts = sorted(Path(path).parent.glob(Path(path).name + ".part*"))
    if parts:                                        # model shipped in chunks (host upload limits): join once into /tmp
        dest = Path(cache_dir) / Path(path).name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as out:
                for part in parts:
                    out.write(part.read_bytes())
        return dest
    if url:
        try:
            return _fetch(url, Path(cache_dir) / Path(path).name)
        except Exception as e:                       # a broken link must not take the whole app down
            print("model download failed:", url, e)
    return None


def load_models(finder_path, finder_classes, classifier_path, classifier_classes,
                finder_url=None, classifier_url=None, cache_dir="/tmp/onioneye/models"):
    """Either model may be missing; the pipeline adapts (see services/pipeline.py)."""
    fp = _resolve(finder_path, finder_url, cache_dir)
    cp = _resolve(classifier_path, classifier_url, cache_dir)
    finder = Finder(fp, finder_classes) if fp else None
    classifier = Classifier(cp, classifier_classes) if cp else None
    return finder, classifier
