"""OnionEye data building helpers, used by the Colab notebooks in ml/notebooks/.

Dataset 1 (finder):   onion + coin instance segmentation, YOLO format.
Dataset 2 (defects):  one crop per onion, ImageFolder format, 5 classes.
"""
import glob
import hashlib
import os
import random
import shutil
from collections import Counter, defaultdict

import yaml

# ----------------------------------------------------------------------------- config
FINDER_CLASSES = ["onion", "coin"]
DEFECT_CLASSES = ["good", "black_mould", "rotten", "sprouted", "damaged"]
SEVERITY = ["rotten", "black_mould", "sprouted", "damaged"]          # worst first

# Roboflow sources. map: source class name (lower-case) -> our class, or None to drop.
FINDER_SOURCES = [
    dict(name="csrp", ws="csrp-onion-dataset", proj="onion-segmentation",
         map={"red-onion": "onion", "yellow-onion": "onion", "reference-object": "coin"}),
    dict(name="8hifs", ws="rishabh-thakur", proj="onion-detection-8hifs", map={"onion": "onion"}),
    dict(name="wagk9", ws="yolo-custom-object-detection", proj="instance-segmentation-wagk9", map={"onion": "onion"}),
]
FINDER_TEST_SOURCES = [
    dict(name="thesis1", ws="paul-angelo-lavarias-ii", proj="onionthesis1", map={"onion": "onion", "not onion": None}),
]

# Defect sources with boxes (spots or whole onions). "whole_good": boxes of these classes are whole onions
# that count as good if no defect box sits inside them. "good_if_unboxed": an onion crop with no box counts as good.
DEFECT_BOX_SOURCES = [
    dict(name="vcsiy", ws="raj-ujydl", proj="onion-disease-vcsiy", good_if_unboxed=False,
         map={"good": "good", "black_mould": "black_mould", "completely_spoiled": "rotten",
              "damaged": "damaged", "sprouted": "sprouted"}, whole_good=["good"]),
    dict(name="ciqkj", ws="onion-grading", proj="onions-ciqkj", good_if_unboxed=False,
         map={"rotten": "rotten", "sprout": "sprouted"}, whole_good=[]),
    dict(name="urad", ws="urad", proj="onion-vi5f2", good_if_unboxed=False,
         map={"rot": "rotten", "mold": "black_mould", "sprout": "sprouted", "doublesplit": "damaged",
              "brownonion": "good", "redonion": "good", "whiteonion": "good"},
         whole_good=["brownonion", "redonion", "whiteonion"]),
]
# Defect sources that are already one label per image (Roboflow classification, "folder" export)
DEFECT_FOLDER_SOURCES = [
    dict(name="sorting", ws="harish-ajankar-waojo", proj="onion_sorting",
         map={"good_onion": "good", "black_mold": "black_mould", "softening": "rotten",
              "external_damage": "damaged", "sprouted": "sprouted"}),
]
MENDELEY_GOOD_CAP = 3000


# ----------------------------------------------------------------------------- small utils
def orig_stem(path):
    """Roboflow names augmented copies 'IMG_12_jpg.rf.<hash>.jpg'; the part before '.rf.' is the original photo."""
    return os.path.basename(path).split(".rf.")[0]


def split_of(key, val=10, test=10):
    """Deterministic split by original photo, so copies of one photo never land in two splits."""
    r = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "test" if r < test else "val" if r < test + val else "train"


def read_names(ds_dir):
    names = yaml.safe_load(open(os.path.join(ds_dir, "data.yaml")))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n).strip().lower() for n in names]


def list_images(ds_dir):
    out = []
    for sp in ("train", "valid", "val", "test"):
        for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
            out += glob.glob(os.path.join(ds_dir, sp, "images", f"*.{ext}"))
    return sorted(set(out))


def label_path(img_path):
    p = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    return os.path.splitext(p)[0] + ".txt"


def parse_label_line(line):
    """-> (class_id, list of polygon xy in 0..1). Plain boxes become a 4-point rectangle."""
    parts = line.split()
    if len(parts) < 5:
        return None
    cid, vals = int(parts[0]), [float(v) for v in parts[1:]]
    if len(vals) == 4:
        cx, cy, w, h = vals
        vals = [cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cx - w / 2, cy + h / 2]
    if len(vals) % 2:
        vals = vals[:-1]
    return cid, vals


def poly_box(vals):
    xs, ys = vals[0::2], vals[1::2]
    return min(xs), min(ys), max(xs), max(ys)


# ----------------------------------------------------------------------------- downloads
def rf_latest_version(project):
    nums = []
    for v in project.versions():
        try:
            nums.append(int(str(v.id).rstrip("/").split("/")[-1]))
        except Exception:
            pass
    if nums:
        return max(nums)
    # some Universe projects list no versions through the API: probe newest-first
    for n in range(20, 0, -1):
        try:
            project.version(n)
            return n
        except Exception:
            pass
    raise RuntimeError("no downloadable version found")


def rf_download(rf, src, dest_root, fmt):
    """Downloads the latest version of a Roboflow Universe project. fmt: 'yolov8' (boxes/polygons) or 'folder'."""
    p = rf.workspace(src["ws"]).project(src["proj"])
    v = src.get("version") or rf_latest_version(p)
    loc = os.path.join(dest_root, src["name"])
    if os.path.isdir(loc) and os.listdir(loc):
        print(f"  {src['name']}: already downloaded")
        return loc
    print(f"  {src['ws']}/{src['proj']} v{v} ...")
    p.version(v).download(fmt, location=loc, overwrite=True)
    return loc


def mendeley_download(zip_path):
    """Mendeley 42bcyncfhy (1.6 GB). Tries Mendeley's public API, then the Zenodo mirror."""
    import requests
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1e8:
        return zip_path
    urls = []
    try:
        files = requests.get("https://data.mendeley.com/public-api/datasets/42bcyncfhy/files",
                             params={"folder_id": "root", "version": 1}, timeout=60).json()
        f = max(files, key=lambda x: x.get("size", 0))
        urls.append(f["content_details"]["download_url"])
    except Exception as e:
        print("  Mendeley API failed:", e)
    try:
        rec = requests.get("https://zenodo.org/api/records/20254934", timeout=60).json()
        f = max(rec["files"], key=lambda x: x.get("size", 0))
        urls.append(f["links"]["self"])
    except Exception as e:
        print("  Zenodo API failed:", e)
    for url in urls:
        try:
            with requests.get(url, stream=True, timeout=1200) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as out:
                    for chunk in r.iter_content(1 << 20):
                        out.write(chunk)
            if os.path.getsize(zip_path) > 1e8:
                return zip_path
        except Exception as e:
            print("  download failed:", e)
    raise RuntimeError("Mendeley download failed. Download the zip from "
                       "https://data.mendeley.com/datasets/42bcyncfhy/1 and put it at " + zip_path)


# ----------------------------------------------------------------------------- dataset 1: finder
def build_finder_dataset(sources, test_sources, raw_root, out_dir):
    """Pools every source, remaps classes, re-splits 80/10/10 by original photo. test_sources go to 'test_unseen'."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    splits = ["train", "val", "test", "test_unseen"]
    for sp in splits:
        os.makedirs(os.path.join(out_dir, "images", sp))
        os.makedirs(os.path.join(out_dir, "labels", sp))
    stats, unmapped, dropped = Counter(), Counter(), 0

    for src, is_unseen in [(s, False) for s in sources] + [(s, True) for s in test_sources]:
        d = os.path.join(raw_root, src["name"])
        if not os.path.exists(os.path.join(d, "data.yaml")):
            print("  skipping missing source", src["name"])
            continue
        names = read_names(d)
        for im in list_images(d):
            sp = "test_unseen" if is_unseen else split_of(src["name"] + "/" + orig_stem(im))
            lp = label_path(im)
            out_lines, had = [], False
            for ln in (open(lp).read().splitlines() if os.path.exists(lp) else []):
                pr = parse_label_line(ln)
                if not pr:
                    continue
                had = True
                cid, vals = pr
                src_name = names[cid] if cid < len(names) else f"id{cid}"
                if src_name not in src["map"]:
                    unmapped[(src["name"], src_name)] += 1
                    continue
                dst = src["map"][src_name]
                if dst is None:
                    continue
                out_lines.append(" ".join([str(FINDER_CLASSES.index(dst))] + [f"{v:.5f}" for v in vals]))
                stats[(sp, dst)] += 1
            if had and not out_lines:
                dropped += 1
                continue
            stem = f"{src['name']}__{os.path.splitext(os.path.basename(im))[0]}"
            shutil.copy(im, os.path.join(out_dir, "images", sp, stem + os.path.splitext(im)[1]))
            open(os.path.join(out_dir, "labels", sp, stem + ".txt"), "w").write("\n".join(out_lines))

    cfg = {"path": out_dir, "train": "images/train", "val": "images/val", "test": "images/test",
           "names": {i: c for i, c in enumerate(FINDER_CLASSES)}}
    yaml.safe_dump(cfg, open(os.path.join(out_dir, "data.yaml"), "w"), sort_keys=False)
    cfg_u = dict(cfg, test="images/test_unseen")
    yaml.safe_dump(cfg_u, open(os.path.join(out_dir, "data_unseen.yaml"), "w"), sort_keys=False)
    return stats, unmapped, dropped


# ----------------------------------------------------------------------------- dataset 2: defects
def _inside(inner, outer):
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def label_onion(onion_box, boxes, src):
    """onion_box and boxes in pixels. boxes: list of (source_class_name, box).
    Returns our label or None (= don't use this crop)."""
    found_defects, whole_good = [], False
    for cname, b in boxes:
        dst = src["map"].get(cname)
        if dst is None:
            continue
        if cname in src.get("whole_good", []):
            if _iou(b, onion_box) > 0.5:
                whole_good = True
            continue
        # a defect box counts if it is mostly inside this onion, or it IS this onion (whole-onion label)
        if _inside(b, onion_box) >= 0.5 or _iou(b, onion_box) > 0.5:
            found_defects.append(dst)
    if found_defects:
        return min(found_defects, key=SEVERITY.index)
    if whole_good or src.get("good_if_unboxed"):
        return "good"
    return None


def crop_with_margin(img, box, margin=0.08):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    m = margin * max(x1 - x0, y1 - y0)
    x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
    x1, y1 = int(min(w, x1 + m)), int(min(h, y1 + m))
    return img[y0:y1, x0:x1]


def build_defect_from_boxes(src, raw_root, out_dir, find_onions, min_px=48):
    """find_onions(img_bgr) -> list of onion boxes in pixels (the trained finder, or anything with that signature)."""
    import cv2
    d = os.path.join(raw_root, src["name"])
    names = read_names(d)
    stats, skipped = Counter(), Counter()
    for im_path in list_images(d):
        img = cv2.imread(im_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = []
        lp = label_path(im_path)
        for ln in (open(lp).read().splitlines() if os.path.exists(lp) else []):
            pr = parse_label_line(ln)
            if not pr:
                continue
            cid, vals = pr
            x0, y0, x1, y1 = poly_box(vals)
            boxes.append((names[cid] if cid < len(names) else "?", (x0 * W, y0 * H, x1 * W, y1 * H)))
        onions = list(find_onions(img))
        # whole-onion boxes from the source itself also count as onions (e.g. urad's BrownOnion)
        for cname, b in boxes:
            if cname in src.get("whole_good", []) and all(_iou(b, o) < 0.5 for o in onions):
                onions.append(b)
        if not onions and boxes:
            onions = [(0, 0, W, H)]            # single close-up where the finder saw nothing: use the whole photo
        sp = split_of(src["name"] + "/" + orig_stem(im_path))
        for k, ob in enumerate(onions):
            lab = label_onion(ob, boxes, src)
            if lab is None:
                skipped["no label"] += 1
                continue
            crop = crop_with_margin(img, ob)
            if min(crop.shape[:2]) < min_px:
                skipped["too small"] += 1
                continue
            od = os.path.join(out_dir, sp, lab)
            os.makedirs(od, exist_ok=True)
            cv2.imwrite(os.path.join(od, f"{src['name']}__{os.path.splitext(os.path.basename(im_path))[0]}__{k}.jpg"),
                        crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            stats[(sp, lab)] += 1
    return stats, skipped


def build_defect_from_folders(src, raw_root, out_dir):
    """Roboflow 'folder' export: <split>/<class>/<img>."""
    d = os.path.join(raw_root, src["name"])
    stats = Counter()
    for p in glob.glob(os.path.join(d, "*", "*", "*")):
        cls = os.path.basename(os.path.dirname(p)).strip().lower()
        lab = src["map"].get(cls)
        if lab is None or not p.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        sp = split_of(src["name"] + "/" + orig_stem(p))
        od = os.path.join(out_dir, sp, lab)
        os.makedirs(od, exist_ok=True)
        shutil.copy(p, os.path.join(od, f"{src['name']}__{os.path.basename(p)}"))
        stats[(sp, lab)] += 1
    return stats


def build_defect_from_mendeley(extract_dir, out_dir, cap=MENDELEY_GOOD_CAP, holdout_dir=None):
    """Healthy single bulbs -> 'good' (capped). Unhealthy bulbs -> holdout_dir for a sanity check (not trained on)."""
    good, bad = [], []
    for p in glob.glob(os.path.join(extract_dir, "**", "*.jp*g"), recursive=True):
        parts = [x.lower() for x in p.split(os.sep)]
        if not any("bulb" in x for x in parts):
            continue
        if any("unhealthy" in x for x in parts):
            bad.append(p)
        elif any("healthy" in x for x in parts) and any("single" in x for x in parts):
            good.append(p)
    random.Random(42).shuffle(good)
    stats, seen = Counter(), set()
    for p in good:
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        if len(seen) > cap:
            break
        sp = split_of("mendeley/" + h)
        od = os.path.join(out_dir, sp, "good")
        os.makedirs(od, exist_ok=True)
        shutil.copy(p, os.path.join(od, f"mendeley__{h[:12]}.jpg"))
        stats[(sp, "good")] += 1
    if holdout_dir:
        os.makedirs(holdout_dir, exist_ok=True)
        for p in bad[:1500]:
            h = hashlib.md5(open(p, "rb").read()).hexdigest()
            shutil.copy(p, os.path.join(holdout_dir, f"{h[:12]}.jpg"))
    return stats, len(bad)


def contact_sheet(paths, out_png, cols=8, cell=160, title=""):
    import cv2
    import numpy as np
    paths = list(paths)
    rows = max(1, (len(paths) + cols - 1) // cols)
    sheet = np.full((rows * cell + 30, cols * cell, 3), 255, np.uint8)
    cv2.putText(sheet, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    for i, p in enumerate(paths):
        im = cv2.imread(p)
        if im is None:
            continue
        s = cell / max(im.shape[:2])
        im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))
        r, c = divmod(i, cols)
        sheet[30 + r * cell:30 + r * cell + im.shape[0], c * cell:c * cell + im.shape[1]] = im
    cv2.imwrite(out_png, sheet)
    return out_png
