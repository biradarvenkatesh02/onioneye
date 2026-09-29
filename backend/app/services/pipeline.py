"""Photo in -> graded lot out.

1. Finder model (finder.onnx): a box around every onion + the reference coin
2. Calibration: printed sheet (ArUco markers) if visible, else the coin, else size is "not measured"
3. Sizing per onion (app/cv/measure.py)
4. Classifier model (classifier.onnx): one label per onion crop (good / black_mould / rotten / sprouted / damaged)
5. Grading per onion + lot summary (app/services/grading.py, limits in rules/grading_rules.json)
6. Annotated image

Either model can be missing. No finder: onions are found by colour on the printed sheet.
No classifier: defects are not checked (size-only grading). Both are reported as warnings.
"""
import cv2
import numpy as np

from app.cv.measure import CalibrationError, find_homography, measure, measure_with_coin
from app.services.grading import estimate_weight_g, grade_onion, lot_summary

BUCKET_COLOR = {"A": (60, 180, 75), "URS": (0, 190, 255), "Reject": (40, 40, 220)}   # BGR


def split_detections(dets):
    """Finder output -> onions + the best coin."""
    cand = sorted((d for d in dets if d.cls == "onion"), key=lambda d: -d.conf)
    kept = []
    for d in cand:   # drop a box that mostly sits inside a stronger one (split / double detections)
        x0, y0, x1, y1 = d.box
        area = max(1e-6, (x1 - x0) * (y1 - y0))
        if any(max(0, min(x1, k.box[2]) - max(x0, k.box[0])) * max(0, min(y1, k.box[3]) - max(y0, k.box[1])) / area > 0.6
               for k in kept):
            continue
        kept.append(d)
    onions = [{"box": d.box, "conf": d.conf, "defects": []} for d in kept]
    coin = max((d for d in dets if d.cls == "coin"), key=lambda c: c.conf, default=None)
    return onions, coin


def _crop(img, box, margin=0.08):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    m = margin * max(x1 - x0, y1 - y0)
    return img[int(max(0, y0 - m)):int(min(h, y1 + m)), int(max(0, x0 - m)):int(min(w, x1 + m))]


def _contour_to_photo(contour_topdown, S):
    pts = contour_topdown.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, np.linalg.inv(S)).astype(np.int32)


def inspect(img, finder, classifier, rules, coin_mm):
    warnings = []
    dets = finder(img, conf_thres=rules.get("min_confidence", 0.35)) if finder else []
    onions, coin = split_detections(dets)
    mode = {(True, True): "ai", (True, False): "finder-only", (False, True): "sheet+classifier",
            (False, False): "fallback"}[(finder is not None, classifier is not None)]
    if not finder:
        warnings.append("Finder model not loaded: onions found by colour on the printed sheet.")
    if not classifier:
        warnings.append("Defect model not loaded: rot, mould, sprouting and damage are NOT checked.")

    # ---- calibration + sizing
    calib = {"method": None}
    try:
        find_homography(img)
        has_sheet = True
    except CalibrationError:
        has_sheet = False

    if has_sheet:
        if finder:
            res = measure(img, boxes=[o["box"] for o in onions])
            sizes = res["onions"]
        else:
            res = measure(img)
            sizes = res["onions"]
            onions = []
            for s in sizes:
                c = _contour_to_photo(s["contour_px"], res["S"])
                x, y, w, h = cv2.boundingRect(c)
                onions.append({"box": (x, y, x + w, y + h), "conf": None, "defects": []})
        calib = {"method": "sheet", "markers_found": res["markers_found"]}
        for o, s in zip(onions, sizes):
            if s:
                o.update(diameter_mm=s["diameter_mm"], length_mm=s["length_mm"],
                         contour=_contour_to_photo(s["contour_px"], res["S"]))
                if s.get("touches_edge"):
                    o.setdefault("flags", []).append("crosses the sheet edge: size may be wrong")
    elif coin is not None:
        res = measure_with_coin(img, [o["box"] for o in onions], coin.box, coin_mm)
        calib = {"method": "coin", "coin_mm": coin_mm, "px_per_mm": round(res["px_per_mm"], 3)}
        for o, s in zip(onions, res["onions"]):
            if s:
                o.update(diameter_mm=s["diameter_mm"], length_mm=s["length_mm"], contour=s["contour_px"])
    else:
        warnings.append("No calibration sheet or coin found: sizes not measured.")
        if not finder:
            raise CalibrationError("No finder model loaded and no calibration sheet in the photo. "
                                   "Use the printed sheet, or add finder.onnx.")

    # ---- defects: classify every onion crop
    review_below = rules.get("review_below_confidence", 0.6)
    if classifier:
        for o in onions:
            crop = _crop(img, o["box"])
            if min(crop.shape[:2]) < 16:
                continue
            label, conf, probs = classifier(crop)
            p_not = probs.get("not_onion", 0.0)
            # other produce (ginger, garlic, tomato, potato...): drop only when the classifier is sure,
            # or fairly sure while the finder itself was unsure. Otherwise use the best onion class.
            o["not_onion"] = p_not >= 0.8 or (p_not >= 0.5 and (o.get("conf") or 1) < 0.9)
            if label == "not_onion" and not o["not_onion"]:
                rest = {k: v for k, v in probs.items() if k != "not_onion"}
                label = max(rest, key=rest.get); conf = rest[label] / max(1e-6, 1 - p_not)
            o["defect_label"], o["defect_conf"], o["defect_probs"] = label, conf, probs
            if o["not_onion"]:
                continue
            if label != "good":
                o["defects"].append(label)
            if conf < review_below:
                o.setdefault("flags", []).append(f"unsure ({label} {conf:.0%}): check by hand")
        # the classifier also knows other produce (ginger, garlic, tomato, potato...): drop those detections
        not_onion = [o for o in onions if o.get("not_onion")]
        if not_onion:
            onions = [o for o in onions if o not in not_onion]
            warnings.append(f"{len(not_onion)} object(s) that are not onions were ignored.")
        ignored = len(not_onion)
    else:
        ignored = 0

    # ---- grading
    density = rules.get("density_g_per_cm3", 0.95)
    out = []
    for i, o in enumerate(sorted(onions, key=lambda o: (o["box"][1] // 80, o["box"][0])), 1):
        d = o.get("diameter_mm")
        bucket, reasons = grade_onion(o["defects"], d, rules)
        w = estimate_weight_g(d, density)
        out.append({
            "id": i, "box": [round(v) for v in o["box"]], "bucket": bucket, "reasons": reasons,
            "defects": sorted(set(o["defects"])), "diameter_mm": d, "length_mm": o.get("length_mm"),
            "weight_g": round(w, 1) if w else None,
            "confidence": round(o["conf"], 3) if o.get("conf") is not None else None,
            "defect_label": o.get("defect_label"), "defect_conf": round(o["defect_conf"], 3) if o.get("defect_conf") else None,
            "defect_probs": o.get("defect_probs"),
            "flags": o.get("flags", []), "_contour": o.get("contour"),
        })
    summary = lot_summary(out, rules)
    annotated = annotate(img, out)
    for o in out:
        o.pop("_contour")
    return {"mode": mode, "calibration": calib, "onions": out, "summary": summary, "warnings": warnings,
            "ignored_objects": ignored}, annotated


def annotate(img, onions):
    vis = img.copy()
    t = max(2, int(round(max(img.shape[:2]) / 500)))
    fs = max(0.6, max(img.shape[:2]) / 1400)
    for o in onions:
        col = BUCKET_COLOR[o["bucket"]]
        if o.get("_contour") is not None:
            cv2.drawContours(vis, [o["_contour"]], -1, col, t)
        else:
            x1, y1, x2, y2 = o["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, t)
        label = f"{o['id']}. {o['bucket']}" + (f" {o['diameter_mm']:.0f}mm" if o["diameter_mm"] else "")
        x1, y1 = o["box"][0], max(20, o["box"][1] - 8)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 6, y1 + 4), col, -1)
        cv2.putText(vis, label, (x1 + 3, y1), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t)
    return vis
