"""Run from backend/:  python -m pytest -q"""
from pathlib import Path

import cv2
import numpy as np

from app.config import FINDER_CLASSES, RULES_PATH
from app.ml.detector import Detection, classify_preprocess, parse_yolo_output
from app.services.grading import grade_onion, load_rules, lot_summary
from app.services.pipeline import inspect, split_detections
from tests.synthetic import TRUTH, sheet_photo

RULES = load_rules(RULES_PATH)
ASSETS = Path(__file__).parent / "assets"


# ---------- grading rules
def test_grade_rules():
    assert grade_onion([], 55, RULES)[0] == "A"
    assert grade_onion([], 40, RULES)[0] == "URS"          # below Grade A size, within URS
    assert grade_onion([], 30, RULES)[0] == "Reject"       # undersized
    assert grade_onion([], 75, RULES)[0] == "Reject"       # oversized
    assert grade_onion(["black_mould"], 55, RULES)[0] == "URS"
    assert grade_onion(["sprouted"], 55, RULES)[0] == "Reject"
    assert grade_onion([], None, RULES) == ("A", ["size not measured"])
    closed = {**RULES, "urs_open": False}
    assert grade_onion([], 40, closed)[0] == "Reject"


def test_lot_summary():
    on = [{"bucket": "A", "weight_g": 80}] * 9 + [{"bucket": "URS", "weight_g": 40}]
    s = lot_summary(on, RULES)
    assert s["pct_by_count"]["A"] == 90.0 and s["decision"] == "Accept as Grade A"
    s2 = lot_summary(on[:5] + [{"bucket": "Reject", "weight_g": 50}] * 5, RULES)
    assert s2["decision"] == "Reject / re-sort"


# ---------- YOLO output parsing
def test_parse_yolo_output_seg_head():
    nc = len(FINDER_CLASSES)
    out = np.random.rand(1, 4 + nc + 32, 3).astype(np.float32) * 0.01   # mask coefficients must be ignored
    # two overlapping 'onion' boxes (NMS keeps one) + one 'coin'; letterbox r=0.5, pad (0, 80)
    out[0, :4, 0] = [200, 280, 100, 100]; out[0, 4 + 0, 0] = 0.9
    out[0, :4, 1] = [205, 282, 100, 100]; out[0, 4 + 0, 1] = 0.6
    out[0, :4, 2] = [400, 380, 40, 40];   out[0, 4 + 1, 2] = 0.8
    out[0, 4 + nc:, :] = 5.0                                              # large mask coeffs: not scores
    dets = parse_yolo_output(out, FINDER_CLASSES, r=0.5, pad_x=0, pad_y=80, img_w=1280, img_h=960)
    assert [d.cls for d in dets] == ["onion", "coin"]
    assert np.allclose(dets[0].box, (300, 300, 500, 500))


def test_classify_preprocess():
    x = classify_preprocess(np.zeros((300, 150, 3), np.uint8))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32


def test_split_detections():
    dets = [Detection("onion", .9, (0, 0, 100, 100)), Detection("coin", .7, (500, 0, 540, 40)),
            Detection("coin", .9, (600, 0, 640, 40))]
    onions, coin = split_detections(dets)
    assert len(onions) == 1 and coin.conf == .9


# ---------- whole pipeline, no model (printed sheet, colour segmentation)
def test_fallback_sheet_sizes():
    photo, _, _ = sheet_photo()
    result, annotated = inspect(photo, None, None, RULES, 27.0)
    assert result["mode"] == "fallback" and result["calibration"]["method"] == "sheet"
    got = sorted(o["diameter_mm"] for o in result["onions"])
    want = sorted(t[2] for t in TRUTH)
    assert len(got) == len(want)
    assert max(abs(g - w) for g, w in zip(got, want)) < 2.0, (got, want)
    assert annotated.shape == photo.shape


# ---------- whole pipeline with an AI model, coin mode, on a real CSRP photo
class FakeFinder:
    """Stands in for finder.onnx: boxes marked by hand on the test photo."""
    def __call__(self, img, conf_thres=0.25):
        return [Detection("coin", .95, (236, 186, 321, 272)),
                Detection("onion", .9, (58, 168, 140, 256)),     # handwritten caliper: 23.87 mm
                Detection("onion", .9, (355, 22, 602, 260)),     # 72.07 mm
                Detection("onion", .9, (244, 423, 350, 525))]    # 32.65 mm


class FakeClassifier:
    """Stands in for classifier.onnx: calls the biggest onion black_mould, the rest good (one unsure)."""
    def __call__(self, crop):
        if crop.shape[1] > 200:
            return "black_mould", 0.9, {}
        if crop.shape[1] > 100:
            return "good", 0.5, {}
        return "good", 0.95, {}


def test_coin_mode_real_photo():
    img = cv2.imread(str(ASSETS / "csrp_multiple_onions.jpg"))
    result, _ = inspect(img, FakeFinder(), FakeClassifier(), RULES, 27.0)
    assert result["mode"] == "ai" and result["calibration"]["method"] == "coin"
    d = {round(o["box"][0]): o for o in result["onions"]}
    for x, true_mm in [(58, 23.87), (355, 72.07), (244, 32.65)]:
        assert abs(d[x]["diameter_mm"] - true_mm) < 4.0, (d[x]["diameter_mm"], true_mm)
    assert d[355]["defects"] == ["black_mould"] and d[355]["bucket"] == "Reject"   # oversized beats mould
    assert d[58]["bucket"] == "Reject"                                              # undersized
    assert any("check by hand" in f for f in d[244]["flags"])                       # low confidence flagged
    assert result["summary"]["decision"] == "Reject / re-sort"


def test_sheet_with_classifier_only():
    photo, _, _ = sheet_photo()
    result, _ = inspect(photo, None, FakeClassifier(), RULES, 27.0)
    assert result["mode"] == "sheet+classifier"
    assert all(o["defect_label"] for o in result["onions"])
