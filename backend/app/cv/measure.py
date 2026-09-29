"""OnionEye sizing: photo of onions on the calibration sheet -> diameter (mm) of each onion.

No training data needed. Steps:
  1. find the 4 ArUco markers -> homography from the photo to the flat sheet (in mm)
  2. warp the photo to a top-down view at PX_PER_MM
  3. onions = "not white paper" pixels inside the placement area (the sheet is white, so this is easy)
  4. per onion: diameter = short side of the minimum-area rectangle (what a sizing ring measures),
     plus the long side, area and an equivalent-circle diameter

Usage:
    python measure.py photo.jpg [out.jpg]
From the backend:
    res = measure(img_bgr)              # onions found by colour alone
    res = measure(img_bgr, boxes=[...]) # or: one onion per detector box (x1,y1,x2,y2 in photo pixels)
"""
import sys
import cv2
import numpy as np

from .calib_sheet import DICT, SHEET_W, SHEET_H, TRAY, marker_corners_mm

PX_PER_MM = 4.0          # resolution of the top-down view
MIN_ONION_MM = 15.0      # blobs smaller than this are ignored (dirt, peel flakes)


class CalibrationError(Exception):
    pass


def find_homography(img):
    """Photo -> sheet-mm homography from the ArUco markers. Needs at least 2 markers (8 points)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        raise CalibrationError("No calibration markers found. Keep all 4 corner markers in the photo.")
    src, dst, found = [], [], []
    for c, i in zip(corners, ids.flatten()):
        if i in (0, 1, 2, 3):
            src.append(c.reshape(4, 2)); dst.append(marker_corners_mm(int(i))); found.append(int(i))
    if len(found) < 2:
        raise CalibrationError(f"Only marker(s) {found} visible; need at least 2 (all 4 is best).")
    H, _ = cv2.findHomography(np.concatenate(src), np.concatenate(dst), cv2.RANSAC, 3.0)
    return H, sorted(found)


def onion_mask(top):
    """White paper vs onion. Onion skin is coloured (red/brown/yellow) or darker than paper."""
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    # paper brightness varies with lighting -> compare with the sheet's own median brightness
    v_paper = np.median(hsv[..., 2])
    mask = ((hsv[..., 1] > 45) | (hsv[..., 2] < 0.72 * v_paper)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    # keep only the placement area (drops markers, text, ruler)
    x0, y0, x1, y1 = [int(v * PX_PER_MM) for v in TRAY]
    keep = np.zeros_like(mask); keep[y0:y1, x0:x1] = 255
    return cv2.bitwise_and(mask, keep)


def contour_stats(cnt):
    area_mm2 = cv2.contourArea(cnt) / PX_PER_MM ** 2
    (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
    short, long_ = sorted([w / PX_PER_MM, h / PX_PER_MM])
    return {
        "diameter_mm": round(short, 1),                      # ring-gauge size, used for grading
        "length_mm": round(long_, 1),
        "equiv_diameter_mm": round(2 * np.sqrt(area_mm2 / np.pi), 1),
        "area_mm2": round(area_mm2, 0),
        "center_mm": (round(cx / PX_PER_MM, 1), round(cy / PX_PER_MM, 1)),
    }


def measure(img, boxes=None):
    H, markers = find_homography(img)
    S = np.diag([PX_PER_MM, PX_PER_MM, 1.0]) @ H
    size = (int(SHEET_W * PX_PER_MM), int(SHEET_H * PX_PER_MM))
    top = cv2.warpPerspective(img, S, size, flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    mask = onion_mask(top)

    regions = []
    if boxes is None:
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
        regions = [(lab == i).astype(np.uint8) * 255 for i in range(1, n)]
    else:
        for (x1, y1, x2, y2) in boxes:   # map each detector box to the top-down view, keep the biggest blob in it
            pts = cv2.perspectiveTransform(np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]), S)[0]
            poly = np.zeros_like(mask); cv2.fillPoly(poly, [pts.astype(np.int32)], 255)
            regions.append(cv2.bitwise_and(mask, poly))

    onions = []
    for r in regions:
        cnts, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            onions.append(None); continue
        c = max(cnts, key=cv2.contourArea)
        s = contour_stats(c)
        if boxes is None and s["equiv_diameter_mm"] < MIN_ONION_MM:
            continue
        # an onion cut by the placement-area edge would be measured too small -> flag it
        x, y, w, h = cv2.boundingRect(c)
        tx0, ty0, tx1, ty1 = [int(v * PX_PER_MM) for v in TRAY]
        s["touches_edge"] = x <= tx0 + 1 or y <= ty0 + 1 or x + w >= tx1 - 1 or y + h >= ty1 - 1
        s["contour_px"] = c
        onions.append(s)
    return {"markers_found": markers, "onions": onions, "topdown": top, "mask": mask, "S": S}


COIN_MM = {"inr10": 27.0, "inr5": 23.0, "inr2": 23.0, "inr1": 21.93}   # current Indian coins


def onion_outline_in_box(img, box):
    """Separate onion from any background inside its detector box (GrabCut, seeded by the box)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    pad = int(0.08 * max(x2 - x1, y2 - y1))
    X1, Y1 = max(0, x1 - pad), max(0, y1 - pad)
    X2, Y2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
    crop = img[Y1:Y2, X1:X2]
    mask = np.zeros(crop.shape[:2], np.uint8)
    rect = (x1 - X1, y1 - Y1, max(1, x2 - x1), max(1, y2 - y1))
    cv2.grabCut(crop, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea) + np.array([X1, Y1])


def measure_with_coin(img, onion_boxes, coin_box, coin_mm=COIN_MM["inr10"]):
    """Coin mode: the detector gives onion boxes + the coin box; mm per pixel comes from the coin.
    Assumes a roughly top-down photo (the coin looks round). Coin diameter in px = longer side of its box
    (the longer side is the least shortened by a slight tilt)."""
    cx1, cy1, cx2, cy2 = coin_box
    px_per_mm = max(cx2 - cx1, cy2 - cy1) / coin_mm
    onions = []
    for b in onion_boxes:
        c = onion_outline_in_box(img, b)
        if c is None:
            onions.append(None); continue
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        short, long_ = sorted([w / px_per_mm, h / px_per_mm])
        area = cv2.contourArea(c) / px_per_mm ** 2
        onions.append({"diameter_mm": round(short, 1), "length_mm": round(long_, 1),
                       "equiv_diameter_mm": round(2 * np.sqrt(area / np.pi), 1), "contour_px": c})
    return {"px_per_mm": px_per_mm, "onions": onions}


def draw(res):
    out = res["topdown"].copy()
    for i, o in enumerate(res["onions"]):
        if o is None: continue
        cv2.drawContours(out, [o["contour_px"]], -1, (0, 200, 0), 3)
        cx, cy = [int(v * PX_PER_MM) for v in o["center_mm"]]
        cv2.putText(out, f"{o['diameter_mm']:.0f} mm", (cx - 45, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(out, f"{o['diameter_mm']:.0f} mm", (cx - 45, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


if __name__ == "__main__":   # python -m app.cv.measure photo.jpg out.jpg
    img = cv2.imread(sys.argv[1])
    res = measure(img)
    print("markers:", res["markers_found"])
    for i, o in enumerate(res["onions"], 1):
        print(f"onion {i}: diameter {o['diameter_mm']} mm, length {o['length_mm']} mm, equiv {o['equiv_diameter_mm']} mm")
    if len(sys.argv) > 2:
        cv2.imwrite(sys.argv[2], draw(res))
