"""Fake phone photo: onions of known size on the calibration sheet, in perspective."""
import cv2
import numpy as np

from app.cv.calib_sheet import MARKER_MM, MARKER_TL, SHEET_H, SHEET_W, marker_image

# (centre x mm, centre y mm, diameter mm, length mm, angle)
TRUTH = [(55, 90, 40, 46, 20), (150, 95, 55, 60, 70), (60, 160, 68, 72, 0), (150, 165, 32, 38, 45), (110, 210, 50, 50, 0)]


def sheet_photo(truth=TRUTH, seed=0):
    rng = np.random.default_rng(seed)
    R = 6.0
    sheet = np.full((int(SHEET_H * R), int(SHEET_W * R), 3), 250, np.uint8)
    for mid, (x, y) in MARKER_TL.items():
        m = cv2.cvtColor(marker_image(mid, int(MARKER_MM * R)), cv2.COLOR_GRAY2BGR)
        sheet[int(y * R):int(y * R) + m.shape[0], int(x * R):int(x * R) + m.shape[1]] = m
    for cx, cy, d, L, ang in truth:
        col = tuple(int(v) for v in rng.integers([30, 30, 110], [80, 110, 190]))
        region = np.zeros(sheet.shape[:2], np.uint8)
        cv2.ellipse(region, (int(cx * R), int(cy * R)), (int(L / 2 * R), int(d / 2 * R)), ang, 0, 360, 255, -1)
        sheet[region > 0] = col
    h, w = sheet.shape[:2]
    P = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
                                    np.float32([[520, 380], [2480, 520], [2600, 3620], [380, 3480]]))
    photo = np.full((4000, 3000, 3), (60, 70, 80), np.uint8)
    photo = cv2.warpPerspective(sheet, P, (3000, 4000), dst=photo, borderMode=cv2.BORDER_TRANSPARENT)
    return cv2.GaussianBlur(photo, (5, 5), 0), P, R
