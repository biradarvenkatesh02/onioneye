"""OnionEye calibration sheet: layout constants shared by the PDF generator and the measurer.

A4 portrait (210 x 297 mm). Four ArUco markers (DICT_4X4_50, ids 0-3) sit in the corners.
Onions go on the white area between them. Every length below is in millimetres,
measured from the sheet's top-left corner.
"""
import cv2
import numpy as np

SHEET_W, SHEET_H = 210.0, 297.0
MARKER_MM = 30.0
MARGIN_MM = 12.0
DICT = cv2.aruco.DICT_4X4_50

# top-left corner (x, y) of each marker, by id
MARKER_TL = {
    0: (MARGIN_MM, MARGIN_MM),                                           # top-left
    1: (SHEET_W - MARGIN_MM - MARKER_MM, MARGIN_MM),                     # top-right
    2: (SHEET_W - MARGIN_MM - MARKER_MM, SHEET_H - MARGIN_MM - MARKER_MM),  # bottom-right
    3: (MARGIN_MM, SHEET_H - MARGIN_MM - MARKER_MM),                     # bottom-left
}

# Placement area for onions (inside the markers), used to ignore stuff off the sheet
TRAY = (MARGIN_MM, MARGIN_MM + MARKER_MM + 6, SHEET_W - MARGIN_MM, SHEET_H - MARGIN_MM - MARKER_MM - 6)


def marker_corners_mm(mid):
    """The 4 corners of marker `mid` in mm, in OpenCV's order: TL, TR, BR, BL."""
    x, y = MARKER_TL[mid]
    s = MARKER_MM
    return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float32)


def marker_image(mid, px):
    return cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICT), mid, px)
