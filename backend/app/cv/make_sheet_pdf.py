"""Generates onioneye_calibration_sheet_A4.pdf. Print at 100% / "Actual size" (NOT "fit to page")."""
import os, sys, tempfile
import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

sys.path.insert(0, os.path.dirname(__file__))
from app.cv.calib_sheet import SHEET_H, MARKER_MM, MARKER_TL, TRAY, marker_image

out = sys.argv[1] if len(sys.argv) > 1 else "onioneye_calibration_sheet_A4.pdf"
c = canvas.Canvas(out, pagesize=A4)
Y = lambda y_top_mm: (SHEET_H - y_top_mm) * mm   # reportlab's origin is bottom-left

tmp = tempfile.mkdtemp()
for mid, (x, y) in MARKER_TL.items():
    p = os.path.join(tmp, f"m{mid}.png")
    cv2.imwrite(p, marker_image(mid, 600))
    c.drawImage(p, x * mm, Y(y + MARKER_MM), MARKER_MM * mm, MARKER_MM * mm)

# light border of the placement area
x0, y0, x1, y1 = TRAY
c.setStrokeGray(0.8); c.setDash(3, 3)
c.rect(x0 * mm, Y(y1), (x1 - x0) * mm, (y1 - y0) * mm)
c.setDash()

# 100 mm check ruler (user verifies the print scale with a real ruler)
c.setStrokeGray(0); c.setLineWidth(0.8)
rx, ry = 55, 20
c.line(rx * mm, Y(ry), (rx + 100) * mm, Y(ry))
for i in range(0, 101, 10):
    c.line((rx + i) * mm, Y(ry), (rx + i) * mm, Y(ry + (3 if i % 50 else 5)))
c.setFont("Helvetica", 7)
c.drawCentredString((rx + 50) * mm, Y(ry + 9), "This line must measure exactly 100 mm. If not, reprint at 100% / Actual size.")

c.setFont("Helvetica-Bold", 10)
c.drawCentredString(105 * mm, Y(282), "OnionEye calibration sheet  ·  SIH26031")
c.setFont("Helvetica", 7.5)
c.drawCentredString(105 * mm, Y(287), "Place onions inside the dashed area, not touching each other or the markers. Photo from straight above; all 4 markers visible.")
c.save()
print("wrote", out)
