"""Headless check of the image zoom/pan behaviour."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QPoint, QPointF, QEvent
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent, QWheelEvent

from tem_analyzer.gui import ImageLabel

app = QApplication([])

img = np.zeros((400, 800, 3), np.uint8)
img[:, :, 1] = np.linspace(0, 255, 800).astype(np.uint8)
qimg = QImage(img.data, 800, 400, 3 * 800, QImage.Format_RGB888)

label = ImageLabel()
label.resize(600, 400)
label.set_image(QPixmap.fromImage(qimg))

fails = []


def check(name, got, want, tol=1e-6):
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got}  (기대 {want})")
    if not ok:
        fails.append(name)


print("맞춤 상태")
# 800x400 into 600x400 -> fits at 600x300, i.e. 75 %
check("zoom_percent", label.zoom_percent(), 75.0, 0.5)
c0 = label._to_image_coords(QPoint(300, 200))
check("중앙 -> image x", c0[0], 400.0, 1.0)
check("중앙 -> image y", c0[1], 200.0, 1.0)

print("\n휠로 확대 (커서 = 중앙)")
wheel = QWheelEvent(QPointF(300, 200), QPointF(300, 200), QPoint(), QPoint(0, 120),
                    Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
label.wheelEvent(wheel)
check("zoom factor", label._zoom, 1.25, 1e-6)
c1 = label._to_image_coords(QPoint(300, 200))
check("커서 아래 지점 유지 x", c1[0], c0[0], 2.0)
check("커서 아래 지점 유지 y", c1[1], c0[1], 2.0)

print("\n휠로 확대 (커서 = 좌상단 근처)")
# Letterboxing pins the short axis until the image outgrows the widget, so use
# a matching aspect ratio to test the anchor on both axes at once.
label.resize(800, 400)
label.reset_view()
anchor = QPoint(200, 150)
before = label._to_image_coords(anchor)
for _ in range(3):
    label.wheelEvent(QWheelEvent(QPointF(anchor), QPointF(anchor), QPoint(), QPoint(0, 120),
                                 Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False))
after = label._to_image_coords(anchor)
check("앵커 고정 x", after[0], before[0], 3.0)
check("앵커 고정 y", after[1], before[1], 3.0)
check("zoom factor", label._zoom, 1.25 ** 3, 1e-6)

print("\n드래그로 이동")
pan_before = QPoint(label._pan)
label.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(300, 200),
                                  Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
label.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, QPointF(360, 200),
                                 Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
label.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, QPointF(360, 200),
                                    Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
check("pan x 이동", label._pan.x() - pan_before.x(), 60.0, 1.0)

print("\n짧은 클릭은 이동이 아니라 클릭")
clicks = []
label.click_callback = lambda x, y: clicks.append((x, y))
label.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(300, 200),
                                  Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
label.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, QPointF(301, 200),
                                 Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
label.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, QPointF(301, 200),
                                    Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
check("클릭 콜백 횟수", len(clicks), 1)

print("\n확대 상태에서 스케일바 측정 좌표")
label.measure_mode = True
measured = []
label.measure_callback = lambda a, b: measured.append((a, b))
p0, p1 = QPoint(200, 200), QPoint(400, 200)
i0, i1 = label._to_image_coords(p0), label._to_image_coords(p1)
label.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(p0),
                                  Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
label.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, QPointF(p1),
                                    Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
check("측정 콜백 횟수", len(measured), 1)
drag_px = measured[0][1][0] - measured[0][0][0]
expected = i1[0] - i0[0]
check("확대 배율 반영된 길이", drag_px, expected, 0.5)
# 800x400 widget now fits 1:1, so 200 widget px = 200 / zoom image px
check("실제 이미지 px", drag_px, 200 / (1.0 * 1.25 ** 3), 1.0)

print("\n최소 배율은 '맞춤' 이하로 내려가지 않음")
label.set_zoom(0.2)
check("clamp", label._zoom, 1.0)

print("\n" + ("전체 통과" if not fails else f"실패: {fails}"))
sys.exit(1 if fails else 0)
