import sys
import os

from . import qt_bootstrap

qt_bootstrap.configure()  # must run before Qt is loaded

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QSplitter,
    QMessageBox, QHeaderView, QComboBox, QCheckBox,
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import openpyxl

from .analyzer import (
    ScaleBarDetector, ParticleAnalyzer, HAS_TESSERACT, load_image, save_image,
)


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 1px solid #ccc; background: #222;")
        self._pixmap = None
        self.click_callback = None
        self.measure_callback = None
        self.measure_mode = False
        self._drag_start = None
        self._drag_end = None

    def set_image(self, pixmap):
        self._pixmap = pixmap
        self._drag_start = self._drag_end = None
        self._update_display()

    def _scaled(self):
        return self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    def _update_display(self):
        if not self._pixmap:
            return
        scaled = self._scaled()
        if self._drag_start and self._drag_end:
            scaled = scaled.copy()
            painter = QPainter(scaled)
            off = self._offset(scaled)
            pen = QPen(QColor(255, 60, 60), 2)
            painter.setPen(pen)
            a = self._drag_start - off
            b = self._drag_end - off
            painter.drawLine(a, b)
            for end in (a, b):
                painter.drawLine(end.x(), end.y() - 6, end.x(), end.y() + 6)
            painter.end()
        super().setPixmap(scaled)

    def _offset(self, scaled):
        return QPoint(int((self.width() - scaled.width()) / 2),
                      int((self.height() - scaled.height()) / 2))

    def _to_image_coords(self, pos):
        scaled = self._scaled()
        off = self._offset(scaled)
        px, py = pos.x() - off.x(), pos.y() - off.y()
        if not (0 <= px < scaled.width() and 0 <= py < scaled.height()):
            return None
        return (px * self._pixmap.width() / scaled.width(),
                py * self._pixmap.height() / scaled.height())

    def resizeEvent(self, event):
        self._update_display()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self._pixmap and self.measure_mode:
            self._drag_start = event.pos()
            self._drag_end = event.pos()
            self._update_display()
        elif self._pixmap and self.click_callback:
            coords = self._to_image_coords(event.pos())
            if coords:
                self.click_callback(*coords)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.measure_mode and self._drag_start:
            self._drag_end = event.pos()
            self._update_display()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.measure_mode and self._drag_start:
            self._drag_end = event.pos()
            self._update_display()
            start = self._to_image_coords(self._drag_start)
            end = self._to_image_coords(self._drag_end)
            if start and end and self.measure_callback:
                self.measure_callback(start, end)
        super().mouseReleaseEvent(event)


class HistogramCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setMinimumHeight(250)

    def plot(self, diameters, unit="nm"):
        self.ax.clear()
        if not diameters:
            self.draw()
            return

        n_bins = max(5, min(30, len(diameters) // 3))
        self.ax.hist(diameters, bins=n_bins, color="#4A90D9", edgecolor="#2C5F8A", alpha=0.85)
        self.ax.set_xlabel(f"Diameter ({unit})", fontsize=10)
        self.ax.set_ylabel("Count", fontsize=10)
        self.ax.set_title("Particle Size Distribution", fontsize=11, fontweight="bold")

        mean_val = np.mean(diameters)
        self.ax.axvline(mean_val, color="#E74C3C", linestyle="--", linewidth=1.5,
                        label=f"Mean: {mean_val:.1f} {unit}")
        self.ax.legend(fontsize=9)
        self.fig.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TEM Particle Analyzer")
        self.setMinimumSize(1200, 800)
        self.image = None
        self.original_image = None
        self.particles = []
        self.result_image = None
        self.nm_per_px = None
        self.scale_text = None
        self.unit = "nm"

        self._build_ui()
        self.statusBar().showMessage("이미지를 로드해주세요.")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("이미지 열기")
        self.btn_load.setMinimumHeight(36)
        self.btn_load.clicked.connect(self._load_image)
        btn_layout.addWidget(self.btn_load)

        self.btn_analyze = QPushButton("분석 실행")
        self.btn_analyze.setMinimumHeight(36)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self._run_analysis)
        btn_layout.addWidget(self.btn_analyze)

        self.btn_export = QPushButton("Excel 내보내기")
        self.btn_export.setMinimumHeight(36)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_excel)
        btn_layout.addWidget(self.btn_export)

        self.btn_save_image = QPushButton("결과 이미지 저장")
        self.btn_save_image.setMinimumHeight(36)
        self.btn_save_image.setEnabled(False)
        self.btn_save_image.clicked.connect(self._save_result_image)
        btn_layout.addWidget(self.btn_save_image)
        left_layout.addLayout(btn_layout)

        self.image_label = ImageLabel()
        self.image_label.click_callback = self._on_image_click
        self.image_label.measure_callback = self._on_scalebar_measured
        left_layout.addWidget(self.image_label, stretch=3)

        self.histogram = HistogramCanvas()
        left_layout.addWidget(self.histogram, stretch=1)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        scale_group = QGroupBox("스케일바 설정")
        scale_form = QFormLayout()

        if HAS_TESSERACT:
            self.scale_status = QLabel("자동 감지 대기 중")
        else:
            self.scale_status = QLabel("OCR 미설치 - 수동 입력 모드")
        scale_form.addRow("상태:", self.scale_status)

        self.chk_manual = QCheckBox("수동 입력 사용")
        scale_form.addRow(self.chk_manual)

        self.spin_bar_px = QDoubleSpinBox()
        self.spin_bar_px.setRange(1, 10000)
        self.spin_bar_px.setValue(100)
        self.spin_bar_px.setEnabled(False)
        scale_form.addRow("스케일바 길이 (px):", self.spin_bar_px)

        self.btn_measure = QPushButton("이미지에서 스케일바 재기")
        self.btn_measure.setCheckable(True)
        self.btn_measure.setEnabled(False)
        self.btn_measure.setToolTip(
            "누른 뒤 이미지 위의 스케일바 양 끝을 드래그하면 픽셀 길이가 자동 입력됩니다")
        self.btn_measure.toggled.connect(self._toggle_measure_mode)
        scale_form.addRow(self.btn_measure)

        self.spin_bar_real = QDoubleSpinBox()
        self.spin_bar_real.setRange(0.01, 100000)
        self.spin_bar_real.setValue(100)
        self.spin_bar_real.setEnabled(False)
        scale_form.addRow("실제 길이:", self.spin_bar_real)

        self.combo_unit = QComboBox()
        self.combo_unit.addItems(["nm", "μm", "mm"])
        self.combo_unit.setEnabled(False)
        scale_form.addRow("단위:", self.combo_unit)

        # Connect only once the widgets the handler touches exist: checking the
        # box emits toggled immediately, and without OCR that fired before the
        # spin boxes were built.
        self.chk_manual.toggled.connect(self._toggle_manual_scale)
        if not HAS_TESSERACT:
            self.chk_manual.setChecked(True)

        scale_group.setLayout(scale_form)
        right_layout.addWidget(scale_group)

        param_group = QGroupBox("분석 파라미터")
        param_form = QFormLayout()

        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(10, 10000)
        self.spin_min_area.setValue(100)
        param_form.addRow("최소 면적 (px):", self.spin_min_area)

        self.spin_max_area = QSpinBox()
        self.spin_max_area.setRange(0, 1000000)
        self.spin_max_area.setValue(0)
        self.spin_max_area.setSpecialValueText("제한 없음")
        param_form.addRow("최대 면적 (px):", self.spin_max_area)

        self.spin_circularity = QDoubleSpinBox()
        self.spin_circularity.setRange(0.1, 1.0)
        self.spin_circularity.setValue(0.5)
        self.spin_circularity.setSingleStep(0.05)
        param_form.addRow("원형도 필터:", self.spin_circularity)

        self.chk_watershed = QCheckBox("겹침 분리 (Watershed)")
        self.chk_watershed.setChecked(True)
        param_form.addRow(self.chk_watershed)

        self.chk_hollow = QCheckBox("Hollow 입자 모드")
        self.chk_hollow.setChecked(True)
        self.chk_hollow.setToolTip(
            "속이 빈 입자(실리카 등)의 링 형태를 채워서 검출합니다.\n"
            "일반 입자에서도 손해가 없으므로 켜 두어도 됩니다.")
        param_form.addRow(self.chk_hollow)

        self.chk_core = QCheckBox("코어 검출 (Yolk-shell)")
        self.chk_core.setChecked(False)
        self.chk_core.setToolTip("중공 입자 내부의 코어 입자를 감지하고 보유 비율을 계산")
        param_form.addRow(self.chk_core)

        param_group.setLayout(param_form)
        right_layout.addWidget(param_group)

        stats_group = QGroupBox("통계 결과")
        stats_form = QFormLayout()
        self.lbl_count = QLabel("-")
        self.lbl_mean = QLabel("-")
        self.lbl_std = QLabel("-")
        self.lbl_min = QLabel("-")
        self.lbl_max = QLabel("-")
        self.lbl_d10 = QLabel("-")
        self.lbl_d50 = QLabel("-")
        self.lbl_d90 = QLabel("-")
        self.lbl_core = QLabel("-")
        stats_form.addRow("입자 수:", self.lbl_count)
        stats_form.addRow("평균 직경:", self.lbl_mean)
        stats_form.addRow("표준편차:", self.lbl_std)
        stats_form.addRow("최소:", self.lbl_min)
        stats_form.addRow("최대:", self.lbl_max)
        stats_form.addRow("D10:", self.lbl_d10)
        stats_form.addRow("D50:", self.lbl_d50)
        stats_form.addRow("D90:", self.lbl_d90)
        stats_form.addRow("코어 보유율:", self.lbl_core)
        stats_group.setLayout(stats_form)
        right_layout.addWidget(stats_group)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["#", "직경", "면적"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSortingEnabled(True)
        right_layout.addWidget(self.table)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def _toggle_manual_scale(self, checked):
        self.spin_bar_px.setEnabled(checked)
        self.spin_bar_real.setEnabled(checked)
        self.combo_unit.setEnabled(checked)
        self.btn_measure.setEnabled(checked and self.original_image is not None)
        if not checked:
            self.btn_measure.setChecked(False)
        if checked:
            self.scale_status.setText("수동 입력 모드")

    def _toggle_measure_mode(self, checked):
        self.image_label.measure_mode = checked
        self.image_label.setCursor(Qt.CrossCursor if checked else Qt.ArrowCursor)
        if checked:
            self.statusBar().showMessage(
                "이미지 위의 스케일바 왼쪽 끝에서 오른쪽 끝까지 드래그하세요.")

    def _on_scalebar_measured(self, start, end):
        scale = getattr(self, "_display_scale", 1.0) or 1.0
        p0 = (start[0] / scale, start[1] / scale)
        p1 = (end[0] / scale, end[1] / scale)
        length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if length < 2:
            return

        # A couple of pixels of drag error multiply into every diameter, so
        # snap to the bar actually under the drag where one can be found.
        snapped = ParticleAnalyzer.snap_scalebar(self.original_image, p0, p1)
        real = self.spin_bar_real.value()
        unit = self.combo_unit.currentText()
        if snapped:
            self.spin_bar_px.setValue(snapped)
            note = f"드래그 {length:.1f} px → 스케일바에 맞춤 {snapped:.1f} px."
        else:
            self.spin_bar_px.setValue(length)
            note = (f"스케일바 길이 {length:.1f} px (드래그 값 그대로 사용 — "
                    "이미지에서 막대를 찾지 못했습니다).")
        self.btn_measure.setChecked(False)
        self.statusBar().showMessage(
            f"{note}  '실제 길이'가 {real:g} {unit}가 맞는지 확인하세요.")

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "TEM 이미지 열기", "",
            "이미지 파일 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;모든 파일 (*)"
        )
        if not path:
            return

        self.image = load_image(path)
        if self.image is None:
            QMessageBox.warning(self, "오류", "이미지를 읽을 수 없습니다.")
            return

        self.original_image = self.image.copy()

        if not self.chk_manual.isChecked():
            detector = ScaleBarDetector()
            nm_per_px, scale_text = detector.detect(self.image)
            if nm_per_px:
                self.nm_per_px = nm_per_px
                self.scale_text = scale_text
                self.scale_status.setText(f"감지됨: {scale_text} ({nm_per_px:.4f} nm/px)")
                self.statusBar().showMessage(f"스케일바 자동 감지: {scale_text}")
            else:
                self.scale_status.setText("자동 감지 실패 - 수동 입력 필요")
                self.chk_manual.setChecked(True)
                self.statusBar().showMessage("스케일바 자동 감지 실패. 수동으로 입력해주세요.")

        self._display_image(self.image)
        self.btn_analyze.setEnabled(True)
        self.btn_measure.setEnabled(self.chk_manual.isChecked())
        self.statusBar().showMessage(
            f"이미지 로드 완료: {os.path.basename(path)}  —  "
            "'이미지에서 스케일바 재기'로 배율을 먼저 맞추세요.")

    def _run_analysis(self):
        if self.image is None:
            return

        if self.chk_manual.isChecked():
            bar_px = self.spin_bar_px.value()
            bar_real = self.spin_bar_real.value()
            unit = self.combo_unit.currentText()
            unit_to_nm = {"nm": 1, "μm": 1000, "mm": 1e6}
            self.nm_per_px = (bar_real * unit_to_nm[unit]) / bar_px
            self.unit = "nm"

        max_area = self.spin_max_area.value()
        if max_area == 0:
            max_area = None

        # Analysis runs on the UI thread and takes seconds on full-resolution
        # TEM images, so make the wait visible instead of looking frozen.
        self.statusBar().showMessage("분석 중...")
        self.btn_analyze.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            analyzer = ParticleAnalyzer(nm_per_px=self.nm_per_px)
            self.particles = analyzer.analyze(
                self.original_image,
                min_area_px=self.spin_min_area.value(),
                max_area_px=max_area,
                circularity_thresh=self.spin_circularity.value(),
                use_watershed=self.chk_watershed.isChecked(),
                hollow=self.chk_hollow.isChecked(),
                detect_cores=self.chk_core.isChecked(),
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_analyze.setEnabled(True)

        self._draw_results()
        stats = ParticleAnalyzer.compute_statistics(self.particles)
        self._update_stats(stats)
        self._update_table()
        self._update_histogram()
        valid = self._valid_particles()
        self.btn_export.setEnabled(bool(valid))
        n_exc = len(self.particles) - len(valid)
        n_approx = sum(1 for p in valid if p.get("approx"))
        msg = f"분석 완료: {len(valid)}개 입자 검출"
        if n_approx:
            msg += f" (근사 {n_approx}개 포함)"
        if n_exc:
            msg += f", 판별불가 {n_exc}개 제외"
        n_overlap, _ = self._overlapping_pairs()
        if n_overlap:
            msg += f"  |  겹치는 검출 {n_overlap}쌍 (보라색) - 중복 여부 확인 필요"
        self.statusBar().showMessage(msg)

    def _valid_particles(self):
        return [p for p in self.particles if not p.get("excluded")]

    def _overlapping_pairs(self):
        """Valid detections that sit substantially on top of one another.

        Two circles covering the same particle inflate the count without
        looking obviously wrong in a crowded field, so they are called out
        rather than left to be spotted by eye.
        """
        valid = self._valid_particles()
        flagged = set()
        pairs = 0
        for i, p in enumerate(valid):
            for j in range(i + 1, len(valid)):
                q = valid[j]
                d = np.hypot(p["center_x"] - q["center_x"],
                             p["center_y"] - q["center_y"])
                if d < 0.55 * (p["radius_px"] + q["radius_px"]):
                    pairs += 1
                    flagged.add(id(p))
                    flagged.add(id(q))
        return pairs, flagged

    def _draw_results(self):
        h, w = self.original_image.shape[:2]
        scale = max(1, int(round(900 / max(h, w))))
        self._result_scale = scale
        display = cv2.resize(self.original_image, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC) if scale > 1 else self.original_image.copy()
        thickness = max(2, scale)
        num = 0
        for p in self.particles:
            cx, cy = p["center_x"] * scale, p["center_y"] * scale
            r = int(p["radius_px"]) * scale
            excluded = p.get("excluded", False)
            if excluded:
                color = (0, 0, 255)
            elif p.get("approx"):
                color = (0, 220, 220)
            else:
                color = (0, 220, 0)
            if p.get("contour") is not None:
                scaled_cnt = (p["contour"].astype(np.int32) * scale)
                self._draw_boundary(display, scaled_cnt, r, color, thickness)
            else:
                cv2.circle(display, (cx, cy), r, color, thickness)
            if excluded:
                continue
            num += 1
            if p.get("has_core"):
                cv2.drawMarker(display, (cx, cy), (0, 165, 255),
                               cv2.MARKER_CROSS, max(10, r // 3), thickness)
            txt = str(num)
            pos = (cx - 8 * len(txt), cy - 6)
            font_scale = 0.45 + 0.1 * scale
            cv2.putText(display, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), 3)
            cv2.putText(display, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (80, 255, 255), 1)

        legend = [("OK", (0, 220, 0)), ("Approx", (0, 220, 220)),
                  ("Overlap", (255, 0, 255)), ("Excluded", (0, 0, 255))]
        lx = 10
        for label, color in legend:
            cv2.rectangle(display, (lx, 10), (lx + 18, 28), color, -1)
            cv2.putText(display, label, (lx + 24, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3)
            cv2.putText(display, label, (lx + 24, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)
            lx += 24 + 13 * len(label) + 16
        self.result_image = display
        self.btn_save_image.setEnabled(True)
        self._display_image(display)

    @staticmethod
    def _draw_boundary(display, contour, r, color, thickness):
        pts = contour.reshape(-1, 2)
        max_gap = max(6, r * 0.35)
        segment = [pts[0]]
        segments = []
        for q in pts[1:]:
            if np.hypot(q[0] - segment[-1][0], q[1] - segment[-1][1]) <= max_gap:
                segment.append(q)
            else:
                segments.append(segment)
                segment = [q]
        if np.hypot(pts[0][0] - segment[-1][0], pts[0][1] - segment[-1][1]) <= max_gap and segments:
            segments[0] = segment + segments[0]
        else:
            segments.append(segment)
        for seg in segments:
            if len(seg) >= 2:
                cv2.polylines(display, [np.array(seg, np.int32).reshape(-1, 1, 2)],
                              False, color, thickness)

    def _on_image_click(self, ix, iy):
        if not self.particles or "has_core" not in (self.particles[0] if self.particles else {}):
            return
        scale = getattr(self, "_display_scale", 1.0) or 1.0
        ox, oy = ix / scale, iy / scale
        best = None
        best_d = None
        for p in self.particles:
            if p.get("excluded"):
                continue
            d = np.hypot(p["center_x"] - ox, p["center_y"] - oy)
            if d <= p["radius_px"] and (best_d is None or d < best_d):
                best = p
                best_d = d
        if best is None:
            return
        best["has_core"] = not best["has_core"]
        self._draw_results()
        stats = ParticleAnalyzer.compute_statistics(self.particles)
        self._update_stats(stats)
        self._update_table()
        state = "체크" if best["has_core"] else "해제"
        self.statusBar().showMessage(f"코어 {state}: ({best['center_x']}, {best['center_y']}) 입자")

    def _save_result_image(self):
        if self.result_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "결과 이미지 저장", "particle_result.png",
            "PNG 이미지 (*.png);;JPEG 이미지 (*.jpg)"
        )
        if not path:
            return
        if not save_image(path, self.result_image):
            QMessageBox.warning(self, "오류", "이미지를 저장할 수 없습니다.")
            return
        self.statusBar().showMessage(f"결과 이미지 저장 완료: {path}")

    def _display_image(self, cv_img):
        # The result view is rendered at an integer upscale, so remember the
        # factor that maps what is on screen back to original image pixels.
        if self.original_image is not None:
            self._display_scale = cv_img.shape[1] / self.original_image.shape[1]
        else:
            self._display_scale = 1.0
        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            qimg = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.set_image(QPixmap.fromImage(qimg))

    def _update_stats(self, stats):
        if not stats:
            return
        u = self.unit
        self.lbl_count.setText(str(stats["count"]))
        self.lbl_mean.setText(f"{stats['mean']:.2f} {u}")
        self.lbl_std.setText(f"{stats['std']:.2f} {u}")
        self.lbl_min.setText(f"{stats['min']:.2f} {u}")
        self.lbl_max.setText(f"{stats['max']:.2f} {u}")
        self.lbl_d10.setText(f"{stats['d10']:.2f} {u}")
        self.lbl_d50.setText(f"{stats['d50']:.2f} {u}")
        self.lbl_d90.setText(f"{stats['d90']:.2f} {u}")
        if "core_ratio" in stats:
            self.lbl_core.setText(
                f"{stats['core_count']}/{stats['count']} ({stats['core_ratio']*100:.1f}%)")
        else:
            self.lbl_core.setText("-")

    @staticmethod
    def _numeric_item(value, text):
        item = QTableWidgetItem()
        item.setData(Qt.DisplayRole, float(f"{value:.2f}"))
        item.setToolTip(text)
        return item

    def _update_table(self):
        particles = self._valid_particles()
        has_core_col = bool(particles) and "has_core" in particles[0]
        # Pixel diameter is shown next to the calibrated value so a mismatch
        # with hand measurements can be pinned on the scale or on the sizing.
        if has_core_col:
            self.table.setColumnCount(5)
            self.table.setHorizontalHeaderLabels(["#", "직경", "직경(px)", "면적", "코어"])
        else:
            self.table.setColumnCount(4)
            self.table.setHorizontalHeaderLabels(["#", "직경", "직경(px)", "면적"])
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(particles))
        u = self.unit
        for i, p in enumerate(particles):
            num = str(i + 1) + (" (근사)" if p.get("approx") else "")
            self.table.setItem(i, 0, self._numeric_item(i + 1, num))
            self.table.setItem(i, 1, self._numeric_item(p["diameter"], f"{p['diameter']:.2f} {u}"))
            self.table.setItem(i, 2, self._numeric_item(p["radius_px"] * 2, "px"))
            self.table.setItem(i, 3, self._numeric_item(p["area"], f"{u}²"))
            if has_core_col:
                self.table.setItem(i, 4, QTableWidgetItem("유" if p["has_core"] else "무"))
        self.table.setSortingEnabled(True)

    def _update_histogram(self):
        diameters = [p["diameter"] for p in self._valid_particles()]
        self.histogram.plot(diameters, self.unit)

    def _export_excel(self):
        if not self.particles:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Excel로 내보내기", "particle_analysis.xlsx",
            "Excel 파일 (*.xlsx)"
        )
        if not path:
            return

        wb = openpyxl.Workbook()
        ws_data = wb.active
        ws_data.title = "Particle Data"

        u = self.unit
        particles = self._valid_particles()
        has_core = bool(particles) and "has_core" in particles[0]
        headers = ["#", f"Diameter ({u})", "Diameter (px)", f"Area ({u}²)",
                   "Center X (px)", "Center Y (px)"]
        headers += ["Approx"]
        if has_core:
            headers += ["Has Core"]
        ws_data.append(headers)
        for i, p in enumerate(particles):
            row = [i + 1, p["diameter"], p["radius_px"] * 2, p["area"],
                   p["center_x"], p["center_y"]]
            row += ["Y" if p.get("approx") else "N"]
            if has_core:
                row += ["Y" if p["has_core"] else "N"]
            ws_data.append(row)

        ws_stats = wb.create_sheet("Statistics")
        stats = ParticleAnalyzer.compute_statistics(self.particles)
        ws_stats.append(["Metric", "Value", "Unit"])
        ws_stats.append(["Count", stats["count"], ""])
        ws_stats.append(["Mean Diameter", stats["mean"], u])
        ws_stats.append(["Std Dev", stats["std"], u])
        ws_stats.append(["Min", stats["min"], u])
        ws_stats.append(["Max", stats["max"], u])
        ws_stats.append(["D10", stats["d10"], u])
        ws_stats.append(["D50", stats["d50"], u])
        ws_stats.append(["D90", stats["d90"], u])
        if "core_ratio" in stats:
            ws_stats.append(["Core Count", stats["core_count"], ""])
            ws_stats.append(["Core Ratio", stats["core_ratio"], ""])

        if self.nm_per_px:
            ws_stats.append([])
            ws_stats.append(["Scale", f"{self.nm_per_px:.4f}", "nm/px"])

        if self.result_image is not None:
            try:
                import tempfile
                from openpyxl.drawing.image import Image as XLImage
                # Close the handle before writing: Windows will not let the
                # image be written while the temp file is still open.
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.close()
                if save_image(tmp.name, self.result_image):
                    ws_img = wb.create_sheet("Result Image")
                    ws_img.add_image(XLImage(tmp.name), "A1")
            except Exception:
                pass

        wb.save(path)
        self.statusBar().showMessage(f"Excel 저장 완료: {path}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
