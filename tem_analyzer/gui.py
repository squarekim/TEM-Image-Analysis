import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QSplitter,
    QMessageBox, QStatusBar, QHeaderView, QComboBox, QCheckBox,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import openpyxl

from .analyzer import ScaleBarDetector, ParticleAnalyzer, HAS_TESSERACT


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 1px solid #ccc; background: #222;")
        self._pixmap = None

    def set_image(self, pixmap):
        self._pixmap = pixmap
        self._update_display()

    def _update_display(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled)

    def resizeEvent(self, event):
        self._update_display()
        super().resizeEvent(event)


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
        left_layout.addLayout(btn_layout)

        self.image_label = ImageLabel()
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
        self.chk_manual.toggled.connect(self._toggle_manual_scale)
        if not HAS_TESSERACT:
            self.chk_manual.setChecked(True)
        scale_form.addRow(self.chk_manual)

        self.spin_bar_px = QDoubleSpinBox()
        self.spin_bar_px.setRange(1, 10000)
        self.spin_bar_px.setValue(100)
        self.spin_bar_px.setEnabled(False)
        scale_form.addRow("스케일바 길이 (px):", self.spin_bar_px)

        self.spin_bar_real = QDoubleSpinBox()
        self.spin_bar_real.setRange(0.01, 100000)
        self.spin_bar_real.setValue(100)
        self.spin_bar_real.setEnabled(False)
        scale_form.addRow("실제 길이:", self.spin_bar_real)

        self.combo_unit = QComboBox()
        self.combo_unit.addItems(["nm", "μm", "mm"])
        self.combo_unit.setEnabled(False)
        scale_form.addRow("단위:", self.combo_unit)

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
        self.chk_hollow.setChecked(False)
        self.chk_hollow.setToolTip("속이 빈 입자 (실리카 등)의 링 형태를 채워서 검출")
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
        right_layout.addWidget(self.table)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def _toggle_manual_scale(self, checked):
        self.spin_bar_px.setEnabled(checked)
        self.spin_bar_real.setEnabled(checked)
        self.combo_unit.setEnabled(checked)
        if checked:
            self.scale_status.setText("수동 입력 모드")

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "TEM 이미지 열기", "",
            "이미지 파일 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;모든 파일 (*)"
        )
        if not path:
            return

        self.image = cv2.imread(path)
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
        self.statusBar().showMessage(f"이미지 로드 완료: {os.path.basename(path)}")

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

        self._draw_results()
        stats = ParticleAnalyzer.compute_statistics(self.particles)
        self._update_stats(stats)
        self._update_table()
        self._update_histogram()
        self.btn_export.setEnabled(bool(self.particles))
        self.statusBar().showMessage(f"분석 완료: {len(self.particles)}개 입자 검출")

    def _draw_results(self):
        display = self.original_image.copy()
        for i, p in enumerate(self.particles):
            cx, cy = p["center_x"], p["center_y"]
            r = int(p["radius_px"])
            cv2.circle(display, (cx, cy), r, (0, 255, 0), 2)
            if p.get("has_core"):
                cv2.circle(display, (cx, cy), int(p["core_radius_px"]), (0, 165, 255), 2)
            cv2.putText(display, str(i + 1), (cx - 10, cy - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        self._display_image(display)

    def _display_image(self, cv_img):
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
            txt = f"{stats['core_count']}/{stats['count']} ({stats['core_ratio']*100:.1f}%)"
            if "core_mean" in stats:
                txt += f", 평균 {stats['core_mean']:.2f} {u}"
            self.lbl_core.setText(txt)
        else:
            self.lbl_core.setText("-")

    def _update_table(self):
        has_core_col = bool(self.particles) and "has_core" in self.particles[0]
        if has_core_col:
            self.table.setColumnCount(4)
            self.table.setHorizontalHeaderLabels(["#", "직경", "면적", "코어"])
        else:
            self.table.setColumnCount(3)
            self.table.setHorizontalHeaderLabels(["#", "직경", "면적"])
        self.table.setRowCount(len(self.particles))
        u = self.unit
        for i, p in enumerate(self.particles):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QTableWidgetItem(f"{p['diameter']:.2f} {u}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{p['area']:.2f} {u}²"))
            if has_core_col:
                core_txt = f"{p['core_diameter']:.2f} {u}" if p["has_core"] else "없음"
                self.table.setItem(i, 3, QTableWidgetItem(core_txt))

    def _update_histogram(self):
        diameters = [p["diameter"] for p in self.particles]
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
        has_core = bool(self.particles) and "has_core" in self.particles[0]
        headers = ["#", f"Diameter ({u})", f"Area ({u}²)", "Center X (px)", "Center Y (px)"]
        if has_core:
            headers += ["Has Core", f"Core Diameter ({u})"]
        ws_data.append(headers)
        for i, p in enumerate(self.particles):
            row = [i + 1, p["diameter"], p["area"], p["center_x"], p["center_y"]]
            if has_core:
                row += ["Y" if p["has_core"] else "N",
                        p["core_diameter"] if p["has_core"] else None]
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
            if "core_mean" in stats:
                ws_stats.append(["Core Mean Diameter", stats["core_mean"], u])

        if self.nm_per_px:
            ws_stats.append([])
            ws_stats.append(["Scale", f"{self.nm_per_px:.4f}", "nm/px"])

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
