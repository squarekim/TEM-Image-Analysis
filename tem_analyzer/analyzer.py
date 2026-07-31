import cv2
import numpy as np
import pytesseract
import re


class ScaleBarDetector:
    def detect(self, image):
        h, w = image.shape[:2]
        roi = image[int(h * 0.85):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        scale_text = self._extract_scale_text(gray)
        if scale_text is None:
            return None, None

        scale_value, unit = scale_text
        bar_length_px = self._find_bar_length(gray)
        if bar_length_px is None:
            return None, None

        nm_per_px = self._to_nm(scale_value, unit) / bar_length_px
        return nm_per_px, f"{scale_value} {unit}"

    def _extract_scale_text(self, gray_roi):
        configs = [
            "--psm 7 -c tessedit_char_whitelist=0123456789.numkμµ ",
            "--psm 7",
            "--psm 6",
        ]
        for config in configs:
            try:
                text = pytesseract.image_to_string(gray_roi, config=config).strip()
                result = self._parse_scale_text(text)
                if result:
                    return result
            except Exception:
                continue
        return None

    def _parse_scale_text(self, text):
        text = text.replace("µ", "μ").replace("u", "μ").lower()
        pattern = r"(\d+\.?\d*)\s*(nm|μm|mm|um)"
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit == "um":
                unit = "μm"
            return value, unit
        return None

    def _to_nm(self, value, unit):
        conversions = {"nm": 1, "μm": 1000, "mm": 1e6}
        return value * conversions.get(unit, 1)

    def _find_bar_length(self, gray_roi):
        _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)
            if aspect > 5 and cw > 30:
                if best is None or cw > best:
                    best = cw
        return best


class ParticleAnalyzer:
    def __init__(self, nm_per_px=None):
        self.nm_per_px = nm_per_px

    def analyze(self, image, min_area_px=50, max_area_px=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        analysis_region = gray[:int(h * 0.85), :]

        preprocessed = self._preprocess(analysis_region)
        contours = self._find_contours(preprocessed, min_area_px, max_area_px)
        particles = self._measure_particles(contours)
        return particles

    def _preprocess(self, gray):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 5
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        return binary

    def _find_contours(self, binary, min_area_px, max_area_px):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_px:
                continue
            if max_area_px and area > max_area_px:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.3:
                continue
            filtered.append(cnt)
        return filtered

    def _measure_particles(self, contours):
        particles = []
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            diameter_px = radius * 2

            if self.nm_per_px:
                diameter_nm = diameter_px * self.nm_per_px
                area_nm2 = area_px * (self.nm_per_px ** 2)
            else:
                diameter_nm = diameter_px
                area_nm2 = area_px

            particles.append({
                "center_x": int(cx),
                "center_y": int(cy),
                "diameter_px": diameter_px,
                "radius_px": radius,
                "diameter": diameter_nm,
                "area": area_nm2,
                "contour": cnt,
            })
        return particles

    @staticmethod
    def compute_statistics(particles):
        if not particles:
            return {}
        diameters = np.array([p["diameter"] for p in particles])
        diameters_sorted = np.sort(diameters)
        return {
            "count": len(diameters),
            "mean": float(np.mean(diameters)),
            "std": float(np.std(diameters)),
            "min": float(np.min(diameters)),
            "max": float(np.max(diameters)),
            "d10": float(np.percentile(diameters_sorted, 10)),
            "d50": float(np.percentile(diameters_sorted, 50)),
            "d90": float(np.percentile(diameters_sorted, 90)),
        }
