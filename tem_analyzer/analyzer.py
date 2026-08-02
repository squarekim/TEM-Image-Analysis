import cv2
import numpy as np
import re

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


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
        if not HAS_TESSERACT:
            return None
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

    def analyze(self, image, min_area_px=100, max_area_px=None,
                circularity_thresh=0.5, use_watershed=True, hollow=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        cutoff = self._find_scalebar_top(gray)
        analysis_region = gray[:cutoff, :]

        binary = self._preprocess(analysis_region)
        if hollow:
            binary = self._enhance_hollow(binary, analysis_region)
        binary = self._remove_edge_objects(binary)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        particles = []
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

            is_round = circularity >= circularity_thresh
            is_filled = self._is_spherical(cnt, area)

            if is_round and is_filled:
                p = self._measure_single(cnt, area)
                if p:
                    particles.append(p)
            elif use_watershed:
                separated = self._split_overlapping(cnt, binary, analysis_region,
                                                    min_area_px, circularity_thresh)
                particles.extend(separated)

        use_hough = not hollow and len(particles) < 3
        if not use_hough and not hollow and particles:
            px_areas = [np.pi * p["radius_px"] ** 2 for p in particles]
            median_area = np.median(px_areas)
            img_area = analysis_region.shape[0] * analysis_region.shape[1]
            if median_area < img_area * 0.005:
                use_hough = True

        if use_hough:
            hough_particles = self._detect_hough(analysis_region, min_area_px)
            if hough_particles:
                hough_median = np.median([np.pi * p["radius_px"] ** 2 for p in hough_particles])
                contour_median = np.median(px_areas) if particles else 0
                if hough_median > contour_median * 2 or len(hough_particles) > len(particles):
                    particles = hough_particles

        return particles

    def _detect_hough(self, gray, min_area_px):
        h, w = gray.shape
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        min_r = max(10, int(np.sqrt(min_area_px / np.pi)))
        max_r = min(h, w) // 3
        min_dist = max(30, min_r * 3)

        for param2 in [45, 40, 35, 30]:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist,
                param1=80, param2=param2, minRadius=min_r, maxRadius=max_r
            )
            if circles is not None and len(circles[0]) >= 3:
                particles = []
                for cx, cy, r in circles[0]:
                    cx, cy, r = int(cx), int(cy), int(r)
                    inside_x = max(0, min(cx + r, w) - max(cx - r, 0))
                    inside_y = max(0, min(cy + r, h) - max(cy - r, 0))
                    if inside_x < r or inside_y < r:
                        continue
                    area_px = np.pi * r * r
                    p = self._measure_circle(cx, cy, r, area_px)
                    particles.append(p)
                return particles

        return []

    def _measure_circle(self, cx, cy, radius, area_px):
        diameter_px = radius * 2
        if self.nm_per_px:
            diameter_nm = diameter_px * self.nm_per_px
            area_nm2 = area_px * (self.nm_per_px ** 2)
        else:
            diameter_nm = diameter_px
            area_nm2 = area_px
        return {
            "center_x": cx,
            "center_y": cy,
            "diameter_px": diameter_px,
            "radius_px": radius,
            "diameter": diameter_nm,
            "area": area_nm2,
            "contour": None,
        }

    @staticmethod
    def _is_spherical(cnt, area):
        _, radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * radius * radius
        fill_ratio = area / circle_area if circle_area > 0 else 0
        return fill_ratio > 0.75

    @staticmethod
    def _find_scalebar_top(gray):
        h, w = gray.shape
        for row in range(h - 1, int(h * 0.7), -1):
            line = gray[row, :]
            dark_ratio = np.sum(line < 30) / w
            if dark_ratio > 0.5:
                return row
        mean_brightness = np.mean(gray, axis=1)
        for row in range(h - 1, int(h * 0.75), -1):
            if mean_brightness[row] < mean_brightness[int(h * 0.5)] * 0.6:
                for start_row in range(row, int(h * 0.75), -1):
                    if mean_brightness[start_row] >= mean_brightness[int(h * 0.5)] * 0.8:
                        return start_row
                return row
        return int(h * 0.9)

    @staticmethod
    def _enhance_hollow(binary, gray):
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(denoised, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        combined = cv2.bitwise_or(binary, edges)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_open, iterations=2)

        filled = ParticleAnalyzer._fill_holes(combined)
        filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel_open, iterations=1)
        return filled

    @staticmethod
    def _fill_holes(binary):
        h, w = binary.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        inv = cv2.bitwise_not(binary)
        cv2.floodFill(inv, flood_mask, (0, 0), 0)
        return cv2.bitwise_or(binary, inv)

    def _preprocess(self, gray):
        h, w = gray.shape
        small_image = min(h, w) < 500

        if small_image:
            denoised = cv2.GaussianBlur(gray, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        else:
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        enhanced = clahe.apply(denoised)

        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if small_image:
            binary = otsu
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        else:
            adaptive = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 5
            )
            binary = cv2.bitwise_or(otsu, adaptive)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        return binary

    def _remove_edge_objects(self, binary):
        h, w = binary.shape
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = binary.copy()
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
                cv2.drawContours(mask, [cnt], -1, 0, -1)
        return mask

    def _split_overlapping(self, cnt, binary, gray, min_area_px, circularity_thresh):
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_val = dist.max()
        if max_val < 5:
            return []

        smoothed = cv2.GaussianBlur(dist, (0, 0), sigmaX=max(1, max_val * 0.15))
        ksize = max(3, int(max_val * 0.5)) | 1
        dilated = cv2.dilate(smoothed, np.ones((ksize, ksize)))
        local_max = (smoothed == dilated) & (smoothed > max_val * 0.2) & (mask > 0)
        local_max = local_max.astype(np.uint8)

        num_peaks, peak_labels = cv2.connectedComponents(local_max)
        if num_peaks <= 2:
            return []

        sure_fg = np.zeros_like(binary)
        for label_id in range(1, num_peaks):
            peak_mask = (peak_labels == label_id).astype(np.uint8)
            kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            peak_mask = cv2.dilate(peak_mask, kernel_sm, iterations=2)
            sure_fg[peak_mask > 0] = 255

        sure_bg = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers[sure_bg == 0] = 1

        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(gray.shape) == 2 else gray.copy()
        cv2.watershed(color, markers)

        expected_particle_area = np.pi * (max_val * 0.8) ** 2

        results = []
        for label_id in range(2, num_labels + 1):
            seg_mask = np.uint8(markers == label_id) * 255
            seg_contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not seg_contours:
                continue
            seg_cnt = max(seg_contours, key=cv2.contourArea)
            seg_area = cv2.contourArea(seg_cnt)
            if seg_area < min_area_px:
                continue
            if seg_area < expected_particle_area * 0.15:
                continue
            seg_perim = cv2.arcLength(seg_cnt, True)
            if seg_perim == 0:
                continue
            circ = 4 * np.pi * seg_area / (seg_perim * seg_perim)
            if circ < circularity_thresh * 0.7:
                continue
            p = self._measure_single(seg_cnt, seg_area)
            if p:
                results.append(p)

        return results

    def _measure_single(self, cnt, area_px):
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        diameter_px = radius * 2

        if self.nm_per_px:
            diameter_nm = diameter_px * self.nm_per_px
            area_nm2 = area_px * (self.nm_per_px ** 2)
        else:
            diameter_nm = diameter_px
            area_nm2 = area_px

        return {
            "center_x": int(cx),
            "center_y": int(cy),
            "diameter_px": diameter_px,
            "radius_px": radius,
            "diameter": diameter_nm,
            "area": area_nm2,
            "contour": cnt,
        }

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
