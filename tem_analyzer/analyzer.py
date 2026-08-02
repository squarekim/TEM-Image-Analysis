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
                circularity_thresh=0.5, use_watershed=True, hollow=False,
                detect_cores=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        cutoff = self._find_scalebar_top(gray)
        analysis_region = gray[:cutoff, :]
        scalebar_rect = self._find_scalebar_rect(gray) if cutoff >= h else None

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

        if not hollow:
            hough_particles = self._detect_hough(analysis_region, min_area_px)
            if hough_particles and particles:
                px_areas = [np.pi * p["radius_px"] ** 2 for p in particles]
                hough_areas = [np.pi * p["radius_px"] ** 2 for p in hough_particles]
                contour_median = np.median(px_areas)
                hough_median = np.median(hough_areas)
                if hough_median > contour_median * 3:
                    particles = hough_particles
                elif len(hough_particles) > len(particles) * 1.5:
                    particles = hough_particles
                elif len(particles) < 3:
                    particles = hough_particles
            elif hough_particles and not particles:
                particles = hough_particles

        if scalebar_rect:
            sx, sy, sw, sh = scalebar_rect
            margin = 8
            kept = []
            for p in particles:
                cx, cy, r = p["center_x"], p["center_y"], p["radius_px"]
                if (sy - margin - r < cy < sy + sh + margin + r
                        and sx - margin - r < cx < sx + sw + margin + r):
                    continue
                kept.append(p)
            particles = kept

        if detect_cores:
            for p in particles:
                p["has_core"] = self._detect_core(
                    gray, p["center_x"], p["center_y"], int(p["radius_px"]))

        return particles

    @staticmethod
    def _detect_core(gray, cx, cy, r):
        h, w = gray.shape
        ri = int(r * 0.75)
        x0, x1 = max(0, cx - ri), min(w, cx + ri)
        y0, y1 = max(0, cy - ri), min(h, cy + ri)
        if x1 - x0 < 6 or y1 - y0 < 6:
            return False
        roi = gray[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= ri * ri
        vals = roi[mask]
        if len(vals) < 30:
            return False

        ring_vals = []
        a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        for f in (0.88, 0.95):
            xs = (cx + r * f * np.cos(a)).astype(int)
            ys = (cy + r * f * np.sin(a)).astype(int)
            v = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            ring_vals.extend(gray[ys[v], xs[v]])
        rim = np.percentile(ring_vals, 25)

        p10, p75 = np.percentile(vals, [10, 75])
        if p75 - rim < 15:
            return False
        if p75 - p10 < 20:
            return False

        t = (p10 + p75) / 2
        dark = ((roi < t) & mask).astype(np.uint8) * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < np.pi * ri * ri * 0.03:
            return False
        (ccx, ccy), _ = cv2.minEnclosingCircle(cnt)
        if np.hypot(ccx + x0 - cx, ccy + y0 - cy) > r * 0.7:
            return False
        return True

    def _detect_hough(self, gray, min_area_px):
        h, w = gray.shape
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        min_r = max(5, int(np.sqrt(min_area_px / np.pi)))
        max_r = min(h, w) // 4

        est = None
        for param2 in [45, 40, 35, 30]:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(10, min_r),
                param1=80, param2=param2, minRadius=min_r, maxRadius=max_r
            )
            if circles is not None and len(circles[0]) >= 5:
                est = self._radius_mode(circles[0][:, 2])
                break
        if est is None:
            return []

        min_r2 = max(min_r, int(est * 0.6))
        max_r2 = int(est * 1.5)
        min_dist2 = max(10, int(est * 1.4))

        best = None
        for param2 in [40, 35, 30, 25]:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist2,
                param1=80, param2=param2, minRadius=min_r2, maxRadius=max_r2
            )
            if circles is None:
                continue
            if best is None or len(circles[0]) > len(best):
                best = circles[0]
        if best is None:
            return []

        blur3 = cv2.GaussianBlur(gray, (3, 3), 0)
        gx = cv2.Sobel(blur3, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur3, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.2), 40, 120)
        eys, exs = np.nonzero(edges)
        edge_pts = np.column_stack([exs, eys])
        bg_mask = self._background_mask(blur3, best)
        rng = np.random.default_rng(0)

        particles = []
        for cx, cy, r in best:
            cx, cy, r = int(cx), int(cy), int(r)
            refined = self._ransac_refine(edge_pts, bg_mask, cx, cy, r, w, h, rng)
            if refined is not None:
                cx, cy, r = refined
            else:
                cx, cy, r = self._refine_circle(grad_mag, cx, cy, r, w, h)
            inside_x = max(0, min(cx + r, w) - max(cx - r, 0))
            inside_y = max(0, min(cy + r, h) - max(cy - r, 0))
            if inside_x < r or inside_y < r:
                continue
            area_px = np.pi * r * r
            particles.append(self._measure_circle(cx, cy, r, area_px))

        particles = self._split_oversized(particles, blurred, edge_pts, bg_mask, w, h, rng)
        return particles

    def _split_oversized(self, particles, blurred, edge_pts, bg_mask, w, h, rng):
        if len(particles) < 5:
            return particles
        radii = np.array([p["radius_px"] for p in particles])
        med = np.median(radii)
        result = []
        for p in particles:
            cx, cy, r = p["center_x"], p["center_y"], int(p["radius_px"])
            if r <= med * 1.35:
                result.append(p)
                continue
            subs = self._rehough_region(blurred, cx, cy, r, med, w, h)
            if not subs or len(subs) < 2:
                result.append(p)
                continue
            for scx, scy, sr in subs:
                refined = self._ransac_refine(edge_pts, bg_mask, scx, scy, sr, w, h, rng)
                if refined is not None:
                    scx, scy, sr = refined
                result.append(self._measure_circle(scx, scy, sr, np.pi * sr * sr))
        return result

    @staticmethod
    def _rehough_region(blurred, cx, cy, r, med, w, h):
        pad = int(r * 1.2)
        x0, x1 = max(0, cx - pad), min(w, cx + pad)
        y0, y1 = max(0, cy - pad), min(h, cy + pad)
        roi = blurred[y0:y1, x0:x1]
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            return None
        min_r = max(3, int(med * 0.7))
        max_r = int(med * 1.25)
        for param2 in [30, 25, 20]:
            circles = cv2.HoughCircles(
                roi, cv2.HOUGH_GRADIENT, dp=1.2,
                minDist=max(8, int(med * 1.2)),
                param1=80, param2=param2,
                minRadius=min_r, maxRadius=max_r
            )
            if circles is None:
                continue
            subs = []
            for scx, scy, sr in circles[0]:
                gcx, gcy = int(scx) + x0, int(scy) + y0
                if np.hypot(gcx - cx, gcy - cy) <= r:
                    subs.append((gcx, gcy, int(sr)))
            if len(subs) >= 2:
                return subs
        return None

    @staticmethod
    def _background_mask(blur, circles):
        h, w = blur.shape
        interior = []
        for cx, cy, r in circles:
            ri = max(1, int(r * 0.5))
            a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            xs = (cx + ri * np.cos(a)).astype(int)
            ys = (cy + ri * np.sin(a)).astype(int)
            v = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            interior.extend(blur[ys[v], xs[v]])
        particle_level = np.median(interior) if interior else 0
        bg_level = np.percentile(blur, 95)
        if bg_level - particle_level < 15:
            return np.zeros_like(blur, dtype=np.float32)
        return (blur > (particle_level + bg_level) / 2).astype(np.float32)

    @staticmethod
    def _bg_fraction(bg_mask, cx, cy, r, w, h):
        total, count = 0.0, 0
        for f in (0.3, 0.5, 0.7, 0.85):
            ri = max(1, int(r * f))
            n = max(8, int(2 * np.pi * ri / 3))
            a = np.linspace(0, 2 * np.pi, n, endpoint=False)
            xs = (cx + ri * np.cos(a)).astype(int)
            ys = (cy + ri * np.sin(a)).astype(int)
            v = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            total += bg_mask[ys[v], xs[v]].sum()
            count += v.sum()
        return total / max(count, 1)

    @staticmethod
    def _circle_from_3pts(p1, p2, p3):
        ax, ay = p1
        bx, by = p2
        cx, cy = p3
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-6:
            return None
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
              + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
              + (cx * cx + cy * cy) * (bx - ax)) / d
        return ux, uy, np.hypot(ax - ux, ay - uy)

    def _ransac_refine(self, edge_pts, bg_mask, cx, cy, r0, w, h, rng,
                       n_iter=400, tol=2.0):
        if len(edge_pts) == 0:
            return None
        d = np.hypot(edge_pts[:, 0] - cx, edge_pts[:, 1] - cy)
        sel = edge_pts[(d > r0 * 0.6) & (d < r0 * 1.45)]
        if len(sel) < 10:
            return None
        best = None
        best_score = -1.0
        n = len(sel)
        for _ in range(n_iter):
            idx = rng.choice(n, 3, replace=False)
            c = self._circle_from_3pts(sel[idx[0]], sel[idx[1]], sel[idx[2]])
            if c is None:
                continue
            ux, uy, r = c
            if not (r0 * 0.7 <= r <= r0 * 1.4):
                continue
            if np.hypot(ux - cx, uy - cy) > r0 * 0.45:
                continue
            dd = np.abs(np.hypot(sel[:, 0] - ux, sel[:, 1] - uy) - r)
            inliers = np.sum(dd < tol)
            score = inliers * (1.0 - self._bg_fraction(bg_mask, int(ux), int(uy), int(r), w, h)) ** 2
            if score > best_score:
                best_score = score
                best = (ux, uy, r)
        if best is None:
            return None
        ux, uy, r = best
        dd = np.abs(np.hypot(sel[:, 0] - ux, sel[:, 1] - uy) - r)
        pts = sel[dd < tol].astype(np.float64)
        if len(pts) >= 6:
            A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
            b = pts[:, 0] ** 2 + pts[:, 1] ** 2
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            ux, uy = sol[0], sol[1]
            r = np.sqrt(sol[2] + ux * ux + uy * uy)
        return int(round(ux)), int(round(uy)), int(round(r))

    @staticmethod
    def _refine_circle(grad_mag, cx, cy, r, w, h, search=0.35, n_angles=72):
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        best = (cx, cy, r)
        best_score = -1.0
        r_lo, r_hi = max(3, int(r * (1 - search))), int(r * (1 + search))
        for dx in (-2, 0, 2):
            for dy in (-2, 0, 2):
                for rr in range(r_lo, r_hi + 1):
                    xs = (cx + dx + rr * cos_a).astype(int)
                    ys = (cy + dy + rr * sin_a).astype(int)
                    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                    if valid.sum() < n_angles * 0.6:
                        continue
                    score = np.median(grad_mag[ys[valid], xs[valid]])
                    if score > best_score:
                        best_score = score
                        best = (cx + dx, cy + dy, rr)
        return best

    @staticmethod
    def _radius_mode(radii):
        hist, edges = np.histogram(radii, bins=max(5, int(len(radii) ** 0.5)))
        idx = np.argmax(hist)
        return (edges[idx] + edges[idx + 1]) / 2

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
        return h

    @staticmethod
    def _find_scalebar_rect(gray):
        h, w = gray.shape
        y0 = int(h * 0.8)
        bottom = gray[y0:, :]
        _, dark = cv2.threshold(bottom, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > w * 0.1 and ch < 20 and cw / max(ch, 1) > 4:
                return (x, y0 + y, cw, ch)
        return None

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
        stats = {
            "count": len(diameters),
            "mean": float(np.mean(diameters)),
            "std": float(np.std(diameters)),
            "min": float(np.min(diameters)),
            "max": float(np.max(diameters)),
            "d10": float(np.percentile(diameters_sorted, 10)),
            "d50": float(np.percentile(diameters_sorted, 50)),
            "d90": float(np.percentile(diameters_sorted, 90)),
        }
        if particles and "has_core" in particles[0]:
            core_count = sum(1 for p in particles if p["has_core"])
            stats["core_count"] = core_count
            stats["core_ratio"] = core_count / len(particles)
        return stats
