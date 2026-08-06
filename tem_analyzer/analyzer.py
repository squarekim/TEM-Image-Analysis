import cv2
import numpy as np
import os
import re

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


def load_image(path):
    """Load a TEM image as 8-bit BGR.

    TEM cameras commonly write 16-bit or float TIFFs whose values occupy only
    part of the container's range (e.g. 12-bit data in a 16-bit file). Reading
    those with a plain cv2.imread crushes the contrast, so high-bit-depth data
    is rescaled from its actual intensity range instead.

    The file is read through numpy rather than cv2.imread because on Windows
    cv2.imread opens paths using the ANSI codepage and simply returns None for
    any path containing non-ASCII characters - a Korean user name in
    C:\\Users\\... is enough to make every image look unreadable.
    """
    try:
        raw = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if raw.size == 0:
        return None

    data = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if data is None:
        return None

    if data.ndim == 3 and data.shape[2] == 4:
        data = data[:, :, :3]
    elif data.ndim == 3 and data.shape[2] == 2:
        data = data[:, :, 0]

    if data.dtype != np.uint8:
        values = data[np.isfinite(data)] if data.dtype.kind == "f" else data
        if values.size == 0:
            return None
        lo, hi = np.percentile(values, [0.1, 99.9])
        if hi <= lo:
            lo, hi = float(np.min(values)), float(np.max(values))
        if hi <= lo:
            data = np.zeros(data.shape, np.uint8)
        else:
            scaled = (data.astype(np.float32) - lo) * (255.0 / (hi - lo))
            data = np.clip(np.nan_to_num(scaled), 0, 255).astype(np.uint8)

    if data.ndim == 2:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    return data


def save_image(path, image):
    """Write an image, tolerating non-ASCII paths on Windows.

    cv2.imwrite has the same ANSI-codepage limitation as cv2.imread, so the
    encode and the write are done separately.
    """
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError:
        return False
    return True


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

        # Neither detector dominates, so both run and their results are merged.
        # Seed-and-trace validates every circle against the actual edge and so
        # separates touching particles the contour path fuses into one blob, but
        # its radius band is set from the modal particle size and it misses
        # anything far off that size. The contour path has the opposite profile:
        # it handles a wide size range but merges neighbours that touch.
        for p in particles:
            p.setdefault("excluded", False)
            p.setdefault("approx", False)
        hough_particles = self._detect_hough(analysis_region, min_area_px)
        particles = self._merge_detections(hough_particles, particles)

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
                if p.get("excluded"):
                    p["has_core"] = False
                else:
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

        p10, p75 = np.percentile(vals, [10, 75])
        if p75 - p10 < 12:
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

        seeds = [(int(cx), int(cy), int(r)) for cx, cy, r in best]
        med = float(np.median([s[2] for s in seeds]))

        expanded = []
        for cx, cy, r in seeds:
            if r > med * 1.35:
                subs = self._rehough_region(blurred, cx, cy, r, med, w, h)
                if subs and len(subs) >= 2:
                    expanded.extend(subs)
                    continue
            expanded.append((cx, cy, r))

        blur3 = cv2.GaussianBlur(gray, (3, 3), 0)
        gx = cv2.Sobel(blur3, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur3, cv2.CV_32F, 0, 1, ksize=3)

        traces = []
        all_scores = []
        for cx, cy, r in expanded:
            trace = self._trace_boundary(gx, gy, cx, cy, r, w, h)
            traces.append(trace)
            all_scores.extend(trace[2][trace[2] > 0])
        if not all_scores:
            return []
        score_thresh = np.percentile(all_scores, 30)

        particles = []
        for (cx, cy, r), (angles, r_ang, s_ang) in zip(expanded, traces):
            fit = self._robust_circle_fit(cx, cy, angles, r_ang, s_ang,
                                          score_thresh, r0=r)
            if fit is None:
                continue
            ux, uy, rr, coverage, rms, pts = fit
            ux, uy, rr = int(round(ux)), int(round(uy)), rr
            p = self._measure_circle(ux, uy, rr, np.pi * rr * rr)
            p["contour"] = pts.reshape(-1, 1, 2).astype(np.int32)
            center_inside = 4 <= ux <= w - 4 and 4 <= uy <= h - 4

            ring = np.linspace(0, 2 * np.pi, 96, endpoint=False)
            rxs = ux + rr * np.cos(ring)
            rys = uy + rr * np.sin(ring)
            ring_inside = ((rxs >= 0) & (rxs < w) & (rys >= 0) & (rys < h)).mean()

            good = (s_ang > score_thresh) & ~np.isnan(r_ang)
            r_from_fit = np.hypot(cx + np.where(np.isnan(r_ang), 0, r_ang) * np.cos(angles) - ux,
                                  cy + np.where(np.isnan(r_ang), 0, r_ang) * np.sin(angles) - uy)
            spread = self._smoothed_spread(r_from_fit, good)

            usable = center_inside and ring_inside >= 0.5 and spread <= 1.6
            if usable and coverage >= 0.5 and rms <= 0.10:
                p["approx"] = False
                p["excluded"] = False
            elif usable and coverage >= 0.25 and rms <= 0.30:
                p["approx"] = True
                p["excluded"] = False
            else:
                p["approx"] = False
                p["excluded"] = True
            particles.append(p)

        particles.sort(key=lambda p: (p.get("excluded", False), p.get("approx", False)))
        particles = self._dedup(particles, med)

        for i, p in enumerate(particles):
            if p.get("excluded"):
                continue
            for q in particles[i + 1:]:
                if q.get("excluded"):
                    continue
                d = np.hypot(p["center_x"] - q["center_x"], p["center_y"] - q["center_y"])
                if d < 0.8 * max(p["radius_px"], q["radius_px"]):
                    p["excluded"] = True
                    q["excluded"] = True
                    p["approx"] = False
                    q["approx"] = False
        return particles

    @staticmethod
    def _trace_boundary(gx, gy, cx, cy, r0, w, h, n_angles=96):
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        radii = np.arange(max(2, r0 * 0.55), r0 * 1.5, 0.5)
        cos_a, sin_a = np.cos(angles), np.sin(angles)

        xs = (cx + np.outer(cos_a, radii)).astype(int)
        ys = (cy + np.outer(sin_a, radii)).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

        rows = np.clip(ys, 0, h - 1)
        cols = np.clip(xs, 0, w - 1)
        outward = (gx[rows, cols] * cos_a[:, None] + gy[rows, cols] * sin_a[:, None])
        outward = np.where(valid, outward, -np.inf)

        best = np.argmax(outward, axis=1)
        enough = valid.sum(axis=1) >= len(radii) * 0.5
        r_best = np.where(enough, radii[best], np.nan)
        s_best = np.where(enough, outward[np.arange(n_angles), best], 0.0)
        return angles, r_best, s_best

    @staticmethod
    def _smoothed_spread(r_ang, good, win=5):
        n = len(r_ang)
        idx = np.where(good)[0]
        if len(idx) < 8:
            return np.inf
        half = win // 2
        neighbors = (idx[:, None] + np.arange(-half, half + 1)[None, :]) % n
        window = np.where(good[neighbors], r_ang[neighbors], np.nan)
        smoothed = np.nanmedian(window, axis=1)
        return np.percentile(smoothed, 90) / max(np.percentile(smoothed, 10), 1)

    @staticmethod
    def _robust_circle_fit(cx, cy, angles, r_ang, s_ang, score_thresh, r0=None):
        good = (s_ang > score_thresh) & ~np.isnan(r_ang)
        coverage = good.mean()
        if good.sum() < 8:
            return None
        xs = cx + r_ang[good] * np.cos(angles[good])
        ys = cy + r_ang[good] * np.sin(angles[good])
        pts = np.column_stack([xs, ys])
        ux, uy, r = cx, cy, np.median(r_ang[good])
        for _ in range(3):
            A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
            b = pts[:, 0] ** 2 + pts[:, 1] ** 2
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            ux, uy = sol[0], sol[1]
            r = np.sqrt(sol[2] + ux * ux + uy * uy)
            resid = np.abs(np.hypot(pts[:, 0] - ux, pts[:, 1] - uy) - r)
            mad = np.median(resid) + 1e-6
            keep = resid < 3 * mad + 1
            if keep.all():
                break
            pts = pts[keep]
            if len(pts) < 8:
                break

        # Where particles touch, the contact side has no background to give an
        # outward gradient, so the trace covers only a shallow arc - and a
        # circle through a shallow arc can be arbitrarily large. The Hough seed
        # radius is reliable, so a fit that runs away from it is replaced by the
        # seed geometry with a radius taken from the traced edge directly.
        if r0 is not None and not (0.65 * r0 <= r <= 1.45 * r0):
            ux, uy = cx, cy
            r = float(np.median(r_ang[good]))
            pts = np.column_stack([cx + r_ang[good] * np.cos(angles[good]),
                                   cy + r_ang[good] * np.sin(angles[good])])

        resid = np.abs(np.hypot(pts[:, 0] - ux, pts[:, 1] - uy) - r)
        rms = np.sqrt(np.mean(resid ** 2)) / max(r, 1)
        return ux, uy, r, coverage, rms, pts

    @staticmethod
    def _merge_detections(primary, secondary):
        """Combine two detector outputs, keeping ``primary`` where they clash.

        A particle from ``secondary`` is only added where nothing already
        occupies that spot, so a contour blob spanning a touching pair is
        dropped in favour of the two traced circles inside it.
        """
        merged = [p for p in primary if not p.get("excluded")]
        rejected = [p for p in primary if p.get("excluded")]

        for cand in secondary:
            if cand.get("excluded"):
                continue
            clash = False
            for kept in merged:
                d = np.hypot(cand["center_x"] - kept["center_x"],
                             cand["center_y"] - kept["center_y"])
                if d < 0.8 * max(cand["radius_px"], kept["radius_px"]):
                    clash = True
                    break
            if not clash:
                merged.append(cand)

        # A circle that swallows two accepted centres is a fused pair, not a
        # particle, whichever detector produced it.
        keep = []
        for p in merged:
            inside = sum(
                1
                for q in merged
                if q is not p
                and np.hypot(p["center_x"] - q["center_x"],
                             p["center_y"] - q["center_y"]) < p["radius_px"]
            )
            if inside >= 2:
                p["excluded"] = True
                p["approx"] = False
                rejected.append(p)
            else:
                keep.append(p)
        return keep + rejected

    @staticmethod
    def _dedup(particles, med):
        kept = []
        for p in particles:
            dup = False
            for q in kept:
                d = np.hypot(p["center_x"] - q["center_x"], p["center_y"] - q["center_y"])
                if d < med * 0.7:
                    dup = True
                    break
            if not dup:
                kept.append(p)
        return kept

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
        for param2 in [30, 25, 20, 15]:
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
                gcx, gcy = scx + x0, scy + y0
                if np.hypot(gcx - cx, gcy - cy) <= r * 1.1:
                    subs.append((int(round(gcx)), int(round(gcy)), int(round(sr))))
            if len(subs) >= 2:
                return subs
        return None

    @staticmethod
    def _radius_mode(radii):
        hist, edges = np.histogram(radii, bins=max(5, int(len(radii) ** 0.5)))
        idx = np.argmax(hist)
        return (edges[idx] + edges[idx + 1]) / 2

    def _measure_circle(self, cx, cy, radius, area_px, contour=None):
        """Build a particle record, converting to nm when a scale is known."""
        scale = self.nm_per_px or 1.0
        return {
            "center_x": int(cx),
            "center_y": int(cy),
            "radius_px": radius,
            "diameter": radius * 2 * scale,
            "area": area_px * scale ** 2,
            "contour": contour,
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
    def snap_scalebar(image, start, end):
        """Refine a rough drag along a scale bar to the bar's true extent.

        A hand drag over a 60-100 px bar is easily a couple of pixels short or
        long, and that error multiplies into every diameter, so the drag is
        only used to locate the bar: its actual ends are then read from the
        image. Returns the refined length in pixels, or None when no bar-like
        run is found under the drag.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape
        (x0, y0), (x1, y1) = start, end
        length = np.hypot(x1 - x0, y1 - y0)
        if length < 8 or abs(x1 - x0) < abs(y1 - y0):
            return None  # scale bars are horizontal; a vertical drag is not one

        # Scan a window around the drag row by row: the bar is often only a
        # couple of pixels tall, so the drag itself may not lie on it.
        pad = max(6, int(length * 0.2))
        ry0, ry1 = max(0, int(min(y0, y1)) - pad), min(h, int(max(y0, y1)) + pad + 1)
        rx0, rx1 = max(0, int(min(x0, x1)) - pad), min(w, int(max(x0, x1)) + pad + 1)
        roi = gray[ry0:ry1, rx0:rx1]
        if roi.size == 0 or roi.shape[1] < 8:
            return None

        drag_lo, drag_hi = int(min(x0, x1)) - rx0, int(max(x0, x1)) - rx0
        best = None
        for dark_bar in (True, False):
            level = np.percentile(roi, 25 if dark_bar else 75)
            mask = roi < level if dark_bar else roi > level
            for ri, on in enumerate(mask):
                idx = np.flatnonzero(np.diff(np.r_[0, on.view(np.int8), 0]))
                if len(idx) < 2:
                    continue
                for s, e in zip(idx[::2], idx[1::2]):
                    run = e - s
                    # the run has to be roughly what the user pointed at
                    if run < 8 or run > length * 1.6 or run < length * 0.5:
                        continue
                    overlap = min(e, drag_hi) - max(s, drag_lo)
                    if overlap < 0.6 * min(run, drag_hi - drag_lo + 1):
                        continue
                    # A scale bar is a thin horizontal rule; a particle edge
                    # produces a run too, but sits in a tall dark region.
                    col = mask[:, (s + e) // 2]
                    top = bottom = ri
                    while top > 0 and col[top - 1]:
                        top -= 1
                    while bottom < len(col) - 1 and col[bottom + 1]:
                        bottom += 1
                    if (bottom - top + 1) > max(6, run * 0.35):
                        continue

                    # Sub-pixel ends: interpolate where the profile crosses the
                    # midpoint between bar and background, so a bar does not
                    # round to whole pixels.
                    prof = roi[ri].astype(np.float32)
                    inside = prof[s:e].mean()
                    outside = np.concatenate([prof[max(0, s - 4):s], prof[e:e + 4]])
                    refined = float(run)
                    if outside.size:
                        half = (inside + outside.mean()) / 2
                        left, right = float(s), float(e - 1)
                        if s > 0 and prof[s - 1] != prof[s]:
                            left = s - 1 + (half - prof[s - 1]) / (prof[s] - prof[s - 1])
                        if e < len(prof) and prof[e] != prof[e - 1]:
                            right = e - 1 + (half - prof[e - 1]) / (prof[e] - prof[e - 1])
                        if right > left:
                            refined = right - left

                    if best is None or refined > best:
                        best = refined
        return float(best) if best else None

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
        # Work inside the contour's bounding box: a full-frame mask and distance
        # transform per contour is what made large images unusably slow. The pad
        # covers the widest dilation used below so results match the full-frame
        # computation.
        pad = 8
        bx, by, bw, bh = cv2.boundingRect(cnt)
        x0 = max(0, bx - pad)
        y0 = max(0, by - pad)
        x1 = min(binary.shape[1], bx + bw + pad)
        y1 = min(binary.shape[0], by + bh + pad)
        offset = np.array([[x0, y0]], dtype=cnt.dtype)
        cnt = cnt - offset
        binary = binary[y0:y1, x0:x1]
        gray = gray[y0:y1, x0:x1]

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
            p = self._measure_single(seg_cnt + offset, seg_area)
            if p:
                results.append(p)

        return results

    def _measure_single(self, cnt, area_px):
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        return self._measure_circle(cx, cy, radius, area_px, contour=cnt)

    @staticmethod
    def compute_statistics(particles):
        excluded_count = sum(1 for p in particles if p.get("excluded"))
        particles = [p for p in particles if not p.get("excluded")]
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
            "excluded": excluded_count,
            "approx": sum(1 for p in particles if p.get("approx")),
        }
        if particles and "has_core" in particles[0]:
            core_count = sum(1 for p in particles if p["has_core"])
            stats["core_count"] = core_count
            stats["core_ratio"] = core_count / len(particles)
        return stats
