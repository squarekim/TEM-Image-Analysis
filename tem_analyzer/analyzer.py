import cv2
import warnings

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
    #: Where across the rim the boundary is taken when the operator fixes it by
    #: hand: 0 at the rim's inner flank, 0.5 on the rim itself, 1 at its outer
    #: flank, measured at half height on each side. The particle size is the
    #: outer diameter - the outside of the shell wall - so the manual default
    #: sits at the outer flank. Left on automatic (the product default) the
    #: analyzer picks this per ray instead; see `_outer_by_level`.
    DEFAULT_EDGE_LEVEL = 0.95

    #: A ray faces a touching neighbour, rather than open background, when the
    #: level just outside the dark band is nearly the particle's own interior
    #: level - the neighbour's interior is as bright as ours, a gap is not.
    #: Judged as a fraction of the band's own contrast so it does not depend on
    #: exposure or magnification.
    CONTACT_GAP_FRAC = 0.30

    def __init__(self, nm_per_px=None, edge_level="auto"):
        self.nm_per_px = nm_per_px
        #: None means decide per ray; a float pins every ray to that position.
        self.edge_level = (None if edge_level is None or edge_level == "auto"
                           else float(np.clip(edge_level, 0.0, 1.0)))

    def analyze(self, image, min_area_px=100, max_area_px=None,
                circularity_thresh=0.5, use_watershed=True, hollow=False,
                detect_cores=False, measure_shell=False):
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

        # Shell-enclosed interiors take precedence over both ray-based paths.
        # Rays from a centre cannot tell this particle's shell from a
        # neighbour's when the interior and the gaps have the same brightness -
        # the circle then rests on an envelope of several particles' walls,
        # displaced and oversized. Enclosure can tell: a bright interior is
        # bounded by its own shell and nothing else, so a particle found this
        # way owns its boundary by construction. On images without bright
        # enclosed interiors (solid particles on a bright background) the path
        # finds nothing and changes nothing.
        enclosed = self._detect_enclosed(analysis_region, min_area_px)
        particles = self._merge_detections(enclosed, particles)

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

        self._reject_implausible_interiors(particles, analysis_region)
        self._reject_clipped(particles, analysis_region.shape)
        self._resolve_overlaps(particles, cv2.GaussianBlur(analysis_region, (0, 0), 1.5))
        self._reject_buried(particles)

        if measure_shell:
            blur3 = cv2.GaussianBlur(analysis_region, (3, 3), 0)
            sgx = cv2.Sobel(blur3, cv2.CV_32F, 1, 0, ksize=3)
            sgy = cv2.Sobel(blur3, cv2.CV_32F, 0, 1, ksize=3)
            rh, rw = analysis_region.shape
            for p in particles:
                inner = None
                if not p.get("excluded"):
                    inner = self._measure_shell(sgx, sgy, blur3,
                                                p["center_x"], p["center_y"],
                                                p["radius_px"], rw, rh)
                self._record_shell(p, inner)

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
        max_r = max(min_r + 1, min(h, w) // 3)

        est = self._estimate_radius(gray, blurred, min_r, max_r, w, h)
        if est is None:
            return []
        gx, gy = self._gradients(gray, est)

        min_r2 = max(min_r, int(est * 0.6))
        max_r2 = int(est * 1.5)
        min_dist2 = max(10, int(est * 1.4))

        # Locating individual seeds uses a gentler blur than the band scan:
        # enough to stop grain from dominating a wide radius sweep, but not so
        # much that neighbours which touch are fused - the thing this stage
        # exists to keep apart.
        seed_img, seed_param1 = self._smooth_for_scale(gray, blurred, est,
                                                       factor=0.025)
        best = None
        for param2 in [40, 35, 30, 25]:
            circles = cv2.HoughCircles(
                seed_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist2,
                param1=seed_param1, param2=param2,
                minRadius=min_r2, maxRadius=max_r2
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
                subs = self._rehough_region(seed_img, cx, cy, r, med, w, h)
                if subs and len(subs) >= 2:
                    expanded.extend(subs)
                    continue
            expanded.append((cx, cy, r))

        # Anchor each seed on the rim it belongs to before anything is traced
        # from it. A seed sitting off-centre traces a boundary made partly of
        # its own rim and partly of a neighbour's, and the circle fitted to
        # that mixture is both displaced and too large: on the real
        # micrographs a sixth of all detections were more than 10% of a radius
        # out of position, and fitting to the rim points alone pulled the
        # radius back to 0.93 of what was reported.
        expanded = [self._refine_seed(blurred, cx, cy, r, w, h)
                    for cx, cy, r in expanded]

        traces = []
        strong_scores, outer_scores = [], []
        for cx, cy, r in expanded:
            trace = self._trace_boundary(gx, gy, cx, cy, r, w, h)
            traces.append(trace)
            strong_scores.extend(trace[2][trace[2] > 0])
            outer_scores.extend(trace[4][trace[4] > 0])
        if not strong_scores:
            return []
        # The outer transition of a rim is the weaker of the two, so judging it
        # against the strong edges' threshold would discard nearly all of it.
        score_thresh = np.percentile(strong_scores, 30)
        outer_thresh = score_thresh * 0.4

        # Fit both transitions everywhere first, then let the field as a whole
        # decide which one is the boundary. Particles in one image are the same
        # kind of object imaged the same way, so a rim resolved on one is a rim
        # on all; choosing per particle is what made neighbouring particles come
        # out a rim-thickness apart in diameter for no physical reason.
        candidates = [self._boundary_candidates(cx, cy, trace[0], trace, score_thresh,
                                                r, outer_thresh, blurred)
                      for (cx, cy, r), trace in zip(expanded, traces)]
        decidable = [c for c in candidates if c["strong"] is not None and c["outer"] is not None]
        use_outer = bool(decidable) and sum(c["qualifies"] for c in decidable) >= 0.5 * len(decidable)

        particles = []
        for (cx, cy, r), trace, candidate in zip(expanded, traces, candidates):
            angles = trace[0]
            fit, r_ang, s_ang, used_thresh = self._select_boundary(candidate, use_outer)
            if fit is None:
                continue
            ux, uy, rr, coverage, rms, (pts, measured) = fit
            ux, uy, rr = int(round(ux)), int(round(uy)), rr
            p = self._measure_circle(ux, uy, rr, np.pi * rr * rr)
            p["coverage"] = float(coverage)
            p["contour"] = pts.reshape(-1, 1, 2).astype(np.float32)
            p["contour_measured"] = measured
            center_inside = 4 <= ux <= w - 4 and 4 <= uy <= h - 4

            ring = np.linspace(0, 2 * np.pi, 96, endpoint=False)
            rxs = ux + rr * np.cos(ring)
            rys = uy + rr * np.sin(ring)
            ring_inside = ((rxs >= 0) & (rxs < w) & (rys >= 0) & (rys < h)).mean()

            good = (s_ang > used_thresh) & ~np.isnan(r_ang)
            r_from_fit = np.hypot(cx + np.where(np.isnan(r_ang), 0, r_ang) * np.cos(angles) - ux,
                                  cy + np.where(np.isnan(r_ang), 0, r_ang) * np.sin(angles) - uy)
            spread = self._smoothed_spread(r_from_fit, good)

            usable = center_inside and ring_inside >= 0.5 and spread <= 1.6
            if usable and coverage >= 0.5 and rms <= 0.10:
                p["approx"] = False
                p["excluded"] = False
            elif usable and coverage >= 0.20 and rms <= 0.30:
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
    def _smooth_for_scale(gray, default, radius, factor=0.04):
        """Image and Canny threshold matched to the circle size being sought.

        Searching every size on the same lightly blurred image makes the
        large-radius passes grind through grain-scale edge points that cannot
        belong to a circle that big. Blurring to the band's scale removes them,
        but it also softens the real edges, so the gradient threshold has to
        come down with it or nothing is detected at all. Small particles blur
        to under a pixel and simply keep the default image.
        """
        sigma = radius * factor
        if sigma <= 1.5:
            return default, 80
        sigma = min(sigma, 8.0)
        return cv2.GaussianBlur(gray, (0, 0), sigma), max(20, int(80 * 1.5 / sigma))

    @staticmethod
    def _gradients(gray, radius):
        """Edge gradients smoothed at the scale of the particles being traced.

        A fixed small blur leaves image grain as the strongest local gradient,
        so on a noisy micrograph of large particles each ray latches onto a
        different speckle and the traced outline zigzags instead of following
        the boundary. Smoothing proportionally keeps the particle edge - a
        large-scale feature - while grain averages away.
        """
        sigma = float(np.clip(radius * 0.02, 0.8, 6.0))
        smooth = cv2.GaussianBlur(gray, (0, 0), sigma)
        return (cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3),
                cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3))

    def _estimate_radius(self, gray, blurred, min_r, max_r, w, h):
        """Pick the particle size scale, judging candidates by their edges.

        Taking the modal radius of one wide Hough sweep lets image grain decide
        the answer: a noisy micrograph of a few large particles yields hundreds
        of circles the size of the speckle, swamping the handful of real ones.
        Instead each octave of radii is sampled separately and its circles are
        traced, each at the smoothing that scale calls for; a band only scores
        for circles whose boundary actually looks like a particle edge, which
        grain cannot fake.
        """
        best_score, best_est = 0.0, None
        lo = min_r
        while lo < max_r:
            hi = min(max_r, lo * 2)
            mid = (lo + hi) / 2
            # Smooth to the band's own scale before searching it. Feeding the
            # raw image to every band makes the large-radius searches crawl
            # through grain-scale edge points that cannot belong to a circle
            # that big.
            band_img, band_param1 = self._smooth_for_scale(gray, blurred, mid)
            circles = None
            for param2 in [40, 30]:
                found = cv2.HoughCircles(
                    band_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(10, lo),
                    param1=band_param1, param2=param2,
                    minRadius=int(lo), maxRadius=int(hi)
                )
                if found is not None and len(found[0]) >= 3:
                    circles = found[0]
                    break
            lo = hi
            if circles is None:
                continue

            gx, gy = self._gradients(gray, mid)
            sample = circles[:16]
            traces = [self._trace_boundary(gx, gy, int(cx), int(cy), int(r), w, h)
                      for cx, cy, r in sample]
            scores = [v for _, _, s_ang, _, _ in traces for v in s_ang[s_ang > 0]]
            if not scores:
                continue
            thresh = np.percentile(scores, 30)
            passed = 0
            for (cx, cy, r), trace in zip(sample, traces):
                fit = self._robust_circle_fit(int(cx), int(cy), trace[0],
                                              trace[1], trace[2], thresh,
                                              r0=int(r))
                if fit and fit[3] >= 0.5 and fit[4] <= 0.10:
                    passed += 1

            # Scale the validated fraction by the population so a band with a
            # few solid circles beats one with many unconvincing ones.
            score = (passed / len(sample)) * len(circles)
            if score > best_score:
                best_score = score
                best_est = self._radius_mode(circles[:, 2])
        return best_est

    def _choose_boundary(self, cx, cy, angles, trace, score_thresh, r0,
                         outer_thresh=None, blurred=None):
        """Fit the outer transition when it holds up, else the strongest one.

        A particle ringed by a dark rim offers two transitions and the outer
        one is the boundary a person measures, but on a plainly edged particle
        the outermost crest above threshold is often just noise. Fitting both
        and comparing settles it: a real outer rim is circular and fits about
        as well as the strong edge, whereas noise does not.
        """
        candidate = self._boundary_candidates(cx, cy, angles, trace, score_thresh,
                                              r0, outer_thresh, blurred)
        return self._select_boundary(candidate, candidate["qualifies"])

    @staticmethod
    def _row_percentile(a, q):
        """``np.nanpercentile(a, q, axis=1)``, without the per-row Python loop.

        numpy reduces along an axis by calling the 1-D routine once per row
        through ``apply_along_axis``. At 180 rays per particle and hundreds of
        particles that is tens of thousands of calls, and it was the single
        largest cost in a full-image analysis - a third of the total. Sorting
        the whole array at once puts NaNs at the end of each row by definition,
        so the valid count per row is all that is needed to index the same
        value numpy would have interpolated.
        """
        a = np.sort(a, axis=1)
        counts = np.count_nonzero(~np.isnan(a), axis=1)
        out = np.full(len(a), np.nan)
        rows = np.flatnonzero(counts > 0)
        if not rows.size:
            return out
        pos = (counts[rows] - 1) * (q / 100.0)
        lo = np.floor(pos).astype(np.intp)
        hi = np.ceil(pos).astype(np.intp)
        w = pos - lo
        out[rows] = a[rows, lo] * (1.0 - w) + a[rows, hi] * w
        return out

    def _outer_by_level(self, blurred, cx, cy, r0, angles, w, h, frac=None):
        """Radius where the rim's darkening has faded back to the surroundings.

        A gradient crest is only a boundary when the edge is sharp. Where the
        rim fades outward over tens of pixels - which is what high
        magnification makes of an edge that looked crisp at low - there is no
        crest to find out there, so the crest search settles either on the
        rim's inner flank or somewhere out in the tail, and which one it picks
        swings with noise.

        Half-recovery does not care how gradual the fade is: it asks where the
        profile has come back half way from the rim's extreme to the level
        outside, which is the point the eye reads as the edge and the standard
        sub-pixel edge criterion. Returns per-angle radii and strengths.
        """
        radii = np.arange(max(2.0, r0 * 0.5), r0 * 1.6, 0.5)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        xs = (cx + np.outer(cos_a, radii)).astype(int)
        ys = (cy + np.outer(sin_a, radii)).astype(int)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        prof = blurred[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)].astype(np.float32)

        band = (radii >= 0.65 * r0) & (radii <= 1.25 * r0)
        far = radii >= 1.35 * r0
        core = (radii >= 0.5 * r0) & (radii <= 0.62 * r0)
        if not band.any() or not far.any():
            return np.full(len(angles), np.nan), np.zeros(len(angles)), 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # An upper percentile, not the median: past the rim the profile is
            # still climbing towards the surroundings, so the middle of that
            # stretch reads low - 115 against a true 125 on the real
            # micrographs. That drags the half-recovery target down with it and
            # landed the boundary at 30-40% recovery when 50% was asked for, by
            # different amounts at different magnifications. Reading the top of
            # the stretch instead lands it at 45-47% and halves the gap between
            # the two magnifications, without sampling further out where the
            # neighbouring particle would be.
            outside = self._row_percentile(
                np.where(inside[:, far], prof[:, far], np.nan), 80.0)
            interior = (self._row_percentile(
                np.where(inside[:, core], prof[:, core], np.nan), 50.0)
                if core.any() else np.full(len(angles), np.nan))
        r_out = np.full(len(angles), np.nan)
        s_out = np.zeros(len(angles))
        band_idx = np.flatnonzero(band)
        rim_votes = []

        for i in range(len(angles)):
            if not np.isfinite(outside[i]) or inside[i].sum() < len(radii) * 0.5:
                continue
            seg = prof[i, band_idx]
            # The rim may be darker or brighter than its surroundings; take
            # whichever extreme departs further from the outside level.
            lo, hi = seg.min(), seg.max()
            dark = (outside[i] - lo) >= (hi - outside[i])
            k = band_idx[int(np.argmin(seg) if dark else np.argmax(seg))]
            extreme = prof[i, k]
            contrast = abs(outside[i] - extreme)
            if contrast < 4:
                continue
            # A rim is a band that stands out from the particle's own interior,
            # not merely a step from interior to background. Asking instead
            # whether a dark dip sits between the steepest point and this one
            # cannot work here: half-recovery is on the far flank of that same
            # dip, so the profile between them only ever rises.
            if np.isfinite(interior[i]):
                rim_votes.append(abs(interior[i] - extreme)
                                 >= max(12.0, 0.50 * contrast))
            # Place the boundary anywhere across the rim, not just outside it.
            # Walking outward from the rim's darkest point can never put the
            # edge on the rim itself, let alone on its inner flank, so the
            # setting had almost nothing to move: 3% to 40% spanned 75 to 78 nm
            # on a real particle whose rim is 9% of the radius wide. Measuring
            # the half-height on both flanks gives the rim's full extent, and
            # the setting then slides between them - 0% at the inner flank,
            # 50% on the rim, 100% at the outer flank.
            def half_height(level, forward):
                target = extreme + 0.5 * (level - extreme)
                walk = prof[i, k:] if forward else prof[i, :k + 1][::-1]
                crossed = (walk >= target) if dark else (walk <= target)
                j = int(np.argmax(crossed))
                if not crossed[j]:
                    return None
                step = k + j if forward else k - j
                if step == k:
                    return radii[k]
                prev = step - 1 if forward else step + 1
                a, b = prof[i, prev], prof[i, step]
                t = 0.0 if b == a else (target - a) / (b - a)
                return radii[prev] + t * (radii[step] - radii[prev])

            r_outer_edge = half_height(outside[i], True)
            r_inner_edge = (half_height(interior[i], False)
                            if np.isfinite(interior[i]) else None)
            if r_outer_edge is None:
                continue
            if r_inner_edge is None or r_inner_edge >= r_outer_edge:
                r_inner_edge = radii[k]
            # Where across the band the surface lies is not one number for the
            # whole particle. Facing an open gap the dark band is this
            # particle's own shell wall and the surface is its outer edge.
            # Facing a touching neighbour the band is two walls together and
            # the surface is the contact plane, halfway across it - taking the
            # outer edge there measures through the neighbour's wall as well
            # and inflates the diameter by a full wall thickness. That is the
            # difference the single global setting could not express: pushed
            # out far enough for the free sectors it made neighbouring circles
            # overlap by 12%, and pulled in far enough to stop that it reported
            # the shell's inner edge.
            if frac is None:
                contact = (np.isfinite(interior[i])
                           and abs(interior[i] - outside[i])
                           < self.CONTACT_GAP_FRAC * contrast)
                here = 0.5 if contact else 1.0
            else:
                here = frac
            r_out[i] = r_inner_edge + here * (r_outer_edge - r_inner_edge)
            s_out[i] = contrast
        rim_frac = float(np.mean(rim_votes)) if rim_votes else 0.0
        return r_out, s_out, rim_frac

    def _boundary_candidates(self, cx, cy, angles, trace, score_thresh, r0,
                             outer_thresh=None, blurred=None):
        """Fit both transitions and judge whether the outer one is a real rim."""
        _, r_strong, s_strong, r_outer, s_outer = trace
        if outer_thresh is None:
            outer_thresh = score_thresh
        strong = self._robust_circle_fit(cx, cy, angles, r_strong, s_strong,
                                         score_thresh, r0=r0)

        rim_frac = 0.0
        if blurred is not None:
            h, w = blurred.shape
            r_level, s_level, rim_frac = self._outer_by_level(
                blurred, cx, cy, r0, angles, w, h, frac=self.edge_level)
            if np.isfinite(r_level).sum() >= 8:
                r_outer, s_outer = r_level, s_level
                outer_thresh = max(4.0, np.percentile(s_level[s_level > 0], 25)) \
                    if (s_level > 0).any() else outer_thresh
        outer = self._robust_circle_fit(cx, cy, angles, r_outer, s_outer,
                                        outer_thresh, r0=r0)

        qualifies = False
        if outer is not None and strong is not None and outer[2] > strong[2] + 0.5:
            # The evidence has to be free of scale, or the same sample answers
            # differently at two magnifications - which is backwards, since the
            # higher magnification shows the rim more clearly. So: is the outer
            # boundary seen most of the way round, is it circular, and is there
            # actually a dark band between it and the inner edge? All three are
            # ratios or brightness comparisons, and none of them changes when
            # the same particle is imaged larger.
            # Coverage is deliberately lenient. In a packed field the outer
            # boundary simply does not exist on the contact sides - there is a
            # neighbour there, not background for the profile to recover to -
            # so demanding it most of the way round asks for something the
            # sample cannot supply, and asks for less of it at low
            # magnification than at high. What must hold is that where the
            # outer boundary is seen it is circular, and that a dark band
            # really does lie between it and the inner edge.
            qualifies = bool(
                outer[3] >= 0.30 and outer[4] <= 0.12
                and (blurred is None or rim_frac >= 0.5))

        return {"strong": strong, "outer": outer, "qualifies": qualifies,
                "rim_frac": rim_frac,
                "r_strong": r_strong, "s_strong": s_strong,
                "r_outer": r_outer, "s_outer": s_outer,
                "score_thresh": score_thresh, "outer_thresh": outer_thresh}

    @staticmethod
    def _select_boundary(c, use_outer):
        """Pick one of the two fitted transitions.

        ``use_outer`` is decided once for the whole image rather than per
        particle. Every particle in a field is the same kind of object imaged
        the same way, so the boundary has to be the same feature on all of
        them; deciding one at a time lets some land on the inner flank of the
        rim and others on the outer, and the diameters then differ by the rim
        thickness for no physical reason.
        """
        strong, outer = c["strong"], c["outer"]
        if outer is None:
            return strong, c["r_strong"], c["s_strong"], c["score_thresh"]
        if strong is None:
            return outer, c["r_outer"], c["s_outer"], c["outer_thresh"]
        # Held to a looser standard than the vote itself: the image has already
        # established that these particles have a rim, so a slightly ragged
        # outer edge is still the right feature to measure.
        if use_outer and outer[2] > strong[2] + 0.5 and outer[3] >= 0.25 and outer[4] <= 0.20:
            return outer, c["r_outer"], c["s_outer"], c["outer_thresh"]
        return strong, c["r_strong"], c["s_strong"], c["score_thresh"]

    def _record_shell(self, particle, inner_r):
        """Attach shell thickness and void fraction to a particle record."""
        scale = self.nm_per_px or 1.0
        outer_r = particle["radius_px"]
        if inner_r is None or outer_r <= 0:
            particle["inner_radius_px"] = None
            particle["inner_diameter"] = None
            particle["shell_thickness"] = None
            particle["porosity"] = None
            return
        particle["inner_radius_px"] = inner_r
        particle["inner_diameter"] = inner_r * 2 * scale
        particle["shell_thickness"] = (outer_r - inner_r) * scale
        # Void fraction of a spherical shell goes as the cube of the radii.
        particle["porosity"] = float((inner_r / outer_r) ** 3)

    def _measure_shell(self, gx, gy, blurred, cx, cy, outer_r, w, h, n_angles=96):
        """Find the cavity wall inside a particle of radius ``outer_r``.

        Returns the inner radius, or None when no coherent inner boundary is
        there - a solid particle has none, and reporting a shell for one would
        be worse than reporting nothing. Circular gradients alone are not
        enough for that: a solid particle's own texture produces them, so the
        cavity has to be visibly a different brightness from the shell around
        it before a shell is reported.
        """
        lo = max(2.0, outer_r * 0.25)
        hi = outer_r * 0.92
        if hi - lo < 3:
            return None

        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        radii = np.arange(lo, hi, 0.5)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        xs = (cx + np.outer(cos_a, radii)).astype(int)
        ys = (cy + np.outer(sin_a, radii)).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        rows = np.clip(ys, 0, h - 1)
        cols = np.clip(xs, 0, w - 1)
        radial = gx[rows, cols] * cos_a[:, None] + gy[rows, cols] * sin_a[:, None]
        strength = np.where(valid, np.abs(radial), 0.0)

        peak = strength.max(axis=1)
        best = np.argmax(strength, axis=1)
        usable = (peak > 0) & (valid.sum(axis=1) >= len(radii) * 0.6)
        r_ang = np.where(usable, radii[best], np.nan)
        s_ang = np.where(usable, peak, 0.0)

        live = s_ang[s_ang > 0]
        if live.size < n_angles * 0.5:
            return None
        # Threshold at the 30th percentile so the coverage figure below is a
        # real test rather than an artefact of where the cut was made.
        fit = self._robust_circle_fit(cx, cy, angles, r_ang, s_ang,
                                      np.percentile(live, 30))
        if fit is None:
            return None
        _, _, inner_r, coverage, rms, _ = fit
        if coverage < 0.6 or rms > 0.12:
            return None
        if not (lo <= inner_r <= hi):
            return None

        def ring_level(r_from, r_to):
            samples = []
            for frac in (0.3, 0.5, 0.7):
                r = r_from + (r_to - r_from) * frac
                px = (cx + r * cos_a).astype(int)
                py = (cy + r * sin_a).astype(int)
                v = (px >= 0) & (px < w) & (py >= 0) & (py < h)
                if v.sum() < n_angles * 0.5:
                    return None
                samples.append(np.median(blurred[py[v], px[v]]))
            return float(np.median(samples))

        cavity = ring_level(0.15 * inner_r, 0.8 * inner_r)
        shell = ring_level(inner_r, outer_r)
        outside = ring_level(outer_r * 1.08, outer_r * 1.3)
        if cavity is None or shell is None:
            return None
        # The cavity has to stand out from the shell about as clearly as the
        # particle stands out from its surroundings - and no more clearly. A
        # solid particle's own texture clears a fixed threshold often enough to
        # be useless, and when it does it produces an "inner wall" sharper than
        # the particle's real edge, which no cavity can be: the shell wall is
        # seen through the shell, the outer edge is not.
        inner_contrast = abs(cavity - shell)
        edge_contrast = abs(shell - outside) if outside is not None else inner_contrast
        if inner_contrast < 25:
            return None
        if not (0.55 * edge_contrast <= inner_contrast <= 1.4 * edge_contrast):
            return None
        return float(inner_r)

    @staticmethod
    def _trace_boundary(gx, gy, cx, cy, r0, w, h, n_angles=96):
        """Follow the particle edge outward along rays from (cx, cy).

        The edge is taken as the outermost strong intensity transition, not the
        strongest one. Particles outlined by a dark rim cross two transitions
        going outward - into the rim and out of it - and the outer one is the
        boundary a person measures. Choosing by strength alone can land on
        either, and which one it picks flips with the rim's shape.

        Working from the outermost transition also avoids having to decide
        whether particles are darker or brighter than their surroundings, which
        is not even well defined when the interior and the gaps between
        particles are equally bright and only the rim stands out.
        """
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        radii = np.arange(max(2, r0 * 0.55), r0 * 1.5, 0.5)
        cos_a, sin_a = np.cos(angles), np.sin(angles)

        xs = (cx + np.outer(cos_a, radii)).astype(int)
        ys = (cy + np.outer(sin_a, radii)).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

        rows = np.clip(ys, 0, h - 1)
        cols = np.clip(xs, 0, w - 1)
        radial = (gx[rows, cols] * cos_a[:, None] + gy[rows, cols] * sin_a[:, None])
        strength = np.where(valid, np.abs(radial), 0.0)

        peak = strength.max(axis=1)
        enough = (peak > 0) & (valid.sum(axis=1) >= len(radii) * 0.5)
        rows_idx = np.arange(n_angles)

        strongest = np.argmax(strength, axis=1)

        # Crest of a transition, not its tail: the outermost sample that merely
        # clears the threshold sits on the fading flank of the gradient and
        # reads a radius that is systematically too large.
        crest = np.zeros_like(strength, dtype=bool)
        crest[:, 1:-1] = ((strength[:, 1:-1] >= strength[:, :-2])
                          & (strength[:, 1:-1] >= strength[:, 2:]))
        crest[:, 0] = crest[:, -1] = True
        significant = crest & (strength >= np.maximum(peak * 0.5, 1e-6)[:, None])
        outermost = strength.shape[1] - 1 - np.argmax(significant[:, ::-1], axis=1)
        has_outer = significant.any(axis=1)

        # A rimmed particle offers two crests on every ray - into the dark rim
        # and out of it - and noise decides which one clears the threshold. Left
        # to itself the choice flips from angle to angle and the traced outline
        # saws back and forth across the rim. Real boundaries move smoothly, so
        # the pick is tied to a smoothed running estimate of the radius: still
        # the outermost crest, but only among those near where the neighbouring
        # angles landed.
        outermost, has_outer = ParticleAnalyzer._regularize_trace(
            significant, strength, radii, outermost, has_outer, r0, prefer="outer")
        strongest, enough = ParticleAnalyzer._regularize_trace(
            significant, strength, radii, strongest, enough, r0, prefer="strong")

        def pick(index, ok):
            return (np.where(ok, radii[index], np.nan),
                    np.where(ok, strength[rows_idx, index], 0.0))

        r_strong, s_strong = pick(strongest, enough)
        r_outer, s_outer = pick(outermost, enough & has_outer)
        return angles, r_strong, s_strong, r_outer, s_outer

    @staticmethod
    def overlap_fraction(d, r1, r2):
        """Area shared by two circles, as a fraction of the smaller one's area."""
        small, large = min(r1, r2), max(r1, r2)
        if small <= 0 or d >= r1 + r2:
            return 0.0
        if d <= large - small:
            return 1.0            # the smaller circle is entirely inside
        d1 = (d * d + r1 * r1 - r2 * r2) / (2 * d)
        d2 = d - d1
        area = (r1 * r1 * np.arccos(np.clip(d1 / r1, -1, 1))
                - d1 * np.sqrt(max(r1 * r1 - d1 * d1, 0))
                + r2 * r2 * np.arccos(np.clip(d2 / r2, -1, 1))
                - d2 * np.sqrt(max(r2 * r2 - d2 * d2, 0)))
        return float(area / (np.pi * small * small))

    @staticmethod
    def _ring_evidence(blurred, cx, cy, radius, n_angles=48):
        """Fraction of directions where a dark ring sits at the boundary.

        This is what marks a particle in a micrograph whose gaps are as bright
        as its interiors: not what is inside the circle - interiors read 131 and
        the gaps 128 on the real images, which no interior test can separate -
        but whether the circle is outlined. A phantom sitting in the space
        between particles borrows pieces of its neighbours' rims and is outlined
        over part of its circumference; a real particle is outlined all round.
        """
        h, w = blurred.shape
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        steps = np.arange(0.70, 1.30, 0.02)
        votes = []
        for a in angles:
            xs = (cx + radius * steps * np.cos(a)).astype(int)
            ys = (cy + radius * steps * np.sin(a)).astype(int)
            ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            if ok.sum() < len(steps) * 0.8:
                continue
            prof = blurred[ys[ok], xs[ok]]
            k = int(np.argmin(prof))
            if k == 0 or k == len(prof) - 1:
                votes.append(False)
                continue
            depth = min(prof[:k].max() - prof[k], prof[k:].max() - prof[k])
            votes.append(depth >= 10)
        return float(np.mean(votes)) if votes else 0.0

    @staticmethod
    def _resolve_overlaps(particles, blurred=None, thresh=0.30):
        """Settle detections that sit on top of one another.

        Particles in a packed monolayer really do overlap in projection - up to
        about 0.19 of the smaller one on the packed fixtures, 0.39 where the
        generator deliberately overlaps a pair - so overlap alone cannot mean
        one of them is wrong, and dropping on overlap alone would delete real
        particles.

        What is wrong is a guess lying on top of a confident measurement: an
        approximated circle, most of whose boundary was carried across rather
        than found, overlapping a particle whose edge was actually traced. That
        one is excluded. Where both are equally confident the pair is left
        alone and marked, so it is visible rather than silently resolved.
        """
        for p in particles:
            p.setdefault("overlap", False)
        live = [p for p in particles if not p.get("excluded")]
        for i, p in enumerate(live):
            for q in live[i + 1:]:
                if p.get("excluded") or q.get("excluded"):
                    continue
                d = float(np.hypot(p["center_x"] - q["center_x"],
                                   p["center_y"] - q["center_y"]))
                if ParticleAnalyzer.overlap_fraction(d, p["radius_px"], q["radius_px"]) <= thresh:
                    continue
                if p.get("approx") != q.get("approx"):
                    loser = p if p.get("approx") else q
                    loser["excluded"] = True
                    loser["approx"] = False
                elif blurred is not None:
                    # Equally confident, so ask which one is actually outlined.
                    # Keeping both was inflating the count: a phantom in the gap
                    # between particles can pass every test applied to it alone,
                    # and only loses when set against the particle it overlaps.
                    for cand in (p, q):
                        if "ring_evidence" not in cand:
                            cand["ring_evidence"] = ParticleAnalyzer._ring_evidence(
                                blurred, cand["center_x"], cand["center_y"],
                                cand["radius_px"])
                    loser = p if p["ring_evidence"] < q["ring_evidence"] else q
                    winner = q if loser is p else p
                    loser["excluded"] = True
                    loser["approx"] = False
                    winner["overlap"] = True
                else:
                    p["overlap"] = q["overlap"] = True

    def _detect_enclosed(self, gray, min_area_px, n_angles=180):
        """Particles as bright interiors enclosed by their own dark shell.

        The dark shell network is segmented (Otsu), and each bright region it
        fully encloses is a candidate interior. Gap regions between packed
        particles are also bright and also enclosed, but they are concave
        curved triangles where an interior is a disc, so solidity and
        circularity separate them cleanly (0.57 vs 0.9+ on the real images).

        The outer boundary is then read per angle from the dark run that starts
        at the interior's edge: on automatic, the far side of the run where it
        is this particle's own wall and the middle of it where the run is two
        walls pressed together, which is what makes the reported size the outer
        diameter without measuring through the neighbour as well. A fixed
        edge_level pins every angle to one position across the run instead,
        exactly as it does across a traced rim.
        """
        blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
        _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Enclosure is all-or-nothing: a wall that thins below the threshold
        # for a few pixels opens the interior into the gap beside it, the two
        # merge into one lumpy region, and the particle is not just mismeasured
        # but never proposed - no candidate at all, which is what the missed
        # particles on the real micrographs had in common. Sealing pinholes
        # before labelling costs nothing where the ring is already closed.
        # The ray march still uses the raw mask, so wall thickness - and with
        # it the measured boundary - is unaffected by the repair.
        sealed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(sealed), 8)
        h, w = gray.shape
        dark_mask = dark > 0

        # Which bright regions are open background rather than a particle's
        # inside: the ones that run off the frame. This is what tells a shell
        # wall from a plain edge. A wall has something of the same kind on its
        # far side - a neighbour's interior, or a small enclosed gap - and the
        # particle extends across it, so the boundary is the wall's far side. A
        # solid particle's edge has open background beyond it and the particle
        # stops where the dark starts. Brightness alone cannot tell the two
        # apart: on the grainy fixture the background is as bright as the real
        # micrographs' gaps, and reading its edge as a wall put every particle
        # 13% oversize. Which region the ray lands in says it outright.
        background = {j for j in range(1, n)
                      if stats[j][0] <= 1 or stats[j][1] <= 1
                      or stats[j][0] + stats[j][2] >= w - 1
                      or stats[j][1] + stats[j][3] >= h - 1}

        # Which bright regions are particle interiors is settled before any ray
        # is cast, because a ray needs the answer about the region it lands in,
        # not just the one it started from. Whether a wall is shared is then a
        # fact rather than a guess: the far side is another interior, or it is
        # not. Judging it by the run being thicker than usual - the obvious
        # heuristic - misses every pair whose walls sit close enough to read as
        # one ordinary run, and those are exactly the crowded neighbours where
        # being wrong costs a full wall thickness. A third of the circles on
        # the real micrographs came out that much too large.
        interiors = {}
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            if area < max(min_area_px * 0.3, 60):
                continue
            comp = (labels[y:y + hh, x:x + ww] == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cnts, key=cv2.contourArea)
            a = cv2.contourArea(cnt)
            if a < 40:
                continue
            solidity = a / max(cv2.contourArea(cv2.convexHull(cnt)), 1)
            perim = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * a / max(perim * perim, 1)
            if solidity < 0.88 or circularity < 0.55:
                continue          # a gap between particles, not an interior
            interiors[i] = cnt

        out = []
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        for i, cnt in interiors.items():
            x, y, ww, hh, area = stats[i]
            if x <= 1 or y <= 1 or x + ww >= w - 1 or y + hh >= h - 1:
                continue          # cut by the frame; the outer edge is not all there
            a = cv2.contourArea(cnt)
            m = cv2.moments(cnt)
            cx = m["m10"] / m["m00"] + x
            cy = m["m01"] / m["m00"] + y
            r_in = np.sqrt(a / np.pi)

            # March each ray from inside the interior out across the shell.
            radii = np.arange(max(1.0, r_in * 0.5), r_in * 2.4, 0.5)
            xs = np.clip((cx + np.outer(cos_a, radii)).astype(int), 0, w - 1)
            ys = np.clip((cy + np.outer(sin_a, radii)).astype(int), 0, h - 1)
            on_dark = dark_mask[ys, xs]
            # Every ray at once. The march is a scan for one run of dark per
            # ray with a handful of conditions on it, and doing that a ray at a
            # time in Python cost more than the rest of the detector put
            # together on a full field.
            n_r = len(radii)
            has_dark = on_dark.any(axis=1)
            k0 = np.argmax(on_dark, axis=1)

            # The run ends at the first step whose next two samples are both
            # bright - one bright sample is speckle, not the far side of a
            # wall. Running out of samples ends it too.
            stop = np.ones_like(on_dark)
            stop[:, :n_r - 1] = ~on_dark[:, 1:]
            stop[:, :n_r - 2] &= ~on_dark[:, 2:]
            reach = np.arange(n_r)[None, :] >= k0[:, None]
            k1 = np.argmax(stop & reach, axis=1)

            # A dark run that reaches the end of the search never found the
            # shell's outer edge: it is background, not a wall. On a solid
            # bright disc on a dark ground every ray is like this, and
            # accepting it would draw a circle out at the search limit. Such
            # rays are left unmeasured, so the detector simply finds nothing
            # there and the ray-based path handles the image instead.
            live = has_dark & (stop & reach).any(axis=1) & (k1 < n_r - 1)

            # Leaving a wall means arriving somewhere - a gap or the
            # neighbour's interior - and staying there. On a noisy image whose
            # "wall" is really the background, two bright specks in a row end
            # the run just as convincingly, and the boundary then lands
            # wherever the noise happened to clear: the grainy fixture came out
            # 13% oversized that way. Requiring the brightness to hold past the
            # exit costs nothing on a genuine wall and gives those rays no
            # reading at all, which drops the particle back to the ray-based
            # path that handles it well.
            off = np.arange(1, 9)[None, :]
            pos = k1[:, None] + off
            inside_r = pos < n_r
            past = np.take_along_axis(on_dark, np.clip(pos, 0, n_r - 1), axis=1)
            n_past = inside_r.sum(axis=1)
            with np.errstate(invalid="ignore"):
                past_dark = np.where(inside_r, past, 0).sum(axis=1) / np.maximum(n_past, 1)
            live &= (n_past >= 4) & (past_dark <= 0.25)

            # What the ray lands in has to be a region in its own right - a
            # neighbour's interior or an enclosed gap - and not the open
            # background. In a noisy image the far side of a supposed wall is a
            # scatter of bright specks a few pixels across, which is neither,
            # and those rays are what dragged the grainy fixture oversize even
            # after the sustained-brightness check.
            probe = np.minimum(k1 + 4, n_r - 1)
            beyond_label = labels[ys[np.arange(n_angles), probe],
                                  xs[np.arange(n_angles), probe]]
            area_ok = np.zeros(n_angles, bool)
            shared = np.zeros(n_angles, bool)
            floor = max(60.0, 0.03 * area)
            for j in np.flatnonzero(live):
                lab = beyond_label[j]
                if lab == 0 or lab in background or stats[lab][4] < floor:
                    continue
                area_ok[j] = True
                shared[j] = lab in interiors
            live &= area_ok

            inner_r = np.where(live, radii[k0], np.nan)
            outer_r = np.where(live, radii[k1], np.nan)

            thickness = outer_r - inner_r
            good = np.isfinite(thickness)
            if good.sum() < n_angles * 0.4:
                continue

            # A shell is bright on both sides: the interior it encloses, and
            # the gap or neighbour beyond it. On a solid bright disc on a dark
            # ground the "shell" is the background, dark all the way out, and
            # noise gives it a scatter of false exits - so the level just past
            # the run is dark, well below the interior. That is how this path
            # knows not to fire on an image the ray-based path already handles.
            interior_level = float(blur[np.clip(int(cy), 0, h - 1),
                                        np.clip(int(cx), 0, w - 1)])
            past = np.clip(np.nan_to_num(outer_r + 3, nan=0.0), 0, radii[-1])
            gx = np.clip((cx + past * cos_a).astype(int), 0, w - 1)
            gy = np.clip((cy + past * sin_a).astype(int), 0, h - 1)
            outside_level = float(np.nanmedian(np.where(good, blur[gy, gx], np.nan)))
            interior_disc = float(np.median(blur[
                np.clip((cy + 0.4 * r_in * sin_a).astype(int), 0, h - 1),
                np.clip((cx + 0.4 * r_in * cos_a).astype(int), 0, w - 1)]))
            if outside_level < interior_disc - 30:
                continue

            med_t = float(np.nanmedian(thickness[good]))
            # A run several times the usual thickness is not one wall seen
            # edge-on but a pile the ray never got out of; there is no surface
            # to read in it either way.
            good &= thickness <= max(med_t * 2.0, med_t + 3.0)
            if good.sum() < 12:
                continue

            # The surface, per ray. Where the far side of the run is another
            # particle's interior the run is two walls together and the surface
            # is the contact plane halfway across; where it is a gap the run is
            # this particle's own wall and the surface is its far edge.
            if self.edge_level is None:
                frac = np.where(shared, 0.5, 1.0)
            else:
                frac = self.edge_level
            level_r = inner_r + frac * (outer_r - inner_r)
            pts = np.column_stack([cx + level_r[good] * cos_a[good],
                                   cy + level_r[good] * sin_a[good]])
            ux, uy, rad = float(cx), float(cy), float(np.nanmedian(level_r[good]))
            keep = pts
            for _ in range(3):
                A = np.column_stack([2 * keep[:, 0], 2 * keep[:, 1], np.ones(len(keep))])
                sol, *_ = np.linalg.lstsq(A, keep[:, 0] ** 2 + keep[:, 1] ** 2, rcond=None)
                ux, uy = float(sol[0]), float(sol[1])
                rad = float(np.sqrt(sol[2] + ux * ux + uy * uy))
                resid = np.abs(np.hypot(keep[:, 0] - ux, keep[:, 1] - uy) - rad)
                ok = resid < 2.5 * np.median(resid) + 1
                if ok.all():
                    break
                keep = keep[ok]
                if len(keep) < 12:
                    break
            if not (r_in * 0.8 <= rad <= r_in * 2.2):
                continue
            # The free fit is allowed to move the centre, but not to leave the
            # interior it was found from. A particle whose neighbours crowd one
            # side and whose other side faces gaps gets its wall read more
            # confidently on the open side, and the fit slides that way - the
            # circle keeps the right size but sits off the particle, bulging
            # past it on one edge, which is what reads as "too big" even though
            # the radius is right. The enclosed interior's own centroid has no
            # such lean: it is the region's centre of area, and every ray was
            # cast from it. When the fit disagrees with it by more than a
            # sixth of a radius, the fit is the thing that moved.
            if np.hypot(ux - cx, uy - cy) > 0.16 * rad:
                ux, uy = float(cx), float(cy)
                rad = float(np.nanmedian(level_r[good]))
                keep = np.column_stack([cx + level_r[good] * cos_a[good],
                                        cy + level_r[good] * sin_a[good]])
            if np.pi * rad * rad < min_area_px:
                continue

            resid = np.abs(np.hypot(keep[:, 0] - ux, keep[:, 1] - uy) - rad)
            rms = float(np.sqrt(np.mean(resid ** 2)) / max(rad, 1))
            coverage = float(good.mean())
            free_frac = float((good & ~shared).sum() / max(good.sum(), 1))

            p = self._measure_circle(int(round(ux)), int(round(uy)), rad,
                                     np.pi * rad * rad)
            outline, measured = self._build_outline(ux, uy, rad, keep)
            p["contour"] = outline.reshape(-1, 1, 2).astype(np.float32)
            p["contour_measured"] = measured
            p["coverage"] = coverage
            # Stricter than the traced path's 0.20, and for a reason particular
            # to this one. A shell surrounds its particle, so where the wall is
            # genuinely a wall it is found nearly all the way round - three
            # quarters of the rays on the real micrographs. A gap between two
            # solid particles that merely happen to sit close reads the same on
            # the few rays that point at a neighbour and nowhere else, which is
            # exactly the grainy fixture: a nearly-touching grid whose widest
            # agreement is 0.59. Asking for most of the circumference keeps the
            # walls and drops the coincidences; the particles this turns away
            # are still found by the traced path, just measured its way.
            # The residual bound is tighter than the traced path's too. This
            # path overrides a measurement the traced path already made, so a
            # ragged fit here replaces a good number with a worse one: on the
            # bright-core fixture the loose bound let marginal arcs through and
            # tripled the spread. Where the arcs disagree, the traced path
            # keeps the particle.
            # Two ways to earn the measurement, and the second is what keeps
            # this path from having to demand most of the circumference.
            # A shell is visible against open gaps somewhere, so rays that
            # cross the wall and come out into a gap are direct evidence the
            # wall belongs to this particle. A gap between two solid particles
            # that merely sit close has none of them at all - every ray that
            # found a "wall" was pointing at a neighbour, because the rest run
            # off into background and are discarded. On the grainy fixture the
            # free-ray fraction is exactly zero for every candidate, against a
            # third to four fifths on the real micrographs. Demanding
            # circumference instead of evidence was costing real particles at
            # 0.60 coverage, and the traced path then drew a larger circle
            # spanning the neighbour in their place.
            enough = ((coverage >= 0.35 and free_frac >= 0.05)
                      or coverage >= 0.60)
            p["excluded"] = not (enough and rms <= 0.12)
            # "근사" means the circle rests on a partial arc and the rest is
            # extrapolation. With the wall seen most of the way round and the
            # arcs agreeing closely, it is a measurement, not an estimate;
            # holding it to three quarters of the circumference marked two
            # thirds of a packed field approximate and made the colour
            # meaningless.
            p["approx"] = (not p["excluded"]) and not (coverage >= 0.60 and rms <= 0.10)
            out.append(p)
        return out

    @staticmethod
    def _refine_seed(blurred, cx, cy, r, w, h, n_angles=120, depth_min=8.0):
        """Re-centre a seed on the dark rim around it.

        The rim is looked for as a two-sided dip along each ray - darker than
        both what lies inside it and what lies outside - and a circle is fitted
        to the points found, discarding those that disagree with the rest. Rays
        that meet a neighbour's rim rather than this particle's are the ones
        that disagree, so that rejection is what keeps the seed on its own
        particle.

        The seed is returned unchanged when too few rays find a rim, or when
        the fit wants to move further than a seed error plausibly explains:
        this corrects a placement, it does not go looking for a different
        particle.
        """
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        fracs = np.arange(0.72, 1.32, 0.02)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        xs = (cx + r * np.outer(cos_a, fracs)).astype(int)
        ys = (cy + r * np.outer(sin_a, fracs)).astype(int)
        good = ((xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)).all(axis=1)
        if good.sum() < n_angles * 0.4:
            return cx, cy, r
        prof = blurred[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)].astype(np.float32)
        points = []
        for i in np.flatnonzero(good):
            row = prof[i]
            k = int(np.argmin(row))
            if k == 0 or k == len(row) - 1:
                continue
            if min(row[:k].max() - row[k], row[k:].max() - row[k]) < depth_min:
                continue
            rr = r * fracs[k]
            points.append((cx + rr * cos_a[i], cy + rr * sin_a[i]))
        if len(points) < 20:
            return cx, cy, r
        pts = np.array(points)
        ux, uy, rad = float(cx), float(cy), float(r)
        for _ in range(4):
            A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
            sol, *_ = np.linalg.lstsq(A, pts[:, 0] ** 2 + pts[:, 1] ** 2, rcond=None)
            ux, uy = float(sol[0]), float(sol[1])
            rad = float(np.sqrt(sol[2] + ux * ux + uy * uy))
            resid = np.abs(np.hypot(pts[:, 0] - ux, pts[:, 1] - uy) - rad)
            keep = resid < 2.5 * np.median(resid) + 1
            if keep.all():
                break
            pts = pts[keep]
            if len(pts) < 20:
                break
        if (np.hypot(ux - cx, uy - cy) > 0.30 * r or not 0.70 * r <= rad <= 1.30 * r
                or not (0 <= ux < w and 0 <= uy < h)):
            return cx, cy, r
        return int(round(ux)), int(round(uy)), int(round(rad))

    @staticmethod
    def _reject_clipped(particles, shape):
        """Exclude particles the frame cuts through.

        Half a boundary cannot fix a diameter: what is measured is the visible
        arc, and the circle through it is an extrapolation the image does not
        support. Those circles sit noticeably off their particle and bulge past
        it, which is what a reader sees as "the circle is bigger than the
        particle" - they were a sixth of the detections on the real
        micrographs. Excluding them is what the documentation has always said
        happens; it was simply never implemented.
        """
        h, w = shape[:2]
        for p in particles:
            if p.get("excluded"):
                continue
            cx, cy, r = p["center_x"], p["center_y"], p["radius_px"]
            if cx - r < 0 or cy - r < 0 or cx + r > w or cy + r > h:
                p["excluded"] = True
                p["approx"] = False
                p["clipped"] = True

    @staticmethod
    def _reject_buried(particles, thresh=0.35):
        """Exclude a circle whose area is mostly other particles' area.

        A detection that sits in the space between particles is held up by
        their rims rather than one of its own, and the giveaway is that most of
        what it encloses already belongs to its neighbours. A real particle
        occupies its own ground: on the fixtures where the truth is known, no
        true particle is more than 13% covered by the others, while these sit
        above 40%.

        Removing one frees the area it was contributing, so the worst offender
        goes first and the rest are re-measured, rather than condemning a group
        that only looks buried because of each other.

        Only the neighbours of the one just removed can have changed, though -
        a circle's coverage depends on nothing else - so the rest keep their
        measurement instead of the whole field being rasterised again on every
        round.
        """
        live = [p for p in particles if not p.get("excluded")]
        if len(live) < 2:
            return
        centres = np.array([[p["center_x"], p["center_y"]] for p in live], float)
        radii = np.array([p["radius_px"] for p in live], float)
        gap = (np.hypot(centres[:, None, 0] - centres[None, :, 0],
                        centres[:, None, 1] - centres[None, :, 1])
               - radii[:, None] - radii[None, :])
        np.fill_diagonal(gap, 1.0)
        neighbours = [np.flatnonzero(row <= 0) for row in gap]
        dropped = np.zeros(len(live), bool)

        def coverage(i):
            p = live[i]
            r = int(round(p["radius_px"]))
            if r < 2:
                return 0.0
            cx, cy = p["center_x"], p["center_y"]
            size = 2 * r + 1
            mine = np.zeros((size, size), np.uint8)
            others = np.zeros_like(mine)
            cv2.circle(mine, (r, r), r, 1, -1)
            for j in neighbours[i]:
                if dropped[j]:
                    continue
                q = live[j]
                cv2.circle(others, (q["center_x"] - cx + r, q["center_y"] - cy + r),
                           int(round(q["radius_px"])), 1, -1)
            return float(np.count_nonzero(mine & others)) / max(np.count_nonzero(mine), 1)

        covered = np.array([coverage(i) for i in range(len(live))])
        while True:
            covered[dropped] = 0.0
            i = int(np.argmax(covered))
            if covered[i] <= thresh:
                return
            dropped[i] = True
            live[i]["excluded"] = True
            live[i]["approx"] = False
            for j in neighbours[i]:
                if not dropped[j]:
                    covered[j] = coverage(j)

    @staticmethod
    def _reject_implausible_interiors(particles, gray, floor=20.0, k=6.0):
        """Exclude detections whose inside does not look like the other particles'.

        A jammed monolayer contains two things that fit a circle well but are
        not particles: the curved triangular gap between three touching
        particles, and the lens where two particles overlap in projection. The
        gap reads at the background level and the lens reads darker than any
        real particle, while the particles themselves - being the same material
        imaged the same way - sit in a tight band.

        The band is measured from the detections themselves rather than assumed,
        so a mixed population simply widens it (via the MAD) instead of having
        half of it thrown away. ``floor`` keeps a very uniform sample from
        rejecting on noise-sized deviations.
        """
        candidates = [p for p in particles if not p.get("excluded")]
        if len(candidates) < 5:
            return

        h, w = gray.shape
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        cos_a, sin_a = np.cos(angles), np.sin(angles)

        # Read the ring just inside the wall, not the whole disc. A real
        # preparation contains a minority of particles whose template was never
        # removed, and they are far darker in the middle than the rest - which
        # is precisely what a rule that learns "what a particle looks like"
        # from the population will throw away. Averaged over the whole disc
        # they sat well outside the band and every one of them was excluded.
        # The band just inside the wall is the part that is the same for all of
        # them, cored or not, while a gap or an overlap lens has no such
        # structure and reads its own level there too.
        levels = []
        for p in candidates:
            samples = []
            for frac in (0.60, 0.72, 0.84):
                r = p["radius_px"] * frac
                xs = np.clip((p["center_x"] + r * cos_a).astype(int), 0, w - 1)
                ys = np.clip((p["center_y"] + r * sin_a).astype(int), 0, h - 1)
                samples.append(np.median(blurred[ys, xs]))
            levels.append(float(np.median(samples)))

        levels = np.array(levels)
        centre = np.median(levels)
        mad = np.median(np.abs(levels - centre))
        tol = max(k * mad, floor)
        for p, level in zip(candidates, levels):
            if abs(level - centre) > tol:
                p["excluded"] = True
                p["approx"] = False

    @staticmethod
    def _smooth_radii(r_ang, win=9):
        """Circular median filter that ignores, and then fills, missing angles."""
        n = len(r_ang)
        offsets = np.arange(-(win // 2), win // 2 + 1)
        window = r_ang[(np.arange(n)[:, None] + offsets[None, :]) % n]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
            smoothed = np.nanmedian(window, axis=1)
        if np.isnan(smoothed).any():
            fill = np.nanmedian(r_ang) if not np.isnan(r_ang).all() else np.nan
            smoothed = np.where(np.isnan(smoothed), fill, smoothed)
        return smoothed

    @staticmethod
    def _regularize_trace(significant, strength, radii, index0, ok0, r0,
                          prefer="outer", rounds=3, band=0.18):
        """Re-pick each angle's edge near where its neighbours put theirs.

        ``prefer`` keeps the original selection rule - the outermost crest, or
        the strongest one - but applies it only to candidates close to a
        smoothed running estimate of the radius. Angles whose candidates all sit
        far from that estimate are dropped rather than forced, so a genuine gap
        in the edge stays a gap instead of becoming an invented radius.
        """
        reference = ParticleAnalyzer._smooth_radii(
            np.where(ok0, radii[index0], np.nan))
        if np.isnan(reference).all():
            return index0, ok0

        tol = max(2.0, 0.06 * r0)
        index, ok = index0, ok0
        for _ in range(rounds):
            near = significant & (np.abs(radii[None, :] - reference[:, None]) <= tol)
            # Only ever narrows the accepted angles: an angle the caller had
            # already rejected must not come back just because it now has a
            # candidate near the smoothed radius.
            ok = ok0 & near.any(axis=1)
            if not ok.any():
                return index0, ok0
            if prefer == "strong":
                index = np.argmax(np.where(near, strength, -np.inf), axis=1)
            else:
                index = near.shape[1] - 1 - np.argmax(near[:, ::-1], axis=1)
            updated = ParticleAnalyzer._smooth_radii(np.where(ok, radii[index], np.nan))
            # Local smoothness alone lets the estimate walk: a long run of
            # angles facing the gap between particles finds the *neighbour's*
            # rim, and each one is a small step from the last, so the reference
            # follows them outward and the outline reaches into the gap. These
            # are spherical particles, so hold the estimate to a band about its
            # own median and let those angles go unmatched instead.
            median = np.nanmedian(updated)
            if np.isfinite(median):
                updated = np.clip(updated, median * (1 - band), median * (1 + band))
            if np.allclose(updated, reference, equal_nan=True):
                break
            reference = updated
        return index, ok

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

        # A point more than a few percent off the fitted circle is not on this
        # particle - in a packed field it is a neighbour's rim, found by a ray
        # that crossed the gap between them. The MAD rejection above cannot
        # remove those on its own: enough of them inflate the MAD that lets
        # them stay. Dropping them here keeps the outline off the gaps.
        #
        # Coverage deliberately still counts the angles that had a usable edge
        # at all: a particle pressed against its neighbours is measurable from
        # the arc that is visible, and counting only the survivors here would
        # push exactly those out of the results.
        resid = np.abs(np.hypot(pts[:, 0] - ux, pts[:, 1] - uy) - r)
        tight = resid <= max(0.10 * r, 1.5)
        if tight.sum() >= 8:
            pts, resid = pts[tight], resid[tight]
        rms = np.sqrt(np.mean(resid ** 2)) / max(r, 1)

        outline, measured = ParticleAnalyzer._build_outline(ux, uy, r, pts)
        return ux, uy, r, coverage, rms, (outline, measured)

    @staticmethod
    def _build_outline(ux, uy, r, pts, n=180):
        """A closed outline round the particle, from the points that were found.

        The outline follows the traced edge rather than the fitted circle -
        these particles are not perfectly round, and a drawn circle visibly
        leaves the edge - but it is drawn all the way round. Where a particle
        is pressed against its neighbours there is no edge to find on the
        contact side, and leaving those stretches blank drew a boundary in
        pieces; the radius is carried across them from the arcs on either side
        instead, which is the same thing the fitted circle does for the
        measurement.
        """
        theta = np.arctan2(pts[:, 1] - uy, pts[:, 0] - ux)
        order = np.argsort(theta)
        theta, radial = theta[order], np.hypot(pts[order, 0] - ux, pts[order, 1] - uy)
        if len(radial) >= 5:
            # Median over neighbouring angles: a raw trace steps a pixel at a
            # time between them, and those steps read as a sawtooth once the
            # view is scaled up.
            radial = np.median([np.roll(radial, k) for k in (-2, -1, 0, 1, 2)], axis=0)

        grid = np.linspace(-np.pi, np.pi, n, endpoint=False)
        if len(theta) >= 2:
            # np.interp is periodic once the sequence is wrapped, so gaps are
            # bridged the short way round rather than across the whole circle.
            wrapped_t = np.concatenate([theta - 2 * np.pi, theta, theta + 2 * np.pi])
            wrapped_r = np.tile(radial, 3)
            radius = np.interp(grid, wrapped_t, wrapped_r)
            # Mark which stretches rest on a found edge and which were carried
            # across, so the drawing can say which is which instead of
            # presenting a guess as an observation. The test is nearness to an
            # actual sample: judging it by the width of the bracketing interval
            # instead calls the whole outline inferred as soon as the surviving
            # samples are a little sparse, however well the edge was seen.
            gap = np.abs(grid[:, None] - theta[None, :])
            nearest = np.minimum(gap, 2 * np.pi - gap).min(axis=1)
            step = np.median(np.diff(theta)) if len(theta) > 2 else np.deg2rad(4)
            measured = nearest <= max(np.deg2rad(5), float(step))
        else:
            radius = np.full(n, r)
            measured = np.zeros(n, bool)

        outline = np.column_stack([ux + radius * np.cos(grid), uy + radius * np.sin(grid)])
        return outline, measured

    @staticmethod
    def _merge_detections(primary, secondary):
        """Combine two detector outputs, keeping ``primary`` where they clash.

        A particle from ``secondary`` is only added where nothing already
        occupies that spot, so a contour blob spanning a touching pair is
        dropped in favour of the two traced circles inside it.
        """
        merged = [p for p in primary if not p.get("excluded")]
        rejected = [p for p in primary if p.get("excluded")]

        # A candidate can clash with one accepted before it, so the additions
        # stay sequential - but each candidate is tested against everything
        # kept so far in one comparison rather than a Python loop over it.
        xs = [p["center_x"] for p in merged]
        ys = [p["center_y"] for p in merged]
        rs = [p["radius_px"] for p in merged]
        cx = np.array(xs, float)
        cy = np.array(ys, float)
        cr = np.array(rs, float)
        for cand in secondary:
            if cand.get("excluded"):
                continue
            x, y, r = cand["center_x"], cand["center_y"], cand["radius_px"]
            if cx.size:
                d = np.hypot(cx - x, cy - y)
                if (d < 0.8 * np.maximum(cr, r)).any():
                    continue
            merged.append(cand)
            cx = np.append(cx, x)
            cy = np.append(cy, y)
            cr = np.append(cr, r)

        # A circle that swallows two accepted centres is a fused pair, not a
        # particle, whichever detector produced it.
        keep = []
        if merged:
            near = (np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
                    < cr[:, None])
            np.fill_diagonal(near, False)
            inside = near.sum(axis=1)
        else:
            inside = np.zeros(0, int)
        for p, n in zip(merged, inside):
            if n >= 2:
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
        """Row where the black info bar at the bottom starts, or h if there is none.

        The band has to be walked up to its top edge: returning the first dark
        row met from the bottom returns the band's *bottom*, which leaves the
        whole strip inside the analysis region, and the bar and its lettering
        then get measured as particles.
        """
        h, w = gray.shape
        limit = int(h * 0.7)
        dark = (gray < 30).sum(axis=1) / w > 0.5
        if not dark[h - 1]:
            return h
        row = h - 1
        while row > limit and dark[row - 1]:
            row -= 1
        return row

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

        smoothed = cv2.GaussianBlur(dist, (0, 0), sigmaX=max(1, max_val * 0.08))
        # Keep separate distance peaks in modestly overlapping clusters.  The
        # previous half-radius non-maximum window collapsed a three-particle
        # clump into one or two seeds before watershed ever had a chance to
        # split it; a quarter-radius window still suppresses rim/noise plateaus
        # while preserving one marker per particle centre.
        ksize = max(3, int(max_val * 0.25)) | 1
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
        shelled = [p for p in particles if p.get("shell_thickness") is not None]
        if shelled:
            stats["shell_count"] = len(shelled)
            stats["shell_mean"] = float(np.mean([p["shell_thickness"] for p in shelled]))
            stats["shell_std"] = float(np.std([p["shell_thickness"] for p in shelled]))
            stats["porosity_mean"] = float(np.mean([p["porosity"] for p in shelled]))
            stats["inner_mean"] = float(np.mean([p["inner_diameter"] for p in shelled]))

        if particles and "has_core" in particles[0]:
            core_count = sum(1 for p in particles if p["has_core"])
            stats["core_count"] = core_count
            stats["core_ratio"] = core_count / len(particles)
        return stats
