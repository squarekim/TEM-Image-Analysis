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

    #: How much of the winning band's score a second scale must carry to be
    #: run as well. A genuine second population shows up in numbers; a few
    #: circles at an odd size are stragglers of the first.
    SCALE_MIN_SHARE = 0.10

    #: At most this many size scales are searched. Each one costs a full pass.
    MAX_SCALES = 3

    #: How far a band's traced edges may stray from the circle fitted to them,
    #: as a share of radius, for that band to be considered the particle scale
    #: at all. Real particles' bands measure 0.005-0.025 on every image tested,
    #: real and synthetic; bands that are actually image grain measure
    #: 0.040-0.055. The cut sits in the empty gap between them.
    BAND_MAX_ROUGHNESS = 0.030

    #: How much deeper than the interior the dark band must be, as a share of
    #: its contrast, for one ray to call it a shell wall. Set below what a
    #: shell actually gives (0.4 upwards) rather than at the midpoint, because
    #: the discrimination that matters is done by RIM_MIN_RAYS instead.
    RIM_DEPTH_FRAC = 0.35

    #: What share of a particle's rays must see that wall. This, not the depth,
    #: is what tells a shell from noise. A hollow particle shows its wall on
    #: essentially every ray - the fixtures give 97-100% - while a noisy solid
    #: particle produces troughs deep enough to pass on a scattered 57% of
    #: them. Judging by depth alone, or by a simple majority, put the two on
    #: the same side.
    RIM_MIN_RAYS = 0.85

    #: A ray faces a touching neighbour, rather than open background, when the
    #: level just outside the dark band is nearly the particle's own interior
    #: level - the neighbour's interior is as bright as ours, a gap is not.
    #: Judged as a fraction of the band's own contrast so it does not depend on
    #: exposure or magnification.
    CONTACT_GAP_FRAC = 0.30

    def __init__(self, nm_per_px=None, edge_level="auto", sphere_edge=False):
        self.nm_per_px = nm_per_px
        #: None means decide per ray; a float pins every ray to that position.
        self.edge_level = (None if edge_level is None or edge_level == "auto"
                           else float(np.clip(edge_level, 0.0, 1.0)))
        #: Place the boundary by extrapolating the sphere's own edge profile
        #: rather than by a brightness threshold. See `_sphere_edge_radius`.
        self.sphere_edge = bool(sphere_edge)
        #: Optional callback(fraction, label) for a progress gauge. A dense
        #: field takes tens of seconds and the caller (the GUI) otherwise looks
        #: frozen; this lets it draw a bar. It is advisory only - the numbers
        #: are approximate and a callback that raises is dropped, never allowed
        #: to interrupt the analysis.
        self._progress = None

    def _report(self, frac, label=None):
        if self._progress is None:
            return
        if label is not None:
            self._progress_label = label
        try:
            self._progress(float(min(1.0, max(0.0, frac))),
                           getattr(self, "_progress_label", ""))
        except Exception:
            self._progress = None

    def _tick(self, window, i, n, label=None):
        """Report progress i/n within a (lo, hi) fraction window."""
        if self._progress is None or window is None:
            return
        lo, hi = window
        self._report(lo + (hi - lo) * (min(i, n) / max(1, n)), label)

    def analyze(self, image, min_area_px=100, max_area_px=None,
                circularity_thresh=0.5, use_watershed=True, hollow=False,
                detect_cores=False, measure_shell=False, progress=None):
        self._progress = progress
        try:
            return self._analyze(image, min_area_px, max_area_px, circularity_thresh,
                                 use_watershed, hollow, detect_cores, measure_shell)
        finally:
            self._report(1.0, "완료")
            self._progress = None

    def _analyze(self, image, min_area_px, max_area_px, circularity_thresh,
                 use_watershed, hollow, detect_cores, measure_shell):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        self._report(0.0, "전처리")
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
        self._report(0.08, "입자 검출")
        hough_particles = self._detect_hough(analysis_region, min_area_px,
                                             pw=(0.08, 0.36))
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
        self._report(0.36, "내부 검출")
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

        # Geometry first, judgement second. Every rule below asks what a circle
        # encloses or what lies on it, and the answer is worthless while the
        # circle is in the wrong place: a real particle whose circle was fitted
        # a third too large has no ring on its circumference either, and gets
        # thrown out with the phantoms. Correcting the circle first cost
        # nothing - the corrections only move circles, they do not create them -
        # and recovered particles a reader had marked as missed.
        self._report(0.40, "중심 보정")
        self._recentre_on_ring(particles, analysis_region, pw=(0.40, 0.48))
        self._report(0.48, "크기 보정")
        self._resize_by_ring(particles, analysis_region)
        self._report(0.52, "경계 정렬")
        self._snap_to_wall(particles, analysis_region, pw=(0.52, 0.82))
        self._report(0.82, "판별")
        self._reject_implausible_interiors(particles, analysis_region)
        if self.sphere_edge:
            self._refine_by_sphere_edge(particles, analysis_region)
        self._reject_clipped(particles, analysis_region.shape)
        self._resolve_overlaps(particles, cv2.GaussianBlur(analysis_region, (0, 0), 1.5))
        self._reject_buried(particles)
        self._reject_unoutlined(particles,
                                cv2.GaussianBlur(analysis_region, (0, 0), 1.5))
        self._reject_annotated(particles, analysis_region)

        # Last, and only where nothing is left: a solid dark particle has no
        # bright interior for the main path to enclose, so it is not that its
        # candidate was rejected - there was never a candidate. Asking earlier
        # does not work either, because at that point the field is still full
        # of candidates that will be rejected, and one of them is sitting on
        # the dark particle; it looks covered right up until it is not.
        self._report(0.90, "누락 입자 탐색")
        extra = self._detect_dark_bodies(analysis_region, particles)
        if extra:
            particles = particles + extra
            self._resize_by_ring(particles, analysis_region)
            self._snap_to_wall(particles, analysis_region, pw=(0.90, 0.94))
            self._reject_clipped(particles, analysis_region.shape)
            self._reject_annotated(particles, analysis_region)
        self._report(0.94, "마무리")
        self._flag_defects(particles, analysis_region)

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

    def _detect_hough(self, gray, min_area_px, pw=None):
        """Detect at every size scale the image actually contains.

        One scale is chosen for the whole image, and everything downstream is
        matched to it: the seed search runs from 0.6 to 1.5 of it, and the
        gradients every boundary is traced on are smoothed to it. A sample with
        one size is served perfectly by that and a sample with two is not. On
        the hard fixture, whose scale is set by particles of radius 25-42, the
        four particles of radius 6-9 fall outside the seed search entirely and
        come out 18.5% oversize; the same four, cropped out and analysed on
        their own, come out 2.7% - so it is the company they keep, not the
        particles.

        So the scales are detected and each is run in full. Which extra scales
        are real is the same question the band scan already answers by
        roundness, and only bands that clear it are eligible, so this cannot
        turn image grain into a second population.
        """
        h, w = gray.shape
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        min_r = max(5, int(np.sqrt(min_area_px / np.pi)))
        max_r = max(min_r + 1, min(h, w) // 3)

        scales = self._estimate_radius(gray, blurred, min_r, max_r, w, h)
        if not scales:
            return []
        found = []
        for si, est in enumerate(scales):
            sub = None
            if pw is not None:
                lo, hi = pw
                sub = (lo + (hi - lo) * si / len(scales),
                       lo + (hi - lo) * (si + 1) / len(scales))
            part = self._detect_at_scale(gray, blurred, min_area_px, est,
                                         min_r, w, h, pw=sub)
            found = self._merge_detections(found, part) if found else part
        return found

    def _detect_at_scale(self, gray, blurred, min_area_px, est, min_r, w, h, pw=None):
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

        # The trace loop and the candidate fits below are the bulk of a scale's
        # cost, so the progress window is split between them.
        w_trace = w_cand = None
        if pw is not None:
            lo, hi = pw
            w_trace = (lo, lo + (hi - lo) * 0.55)
            w_cand = (lo + (hi - lo) * 0.55, lo + (hi - lo) * 0.97)
        traces = []
        strong_scores, outer_scores = [], []
        for ti, (cx, cy, r) in enumerate(expanded):
            if ti % 16 == 0:
                self._tick(w_trace, ti, len(expanded))
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
        candidates = []
        for ci, ((cx, cy, r), trace) in enumerate(zip(expanded, traces)):
            if ci % 16 == 0:
                self._tick(w_cand, ci, len(expanded))
            candidates.append(self._boundary_candidates(
                cx, cy, trace[0], trace, score_thresh, r, outer_thresh, blurred))
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
        bands = []
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
            passed, roundness = 0, []
            for (cx, cy, r), trace in zip(sample, traces):
                fit = self._robust_circle_fit(int(cx), int(cy), trace[0],
                                              trace[1], trace[2], thresh,
                                              r0=int(r))
                if not fit:
                    continue
                roundness.append(fit[4])
                if fit[3] >= 0.5 and fit[4] <= 0.10:
                    passed += 1
            if not roundness:
                continue
            bands.append(((passed / len(sample)) * len(circles),
                          float(np.median(roundness)),
                          self._radius_mode(circles[:, 2])))

        if not bands:
            return None
        # Population alone cannot choose the band, because the smaller the
        # circle the more of them any image yields: on a grainy micrograph of a
        # dozen large particles the grain-scale band offered 159 circles
        # against the real band's 8 and won twenty to one, and the program then
        # measured grain. Neither can validated fraction rescue it - the
        # threshold that validates a band is drawn from that band's own edge
        # scores, so a band of pure noise grades itself on a curve and passed
        # 81% of its circles.
        #
        # What grain cannot fake is a round boundary. Fitting a circle to the
        # traced edge and asking how far the edge strays from it separates the
        # two everywhere it was measured: real particles' bands sit at 0.005 to
        # 0.025 of a radius, grain-scale bands at 0.040 to 0.055, on both real
        # micrographs and every fixture. So roundness decides which bands are
        # eligible, and population only breaks ties among bands that are
        # equally round - which is what it was always good for, telling a
        # sparsely populated true scale from a densely populated one.
        # An absolute cut, because that is how the measurement actually falls -
        # the two groups are separated by a wide empty gap, not by a ratio, and
        # a ratio decided the dense fixture on the third decimal place. The
        # relative term only takes over on an image whose fits are all rougher
        # than the cut, where what matters is which band is roundest rather
        # than whether it clears a bar set on other images.
        best_round = min(b[1] for b in bands)
        limit = max(self.BAND_MAX_ROUGHNESS, best_round * 1.5)
        eligible = [b for b in bands if b[1] <= limit]
        eligible.sort(key=lambda b: b[0], reverse=True)

        # Every eligible scale is returned, not just the winner, because an
        # image may honestly contain more than one. A band is only kept if it
        # carries a real share of the population - a scale with a handful of
        # circles beside one with hundreds is a straggler, not a second
        # population - and if it is far enough from a scale already taken to be
        # a different size at all rather than the same one seen twice.
        chosen = []
        for score, _, radius in eligible:
            if score < eligible[0][0] * self.SCALE_MIN_SHARE:
                break
            if any(0.6 <= radius / taken <= 1.7 for taken in chosen):
                continue
            chosen.append(radius)
            if len(chosen) >= self.MAX_SCALES:
                break
        return chosen

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

    def _outer_by_level(self, blurred, cx, cy, r0, angles, w, h, frac=None,
                        wall_only=False, wall_at=None):
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

        # Where to look for the wall. Left open the search spans two thirds of
        # a radius either side, which is fine when the circle is roughly right
        # and wrong when it is not: a small particle wedged among larger ones
        # has its neighbours' walls inside that span, and the median then walks
        # outward onto them - one grew from 40 px to 55 that way, swallowing
        # the gap and part of a neighbour. Once the field has agreed where its
        # walls sit, saying so confines the search to a band around that.
        if wall_at:
            band = (radii >= (wall_at - 0.13) * r0) & (radii <= (wall_at + 0.30) * r0)
        else:
            band = (radii >= 0.65 * r0) & (radii <= 1.25 * r0)
        far = radii >= 1.35 * r0
        core = (radii >= 0.5 * r0) & (radii <= 0.62 * r0)
        if not band.any() or not far.any():
            return (np.full(len(angles), np.nan), np.zeros(len(angles)), 0.0,
                    (float("nan"), float("nan")))

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
        edges = []

        # Hoisted out of the per-angle loop: defining it inside built a fresh
        # closure on every one of the 180 angles, hundreds of thousands per
        # dense field. `prof` and `radii` do not change per angle, so this is
        # the same function; the per-angle state it used to capture (i, k,
        # extreme, dark) is passed in instead.
        def half_height(i, k, extreme, dark, level, forward):
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

        # The per-angle preamble - which extreme is the rim, where it is, and
        # how strong - is the same arithmetic on every angle, so it is done on
        # all of them at once. The old loop called .min()/.max()/argmin/argmax
        # a handful of times per angle, two million tiny reductions on a dense
        # field; this is a fixed few. The scalars each angle used to compute
        # are pulled from these arrays unchanged, so the result is identical.
        seg2d = prof[:, band_idx]
        lo_arr = seg2d.min(axis=1)
        hi_arr = seg2d.max(axis=1)
        # The rim may be darker or brighter than its surroundings; take
        # whichever extreme departs further from the outside level. An
        # out-of-focus edge throws a bright Fresnel fringe just outside the
        # particle, and on these micrographs the gaps between particles read
        # *darker* than the interiors, so the fringe is the bigger departure
        # and this picks it as the rim, putting the boundary a fringe-width
        # out. A caller looking for a shell wall says wall_only, because a wall
        # is dark by construction - a trough darker than the particle's inside
        # *and* darker than what is beyond it - and requiring depth on both
        # sides separates the wall from the fringe and from a dark background.
        with np.errstate(invalid="ignore"):
            dark_arr = (outside - lo_arr) >= (hi_arr - outside)
            if wall_only:
                fin_int = np.isfinite(interior)
                dark_wall = (np.minimum(outside - lo_arr, interior - lo_arr)
                             >= 0.25 * (hi_arr - lo_arr))
                dark_arr = np.where(fin_int, dark_wall, dark_arr)
            k_arr = np.where(dark_arr, band_idx[np.argmin(seg2d, axis=1)],
                             band_idx[np.argmax(seg2d, axis=1)])
            extreme_arr = prof[np.arange(len(angles)), k_arr]
            contrast_arr = np.abs(outside - extreme_arr)
            gate = (np.isfinite(outside)
                    & (inside.sum(axis=1) >= len(radii) * 0.5)
                    & (contrast_arr >= 4))

        for i in np.flatnonzero(gate):
            i = int(i)
            k = int(k_arr[i])
            dark = bool(dark_arr[i])
            extreme = extreme_arr[i]
            contrast = contrast_arr[i]
            # A rim is a band that stands out from the particle's own interior,
            # not merely a step from interior to background. Asking instead
            # whether a dark dip sits between the steepest point and this one
            # cannot work here: half-recovery is on the far flank of that same
            # dip, so the profile between them only ever rises.
            if np.isfinite(interior[i]):
                # How much deeper the band is than the particle's own inside,
                # as a share of the band's total contrast. A solid particle has
                # nothing there - its darkest point *is* its inside - so the
                # share is zero, while a shell puts a real trough between the
                # two. The threshold is deliberately well below what a shell
                # gives, because what separates a shell from noise is not depth
                # but consistency: see the rim_frac test at the call site.
                rim_votes.append(abs(interior[i] - extreme)
                                 >= max(12.0, self.RIM_DEPTH_FRAC * contrast))
            # Place the boundary anywhere across the rim, not just outside it.
            # Walking outward from the rim's darkest point can never put the
            # edge on the rim itself, let alone on its inner flank, so the
            # setting had almost nothing to move: 3% to 40% spanned 75 to 78 nm
            # on a real particle whose rim is 9% of the radius wide. Measuring
            # the half-height on both flanks gives the rim's full extent, and
            # the setting then slides between them - 0% at the inner flank,
            # 50% on the rim, 100% at the outer flank. (half_height is hoisted
            # above this loop; the per-angle state it needs is passed in.)
            if wall_only:
                # The wall's outer flank rises to the shoulder immediately
                # outside it, not to the level far away, and half-height has to
                # be measured against the shoulder the flank actually climbs.
                # Where an out-of-focus edge throws a bright fringe the two are
                # not the same number at all - the fringe stands well above the
                # background - and half-height against the distant level lands
                # a fair way down the flank, inside the wall. Measured by how
                # closely the circles then pack, that read the yolk-shell
                # fields 5-8% small; against the shoulder they pack at 0.99.
                out_seg = prof[i, k:]
                # The *first* crest outward, not the highest one out there:
                # past it the ray has left this particle, and the brightest
                # thing further along is a neighbour's interior, which is
                # brighter than the gap and would pull the target outwards
                # again. Two consecutive falls to call a crest, so a single
                # noisy pixel does not end the climb early.
                stop = len(out_seg) - 1
                for j in range(1, len(out_seg) - 2):
                    if out_seg[j + 1] < out_seg[j] and out_seg[j + 2] < out_seg[j]:
                        stop = j
                        break
                shoulder = float(out_seg[:stop + 1].max())
                # Only a fringe, though. A crest has to stand clear of the
                # surrounding level *and* fall back from it, or it is not a
                # fringe at all - it is where a noisy profile happened to peak
                # on its way to a flat background, and measuring half-height
                # against it reads the particle 17% large (the grainy fixture,
                # where this was found).
                rise = shoulder - outside[i]
                fall = shoulder - float(out_seg[stop:].min())
                level = (shoulder if min(rise, fall) >= 0.25 * contrast
                         else outside[i])
                r_outer_edge = half_height(i, k, extreme, dark, level, True)
            else:
                r_outer_edge = half_height(i, k, extreme, dark, outside[i], True)
            r_inner_edge = (half_height(i, k, extreme, dark, interior[i], False)
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
            edges.append((r_inner_edge, r_outer_edge))
        rim_frac = float(np.mean(rim_votes)) if rim_votes else 0.0
        # The wall's own extent, as one pair of numbers for the whole particle:
        # where its inner flank is and where its outer flank is. What the
        # surface is a fraction *of*.
        span = ((float(np.median([e[0] for e in edges])),
                 float(np.median([e[1] for e in edges])))
                if edges else (float("nan"), float("nan")))
        return r_out, s_out, rim_frac, span

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
            r_level, s_level, rim_frac, _width = self._outer_by_level(
                blurred, cx, cy, r0, angles, w, h, frac=self.edge_level)
            if np.isfinite(r_level).sum() >= 8:
                r_outer, s_outer = r_level, s_level
                outer_thresh = max(4.0, np.percentile(s_level[s_level > 0], 25)) \
                    if (s_level > 0).any() else outer_thresh
        outer = self._robust_circle_fit(cx, cy, angles, r_outer, s_outer,
                                        outer_thresh, r0=r0)

        qualifies = False
        # The half-pixel is load-bearing, and not for the reason its author
        # may have had in mind. Dropping it does let the level boundary be used
        # on shells whose outer flank tapers, where it coincides with the
        # gradient crest - but it also admits the curved gap between three
        # packed particles as a particle, because such a gap has a boundary
        # that passes every other test. Reported ghosts cost more than an
        # inert setting; see _select_boundary and test_bright_core.
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
                and (blurred is None or rim_frac >= self.RIM_MIN_RAYS))

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

    #: How much better outlined one of two overlapping circles must be before
    #: that settles which is the particle, ahead of how confident each fit was.
    #: Below it the two are outlined about equally and the confidence tier is
    #: the better tie-break.
    RING_EVIDENCE_MARGIN = 0.15

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
                evidence = None
                if blurred is not None:
                    for cand in (p, q):
                        if "ring_evidence" not in cand:
                            cand["ring_evidence"] = ParticleAnalyzer._ring_evidence(
                                blurred, cand["center_x"], cand["center_y"],
                                cand["radius_px"])
                    evidence = abs(p["ring_evidence"] - q["ring_evidence"])

                # Which one is actually outlined decides it, whenever the two
                # differ clearly enough to say. Deciding by confidence tier
                # first gets it backwards exactly where it matters: a real
                # particle half-hidden by its neighbour is marked approximate
                # *because* it is half-hidden, and a phantom sitting in the gap
                # can be fitted confidently to pieces of the rims around it. On
                # a real micrograph that pairing threw away a particle outlined
                # on 94% of its circumference and kept the phantom overlapping
                # it, outlined on 56%. The tier only breaks ties now.
                if evidence is not None and evidence >= ParticleAnalyzer.RING_EVIDENCE_MARGIN:
                    loser = p if p["ring_evidence"] < q["ring_evidence"] else q
                    winner = q if loser is p else p
                    loser["excluded"] = True
                    loser["approx"] = False
                    winner["overlap"] = True
                elif p.get("approx") != q.get("approx"):
                    loser = p if p.get("approx") else q
                    loser["excluded"] = True
                    loser["approx"] = False
                elif blurred is not None:
                    loser = p if p["ring_evidence"] < q["ring_evidence"] else q
                    winner = q if loser is p else p
                    loser["excluded"] = True
                    loser["approx"] = False
                    winner["overlap"] = True
                else:
                    p["overlap"] = q["overlap"] = True

    #: A dark body has to be this round, and fill this much of the circle
    #: drawn round it, to be a particle rather than the junction where three
    #: shell walls meet. Measured on real fields: particles a reader marked as
    #: missed score 0.80-0.93 and 0.66-0.96, wall junctions 0.25-0.79 and
    #: 0.28-0.68, and no junction clears both.
    DARK_BODY_ROUNDNESS = 0.75
    DARK_BODY_FILL = 0.65

    #: Share of the detections that may be solid dark bodies before this path
    #: is switched off as meaningless for the image. Real fields of hollow
    #: particles run at 2-5%; a field of solid discs runs at nearly 100%.
    DARK_BODY_MAX_SHARE = 0.25

    def _detect_dark_bodies(self, gray, found):
        """Particles that are solid dark right through, which nothing else finds.

        The main path for a packed field finds a *bright* interior sealed
        inside its own dark shell. A particle whose template was never
        dissolved out, or whose shell collapsed inward, has no bright interior
        at all - it is a dark disc - so it never becomes a candidate, and no
        later rule can recover what was never proposed. These were the
        particles left uncircled on a field a reader went through by hand.

        In the shell mask such a particle is a solid blob hanging off the wall
        network, and the network itself is thin. Opening the mask with a disc
        half a particle wide erases the walls and leaves the blobs. What it
        also leaves is the junction where three walls meet, which is thick for
        the same reason; that one is a concave triangle, so roundness and how
        much of its own circle it fills separate them with nothing in between.

        Only the position and a rough size come from here. Both are then
        corrected like any other detection - the ring search finds the wall and
        the per-ray rule places the surface on it - so a yolk-shell particle
        whose dark core is smaller than the particle still ends up measured at
        its shell.
        """
        live = [p for p in found if not p.get("excluded")]
        if len(live) < 8:
            return []
        r_med = float(np.median([p["radius_px"] for p in live]))
        if r_med < 6:
            return []
        blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
        _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k = int(r_med * 0.55) | 1
        solid = cv2.morphologyEx(
            dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, 8)

        out = []
        for j in range(1, n):
            area = float(stats[j, cv2.CC_STAT_AREA])
            if not (np.pi * (r_med * 0.5) ** 2 <= area <= np.pi * (r_med * 1.8) ** 2):
                continue
            cnts, _ = cv2.findContours((labels == j).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            perim = cv2.arcLength(cnt, True)
            if perim <= 0:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r <= 0:
                continue
            roundness = 4 * np.pi * area / (perim * perim)
            fill = area / (np.pi * r * r)
            if roundness < self.DARK_BODY_ROUNDNESS or fill < self.DARK_BODY_FILL:
                continue
            out.append((cx, cy, r))

        # This path is for the exceptions in a field of hollow particles. Where
        # the particles are solid dark discs to begin with - which is what the
        # `hard` fixture is - every one of them is a dark body, the main path
        # already has them all, and proposing them again only gives the overlap
        # rules a second copy to choose between. One real particle was lost and
        # one spurious circle gained that way before this gate existed.
        if len(out) > self.DARK_BODY_MAX_SHARE * len(live):
            return []
        out = [self._measure_circle(cx, cy, r, np.pi * r * r)
               for cx, cy, r in out
               if not any(np.hypot(p["center_x"] - cx, p["center_y"] - cy)
                          < p["radius_px"] * 0.6 for p in live)]
        for p in out:
            p["excluded"] = False
            p["approx"] = False
            p["dark_body"] = True
        return out

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

    #: How much of the background ring may be occupied by neighbours before the
    #: sphere-edge extrapolation declines to run. It needs real background to
    #: extrapolate to; a neighbour's interior in its place inflates the answer
    #: by an order of magnitude more than the correction is worth.
    SPHERE_MAX_CROWDING = 0.25

    #: Where on the edge profile the extrapolation is fitted, as a share of the
    #: edge's own contrast. Chosen by sweeping: further out the imaging blur
    #: bends the curve and the answer drifts with it (0.20-0.60 moves 1.3 px
    #: between blur 0.8 and 3.0); further in the square-root approximation
    #: fails. At 0.40-0.80 the drift is 0.5 px across that range.
    SPHERE_FIT_BAND = (0.40, 0.80)

    def _sphere_edge_radius(self, blurred, cx, cy, r_seed, n_ang=180):
        """Outer radius from the shape of a sphere's edge, not its brightness.

        A sphere's projected thickness near the rim goes as sqrt(R - b), so the
        contrast against the background does too, and the SQUARE of that
        contrast is a straight line in b whose root is exactly R. Fitting the
        line and reading its root asks nothing about how dark is dark enough,
        so exposure, absorption and shell thickness cannot move the answer.

        This matters because a sphere's edge is not a step and no threshold
        criterion lands on it. Measured against geometry the current boundary
        comes out 1.05 px inside R on a sphere and 1.06 px inside on a hollow
        shell, while on a flat disc - a true step - it is exact to 0.02 px. The
        deficit is the edge shape, not the search. Extrapolation reverses that:
        0.13 px outside R on the shell, and the scatter between particles drops
        from 0.52 px to 0.03 px because the fit uses the whole flank rather
        than one crossing.

        The cost is that a genuine step edge is read 1.1 px too large, and no
        way was found to tell the two apart from the profile - linearity of the
        fit and the shape of the tail were both tried and neither separates
        them. So this is opt-in rather than automatic. Real particles are
        spheres; the flat disc is a drawing convenience that several fixtures
        happen to use.
        """
        h, w = blurred.shape
        ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
        rr = np.arange(r_seed * 0.75, r_seed * 1.45, 0.25)
        if len(rr) < 8:
            return None
        xs = cx + np.outer(np.cos(ang), rr)
        ys = cy + np.outer(np.sin(ang), rr)
        if xs.min() < 0 or ys.min() < 0 or xs.max() >= w - 1 or ys.max() >= h - 1:
            return None
        x0 = xs.astype(np.intp)
        y0 = ys.astype(np.intp)
        fx, fy = xs - x0, ys - y0
        prof = (blurred[y0, x0] * (1 - fx) * (1 - fy)
                + blurred[y0, x0 + 1] * fx * (1 - fy)
                + blurred[y0 + 1, x0] * (1 - fx) * fy
                + blurred[y0 + 1, x0 + 1] * fx * fy)

        lo_f, hi_f = self.SPHERE_FIT_BAND
        far = rr >= r_seed * 1.30
        near = rr <= r_seed * 1.10
        found = []
        for j in range(n_ang):
            p = prof[j]
            contrast = np.clip(np.percentile(p[far], 80) - p, 0.0, None)
            peak = float(contrast[near].max())
            if peak < 6.0:
                continue
            d = contrast ** 2
            band = (d >= (lo_f * peak) ** 2) & (d <= (hi_f * peak) ** 2)
            idx = np.flatnonzero(band)
            if idx.size < 4:
                continue
            # Only the outward side of the peak: the same contrast occurs on
            # the way in, and fitting both sides would average them.
            idx = idx[idx >= int(np.argmax(contrast))]
            if idx.size < 4:
                continue
            slope, intercept = np.polyfit(rr[idx], d[idx], 1)
            if slope >= 0:
                continue
            found.append(-intercept / slope)
        if len(found) < n_ang * 0.4:
            return None
        return float(np.median(found))

    def _refine_by_sphere_edge(self, particles, gray):
        """Re-place each boundary on the sphere's edge, keeping the centre.

        The centre and the outline come from the traced fit and are left alone;
        only the radius moves, because the extrapolation measures a radius and
        nothing else. A particle whose edge the fit cannot read - too near the
        frame, too little contrast - keeps the radius it had.
        """
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
        live = [p for p in particles if not p.get("excluded")]
        ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        for p in live:
            r0 = p["radius_px"]
            # The extrapolation reads its background from a ring outside the
            # particle. In a packed field that ring is inside the neighbours -
            # 63% of it on the median particle of a real micrograph - so the
            # level it calls background is a neighbour's interior, the fitted
            # line is too shallow, and its root lands far outside the particle.
            # On three real samples that inflated every diameter by about 18%,
            # against the 2% the physics predicts, and it went unnoticed
            # because the fixture this was built on has particles standing
            # apart. Where there is no clear background there is nothing to
            # extrapolate to, so the particle keeps the traced boundary.
            xs = p["center_x"] + 1.30 * r0 * cos_a
            ys = p["center_y"] + 1.30 * r0 * sin_a
            crowded = np.zeros(len(ang), bool)
            for q in live:
                if q is p:
                    continue
                crowded |= (np.hypot(xs - q["center_x"], ys - q["center_y"])
                            <= q["radius_px"])
            if crowded.mean() > self.SPHERE_MAX_CROWDING:
                continue
            r = self._sphere_edge_radius(blurred, p["center_x"], p["center_y"], r0)
            if r is None or not (0.7 * r0 <= r <= 1.4 * r0):
                continue
            self._set_radius(p, r)

    def _set_radius(self, p, r):
        """Move a boundary in or out, keeping its centre and outline shape."""
        r0 = p["radius_px"]
        if r0 <= 0:
            return
        p["radius_px"] = r
        p["area_px"] = float(np.pi * r * r)
        p["diameter"] = 2.0 * r * (self.nm_per_px or 1.0)
        if p.get("contour") is not None:
            pts = p["contour"].reshape(-1, 2).astype(np.float64)
            centre = np.array([p["center_x"], p["center_y"]], float)
            p["contour"] = ((centre + (pts - centre) * (r / r0))
                            .reshape(-1, 1, 2).astype(np.float32))

    #: Share of a particle's inside that has to read as solid material before
    #: it is called defective. Measured on the real yolk-shell micrograph, this
    #: is 0.00 for half the field and 0.05 at the 95th percentile, then jumps
    #: to 0.34 at the 97th - an intact shell has nothing inside it at all, so
    #: there is a wide empty gap to put the line in and its exact place hardly
    #: matters.
    DEFECT_DARK_SHARE = 0.20

    def _flag_defects(self, particles, gray):
        """Mark the particles that have something inside them.

        A template that was never dissolved out, or a shell that collapsed and
        folded in on itself, leaves solid material in a cavity that should be
        empty. These are still particles and still have an outer diameter worth
        measuring - a reader asked for them counted, not dropped - so this only
        labels them; nothing here excludes anything.

        What it looks for is material, not darkness in general. The whole
        particle reads darker when it sits on a thicker patch of support film,
        and the interior of a small particle reads darker than a large one's
        because the shell it is seen through is the same thickness either way.
        Neither of those puts a *lump* inside. So the reference is the
        particle's own interior and its own wall - the share of the inside that
        is more than halfway down from one to the other - which is a ratio of
        two levels the particle supplies itself, and so does not move with
        exposure, magnification, or where on the film it sits.
        """
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
        h, w = blurred.shape
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        steps = np.arange(0.80, 1.25, 0.03)
        readings = []
        for p in particles:
            p.setdefault("defect", False)
            if p.get("excluded"):
                continue
            cx, cy, r = p["center_x"], p["center_y"], p["radius_px"]
            xs = np.clip((cx + r * np.outer(cos_a, steps)).astype(int), 0, w - 1)
            ys = np.clip((cy + r * np.outer(sin_a, steps)).astype(int), 0, h - 1)
            wall = float(np.median(blurred[ys, xs].min(axis=1)))

            rr = r * 0.65
            x0, x1 = max(0, int(cx - rr)), min(w, int(cx + rr) + 1)
            y0, y1 = max(0, int(cy - rr)), min(h, int(cy + rr) + 1)
            if x1 - x0 < 5 or y1 - y0 < 5:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            inside = blurred[y0:y1, x0:x1][np.hypot(xx - cx, yy - cy) <= rr]
            if inside.size < 25:
                continue
            level = float(np.percentile(inside, 80))
            readings.append((p, level, wall))

        if not readings:
            return
        # What an intact particle's inside reads, taken from the field itself
        # so that exposure and film thickness cancel.
        typical = float(np.median([lv for _, lv, _ in readings]))
        floor = float(np.median([wl for _, _, wl in readings]))
        for p, level, wall in readings:
            rr = p["radius_px"] * 0.65
            cx, cy = p["center_x"], p["center_y"]
            x0, x1 = max(0, int(cx - rr)), min(w, int(cx + rr) + 1)
            y0, y1 = max(0, int(cy - rr)), min(h, int(cy + rr) + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            inside = blurred[y0:y1, x0:x1][np.hypot(xx - cx, yy - cy) <= rr]
            if level - wall >= 6:
                share = float(np.mean(inside < level - 0.40 * (level - wall)))
            elif typical - floor >= 6:
                # A particle packed solid right through has no wall to measure
                # against - its inside and its edge read the same - so the ratio
                # above is noise over noise. That flatness is itself the answer,
                # as long as what it is flat *at* is the dark end: judged
                # against what an intact particle in the same field reads, the
                # whole disc is then below the halfway mark and the share comes
                # out at 1.
                share = float(np.mean(inside < typical - 0.40 * (typical - floor)))
            else:
                continue
            p["defect_share"] = share
            p["defect"] = share >= self.DEFECT_DARK_SHARE

    #: A wall has to be resolved on this share of the rays before the boundary
    #: is placed by it. Below that the median is taken over too few directions
    #: to be a radius rather than a local reading.
    SNAP_MIN_RAYS = 0.60

    #: Largest move allowed, as a share of the radius, in one pass and in
    #: total. A wall further away than this is not the particle's own - it is
    #: the neighbour behind the gap - and snapping to it would swap one
    #: particle for another. The total matters as much as the step: a small
    #: particle wedged among larger ones has its neighbours' walls inside its
    #: own search band, and two unbounded passes walked one out from 40 px to
    #: 55, swallowing the gap and part of a neighbour. By this point the ring
    #: search has already fixed the gross errors, so the snap only has to
    #: fine-tune and does not need the room.
    SNAP_MAX_MOVE = 0.15

    #: Smoothing before the wall is read, as a share of the wall's own
    #: thickness. See `_snap_sigma` for why it is tied to that and not to the
    #: radius or to a fixed number of pixels.
    SNAP_SMOOTH_FRAC = 0.35

    #: How far a particle's wall may sit from where the rest of the field puts
    #: its wall, as a share of the radius, before the circle is resized to
    #: match. Within one image the wall is the same fraction of the radius on
    #: every particle, so a circle that disagrees by more than this is not a
    #: circle round a differently-built particle - it is the wrong circle.
    RING_TOLERANCE = 0.07

    #: The vote for a coherent ring has to reach this share of the rays. A
    #: phantom in the gap between particles scores 0.35 against 2.1-6.3 for the
    #: particles around it, because the dark arcs it borrows are pieces of
    #: several neighbours at several radii and they do not stack up.
    RING_MIN_VOTE = 1.2

    #: Largest resize the ring search may ask for, as a share of the radius.
    RING_MAX_RESIZE = 0.25

    def _ring_radius(self, blurred, cx, cy, r0, n_angles=96,
                     lo=0.55, hi=1.40, tol=0.03):
        """The radius at which a dark ring closes round this centre.

        A ray cannot tell this particle's wall from a neighbour's; both are
        dark bands crossing it. What tells them apart is that only one of them
        is at the same radius in every direction. Collecting every dark band on
        every ray and asking which radius they agree on finds the particle's
        own wall even when it is not the deepest band on most rays - which is
        the case for a circle that was fitted a third too large, where two
        thirds of the rays find a neighbour's wall first and the median lands
        in the gap between the two.

        Returns the radius and the vote it won, in rays. The vote is itself
        worth having: a phantom sitting between particles borrows arcs from
        several neighbours at several radii, and they do not stack.
        """
        h, w = blurred.shape
        radii = np.arange(r0 * lo, r0 * hi, 0.5)
        if len(radii) < 8:
            return None, 0.0
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        fx = cx + np.outer(np.cos(angles), radii)
        fy = cy + np.outer(np.sin(angles), radii)
        inside = (fx >= 0) & (fx < w) & (fy >= 0) & (fy < h)
        prof = blurred[np.clip(fy, 0, h - 1).astype(int),
                       np.clip(fx, 0, w - 1).astype(int)].astype(np.float32)

        span = float(np.percentile(prof, 90) - np.percentile(prof, 10))
        floor = max(6.0, 0.25 * span)
        usable = inside.mean(axis=1) >= 0.9
        if usable.sum() < n_angles * 0.4:
            return None, 0.0
        prof = prof[usable]

        dip = np.zeros_like(prof, bool)
        dip[:, 1:-1] = (prof[:, 1:-1] <= prof[:, :-2]) & (prof[:, 1:-1] <= prof[:, 2:])
        left = np.maximum.accumulate(prof, axis=1)
        right = np.maximum.accumulate(prof[:, ::-1], axis=1)[:, ::-1]
        deep = np.minimum(left - prof, right - prof) >= floor
        hits = np.count_nonzero(dip & deep, axis=0).astype(np.float64)
        if not hits.any():
            return None, 0.0

        # Smear each vote over the tolerance so that rays agreeing to within a
        # few percent reinforce one another instead of splitting the peak.
        width = max(1.0, tol * r0 / 0.5)
        k = np.exp(-0.5 * (np.arange(-int(3 * width), int(3 * width) + 1) / width) ** 2)
        votes = np.convolve(hits, k, mode="same")
        top = float(votes.max())
        # Among radii that essentially tie, the innermost. A neighbour's wall
        # can lie outside this particle's, never inside it, so when two radii
        # are equally coherent the smaller one is the one that belongs to this
        # particle.
        best = int(np.flatnonzero(votes >= 0.85 * top)[0])
        return float(radii[best]), top / max(1, prof.shape[0])

    def _resize_by_ring(self, particles, gray):
        """Resize circles whose wall is not where the field puts its walls.

        Within one image every particle is the same object imaged the same way,
        so the wall sits at the same fraction of the radius on all of them -
        measured on real micrographs, 0.86 to 0.91 with a spread of a few
        percent. A circle that was fitted round a particle *and* the bright
        halo outside it has its wall at 0.69 or 0.72 of its radius instead, and
        that is the signature: not that the particle is unusual, but that the
        circle is. Rescaling it so its wall sits where everyone else's does
        recovers the particle without any assumption about brightness levels.

        The per-ray placement afterwards (`_snap_to_wall`) still decides exactly
        where on the wall the surface is; this only gets the circle onto the
        right wall first.
        """
        live = [p for p in particles if not p.get("excluded")]
        if len(live) < 8:
            return
        angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        blurred = cv2.GaussianBlur(gray, (0, 0), self._snap_sigma(gray, live, angles))
        found = []
        for p in live:
            r, vote = self._ring_radius(blurred, p["center_x"], p["center_y"],
                                        p["radius_px"])
            p["ring_vote"] = float(vote)
            found.append(r)
        ratios = [r / p["radius_px"] for p, r, in zip(live, found)
                  if r is not None and p["ring_vote"] >= self.RING_MIN_VOTE]
        if len(ratios) < 8:
            return
        # The field's own answer, not a constant: a thicker shell or a
        # different focus moves it, and every particle in the image moves with
        # it.
        typical = float(np.median(ratios))
        for p, r in zip(live, found):
            if r is None or p["ring_vote"] < self.RING_MIN_VOTE:
                continue
            target = r / typical
            if abs(target / p["radius_px"] - 1.0) <= self.RING_TOLERANCE:
                continue
            # The same trap as the snap: a small particle among larger ones has
            # its neighbours' walls inside its own search band, and without a
            # ceiling the ring vote can settle on one of them.
            if not (1.0 - self.RING_MAX_RESIZE <= target / p["radius_px"]
                    <= 1.0 + self.RING_MAX_RESIZE):
                continue
            p["ring_resized"] = float(target / p["radius_px"] - 1.0)
            self._set_radius(p, target)

    def _snap_to_wall(self, particles, gray, pw=None):
        """Put every boundary on the shell wall, whatever found the particle.

        Which of the two fitted transitions a particle's diameter comes from is
        voted on once per image (see `_select_boundary`), and on six of nine
        real micrographs the vote went against the wall - so every particle in
        those images was measured at the *inner flank* of its wall, with the
        whole dark band lying outside the circle. A reader spotted the worst of
        them by eye; measuring it afterwards, the median circle on one field was
        7.2% small and the worst 15%.

        The vote is not wrong to be cautious: it decides which feature is the
        boundary, and getting that wrong on a whole image is expensive. But it
        decides it before anything has been rejected, from evidence that a
        packed field withholds - on the contact sides there is no background for
        the profile to recover to, so the outer transition simply is not there
        to be voted for. This asks the narrower question afterwards, of each
        surviving particle on its own: where is *your* wall, and is the circle
        on it? A particle whose wall is not resolved keeps what it had.

        It reads the same per-ray rule the boundary is defined by - contact
        sides at the wall's middle, free sides at its outer edge - so it moves a
        circle only towards where that rule already says the surface is. Twice,
        because the search window is set from the radius it starts with, and a
        circle that was well inside its wall gets a better window on the second
        pass. Solid particles have no wall for the rule to find and are left
        alone.

        Scored by how closely the circles pack - the diameter against half the
        distance to the nearest neighbours, which needs no brightness model and
        no scale bar, and which a jammed monolayer fixes at just under 1. The
        yolk-shell sample read 0.93 / 0.98 / 0.98 at its three magnifications
        and now reads 0.97 / 0.97 / 0.96; the two hollow-silica samples were
        already close and stay there (0.99-1.03).

        Comparing the *diameters* between magnifications, which is the obvious
        check, cannot settle this and was misleading while it was believed. The
        neighbour spacing in nm - which depends on the centres and the scale
        bar, not on where the boundary is put - differs by up to 6% between two
        magnifications of the same specimen, because the high-magnification
        frame holds twenty particles and they are simply not the same twenty.
        A rule tuned to make those diameters agree is tuned to a difference in
        the specimen.
        """
        live = [p for p in particles if not p.get("excluded")]
        if not live:
            return
        # Carve the caller's window across this method's phases: the two-pass
        # snap loop is the bulk, then re-centring and levelling.
        w_snap = w_rec = w_lev = None
        if pw is not None:
            lo, hi = pw
            w_snap = (lo, lo + (hi - lo) * 0.60)
            w_rec = (lo + (hi - lo) * 0.60, lo + (hi - lo) * 0.72)
            w_lev = (lo + (hi - lo) * 0.72, hi)
        h, w = gray.shape
        angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        blurred = cv2.GaussianBlur(gray, (0, 0), self._snap_sigma(gray, live, angles))
        wall_at = self._field_wall_fraction(blurred, live)
        for pi, p in enumerate(live):
            if pi % 8 == 0:
                self._tick(w_snap, pi, len(live))
            started = p["radius_px"]
            for _ in range(2):
                r0 = p["radius_px"]
                if r0 < 4:
                    break
                r_ang, _s, rim_frac, _span = self._outer_by_level(
                    blurred, p["center_x"], p["center_y"], r0, angles, w, h,
                    frac=self.edge_level, wall_only=True, wall_at=wall_at)
                seen = np.isfinite(r_ang)
                if seen.mean() < self.SNAP_MIN_RAYS or rim_frac < self.RIM_MIN_RAYS:
                    break
                r = float(np.median(r_ang[seen]))
                if not (1.0 - self.SNAP_MAX_MOVE <= r / r0 <= 1.0 + self.SNAP_MAX_MOVE):
                    break
                r = float(np.clip(r, started * (1.0 - self.SNAP_MAX_MOVE),
                                  started * (1.0 + self.SNAP_MAX_MOVE)))
                p["wall_offset"] = float(r / r0 - 1.0)
                self._set_radius(p, r)
                if abs(r / r0 - 1.0) < 0.01:
                    break
        self._recentre_on_wall(live, blurred, wall_at=wall_at, pw=w_rec)
        self._level_across_wall(live, blurred, angles, wall_at=wall_at, pw=w_lev)
        self._flag_irregular(live)

    #: How far above the field's own median a wall fit has to sit before the
    #: particle is called out as not round. Real fields run at a median of
    #: 0.010-0.026 of the radius with a tail reaching 0.10-0.14, so the line is
    #: drawn relative to the field and floored so that a very uniform sample
    #: does not start flagging its own noise.
    IRREGULAR_RMS_FACTOR = 3.0
    IRREGULAR_RMS_FLOOR = 0.04

    def _flag_irregular(self, live):
        """Mark particles whose boundary is not a circle.

        A diameter assumes a sphere. An elongated particle, or two fused
        together, still gets a circle fitted to it and a number reported, and
        nothing in the number says the shape it came from was not round - a
        reader has to spot it by eye, and on a field of four hundred they will
        not spot them all. How far the wall strays from the fitted circle says
        it directly, and it is already measured while re-centring.
        """
        values = [p["wall_rms"] for p in live if "wall_rms" in p]
        if len(values) < 8:
            return
        limit = max(self.IRREGULAR_RMS_FLOOR,
                    self.IRREGULAR_RMS_FACTOR * float(np.median(values)))
        for p in live:
            p["irregular"] = bool(p.get("wall_rms", 0.0) > limit)

    #: Smallest and largest centre move worth making once the wall band is
    #: known, as a share of the radius. The floor is low because by this point
    #: the search is confined to a band a few pixels wide around a wall the
    #: field has already located, so a small answer is a real one rather than
    #: the fit wandering; the earlier pass, which searches from scratch, keeps
    #: its own much higher floor.
    WALL_CENTRE_MIN_MOVE = 0.015
    WALL_CENTRE_MAX_MOVE = 0.20

    #: How ragged the fitted wall may be, as a share of the radius, before the
    #: centre it implies is not worth trusting.
    WALL_CENTRE_MAX_RMS = 0.05

    def _field_wall_fraction(self, blurred, live):
        """Where the field puts its walls, as a share of the radius."""
        found = []
        for p in live:
            r, vote = self._ring_radius(blurred, p["center_x"], p["center_y"],
                                        p["radius_px"])
            if r and vote >= self.RING_MIN_VOTE:
                found.append(r / p["radius_px"])
        return float(np.median(found)) if len(found) >= 8 else None

    def _recentre_on_wall(self, live, blurred, n_angles=180, wall_at=None, pw=None):
        """Re-centre once more, now that the wall's radius is known.

        The first re-centring searches for the ring from scratch, over a wide
        band, so it has to insist on a large move before it believes itself -
        a circle less than a sixteenth of a radius out is left alone, because
        at that size the fit cannot tell a displaced circle from its own noise.
        By this point the field has agreed where the wall is, to within a few
        percent of the radius. Looking only in that band makes a small answer
        trustworthy: the darkest point along each ray is this particle's own
        wall by construction, so the circle through them is the particle's.

        Half the field moves by less than 2% of a radius and it is not worth
        arguing about, but the tail is: on a real yolk-shell field the worst
        run to 16%, which is a circle visibly off its particle.
        """
        if len(live) < 8:
            return
        h, w = blurred.shape
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        if wall_at is None:
            wall_at = self._field_wall_fraction(blurred, live)
        if wall_at is None:
            return

        for pi, p in enumerate(live):
            if pi % 16 == 0:
                self._tick(pw, pi, len(live))
            r0 = p["radius_px"]
            if r0 < 6:
                continue
            band = np.arange(r0 * wall_at * 0.82, r0 * wall_at * 1.18, 0.25)
            if len(band) < 4:
                continue
            fx = p["center_x"] + np.outer(cos_a, band)
            fy = p["center_y"] + np.outer(sin_a, band)
            seen = ((fx >= 0) & (fx < w) & (fy >= 0) & (fy < h)).mean(axis=1) >= 0.95
            if seen.sum() < n_angles * 0.5:
                continue
            prof = blurred[np.clip(fy, 0, h - 1).astype(int),
                           np.clip(fx, 0, w - 1).astype(int)]
            k = np.argmin(prof, axis=1)
            px = (p["center_x"] + band[k] * cos_a)[seen]
            py = (p["center_y"] + band[k] * sin_a)[seen]
            ux = uy = rad = None
            for _ in range(4):
                design = np.column_stack([2 * px, 2 * py, np.ones(len(px))])
                sol, *_ = np.linalg.lstsq(design, px ** 2 + py ** 2, rcond=None)
                ux, uy = float(sol[0]), float(sol[1])
                rad = float(np.sqrt(max(sol[2] + ux * ux + uy * uy, 1e-6)))
                resid = np.abs(np.hypot(px - ux, py - uy) - rad)
                keep = resid < 2.0 * np.median(resid) + 0.4
                if keep.all() or keep.sum() < n_angles * 0.35:
                    break
                px, py = px[keep], py[keep]
            if rad is None or rad <= 0:
                continue
            rms = float(np.median(np.abs(np.hypot(px - ux, py - uy) - rad))) / rad
            p["wall_rms"] = rms
            move = np.hypot(ux - p["center_x"], uy - p["center_y"]) / r0
            if rms > self.WALL_CENTRE_MAX_RMS:
                continue
            if not (self.WALL_CENTRE_MIN_MOVE < move <= self.WALL_CENTRE_MAX_MOVE):
                continue
            dx = int(round(ux)) - p["center_x"]
            dy = int(round(uy)) - p["center_y"]
            if dx == 0 and dy == 0:
                continue
            p["center_x"] += dx
            p["center_y"] += dy
            p["wall_recentre"] = float(move)
            if p.get("contour") is not None:
                pts = p["contour"].reshape(-1, 2).astype(np.float32)
                pts[:, 0] += dx
                pts[:, 1] += dy
                p["contour"] = pts.reshape(-1, 1, 2)

    #: How far a circle may sit from where the field as a whole sits across its
    #: wall, as a share of the wall's thickness, before it is brought into
    #: line. Below this the move is smaller than the noise it is correcting.
    WALL_POSITION_TOLERANCE = 0.10

    def _level_across_wall(self, live, blurred, angles, wall_at=None, pw=None):
        """Put every circle at the same place across its wall as the rest.

        Where on the wall the surface lies is decided ray by ray - the middle
        where a neighbour is pressed against this particle, the outer edge
        where the wall faces open space - and that is right for each ray. But
        the *mixture* is not a property of the particle: it is how many
        neighbours it happens to have touching, and it varies from particle to
        particle in a way the particle's size does not. Measured across one
        real field, circles landed anywhere from 0.60 to 0.98 of the way across
        their own wall, which is a spread of about 8% in radius between two
        particles that are the same size.

        A reader picked ten of them out as too small. They are not a separate
        kind of error - on every measure available they sit inside the field's
        own spread, and half of them are on the *large* side of it - which is
        exactly what a scatter looks like when someone marks its tail. So this
        does not try to find them. It removes the scatter: the particles in one
        image are the same object imaged the same way, so the surface is at the
        same place on the wall for all of them, and the field's median says
        where. The median is left where it was, so nothing about the average
        size changes; only the disagreement between particles shrinks.
        """
        h, w = blurred.shape
        spans = {}
        for pi, p in enumerate(live):
            if pi % 16 == 0:
                self._tick(pw, pi, len(live))
            if p["radius_px"] < 4:
                continue
            _r, _s, rim, (inner, outer) = self._outer_by_level(
                blurred, p["center_x"], p["center_y"], p["radius_px"], angles,
                w, h, frac=self.edge_level, wall_only=True, wall_at=wall_at)
            if rim < self.RIM_MIN_RAYS or not np.isfinite(inner) or outer <= inner:
                continue
            spans[id(p)] = (inner, outer)
        if len(spans) < 8:
            return
        places = {k: (p["radius_px"] - spans[k][0]) / (spans[k][1] - spans[k][0])
                  for p in live if (k := id(p)) in spans}
        common = float(np.median(list(places.values())))
        for p in live:
            k = id(p)
            if k not in spans:
                continue
            if abs(places[k] - common) <= self.WALL_POSITION_TOLERANCE:
                continue
            inner, outer = spans[k]
            r = inner + common * (outer - inner)
            if not (0.88 <= r / p["radius_px"] <= 1.14):
                continue
            p["wall_place"] = float(places[k])
            self._set_radius(p, r)

    def _snap_sigma(self, gray, live, angles):
        """How much to smooth before reading the wall: a share of the wall.

        Not a fixed number of pixels, and not a share of the radius either.
        Smoothing widens the dark band on both flanks, so the point a given
        share across it lands on moves outward - and by how much depends on the
        blur measured against *the band*, not against the particle. A fixed
        pixel count therefore reads the same specimen differently at two
        magnifications (2% at a radius of 60 px, 0.5% at 180). A share of the
        radius fixes that but not the other half: a shell 15% of the radius
        thick and one 7% thick then get the same blur relative to a particle
        and twice the difference relative to the wall, which read the thin-
        walled fixture 7% large while the thick-walled real sample needed
        exactly that much smoothing to hold still.

        So the wall is measured first, lightly, and the smoothing set from what
        comes back. Both scales then follow the feature being measured, and the
        thin-ring fixture and the real yolk-shell sample can be right at once.
        """
        h, w = gray.shape
        radii = [p["radius_px"] for p in live]
        light = cv2.GaussianBlur(gray, (0, 0), max(0.5, 0.02 * float(np.median(radii))))
        # A sample is enough for a per-image median, and this pass is only here
        # to set one number.
        step = max(1, len(live) // 60)
        widths = []
        for p in live[::step]:
            if p["radius_px"] < 4:
                continue
            _r, _s, _rim, (inner, outer) = self._outer_by_level(
                light, p["center_x"], p["center_y"], p["radius_px"], angles, w, h,
                frac=self.edge_level, wall_only=True)
            if np.isfinite(outer) and np.isfinite(inner):
                widths.append(outer - inner)
        if not widths:
            # No wall found anywhere - the snap will decline on every particle
            # anyway, so this only has to be a sane number.
            return max(0.5, 0.02 * float(np.median(radii)))
        return float(max(0.5, self.SNAP_SMOOTH_FRAC * np.median(widths)))

    #: A circle may be moved onto its ring by at most this share of its radius.
    #: Beyond it the ring being fitted is more likely a neighbour's than its
    #: own, and moving there would swap one particle for another.
    RECENTRE_MAX_MOVE = 0.25

    #: Below this the move is inside the noise of the fit and not worth making.
    RECENTRE_MIN_MOVE = 0.06

    def _recentre_on_ring(self, particles, gray, n_angles=180, pw=None):
        """Put each circle back on the ring it is supposed to be on.

        The boundary is fitted to points traced outward from a seed, and a seed
        that starts off-centre traces a boundary made partly of its own rim and
        partly of a neighbour's. The circle through that mixture is displaced,
        and it stays displaced however good the fit was, because the fit is
        faithful to points that were wrong. Seeds are already re-centred before
        tracing; this catches what survives, and on real micrographs that was
        one detection in seven sitting more than 10% of a radius off its
        particle - the thing a reader notices first, because a circle a tenth
        out of place visibly bulges past the particle on one side.

        The ring is found independently of any of that: along each ray, the
        darkest point that dips on both sides. Only the centre moves. The
        radius is left alone because it was measured against a criterion this
        does not reproduce, and re-deriving it here would silently replace the
        boundary rule with a darkest-point one.
        """
        live = [p for p in particles if not p.get("excluded")]
        if not live:
            return
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
        h, w = gray.shape
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        fracs = np.arange(0.78, 1.26, 0.02)
        for pi, p in enumerate(live):
            if pi % 16 == 0:
                self._tick(pw, pi, len(live))
            cx, cy, r = p["center_x"], p["center_y"], p["radius_px"]
            # The per-angle darkest-point search, over all angles at once. The
            # arithmetic is kept in the same order the scalar loop used -
            # (r*fracs) formed first, then multiplied by cos/sin per angle - so
            # the .astype(int) truncation, and with it the pixel each ray
            # samples, is bit-for-bit what the loop produced.
            rf = r * fracs
            xs = (cx + rf[None, :] * cos_a[:, None]).astype(int)
            ys = (cy + rf[None, :] * sin_a[:, None]).astype(int)
            allin = ((xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)).all(axis=1)
            prof = blurred[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)]
            k = np.argmin(prof, axis=1)
            valley = prof.min(axis=1)
            fwd = np.maximum.accumulate(prof, axis=1)
            rev = np.maximum.accumulate(prof[:, ::-1], axis=1)[:, ::-1]
            rows = np.arange(len(angles))
            # prof[:k].max() is the running max up to k-1; prof[k:].max() the
            # running max from k. Guard k-1 for the k==0 rays, which are
            # dropped anyway.
            left_max = fwd[rows, np.maximum(k - 1, 0)]
            right_max = rev[rows, k]
            depth = np.minimum(left_max - valley, right_max - valley)
            good = allin & (k != 0) & (k != prof.shape[1] - 1) & (depth >= 8)
            if good.sum() < n_angles * 0.22:
                continue
            rr = r * fracs[k[good]]
            P = np.column_stack([cx + rr * cos_a[good], cy + rr * sin_a[good]])
            ux = uy = rad = None
            resid = None
            for _ in range(3):
                A = np.column_stack([2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))])
                sol, *_ = np.linalg.lstsq(A, P[:, 0] ** 2 + P[:, 1] ** 2, rcond=None)
                ux, uy = float(sol[0]), float(sol[1])
                rad = float(np.sqrt(sol[2] + ux * ux + uy * uy))
                resid = np.abs(np.hypot(P[:, 0] - ux, P[:, 1] - uy) - rad)
                keep = resid < 2.5 * np.median(resid) + 1
                if keep.all():
                    break
                P = P[keep]
                if len(P) < n_angles * 0.22:
                    break
            if rad is None or rad <= 0:
                continue
            rms = float(np.median(resid)) / rad
            move = np.hypot(ux - cx, uy - cy) / max(r, 1)
            if rms > 0.08 or not (ParticleAnalyzer.RECENTRE_MIN_MOVE < move
                                  <= ParticleAnalyzer.RECENTRE_MAX_MOVE):
                continue
            dx, dy = int(round(ux)) - cx, int(round(uy)) - cy
            p["center_x"] += dx
            p["center_y"] += dy
            if p.get("contour") is not None:
                c = p["contour"].reshape(-1, 2).astype(np.float32)
                c[:, 0] += dx
                c[:, 1] += dy
                p["contour"] = c.reshape(-1, 1, 2)

    #: A particle is outlined all the way round; a phantom lying in the space
    #: between particles borrows pieces of its neighbours' rims and is outlined
    #: over part of its circumference. On real micrographs every detection a
    #: reader marked as "nothing is there" measured 0.73 or below, against a
    #: median of 0.92 for the ones they accepted.
    MIN_OUTLINED = 0.75

    #: The rule only applies to images whose particles are outlined at all.
    #: Measured per image, this is 0.91-1.00 where they are and 0.00 where the
    #: particles are dark discs on a bright ground, whose darkest point along
    #: any ray is inside them rather than at the boundary. Nothing in between
    #: was observed, so the gate is not delicate.
    OUTLINED_FIELD = 0.80

    @staticmethod
    def _reject_unoutlined(particles, blurred):
        """Exclude circles that are not outlined most of the way round.

        The same measurement already settles which of two overlapping circles
        is the particle. It was never applied to a circle standing on its own,
        and a phantom in the gap between three particles has nothing to be set
        against - it passes every test applied to it alone. Asking it directly
        catches all seventeen that a reader marked as phantoms across four real
        micrographs, at the cost of 7% of the detections they had accepted,
        some of which are errors they did not mark.

        It is asked only where being outlined means something. On a dark disc
        on a bright ground the darkest point along a ray is the middle of the
        particle, so the measure reads zero for every particle and applying it
        would empty the image - which it did, on three fixtures, before this
        was gated.
        """
        live = [p for p in particles if not p.get("excluded")]
        if len(live) < 5:
            return
        for p in live:
            if "ring_evidence" not in p:
                p["ring_evidence"] = ParticleAnalyzer._ring_evidence(
                    blurred, p["center_x"], p["center_y"], p["radius_px"])
        if np.median([p["ring_evidence"] for p in live]) < ParticleAnalyzer.OUTLINED_FIELD:
            return
        for p in live:
            if p["ring_evidence"] < ParticleAnalyzer.MIN_OUTLINED:
                p["excluded"] = True
                p["approx"] = False
                p["unoutlined"] = True

    #: How much of a particle's circumference must lie inside the frame for
    #: its diameter to be worth fitting. Chosen by measurement: on three real
    #: specimens photographed at two or three magnifications each, this is
    #: where the magnifications agree most closely.
    MIN_VISIBLE_ARC = 0.65

    #: How much of a particle may lie under the scale bar and its lettering
    #: before it is dropped. A particle grazing the annotation is still
    #: measurable; one behind it is not, and neither its edge nor its interior
    #: means anything there.
    MAX_ANNOTATED = 0.15

    @staticmethod
    def _annotation_box(gray):
        """Where the scale bar and its caption sit, when they are drawn over
        the micrograph rather than in a black strip below it.

        The bar is found by shape rather than brightness: a solid, very bright
        rectangle at least 60 px wide and four times wider than it is tall.
        Brightness alone cannot find it, because the gaps between particles in
        these samples reach 254 as well. The caption sits directly above the
        bar and scales with it, so the box extends upward by a fraction of the
        bar's length. Returns None when there is no such bar.
        """
        h, w = gray.shape
        top = int(h * 0.75)
        mask = (gray[top:, :] > 235).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        best = None
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if bw < 60 or bh < 1 or bw / max(bh, 1) < 4:
                continue
            if area < bw * bh * 0.6:
                continue
            if best is None or bw > best[2]:
                best = (x, y, bw, bh)
        if best is None:
            return None
        x, y, bw, _ = best
        pad = int(bw * 0.08)
        return (max(0, x - pad), max(0, top + y - int(bw * 0.45)),
                min(w, x + bw + pad), h)

    @staticmethod
    def _reject_annotated(particles, gray):
        """Drop particles the scale bar or its caption is drawn over."""
        box = ParticleAnalyzer._annotation_box(gray)
        if box is None:
            return
        x0, y0, x1, y1 = box
        for p in particles:
            if p.get("excluded"):
                continue
            cx, cy, r = p["center_x"], p["center_y"], p["radius_px"]
            if cx + r < x0 or cx - r > x1 or cy + r < y0 or cy - r > y1:
                continue
            size = 2 * int(round(r)) + 1
            disc = np.zeros((size, size), np.uint8)
            cv2.circle(disc, (int(round(r)), int(round(r))), int(round(r)), 1, -1)
            ann = np.zeros_like(disc)
            ax0 = max(0, x0 - (cx - int(round(r))))
            ay0 = max(0, y0 - (cy - int(round(r))))
            ax1 = min(size, x1 - (cx - int(round(r))))
            ay1 = min(size, y1 - (cy - int(round(r))))
            if ax1 <= ax0 or ay1 <= ay0:
                continue
            ann[ay0:ay1, ax0:ax1] = 1
            hidden = np.count_nonzero(disc & ann) / max(np.count_nonzero(disc), 1)
            if hidden > ParticleAnalyzer.MAX_ANNOTATED:
                p["excluded"] = True
                p["approx"] = False
                p["annotated"] = True

    @staticmethod
    def _reject_clipped(particles, shape, n_angles=180):
        """Exclude only the particles the frame cuts too deeply to measure.

        A diameter does not need a whole circle, it needs enough arc to fix
        one. Dropping every particle the frame touches looked right and was
        not: it removes a different share of each magnification - a wide field
        has many small particles against its edges, a close one has a few large
        ones - and that biased the magnifications apart. Keeping the ones with
        two thirds of their circumference inside the frame closed the gap
        between magnifications of one specimen from 4.13% to 0.90% on one
        sample and from 0.90% to 0.03% on another.

        This reverses an earlier decision, and the reason it can be reversed is
        that the two faults it was covering for are now fixed directly: circles
        are put back on their ring, and circles that are not outlined are
        dropped. A clipped particle no longer sits off its particle, so there
        is no longer a reason to refuse to measure it.
        """
        h, w = shape[:2]
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        for p in particles:
            if p.get("excluded"):
                continue
            cx, cy, r = p["center_x"], p["center_y"], p["radius_px"]
            if cx - r >= 0 and cy - r >= 0 and cx + r <= w and cy + r <= h:
                continue
            xs, ys = cx + r * cos_a, cy + r * sin_a
            inside = float(np.mean((xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)))
            if inside >= ParticleAnalyzer.MIN_VISIBLE_ARC:
                p["partial"] = True
                continue
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

    @classmethod
    def _reject_implausible_interiors(cls, particles, gray, floor=20.0, k=6.0):
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
        ring = cv2.GaussianBlur(gray, (0, 0), 1.5)
        for p, level in zip(candidates, levels):
            if abs(level - centre) <= tol:
                continue
            # A dark reading has two causes and they need opposite answers.
            # The overlap lens between two particles is dark because two walls
            # are stacked there, and it is not a particle. A particle whose
            # template was never removed, or whose shell collapsed inward, is
            # dark because there is material inside it - and it *is* a particle,
            # one the reader wants counted and measured. Earlier attempts to
            # separate them by what is inside the circle all failed, because
            # inside is where the two look alike (see test_shelled).
            #
            # What differs is outside: a particle carries its own wall the whole
            # way round, a lens carries only the pieces of its neighbours' walls
            # that happen to cross it. That measurement is already trusted for
            # exactly this job elsewhere in the pipeline, so it is asked here
            # too - but only of the dark ones. A bright reading means a gap
            # between particles, and a gap has no defensible reading at all.
            if level < centre and cls._ring_evidence(
                    ring, p["center_x"], p["center_y"], p["radius_px"]) >= cls.MIN_OUTLINED:
                p["dark_interior"] = True
                continue
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
        defect_count = sum(1 for p in particles if p.get("defect"))
        stats["defect_count"] = defect_count
        stats["defect_ratio"] = defect_count / len(particles)
        irregular_count = sum(1 for p in particles if p.get("irregular"))
        stats["irregular_count"] = irregular_count
        stats["irregular_ratio"] = irregular_count / len(particles)
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
