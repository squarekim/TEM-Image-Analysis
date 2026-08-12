"""The same sample must measure the same at two magnifications.

Higher magnification resolves the rim more clearly, so it should be the easier
case - but the boundary used to be chosen by a gradient crest, and a rim that
fades outward has no crest out there. The search then settled either on the
rim's inner flank or somewhere in the tail, and which one it picked depended on
how many pixels the rim happened to span. The same particles came out ~12%
smaller at high magnification than at low, which is exactly backwards.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_analyzer.analyzer import ParticleAnalyzer, load_image  # noqa: E402

# What must hold is a direction, not an equality. At low magnification the
# outer fade spans a handful of pixels and is genuinely not resolved, so the two
# magnifications are allowed to disagree about where in the fade the edge sits.
# What may never happen is the high-magnification image measuring the particles
# *smaller* - that is the symptom of falling back to the rim's inner flank, and
# it made the clearer image the less accurate one.
MAX_SHRINK_PCT = 3.0


def build(path, radius, rim_frac, soft_frac, seed=3, count=12):
    """A packed field of rimmed particles; `radius` alone sets the magnification."""
    rng = np.random.RandomState(seed)
    size = 1300
    yy, xx = np.mgrid[0:size, 0:size]
    absorb = np.zeros((size, size), np.float32)

    placed = []
    for _ in range(6000):
        if len(placed) >= count:
            break
        r = rng.randint(int(radius * 0.8), int(radius * 1.2))
        cx = rng.randint(-r // 3, size + r // 3)
        cy = rng.randint(-r // 3, size + r // 3)
        gap = min((np.hypot(cx - px, cy - py) - r - pr for px, py, pr in placed),
                  default=0.0)
        if gap < -0.10 * r or gap > 0.05 * r:
            continue
        placed.append((cx, cy, r))

    for cx, cy, r in placed:
        rim, soft = r * rim_frac, max(1.0, r * soft_frac)
        d = np.hypot(xx - cx, yy - cy)
        absorb[d <= r - rim] += 0.18
        # Dark ring: sharp on the inside, fading outward over `soft` pixels.
        inner = np.clip((d - (r - rim)) / 2.0, 0, 1)
        outer = np.clip(1 - (d - r) / soft, 0, 1)
        absorb += 0.45 * np.minimum(inner, outer) * (d <= r + soft)

    img = 238.0 * np.exp(-absorb)
    img = cv2.GaussianBlur(img, (0, 0), 2.0) + rng.normal(0, 12, (size, size))
    cv2.imwrite(path, cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                                   cv2.COLOR_GRAY2BGR))
    return placed


def measure(path, truth):
    # The fixture's truth is the outer edge of the rim, so it is scored at that
    # convention (edge_level 0.95), independent of the product default which is
    # tuned to the dark-ring convention of real packed hollow silica.
    particles = ParticleAnalyzer(edge_level=0.95).analyze(
        load_image(path), min_area_px=100, circularity_thresh=0.5, hollow=True)
    valid = [p for p in particles if not p.get("excluded")]
    errors = []
    for p in valid:
        near = sorted((np.hypot(p["center_x"] - x, p["center_y"] - y), r)
                      for x, y, r in truth)
        if near and near[0][0] < near[0][1] * 0.6:
            errors.append((p["radius_px"] - near[0][1]) / near[0][1] * 100)
    return float(np.mean(errors)) if errors else float("nan")


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "magnification.png")

    failures = []
    print("같은 시료를 배율만 바꿔 촬영 (외곽 감쇠 = 반지름 대비 %)")
    for soft in (0.02, 0.08, 0.15):
        results = []
        for radius in (60, 180):
            truth = build(path, radius, 0.13, soft)
            results.append(measure(path, truth))
        low, high = results
        shrink = low - high          # positive when high magnification measures smaller
        ok = shrink <= MAX_SHRINK_PCT
        print(f"  {'PASS' if ok else 'FAIL'}  감쇠 {soft * 100:4.0f}%   "
              f"저배율 {low:+6.1f}%   고배율 {high:+6.1f}%   "
              f"고배율이 {shrink:+5.1f}%p 작게 잼  (허용 {MAX_SHRINK_PCT:.0f}%p)")
        if not ok:
            failures.append(f"감쇠 {soft * 100:.0f}%: 고배율이 {shrink:.1f}%p 작게 잼")

    print("\n" + ("전체 통과" if not failures else "실패: " + "; ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
