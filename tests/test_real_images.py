"""Measure real micrographs at three magnifications.

These are real micrographs of the sample this program exists for, and they are
the only images here whose particles were not drawn by a generator.

They are NOT the same specimen. This file used to assert that the three had to
report the same size, on the assumption that only the magnification differed;
the person who supplied them has since said otherwise. So sizes agreeing across
these three is not evidence of accuracy, and sizes disagreeing is not evidence
of a fault - the specimens genuinely differ. Do not reintroduce that assertion
without a pair of images confirmed to be one specimen. What these images are
good for is everything that does not depend on the sample: the scale bar being
read, particles being found at all, and the run not falling over.

The scale bar is read from each image, so the comparison is in nanometres.

Result: the 200 nm and 500 nm images report sizes within 3% of each other,
which given the above is a coincidence worth printing and nothing to lean on.

The 50 nm image used to measure grain rather than particles - 16 nm where the
particles are about 90 - and it is now fixed. The record of how, because the
same trap is easy to fall back into:

_estimate_radius scored each octave band by validated fraction times the
*number* of circles it found. Small circles are always the more numerous, so on
a grainy image of a dozen large particles the grain-scale band offered 159
circles against the real band's 8 and won twenty to one. The validated fraction
could not object, because the threshold that validates a band is drawn from
that band's own edge scores - a band of pure noise grades itself on a curve,
and this one passed 81%.

Two fixes that look obvious do not work. Weighting by the fraction of the frame
covered hands sparse images to a few huge circles that happen to span it
(hollow, touching, solid and dense all break). Comparing edge strength across
bands is meaningless, since each band is searched at its own smoothing.

What grain cannot fake is a round boundary. Fitting a circle to the traced edge
and measuring how far the edge strays from it separates the two on every image
available, real and synthetic: true bands 0.005-0.025 of a radius, grain bands
0.040-0.055, with nothing in between. Roundness now decides which bands are
eligible and population only breaks ties among them.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_analyzer.analyzer import ParticleAnalyzer, load_image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = [("50nm", "tem_50nm.jpg", 50.0), ("200nm", "tem_200nm.jpg", 200.0),
          ("500nm", "tem_500nm.jpg", 500.0)]
KNOWN_BAD = "50nm"


def scale_bar_px(gray):
    """Length of the bright scale bar in the bottom-left corner, in pixels."""
    h, w = gray.shape
    roi = gray[int(h * 0.86):, :int(w * 0.6)]
    best = 0
    for threshold in (200, 215, 230):
        mask = (roi > threshold).astype(np.uint8)
        for row in mask:
            edges = np.flatnonzero(np.diff(np.r_[0, row, 0]))
            for start, end in zip(edges[::2], edges[1::2]):
                if end - start > 60:
                    best = max(best, end - start)
    return best or None


def packing_ratio(particles):
    """Median diameter against the distance to the nearest neighbours.

    In a jammed monolayer the particles are touching or nearly so, which fixes
    half the centre-to-centre distance at the outer radius. That makes this a
    check on where the boundary was put which needs no ground truth, no
    brightness model and no scale bar - only the centres, which are found
    independently of the radius.

    It is the check that caught the boundary sitting on the inner flank of the
    shell wall: the yolk-shell field read 0.93 while its neighbours' walls said
    it should read about 1. Comparing diameters between magnifications of the
    same specimen cannot do that job - the neighbour spacing in nm differs by
    up to 6% between two frames of one specimen, because a high-magnification
    frame holds twenty particles and they are not the same twenty.

    Nearest *three*, not nearest one: the single nearest neighbour is the
    closest of six or so and reads systematically short.
    """
    if len(particles) < 8:
        return None
    centres = np.array([[p["center_x"], p["center_y"]] for p in particles], float)
    radii = np.array([p["radius_px"] for p in particles], float)
    gaps = np.hypot(centres[:, 0][:, None] - centres[:, 0][None, :],
                    centres[:, 1][:, None] - centres[:, 1][None, :])
    np.fill_diagonal(gaps, np.inf)
    half_spacing = np.sort(gaps, axis=1)[:, :3].mean(axis=1) / 2.0
    return float(np.median(radii) / np.median(half_spacing))


# A circle on the inner flank of the shell wall packs at 0.93 and one that has
# swallowed a neighbour's wall at 1.10; a correct one sits just under 1. The
# bar is wide because a sparse field genuinely packs loose - it is here to
# catch the boundary landing on the wrong feature, not to tune it.
PACKING_RANGE = (0.88, 1.08)


def main():
    results = {}
    packing = {}
    print("실제 미크로그래프 3장 (서로 다른 시료 - 직경 비교는 참고용)")
    for label, filename, bar_nm in IMAGES:
        path = os.path.join(HERE, "real", filename)
        if not os.path.exists(path):
            print(f"  SKIP  {label}: {path} 없음")
            continue
        image = load_image(path)
        bar_px = scale_bar_px(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if not bar_px:
            print(f"  SKIP  {label}: 스케일바를 찾지 못함")
            continue

        nm_per_px = bar_nm / bar_px
        particles = ParticleAnalyzer(nm_per_px=nm_per_px).analyze(
            image, min_area_px=100, circularity_thresh=0.5, hollow=True)
        valid = [p for p in particles if not p.get("excluded")]
        if not valid:
            print(f"  FAIL  {label}: 검출 없음")
            continue

        diameter = float(np.median([p["diameter"] for p in valid]))
        radius = float(np.median([p["radius_px"] for p in valid]))
        results[label] = diameter
        packing[label] = packing_ratio(valid)
        packed = ("-" if packing[label] is None else f"{packing[label]:.3f}")
        print(f"        {label:6s} 스케일바 {bar_px:3d}px = {bar_nm:.0f}nm "
              f"({nm_per_px:.3f} nm/px)   검출 {len(valid):4d}   "
              f"반지름중앙 {radius:5.1f}px   직경중앙 {diameter:6.1f} nm   "
              f"이웃대비 {packed}")

    agreeing = {k: v for k, v in results.items() if k != KNOWN_BAD}
    failures = []
    if len(agreeing) >= 2:
        values = np.array(list(agreeing.values()))
        spread = (values.max() - values.min()) / values.mean() * 100
        # Printed, never asserted: these are different specimens, so the two
        # sizes have no reason to match and matching proves nothing.
        print(f"\n  참고  {' / '.join(agreeing)} 직경 차이 {spread:.1f}% "
              "(서로 다른 시료이므로 판정 기준 아님)")

    for label, ratio in packing.items():
        if ratio is None:
            continue
        lo, hi = PACKING_RANGE
        ok = lo <= ratio <= hi
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: 원 지름이 이웃 간격의 "
              f"{ratio:.3f}배 (허용 {lo}-{hi})")
        if not ok:
            failures.append(f"{label}: 이웃 대비 {ratio:.3f} — 경계가 쉘 벽 "
                            "위가 아님")

    if KNOWN_BAD in results:
        # This image used to report 16 nm - image grain - and is now asserted
        # like the rest. The bar is deliberately loose: what failed before was
        # off by a factor of six, and no ground truth exists for this image.
        d = results[KNOWN_BAD]
        ok = 40.0 <= d <= 200.0
        print(f"  {'PASS' if ok else 'FAIL'}  {KNOWN_BAD}: 직경 {d:.1f} nm "
              "(그레인을 세면 20nm 미만으로 나옴)")
        if not ok:
            failures.append(f"{KNOWN_BAD}: 직경 {d:.1f} nm — 입자가 아닌 그레인")

    print("\n" + ("전체 통과" if not failures else "실패: " + "; ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
