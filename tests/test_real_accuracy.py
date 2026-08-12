"""Accuracy on the real 200 nm and 500 nm micrographs, in detail.

These two images are the same specimen at different magnifications, so they
must agree with each other, and that check needs no ground truth. The 50 nm
image is excluded: its size-scale estimate fails outright (see
test_real_images.py), so it measures grain rather than particles.

Three things are measured here:

  agreement   the two images must report the same size distribution
  fidelity    the boundary must land where "경계 기준" says it does - that
              setting is a position across the rim, 0 at its inner flank and 1
              at its outer, so the boundary is checked against the rim's own
              half-height on both sides. A setting an operator cannot calibrate
              against hand measurements is worse than none.
  recall      particles actually found, estimated from the gaps left in the
              covered area that are large enough to hold another particle.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_analyzer.analyzer import ParticleAnalyzer, load_image  # noqa: E402
from tests.test_real_images import scale_bar_px  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = [("200nm", "tem_200nm.jpg", 200.0), ("500nm", "tem_500nm.jpg", 500.0)]

MAX_DISAGREEMENT_PCT = 3.0    # between the two magnifications, on the mean
MAX_FIDELITY_GAP_PCT = 15.0   # requested recovery fraction vs achieved
MIN_RECALL_PCT = 90.0

failures = []


def analyse(filename, bar_nm, edge_level=None):
    image = load_image(os.path.join(HERE, "real", filename))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bar = scale_bar_px(gray)
    analyzer = ParticleAnalyzer(nm_per_px=bar_nm / bar, edge_level=edge_level)
    particles = analyzer.analyze(image, min_area_px=100, circularity_thresh=0.5,
                                 hollow=True)
    return image, gray, [p for p in particles if not p.get("excluded")], particles


def recall_estimate(gray, valid):
    """Fraction found, from gaps big enough to hold another particle."""
    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)
    for p in valid:
        cv2.circle(mask, (p["center_x"], p["center_y"]), int(p["radius_px"]), 255, -1)
    median_r = float(np.median([p["radius_px"] for p in valid]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (max(3, int(median_r)), max(3, int(median_r))))
    free = cv2.morphologyEx(255 - mask, cv2.MORPH_OPEN, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(free, 8)
    missed = sum(1 for i in range(1, count)
                 if stats[i, cv2.CC_STAT_AREA] > np.pi * median_r ** 2 * 0.7)
    return len(valid) / (len(valid) + missed) * 100, missed, mask.mean() / 255


def fidelity(gray, valid, requested):
    """Where the boundary landed across the rim: 0 = inner flank, 1 = outer."""
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    h, w = gray.shape
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    steps = np.arange(0.55, 1.70, 0.01)
    achieved = []
    for p in valid:
        if p.get("approx"):
            continue
        cx, cy, r = p["center_x"], p["center_y"], p["radius_px"]
        if not (r * 1.7 < cx < w - r * 1.7 and r * 1.7 < cy < h - r * 1.7):
            continue
        prof = np.array([np.median(blurred[(cy + r * f * np.sin(angles)).astype(int),
                                           (cx + r * f * np.cos(angles)).astype(int)])
                         for f in steps])
        start = int(np.argmax(steps >= 0.75))
        rim = int(np.argmin(prof[start:int(np.argmax(steps >= 1.15))])) + start
        outside = float(np.percentile(prof[steps >= 1.35], 80))
        interior = float(np.median(prof[steps <= 0.62]))
        if outside - prof[rim] < 15 or interior - prof[rim] < 15:
            continue

        def flank(level, forward):
            target = prof[rim] + 0.5 * (level - prof[rim])
            walk = prof[rim:] if forward else prof[:rim + 1][::-1]
            hit = np.flatnonzero(walk >= target)
            if not hit.size:
                return None
            j = int(hit[0])
            return steps[rim + j] if forward else steps[rim - j]

        outer, inner = flank(outside, True), flank(interior, False)
        if outer is None or inner is None or outer <= inner:
            continue
        achieved.append((1.0 - inner) / (outer - inner))
    return (float(np.median(achieved)) * 100 if achieved else float("nan")), len(achieved)


def check(name, ok, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        failures.append(name)


def main():
    requested = ParticleAnalyzer.DEFAULT_EDGE_LEVEL
    stats = {}
    print("실제 미크로그래프 정확도 (같은 시료, 배율만 다름)\n")
    for label, filename, bar_nm in IMAGES:
        path = os.path.join(HERE, "real", filename)
        if not os.path.exists(path):
            print(f"  SKIP  {label}: 파일 없음")
            return 0
        image, gray, valid, everything = analyse(filename, bar_nm)
        d = np.array([p["diameter"] for p in valid])
        rec, missed, coverage = recall_estimate(gray, valid)
        got, n = fidelity(gray, valid, requested)
        stats[label] = {"mean": d.mean(), "d50": np.percentile(d, 50),
                        "recall": rec, "fidelity": got}
        print(f"  {label}  검출 {len(valid)} (제외 {len(everything) - len(valid)}), "
              f"화면 {coverage * 100:.0f}% 차지")
        print(f"        직경 평균 {d.mean():6.2f}  표준편차 {d.std():5.2f}  "
              f"D10 {np.percentile(d, 10):.1f}  D50 {np.percentile(d, 50):.1f}  "
              f"D90 {np.percentile(d, 90):.1f} nm")
        print(f"        회수율 약 {rec:.0f}% (빈 구역 {missed}곳)   "
              f"경계 도달 {got:.0f}% (요청 {requested * 100:.0f}%, n={n})")

    print()
    means = [stats[k]["mean"] for k in stats]
    gap = abs(means[0] - means[1]) / np.mean(means) * 100
    check("두 배율 일치", gap <= MAX_DISAGREEMENT_PCT,
          f"평균 직경 {means[0]:.2f} vs {means[1]:.2f} nm -> {gap:.2f}% 차이")

    for label in stats:
        got = stats[label]["fidelity"]
        off = abs(got - requested * 100)
        check(f"{label} 경계 기준 충실도", off <= MAX_FIDELITY_GAP_PCT,
              f"요청 {requested * 100:.0f}% -> 도달 {got:.0f}% ({off:.0f}%p 차이)")
        check(f"{label} 회수율", stats[label]["recall"] >= MIN_RECALL_PCT,
              f"{stats[label]['recall']:.0f}%")

    print("\n  경계 기준별 측정값 (손으로 잰 값에 맞추실 때 참고)")
    print(f"    {'기준':>6}{'200nm 평균':>12}{'500nm 평균':>12}{'차이':>9}")
    for level in (0.25, 0.35, 0.50, 0.65, 0.80):
        row = []
        for label, filename, bar_nm in IMAGES:
            _, _, valid, _ = analyse(filename, bar_nm, edge_level=level)
            row.append(float(np.mean([p["diameter"] for p in valid])))
        print(f"    {int(level * 100):5d}%{row[0]:12.2f}{row[1]:12.2f}"
              f"{abs(row[0] - row[1]) / np.mean(row) * 100:8.2f}%")

    print("\n" + ("전체 통과" if not failures else f"실패: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
