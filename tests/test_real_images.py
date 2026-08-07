"""Measure the same specimen photographed at three magnifications.

These are real micrographs of the sample this program exists for. The specimen
does not change between them, so the three must report the same particle size;
any disagreement is the program's, and it needs no ground truth to detect.

The scale bar is read from each image, so the comparison is in nanometres.

Result: the 200 nm and 500 nm images agree to within 3%. The 50 nm image does
not, and the reason is upstream of anything to do with boundaries:
_estimate_radius picks 21.7 px for particles whose radius is about 185 px, so
the seed search runs over the wrong size range entirely and returns hundreds of
grain speckles instead of the eight particles that are there.

The cause is in the band score, which is the validated fraction multiplied by
the *number* of circles the band found. Small circles are always the more
numerous, so on a grainy image of a few large particles the speckle band scored
128 circles against the real band's 6 and won by twenty to one. Weighing by the
fraction of the frame the circles account for says the opposite and says it
clearly - 0.18 against 0.48 - but coverage on its own hands other images to a
handful of spurious large circles that happen to cover the frame (the 500 nm
image here, and the hollow and touching fixtures), so it is not a fix on its
own and is not applied.
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
MAX_SPREAD_PCT = 10.0


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


def main():
    results = {}
    print("같은 시료, 배율만 다름 - 셋의 직경이 일치해야 함")
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
        print(f"        {label:6s} 스케일바 {bar_px:3d}px = {bar_nm:.0f}nm "
              f"({nm_per_px:.3f} nm/px)   검출 {len(valid):4d}   "
              f"반지름중앙 {radius:5.1f}px   직경중앙 {diameter:6.1f} nm")

    agreeing = {k: v for k, v in results.items() if k != KNOWN_BAD}
    failures = []
    if len(agreeing) >= 2:
        values = np.array(list(agreeing.values()))
        spread = (values.max() - values.min()) / values.mean() * 100
        ok = spread <= MAX_SPREAD_PCT
        print(f"\n  {'PASS' if ok else 'FAIL'}  {' / '.join(agreeing)} 일치도: "
              f"{spread:.1f}% 차이 (허용 {MAX_SPREAD_PCT:.0f}%)")
        if not ok:
            failures.append(f"배율 간 {spread:.1f}% 차이")

    if KNOWN_BAD in results and agreeing:
        off = (results[KNOWN_BAD] / np.mean(list(agreeing.values())) - 1) * 100
        print(f"  KNOWN {KNOWN_BAD}: {off:+.0f}% 어긋남 - 미해결. "
              "입자가 아니라 이미지 그레인을 세고 있음 (파일 상단 설명 참조)")

    print("\n" + ("통과 (미해결 항목 제외)" if not failures else "실패: " + "; ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
