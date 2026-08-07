"""Check the two things a jammed monolayer makes the detector report falsely.

1. Detections in the gaps between particles. The curved triangle left between
   three touching particles fits a circle well and is the same size as the
   smaller particles; the lens where two particles overlap in projection has
   an even stronger edge. Neither is a particle, and a person dismisses both
   instantly - "there is nothing there".
2. A small detection sitting on top of a large one. Judging overlap by centre
   distance against the summed radii cannot see this: the large radius alone
   keeps the threshold out of reach, so the pair is never called out even when
   the small circle is entirely inside the large one.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_analyzer.analyzer import ParticleAnalyzer, load_image  # noqa: E402
from tem_analyzer.gui import MainWindow  # noqa: E402
from tests.generate_voids_image import generate_voids_image  # noqa: E402

failures = []


def check(name, got, want, tol=0.0):
    ok = abs(float(got) - float(want)) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got}  (기대 {want})")
    if not ok:
        failures.append(name)


def test_overlap_fraction():
    print("겹침 면적 비율 (작은 원 기준)")
    f = MainWindow._overlap_fraction
    check("떨어져 있음", f(100, 30, 30), 0.0)
    check("접함", f(60, 30, 30), 0.0)
    check("작은 원이 완전히 안에", f(10, 80, 20), 1.0)
    check("동심원", f(0, 80, 20), 1.0)
    check("반지름 같고 중심 일치", f(0, 30, 30), 1.0)
    # Half-overlap of equal circles: centres one radius apart.
    check("같은 크기 절반 겹침", round(f(30, 30, 30), 2), 0.39, 0.02)

    # The case the old centre-distance rule missed: a small detection well
    # inside a large one. Old rule flagged only if d < 0.55 * (r1 + r2).
    d, small, large = 70.0, 22.0, 95.0
    old_rule = d < 0.55 * (small + large)
    print(f"  작은 검출({small:.0f}px)이 큰 입자({large:.0f}px) 안에 {d:.0f}px 거리로 있을 때:")
    print(f"        예전 규칙(중심거리) 플래그: {old_rule}")
    check("겹침 비율로는 잡힘", f(d, small, large) > 0.30, True)


def test_no_void_detections():
    print("\n빽빽한 단층에서 빈틈을 입자로 잡지 않는가")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "voids.png")
    truth = generate_voids_image(path)

    particles = ParticleAnalyzer().analyze(load_image(path), min_area_px=100,
                                           circularity_thresh=0.5, hollow=True)
    valid = [p for p in particles if not p.get("excluded")]

    matched = set()
    for tx, ty, tr in truth:
        best, best_d = None, None
        for i, p in enumerate(valid):
            if i in matched:
                continue
            d = np.hypot(p["center_x"] - tx, p["center_y"] - ty)
            if d < tr * 0.6 and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is not None:
            matched.add(best)

    spurious = len(valid) - len(matched)
    print(f"        실제 {len(truth)}개 / 유효 검출 {len(valid)}개 / 매칭 {len(matched)}개")
    check("빈틈 오검출", spurious, 0)


def main():
    test_overlap_fraction()
    test_no_void_detections()
    print("\n" + ("전체 통과" if not failures else f"실패: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
