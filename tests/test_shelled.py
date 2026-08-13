"""Hollow spheres sitting apart on a support film, scored against real geometry.

This is the arrangement the other fixtures do not cover: particles separated by
open background, each darkening the beam by how much shell the ray crossed. The
fixture is computed from the projection integral rather than drawn, so the
ground truth is the sphere's actual outer radius and not a ring somebody
painted at a chosen place - which makes this the one test here that can say
whether the reported diameter is the outer diameter.

It also has the opposite contrast to the packed fixtures. Here the interior is
*darker* than the background, because a ray through the middle still crosses
two shell thicknesses; in a packed field the gaps are darkened by neighbouring
shells and the interiors come out brighter. A boundary rule that reads contrast
rather than structure passes one of those and fails the other.

Result: the outer diameter comes out 1-3% small across a four-fold range of
particle size, with no false detections. The bias is the half-height criterion
sitting just inside the true edge on a blurred flank, and it shrinks as the
particle gets larger relative to the blur - which is the expected direction.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_analyzer.analyzer import ParticleAnalyzer, load_image  # noqa: E402
from tests.generate_shelled_image import generate_shelled_image  # noqa: E402

SIZE = 1024
MAX_ERROR_PCT = 5.0
MIN_RECALL_PCT = 75.0

CASES = [
    ("작은 입자", 30, 0.18, 40),
    ("중간 입자", 60, 0.18, 40),
    ("큰 입자", 120, 0.18, 16),
    ("두꺼운 쉘", 60, 0.30, 40),
    ("얇은 쉘", 60, 0.08, 40),
]


def measure(path, truth):
    particles = ParticleAnalyzer().analyze(load_image(path), min_area_px=100,
                                           circularity_thresh=0.5, hollow=True)
    valid = [p for p in particles if not p.get("excluded")]
    errors, used = [], set()
    for tx, ty, tr in truth:
        best, best_d = None, None
        for i, p in enumerate(valid):
            if i in used:
                continue
            d = np.hypot(p["center_x"] - tx, p["center_y"] - ty)
            if d < tr * 0.6 and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is not None:
            used.add(best)
            errors.append((valid[best]["radius_px"] - tr) / tr * 100)
    return errors, len(used), len(valid) - len(used)


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "shelled.png")

    failures = []
    print("중공 구, 배경 위에 떨어져 있음 (투영 물리로 생성 - 정답은 외반경)")
    for label, radius, shell, count in CASES:
        truth = generate_shelled_image(path, radius=radius, shell_frac=shell,
                                       count=count, size=SIZE)
        # A particle the frame cuts through has no determinable diameter and is
        # excluded by design, so it cannot count against recall.
        truth = [(x, y, r) for x, y, r in truth
                 if x - r >= 0 and y - r >= 0 and x + r <= SIZE and y + r <= SIZE]
        errors, matched, spurious = measure(path, truth)
        if not errors:
            print(f"  FAIL  {label}: 매칭된 입자 없음")
            failures.append(f"{label}: 검출 실패")
            continue
        mean, std = float(np.mean(errors)), float(np.std(errors))
        recall = matched / max(len(truth), 1) * 100
        ok = abs(mean) <= MAX_ERROR_PCT and recall >= MIN_RECALL_PCT and spurious == 0
        print(f"  {'PASS' if ok else 'FAIL'}  {label:9s} 반지름 {radius:3d}px "
              f"쉘 {shell * 100:2.0f}%   매칭 {matched:2d}/{len(truth):<2d} "
              f"({recall:3.0f}%)   외경 오차 {mean:+5.1f}% (편차 {std:4.1f}%)   "
              f"거짓검출 {spurious}")
        if not ok:
            failures.append(f"{label}: 오차 {mean:+.1f}%, 회수 {recall:.0f}%, "
                            f"거짓 {spurious}")

    print("\n" + ("전체 통과" if not failures else "실패: " + "; ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
