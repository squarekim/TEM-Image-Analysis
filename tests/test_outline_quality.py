"""Check that traced outlines follow the edge smoothly instead of sawing.

A rimmed particle presents two edges on every ray - the inner and outer flank
of the dark rim - and an edge search that judges each angle on its own flips
between them from angle to angle. The outline then zigzags across the rim's
thickness, which both looks wrong and feeds noise into the circle fit and the
quality classification. This measures how far each traced outline departs from
its own smoothed form, as a fraction of the radius.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_analyzer.analyzer import ParticleAnalyzer, load_image  # noqa: E402
from tests.generate_dense_image import generate_dense_tem_image  # noqa: E402
from tests.generate_hollow_image import generate_hollow_tem_image  # noqa: E402
from tests.generate_packed_hollow_image import generate_packed_hollow_image  # noqa: E402
from tests.generate_rimmed_image import generate_rimmed_image  # noqa: E402

# Sawtoothing across a rim runs to 9-10 % of the radius; a smooth outline on
# these fixtures stays near 1 %. The gap is wide enough that 4 % separates the
# two cleanly without being brittle about noise.
MAX_MEAN = 0.025
MAX_WORST = 0.045


def raggedness(particle):
    """Mean |r - smoothed r| / r around the traced outline, or None if too short."""
    contour = particle.get("contour")
    if contour is None:
        return None
    pts = contour.reshape(-1, 2).astype(float)
    dx = pts[:, 0] - particle["center_x"]
    dy = pts[:, 1] - particle["center_y"]
    radius = np.hypot(dx, dy)[np.argsort(np.arctan2(dy, dx))]
    if len(radius) < 7:
        return None
    smoothed = np.median([np.roll(radius, i) for i in range(-3, 4)], axis=0)
    return float(np.mean(np.abs(radius - smoothed)) / max(np.mean(radius), 1))


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench")
    os.makedirs(out, exist_ok=True)

    fixtures = [
        ("hollow", "hollow.png", generate_hollow_tem_image),
        ("dense", "dense.png", generate_dense_tem_image),
        ("rimmed", "rimmed.png", generate_rimmed_image),
        ("packed", "packed.png", generate_packed_hollow_image),
    ]

    failures = []
    for name, filename, build in fixtures:
        path = os.path.join(out, filename)
        build(path)
        particles = ParticleAnalyzer().analyze(load_image(path), min_area_px=100,
                                               circularity_thresh=0.5, hollow=True)
        valid = [p for p in particles if not p.get("excluded")]
        scores = [s for s in (raggedness(p) for p in valid) if s is not None]
        if not scores:
            failures.append(f"{name}: 측정된 윤곽이 없습니다")
            continue

        mean, worst = float(np.mean(scores)), float(np.max(scores))
        ok = mean <= MAX_MEAN and worst <= MAX_WORST
        print(f"  {'PASS' if ok else 'FAIL'}  {name:8s} n={len(scores):3d} "
              f"평균 {mean * 100:4.1f}%  최대 {worst * 100:4.1f}%"
              f"   (허용 평균 {MAX_MEAN * 100:.1f}% / 최대 {MAX_WORST * 100:.1f}%)")
        if not ok:
            failures.append(f"{name}: 평균 {mean * 100:.1f}% 최대 {worst * 100:.1f}%")

    print("\n" + ("전체 통과" if not failures else "실패: " + "; ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
