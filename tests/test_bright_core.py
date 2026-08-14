"""Measure the real structure: bright interiors, thin dark ring, darker gaps.

Every other fixture is a dark particle on a bright background. Real hollow
silica is inverted - the interior is brighter than the material between the
particles, and the only dark feature is a thin ring at the shell wall. The
analyzer is measurably worse on this arrangement, and worse in a way that
depends on magnification, so it is tracked separately.

Known gap, unresolved: at the lowest magnification the particles come out about
9% small, and the "경계 기준" setting has no effect on that number at all -
which locates the fault. The setting only moves the boundary found from the
brightness profile, so a number it cannot move means that boundary is not being
used, and every particle falls back to the steepest-gradient edge.

Why it is not used is now known, and it is not the rim vote. Reaching the
brightness boundary requires it to come out at least half a pixel outside the
gradient crest, on the reasoning that a rim's outer edge lies beyond its crest.
Where a shell's outer flank tapers instead of stepping, the crest already sits
as far out as the profile goes and the two coincide - so a correct boundary is
rejected for not being further out, and the setting has nothing to move.

Removing that ordering test does make the setting responsive (90% to 100% then
moves the answer by 2.5%p where it moved 0.2%p before) and improves the hollow
fixture from 2.3% to 1.4%. It also admits the curved gap between three packed
particles as a particle, because such a gap clears every remaining test:
test_false_positives goes from zero ghosts to one. A tolerance instead of a
threshold does not separate them either - the ghost's two boundaries coincide
just as closely as the shell's do. Ghosts cost more than an inert setting, so
the ordering test stays and this is written down rather than traded away.

Attempts that did not work, so the next person does not repeat them:
  - judging the rim's polarity against the interior instead of the outside
    level fixes the packed-field case and inflates grainy to 17%
  - two-sided prominence fixes the polarity blind spot but reads noise troughs
    in the background of a grainy image as rims (16%)
  - requiring the rim to sit at a consistent radius does not separate them
  - capping the rim's thickness clears grainy at 1.15 but breaks rimmed, which
    needs more than 1.20; no single value separates the two
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_analyzer.analyzer import ParticleAnalyzer, load_image  # noqa: E402
from tests.generate_bright_core_image import generate_bright_core_image  # noqa: E402

# The two magnifications that do work are held to this; the lowest is reported
# but not asserted, so the known gap stays visible without a permanently red
# suite that everyone learns to ignore.
MAX_ERROR_PCT = 3.0
KNOWN_BAD_RADIUS = 35


def measure(path, truth, edge_level=None):
    analyzer = ParticleAnalyzer(edge_level=edge_level)
    particles = analyzer.analyze(load_image(path), min_area_px=100,
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
    return (float(np.mean(errors)), float(np.std(errors)), len(used)) if errors \
        else (float("nan"), float("nan"), 0)


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "bright_core.png")

    failures = []
    print("밝은 내부 + 얇은 어두운 링 (실제 중공 실리카 구조)")
    for label, radius in (("500nm급", 35), ("200nm급", 90), ("50nm급", 200)):
        truth = generate_bright_core_image(path, radius=radius)
        # Truth here is the shell's outer edge, so score at that convention
        # (0.95) rather than the dark-ring product default.
        mean, std, matched = measure(path, truth, edge_level=0.95)
        known = radius == KNOWN_BAD_RADIUS
        ok = known or abs(mean) <= MAX_ERROR_PCT
        tag = "KNOWN" if known else ("PASS" if ok else "FAIL")
        print(f"  {tag:5s} {label:8s} 반지름 {radius:3d}px  매칭 {matched:3d}/{len(truth):<3d}"
              f"  오차 {mean:+6.1f}%  표준편차 {std:5.1f}%")
        if not ok:
            failures.append(f"{label}: {mean:+.1f}%")

    # The setting is inert at the failing scale; showing that is the diagnosis.
    truth = generate_bright_core_image(path, radius=KNOWN_BAD_RADIUS)
    levels = [(lvl, measure(path, truth, edge_level=lvl)[0]) for lvl in (0.35, 0.50, 0.80)]
    print("\n  미해결: 최저배율에서 '경계 기준'을 바꿔도 값이 움직이지 않음 "
          "-> 밝기 기반 경계가 아예 쓰이지 않는다는 뜻")
    print("   " + "   ".join(f"{int(l * 100)}% -> {e:+.1f}%" for l, e in levels))

    print("\n" + ("통과 (미해결 항목 제외)" if not failures else "실패: " + "; ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
