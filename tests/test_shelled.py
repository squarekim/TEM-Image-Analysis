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

It also carries a minority of particles with a dense core - the ones a real
preparation is never entirely free of, where the template was not removed. They
matter out of proportion to their number, because any step that decides what a
particle should look like by learning it from the population will throw away
exactly the particles that look different. This is not hypothetical: judging
the interior over the whole disc excluded every one of them, and the fix was to
read the ring just inside the wall, which is the part cored and hollow
particles have in common.

That fix is not complete, and the limit is worth knowing before someone tries
again. On a real yolk-shell sample the cores fill nearly the whole interior
rather than sitting small in the middle, so they darken the ring just inside
the wall too: six of its seven cored particles are still excluded there. The
obvious repair - exempting any candidate darker inside than at its own boundary,
since a gap or an overlap lens holds less material than the walls around it and
must read brighter - does recover them, and costs more than it gives: two of
nine synthetic cored particles are then kept with the circle sitting on the
core rather than the shell, measured at 57% of true size, which corrupts a size
distribution far worse than losing a particle does. Four ways to tell those two
cases apart were measured and none separates them - the second concentric ring
(ratio 1.96 vs 2.07), its depth, the linearity of the edge fit, and how far the
cavity recovers from core to wall (0.48-0.59 against 0.09-0.22 on the synthetic
fixture, but the real cores are large enough to sit in the same range). A
circle on the shell of a particle whose core fills it looks like a circle on a
core. Excluding them is the safer error: they are a small minority and their
size is close to the population median (99 nm against 95), so the distribution
barely moves.

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
from tests.benchmark_accuracy import visible_arc  # noqa: E402
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


def cored_case():
    """A dark-cored minority must survive, and be sized like the rest."""
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "cored.png")
    failures = []
    print("\n코어를 가진 소수 입자가 섞였을 때 (템플릿이 남은 입자)")
    for fraction in (0.10, 0.25, 0.40):
        truth = generate_shelled_image(path, radius=70, count=32, seed=11,
                                       cored=fraction, size=SIZE)
        truth = [(x, y, r, c) for x, y, r, c in truth
                 if visible_arc(x, y, r, SIZE, SIZE) >= ParticleAnalyzer.MIN_VISIBLE_ARC]
        valid = [p for p in ParticleAnalyzer().analyze(
            load_image(path), min_area_px=100, circularity_thresh=0.5,
            hollow=True, detect_cores=True) if not p.get("excluded")]

        found = {True: 0, False: 0}
        total = {True: 0, False: 0}
        errors, flagged, used = [], 0, set()
        for tx, ty, tr, cored in truth:
            total[cored] += 1
            best, best_d = None, None
            for i, p in enumerate(valid):
                if i in used:
                    continue
                d = np.hypot(p["center_x"] - tx, p["center_y"] - ty)
                if d < tr * 0.6 and (best_d is None or d < best_d):
                    best, best_d = i, d
            if best is None:
                continue
            used.add(best)
            found[cored] += 1
            if cored:
                errors.append((valid[best]["radius_px"] - tr) / tr * 100)
                flagged += bool(valid[best].get("has_core"))

        recall = found[True] / max(total[True], 1) * 100
        plain = found[False] / max(total[False], 1) * 100
        bias = float(np.mean(errors)) if errors else float("nan")
        ok = recall >= 75.0 and plain >= 90.0 and abs(bias) <= MAX_ERROR_PCT
        print(f"  {'PASS' if ok else 'FAIL'}  코어 비율 {fraction * 100:3.0f}%   "
              f"코어입자 {found[True]:2d}/{total[True]:<2d} ({recall:3.0f}%)   "
              f"일반 {found[False]:2d}/{total[False]:<2d} ({plain:3.0f}%)   "
              f"외경 오차 {bias:+5.1f}%   코어로 판정 {flagged}/{found[True]}")
        if not ok:
            failures.append(f"코어 비율 {fraction * 100:.0f}%: 회수 {recall:.0f}%")
    return failures


def sphere_edge_case():
    """The opt-in edge model, and the trade it makes, in numbers."""
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "sphere_edge.png")
    failures = []
    print("\n구 모서리 외삽 (sphere_edge) — 켜고 끄고 비교")
    for radius, shell in ((30, 0.18), (60, 0.18), (120, 0.18), (60, 0.08), (60, 0.30)):
        truth = generate_shelled_image(path, radius=radius, shell_frac=shell,
                                       count=40 if radius < 100 else 16, size=SIZE)
        truth = [(x, y, r) for x, y, r, _ in truth
                 if visible_arc(x, y, r, SIZE, SIZE) >= ParticleAnalyzer.MIN_VISIBLE_ARC]
        row = []
        for flag in (False, True):
            valid = [p for p in ParticleAnalyzer(sphere_edge=flag).analyze(
                load_image(path), min_area_px=100, circularity_thresh=0.5,
                hollow=True) if not p.get("excluded")]
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
            row.append(float(np.mean(errors)) if errors else float("nan"))
        off, on = row
        # The point of the option is that it removes the bias, so that is what
        # is asserted - not that it beats the default by some margin.
        ok = abs(on) <= 1.5
        print(f"  {'PASS' if ok else 'FAIL'}  반지름 {radius:3d}px 쉘 {shell * 100:2.0f}%   "
              f"끔 {off:+6.2f}%   켬 {on:+6.2f}%")
        if not ok:
            failures.append(f"r={radius} 쉘{shell * 100:.0f}%: 켬 {on:+.2f}%")
    print("  참고: 이 옵션은 계단 모서리(두께 일정한 원반)를 1px쯤 크게 읽습니다.")
    print("        benchmark_accuracy를 켜고 돌리면 grainy 0.3%->8.0%, "
          "rimmed 0.9%->3.8%.")
    print("        실제 시료(빽빽한 단층) 3세트로 확인한 결과: 배율 간 일치도는")
    print("        4.19->3.55%, 1.59->0.02%, 6.25->6.38%. 이득이 작고 일정하지")
    print("        않으므로 기본값은 끔, 판단은 사용자 몫입니다.")
    return failures


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "shelled.png")

    failures = []
    print("중공 구, 배경 위에 떨어져 있음 (투영 물리로 생성 - 정답은 외반경)")
    for label, radius, shell, count in CASES:
        truth = generate_shelled_image(path, radius=radius, shell_frac=shell,
                                       count=count, size=SIZE)
        # Scored at the same line the analyzer draws: a particle with two
        # thirds of its circumference inside the frame is measurable.
        truth = [(x, y, r) for x, y, r, _ in truth
                 if visible_arc(x, y, r, SIZE, SIZE) >= ParticleAnalyzer.MIN_VISIBLE_ARC]
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

    failures += cored_case()
    failures += sphere_edge_case()
    print("\n" + ("전체 통과" if not failures else "실패: " + "; ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
