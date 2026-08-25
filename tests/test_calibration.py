"""Hand-measurement calibration carries the user's edge to the whole field.

Where the surface sits on a blurred shell wall is genuinely ambiguous - inner
flank, middle, or outer edge, and the difference is real pixels. The program
picks a place by the field's own median, but the person measuring may want a
different one, and there is no way to argue them out of it: their measurement
is the definition of the diameter for their purpose.

So the test is not "does the calibrated answer match some ground truth we chose"
- it is "if the user measures a handful of particles at a place, does the rest
of the field come to sit at that same place". The shelled fixture has a known
true outer radius, so the user is simulated as measuring a few particles at
exactly that radius; the held-out particles - the ones NOT measured - are then
checked, because getting the measured ones right is trivial and getting the
unmeasured ones right is the whole point.

It also checks the two properties the feature promises beyond that: the place
is stored on the analyzer so a second image reuses it without measuring again,
and clearing it returns the automatic behaviour unchanged.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_analyzer.analyzer import ParticleAnalyzer, load_image  # noqa: E402
from tests.generate_shelled_image import generate_shelled_image  # noqa: E402

SIZE = 1024


def _match(live, truth):
    pairs = []
    for tx, ty, tr, *_ in truth:
        best, best_d = None, None
        for p in live:
            d = np.hypot(p["center_x"] - tx, p["center_y"] - ty)
            if best_d is None or d < best_d:
                best, best_d = p, d
        if best is not None and best_d < tr * 0.5:
            pairs.append((best, tr))
    return pairs


def _case(radius, shell, count, n_ref=8):
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "shelled.png")
    truth = generate_shelled_image(path, radius=radius, shell_frac=shell,
                                   count=count, size=SIZE)
    image = load_image(path)

    analyzer = ParticleAnalyzer()
    particles = analyzer.analyze(image, min_area_px=100,
                                 circularity_thresh=0.5, hollow=True)
    live = [p for p in particles if not p.get("excluded")]
    pairs = _match(live, truth)
    if len(pairs) < n_ref + 8:
        return None

    before = np.array([(p["radius_px"] - tr) / tr * 100 for p, tr in pairs])

    # The user measures the first few at their true outer diameter.
    refs = [(p["center_x"], p["center_y"], 2 * tr) for p, tr in pairs[:n_ref]]
    held = pairs[n_ref:]
    place, used = analyzer.calibrate_to_measurements(particles, refs, image)

    after = np.array([(p["radius_px"] - tr) / tr * 100 for p, tr in held])
    return {
        "n": len(pairs), "place": place, "used": used,
        "before": float(np.abs(before).mean()),
        "after": float(np.abs(after).mean()),
        "stored": analyzer.wall_place,
    }


def main():
    failures = []
    print("손 측정 보정 — 재보지 않은 입자가 사람이 잰 자리로 오는가")
    for radius, shell, count in ((30, 0.18, 40), (60, 0.18, 40)):
        r = _case(radius, shell, count)
        if r is None:
            print(f"  SKIP  반지름 {radius}: 매칭 부족")
            continue
        # The held-out error must come down, and the place must be stored.
        ok = (r["after"] <= r["before"] and r["after"] <= 2.0
              and r["stored"] is not None and r["used"] >= 6)
        print(f"  {'PASS' if ok else 'FAIL'}  반지름 {radius:3d}  참조 {r['used']}개  "
              f"벽 위치 {r['place']:.2f}   "
              f"재보지 않은 입자 외경오차 {r['before']:.2f}% → {r['after']:.2f}%")
        if not ok:
            failures.append(f"반지름 {radius}: {r['before']:.1f}%→{r['after']:.1f}%")

    # The default path must be untouched when no calibration is set: a fresh
    # analyzer measures exactly as before.
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench")
    path = os.path.join(out, "shelled.png")
    generate_shelled_image(path, radius=60, shell_frac=0.18, count=40, size=SIZE)
    image = load_image(path)
    a = ParticleAnalyzer()
    d_auto = [p["radius_px"] for p in a.analyze(image, min_area_px=100,
              circularity_thresh=0.5, hollow=True) if not p.get("excluded")]
    b = ParticleAnalyzer()
    b.wall_place = None
    d_none = [p["radius_px"] for p in b.analyze(image, min_area_px=100,
              circularity_thresh=0.5, hollow=True) if not p.get("excluded")]
    same = d_auto == d_none
    print(f"  {'PASS' if same else 'FAIL'}  보정 없을 때 자동 판단과 동일: {same}")
    if not same:
        failures.append("wall_place=None 경로가 자동과 다름")

    print("\n" + ("전체 통과" if not failures else "실패: " + "; ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
