"""Measure detection and sizing accuracy against images with known geometry.

The synthetic generators know exactly where every particle is and how big it
is, so they can score the analyzer rather than just count what it found:
detections are matched to ground truth by centre distance and the radius error
is reported in pixels and percent.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_analyzer.analyzer import ParticleAnalyzer, load_image  # noqa: E402
from tests.generate_dense_image import generate_dense_tem_image  # noqa: E402
from tests.generate_hard_image import generate_hard_tem_image  # noqa: E402
from tests.generate_hollow_image import generate_hollow_tem_image  # noqa: E402
from tests.generate_test_image import generate_tem_image  # noqa: E402
from tests.generate_touching_image import generate_touching_image  # noqa: E402


def match(truth, found, tol_factor=0.6):
    """Pair each true particle with the nearest unclaimed detection."""
    pairs, used = [], set()
    for tx, ty, tr in truth:
        best, best_d = None, None
        for i, p in enumerate(found):
            if i in used:
                continue
            d = np.hypot(p["center_x"] - tx, p["center_y"] - ty)
            if d < tr * tol_factor and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is not None:
            used.add(best)
            pairs.append((tr, found[best]["radius_px"], best_d))
    return pairs, len(found) - len(used)


def score(name, path, truth, hollow):
    analyzer = ParticleAnalyzer()
    particles = analyzer.analyze(load_image(path), min_area_px=100,
                                 circularity_thresh=0.5, hollow=hollow)
    valid = [p for p in particles if not p.get("excluded")]
    pairs, spurious = match(truth, valid)
    if not pairs:
        print(f"{name}: no matches")
        return None

    err = np.array([f - t for t, f, _ in pairs])
    rel = np.array([(f - t) / t * 100 for t, f, _ in pairs])
    off = np.array([d for _, _, d in pairs])
    print(f"{name}:")
    print(f"  recall     {len(pairs)}/{len(truth)}   spurious {spurious}")
    print(f"  radius err  bias {err.mean():+.2f} px   MAE {np.abs(err).mean():.2f} px"
          f"   RMS {np.sqrt((err ** 2).mean()):.2f} px")
    print(f"  relative    bias {rel.mean():+.2f} %    MAE {np.abs(rel).mean():.2f} %")
    print(f"  centre off  mean {off.mean():.2f} px   max {off.max():.2f} px")
    return {"recall": len(pairs) / len(truth), "mae_px": float(np.abs(err).mean()),
            "mae_pct": float(np.abs(rel).mean()), "bias_px": float(err.mean()),
            "spurious": spurious}


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench")
    os.makedirs(out, exist_ok=True)

    fixtures = [
        ("solid", "solid.png",
         lambda p: [(x, y, r) for x, y, r in generate_tem_image(p)[0]]),
        ("hollow", "hollow.png",
         lambda p: [(x, y, r) for x, y, r, _, _ in generate_hollow_tem_image(p)[0]]),
        ("touching", "touching.png",
         lambda p: generate_touching_image(p)),
        ("hard", "hard.png",
         lambda p: [(x, y, r) for x, y, r, _ in generate_hard_tem_image(p)[0]]),
        ("dense", "dense.png",
         lambda p: [(x, y, r) for x, y, r, _ in generate_dense_tem_image(p)[0]]),
    ]

    results = {}
    for name, filename, build in fixtures:
        path = os.path.join(out, filename)
        truth = build(path)
        print()
        results[name] = score(name, path, truth, hollow=True)

    print("\nsummary")
    for name, r in results.items():
        if r:
            print(f"  {name:9s} recall {r['recall'] * 100:5.1f}%   "
                  f"radius MAE {r['mae_px']:.2f} px ({r['mae_pct']:.1f} %)   "
                  f"spurious {r['spurious']}")
    return results


if __name__ == "__main__":
    main()
