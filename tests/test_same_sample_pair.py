"""One specimen at two magnifications - the only check that can settle a bias.

Everything else here is scored against either a generator's ground truth or
nothing at all. A generator's truth is only as good as the generator: the dense
fixture recorded a radius a pixel inside the particle it drew, and for a year
that read as the analyzer measuring large. Real micrographs have no truth to
compare against, and the three in tests/real are three different specimens, so
their agreeing or disagreeing says nothing.

A single specimen photographed at two magnifications escapes both problems. The
particles are the same particles, so the two images must report the same size
distribution in nanometres, and the only things that can break that are the
scale bar and the boundary rule. Nothing is assumed about what the right answer
is - only that the two must agree.

That makes this the test that decides `sphere_edge`. A sphere's edge fades in
over several pixels rather than stepping, so a brightness threshold lands
inside the true edge by about a pixel - a fixed distance, which is a larger
share of a small particle than a large one. So if the boundary is being placed
by brightness, the higher-magnification image (bigger particles in pixels)
reports a *larger* diameter in nanometres than the lower one, and the gap
between them measures the bias directly. The edge model removes the pixel
offset, so if it is right the two magnifications close up when it is on.

  gap unchanged or worse with sphere_edge on  ->  leave it off
  gap clearly smaller with it on              ->  turn it on

To use: put both images in tests/real/ named <group>_<bar>nm.<ext>, e.g.
batchA_200nm.jpg and batchA_500nm.jpg. Any prefix except "tem" is treated as a
same-specimen group; the three tem_*.jpg images are excluded because they are
not one. The test skips quietly when no pair is present, so it can live in the
suite before the images do.
"""
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_analyzer.analyzer import ParticleAnalyzer, load_image  # noqa: E402
from tests.test_real_images import scale_bar_px  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "real")
PATTERN = re.compile(r"^(.+)_(\d+)nm\.(jpg|jpeg|png|tif|tiff)$", re.I)
NOT_A_GROUP = {"tem"}

#: Two magnifications of one specimen should agree this closely. Loose, because
#: the two frames show different particles of the same population and a few
#: hundred particles still leave a sampling difference of a percent or two.
MAX_GAP_PCT = 5.0


def groups():
    if not os.path.isdir(REAL):
        return {}
    found = defaultdict(list)
    for name in sorted(os.listdir(REAL)):
        m = PATTERN.match(name)
        if m and m.group(1).lower() not in NOT_A_GROUP:
            found[m.group(1)].append((float(m.group(2)), name))
    return {k: sorted(v) for k, v in found.items() if len(v) >= 2}


def measure(filename, bar_nm, sphere_edge):
    image = load_image(os.path.join(REAL, filename))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    bar = scale_bar_px(gray)
    if not bar:
        return None
    analyzer = ParticleAnalyzer(nm_per_px=bar_nm / bar, sphere_edge=sphere_edge)
    valid = [p for p in analyzer.analyze(image, min_area_px=100,
                                         circularity_thresh=0.5, hollow=True)
             if not p.get("excluded")]
    if len(valid) < 10:
        return None
    d = np.array([p["diameter"] for p in valid])
    return {"n": len(d), "bar_px": bar, "nm_per_px": bar_nm / bar,
            "mean": float(d.mean()), "d50": float(np.median(d)),
            "sd": float(d.std())}


def main():
    pairs = groups()
    if not pairs:
        print("SKIP  같은 시료의 두 배율 이미지가 없습니다.")
        print("      tests/real/ 에 <이름>_200nm.jpg, <이름>_500nm.jpg 형태로 넣으면"
              " 이 검사가 동작합니다.")
        print("      (tem_*.jpg 세 장은 서로 다른 시료이므로 제외됩니다)")
        return 0

    failures = []
    print("같은 시료, 배율만 다름 - 두 이미지가 같은 크기를 보고해야 함")
    for group, members in pairs.items():
        print(f"\n  [{group}]")
        for sphere_edge in (False, True):
            label = "구 모서리 보정 켬" if sphere_edge else "구 모서리 보정 끔"
            rows = []
            for bar_nm, filename in members:
                r = measure(filename, bar_nm, sphere_edge)
                if r is None:
                    print(f"    SKIP  {filename}: 스케일바 또는 입자를 찾지 못함")
                    rows = []
                    break
                rows.append((bar_nm, filename, r))
            if len(rows) < 2:
                continue
            for bar_nm, filename, r in rows:
                print(f"    {label}  {bar_nm:5.0f}nm바={r['bar_px']:3d}px "
                      f"({r['nm_per_px']:.3f} nm/px)  검출 {r['n']:4d}  "
                      f"평균 {r['mean']:6.2f}  중앙 {r['d50']:6.2f} nm")
            means = np.array([r["mean"] for _, _, r in rows])
            gap = (means.max() - means.min()) / means.mean() * 100
            print(f"    {label}  ->  배율 간 차이 {gap:.2f}%")
            if sphere_edge:
                ok = gap <= MAX_GAP_PCT
                print(f"  {'PASS' if ok else 'FAIL'}  [{group}] 켠 상태 일치도 "
                      f"{gap:.2f}% (허용 {MAX_GAP_PCT:.0f}%)")
                if not ok:
                    failures.append(f"{group}: {gap:.2f}%")

    print("\n" + ("전체 통과" if not failures else "실패: " + "; ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
