"""Print the radial brightness profile of one particle next to the measured edge.

Where a boundary belongs is a judgement call when the particle is outlined by a
soft dark fringe: the steepest point of the rise, the point where the fringe
stops, and what a person marks by eye need not coincide. Printing the profile
makes the choice visible, so a disagreement with hand measurement can be read
off directly instead of guessed at.

    py tests/profile_tool.py <image> <center_x> <center_y>

The centre coordinates come from the Excel export's "Center X/Y (px)" columns.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_analyzer.analyzer import ParticleAnalyzer, load_image  # noqa: E402


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1

    path, cx, cy = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    image = load_image(path)
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {path}")
        return 1

    particles = ParticleAnalyzer().analyze(image, min_area_px=100,
                                           circularity_thresh=0.5, hollow=True)
    if not particles:
        print("검출된 입자가 없습니다.")
        return 1

    p = min(particles, key=lambda q: np.hypot(q["center_x"] - cx,
                                              q["center_y"] - cy))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    r0 = p["radius_px"]

    print(f"가장 가까운 검출 입자: 중심 ({p['center_x']}, {p['center_y']}), "
          f"반지름 {r0:.1f} px, 직경 {r0 * 2:.1f} px")
    print(f"요청 좌표에서 {np.hypot(p['center_x'] - cx, p['center_y'] - cy):.1f} px 떨어짐")
    print()
    print("  r/r0     r(px)   밝기   프로파일")

    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    rows = []
    for f in np.arange(0.55, 1.55, 0.05):
        rr = r0 * f
        xs = (p["center_x"] + rr * np.cos(angles)).astype(int)
        ys = (p["center_y"] + rr * np.sin(angles)).astype(int)
        v = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        rows.append((f, rr, float(np.median(gray[ys[v], xs[v]])) if v.any() else float("nan")))

    values = [b for _, _, b in rows if np.isfinite(b)]
    lo, hi = min(values), max(values)
    for f, rr, b in rows:
        bar = "#" * int((b - lo) / max(hi - lo, 1) * 40)
        mark = "  <== 측정된 경계" if abs(f - 1.0) < 0.026 else ""
        print(f"  {f:5.2f} {rr:8.1f} {b:7.1f}   {bar}{mark}")

    print()
    print("읽는 법: 어두운 링이 가장 낮은 값입니다. 측정 경계가 링의 바깥쪽 끝")
    print("(밝기가 배경으로 회복되는 지점)보다 안쪽이면 그만큼 작게 측정됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
