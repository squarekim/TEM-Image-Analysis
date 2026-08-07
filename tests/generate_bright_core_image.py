"""Hollow silica as it actually images: bright interiors, thin dark ring, darker gaps.

Every other fixture here is a dark particle on a bright background. Real hollow
silica is the other way round - the interior is *brighter* than the material
between the particles, and the only dark feature is a thin ring at the shell
wall. That inversion matters, because the outer edge is then a transition from
the ring's darkness back up to a gap level that is itself darker than the
particle's inside: the contrast the edge criterion works with is small, and the
level to recover to is not the brightest thing on the ray.

Radius sets the magnification; the sample is otherwise the same, so the three
scales stand in for the same specimen photographed at 500, 200 and 50 nm.
"""
import os

import cv2
import numpy as np


def generate_bright_core_image(output_path, radius=90, count=None, seed=11,
                               size=1024, ring_frac=0.07, grain=9.0):
    rng = np.random.RandomState(seed)
    if count is None:
        # Fill the frame however many particles that takes at this scale.
        count = max(6, int((size / radius) ** 2 * 0.85))

    placed = []
    for _ in range(count * 400):
        if len(placed) >= count:
            break
        r = rng.randint(int(radius * 0.75), int(radius * 1.25))
        cx = rng.randint(-r // 3, size + r // 3)
        cy = rng.randint(-r // 3, size + r // 3)
        if any(np.hypot(cx - px, cy - py) < (r + pr) * 0.99 for px, py, pr in placed):
            continue
        placed.append((cx, cy, r))

    yy, xx = np.mgrid[0:size, 0:size]
    # Gaps between particles are the darker background here.
    img = np.full((size, size), 150.0)
    for cx, cy, r in placed:
        ring = max(2.0, r * ring_frac)
        d = np.hypot(xx - cx, yy - cy)
        img[d <= r] = 178.0                                   # bright interior
        img[(d >= r - ring) & (d <= r)] = 96.0                # dark shell wall

    img = cv2.GaussianBlur(img, (0, 0), max(1.0, radius * 0.012))
    img += rng.normal(0, grain, (size, size))
    img = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    cv2.imwrite(output_path, img)
    return placed


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for name, r in (("bright_core_low", 35), ("bright_core_mid", 90),
                    ("bright_core_high", 200)):
        path = os.path.join(here, f"test_tem_{name}.png")
        truth = generate_bright_core_image(path, radius=r)
        print(f"Generated {path}: {len(truth)} particles, radius ~{r} px")
