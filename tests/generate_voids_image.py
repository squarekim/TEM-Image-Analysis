"""Generate a monolayer packed tightly enough to create interstitial voids.

Two things in a dense monolayer look like particles but are not:

- the curved triangular gap left between three mutually touching particles.
  It is bounded by three arcs, so a circle fits it well, and its size sits in
  the same range as the smaller particles.
- the lens where two particles overlap in projection. TEM contrast is
  absorption, so the lens is darker than either particle and reads as a
  strongly-edged object of its own.

Both are what a person immediately dismisses as "there is nothing there", and
both are what a circle detector happily reports.
"""
import os

import cv2
import numpy as np


def generate_voids_image(output_path, width=1100, height=1100, seed=5):
    rng = np.random.RandomState(seed)

    placed = []
    # Grow a jammed monolayer: each particle is pushed up against the ones
    # already down, so the gaps between them are genuine interstitial voids.
    for _ in range(6000):
        if len(placed) >= 60:
            break
        radius = rng.randint(55, 110)
        cx = rng.randint(-radius // 2, width + radius // 2)
        cy = rng.randint(-radius // 2, height + radius // 2)
        gap = min((np.hypot(cx - px, cy - py) - radius - pr for px, py, pr in placed),
                  default=0.0)
        # Touching (or slightly overlapping) only - never floating free.
        if gap < -0.18 * radius or gap > 0.06 * radius:
            continue
        placed.append((cx, cy, radius))

    yy, xx = np.mgrid[0:height, 0:width]
    absorb = np.zeros((height, width), np.float32)
    for cx, cy, radius in placed:
        rim = max(5, int(radius * rng.uniform(0.10, 0.14)))
        d = np.hypot(xx - cx, yy - cy)
        absorb[d <= radius] += 0.22
        absorb[(d >= radius - rim) & (d <= radius)] += rng.uniform(0.45, 0.60)

    img = 240.0 * np.exp(-absorb)
    img += cv2.GaussianBlur(rng.normal(0, 7, (height, width)), (0, 0), 2.0) * 3
    img = cv2.GaussianBlur(img, (0, 0), 1.4) + rng.normal(0, 5, (height, width))
    img = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    cv2.imwrite(output_path, img)
    return placed


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "test_tem_voids.png")
    truth = generate_voids_image(path)
    print(f"Generated {path} with {len(truth)} particles")
