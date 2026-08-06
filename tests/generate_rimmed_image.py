"""Generate particles whose only contrast is an asymmetric dark rim.

The hardest sizing case in practice: the interior is as bright as the gaps
between particles, so the particle is visible only as a dark ring, and that
ring is sharp on the inside and fades outward. Both flanks of the ring are
edges, and choosing the stronger one measures the inside of the rim while a
person measuring the outer diameter marks the outside of it.
"""
import os

import cv2
import numpy as np


def generate_rimmed_image(output_path, width=1024, height=1024, rmin=38, rmax=62,
                          rim=8, outer_soft=4, num_particles=60, noise=7, seed=5):
    rng = np.random.RandomState(seed)
    base = np.full((height, width), 205, np.float32)

    placed = []
    for _ in range(num_particles * 500):
        if len(placed) >= num_particles:
            break
        r = int(rng.randint(rmin, rmax))
        cx = int(rng.randint(r + 6, width - r - 6))
        cy = int(rng.randint(r + 6, height - r - 6))
        if any(np.hypot(cx - a, cy - b) < r + c + 3 for a, b, c in placed):
            continue
        placed.append((cx, cy, r))

    ring = np.zeros((height, width), np.float32)
    solid = np.zeros((height, width), np.float32)
    for cx, cy, r in placed:
        cv2.circle(ring, (cx, cy), r, 1.0, -1)
        cv2.circle(ring, (cx, cy), r - rim, 0.0, -1)
        cv2.circle(solid, (cx, cy), r, 1.0, -1)

    # Blur only outside the particle so the rim stays crisp on its inner flank.
    softened = cv2.GaussianBlur(ring, (0, 0), outer_soft)
    img = base - np.where(solid > 0.5, ring, softened) * 140
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Generated rimmed image: {output_path}")
    print(f"  Particles: {len(placed)}   rim {rim} px, outer softening {outer_soft} px")
    return placed


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    generate_rimmed_image(os.path.join(here, "test_tem_rimmed.png"))
