"""Generate gray particles outlined by a dark rim, packed close together.

Many TEM samples show each particle as a grey disc ringed by a darker edge -
shell material, or a Fresnel fringe from defocus. Where the boundary is read
from that ring decides the measured diameter, so this fixture records the true
outer radius and the rim thickness for both conventions to be checked against.
"""
import os

import cv2
import numpy as np


def generate_fringed_image(output_path, width=900, height=900, rmin=28, rmax=58,
                           rim_fraction=0.10, num_particles=110, noise=7, seed=4):
    rng = np.random.RandomState(seed)
    img = np.full((height, width), 196, np.float32)

    placed = []
    for _ in range(num_particles * 400):
        if len(placed) >= num_particles:
            break
        r = int(rng.randint(rmin, rmax))
        cx = int(rng.randint(r + 3, width - r - 3))
        cy = int(rng.randint(r + 3, height - r - 3))
        if any(np.hypot(cx - a, cy - b) < r + c - 1 for a, b, c in placed):
            continue
        placed.append((cx, cy, r))

    for cx, cy, r in placed:
        cv2.circle(img, (cx, cy), r, 70, -1)
        cv2.circle(img, (cx, cy), int(r * (1 - rim_fraction)), 150, -1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    radii = np.array([r for _, _, r in placed])
    print(f"Generated fringed dense image: {output_path}")
    print(f"  Particles: {len(placed)}   radius {radii.min()}-{radii.max()}"
          f"   rim {rim_fraction:.0%} of radius")
    return placed


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    generate_fringed_image(os.path.join(here, "test_tem_fringed.png"))
