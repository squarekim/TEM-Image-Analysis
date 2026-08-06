"""Generate a few very large particles buried in heavy image grain.

High-magnification micrographs show only a handful of particles, each filling a
large part of the frame, with strong pixel noise inside them. That noise offers
the circle detector hundreds of speckle-sized candidates, which used to decide
the particle size scale and bury the real particles.
"""
import os

import cv2
import numpy as np


def generate_grainy_image(output_path, width=860, height=860, radius=118,
                          noise=26, grain=1.6, seed=5):
    rng = np.random.RandomState(seed)

    img = np.full((height, width), 118, np.float32)
    placed = []
    pitch = int(radius * 2.08)
    pad = radius + 6  # keep every particle fully inside the frame
    for i in range(3):
        for j in range(3):
            cx, cy = pad + pitch * i, pad + pitch * j
            if not (radius < cx < width - radius and radius < cy < height - radius):
                continue
            cv2.circle(img, (cx, cy), radius, 176, -1)
            placed.append((cx, cy, radius))

    img = cv2.GaussianBlur(img, (5, 5), 0)
    speckle = rng.normal(0, noise, img.shape)
    if grain > 1:
        speckle = cv2.GaussianBlur(speckle, (0, 0), grain) * grain * 1.6
    img = np.clip(img + speckle, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Generated grainy large-particle image: {output_path}")
    print(f"  Particles: {len(placed)}   radius: {radius}   noise sigma: {noise}")
    return placed


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    generate_grainy_image(os.path.join(here, "test_tem_grainy.png"))
