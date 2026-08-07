"""Generate densely packed, rim-outlined hollow particles that overlap.

This is the hard case real samples present. Particles are pressed together and
project on top of each other, and TEM contrast is absorption: where two shells
overlap the image gets darker still, painting lens-shaped patches between
particles that are as dark as the rims themselves. A ray leaving a particle's
centre therefore meets several equally convincing dark edges, and an edge search
that judges each angle independently picks a different one from angle to angle -
which is what turns a traced outline into a sawtooth.
"""
import os

import cv2
import numpy as np


def generate_packed_hollow_image(output_path, width=1200, height=1200,
                                 num_particles=45, seed=13):
    rng = np.random.RandomState(seed)

    yy, xx = np.mgrid[0:height, 0:width]
    # Work in absorption: contributions add, so overlaps darken.
    absorb = np.zeros((height, width), np.float32)

    particles = []
    for _ in range(num_particles * 80):
        if len(particles) >= num_particles:
            break
        radius = rng.randint(45, 95)
        cx = rng.randint(-radius // 3, width + radius // 3)
        cy = rng.randint(-radius // 3, height + radius // 3)
        # Packed hard enough that neighbours genuinely overlap on screen.
        if any(np.hypot(cx - px, cy - py) < (radius + pr) * 0.78
               for px, py, pr, _ in particles):
            continue
        particles.append((cx, cy, radius, max(4, int(radius * rng.uniform(0.07, 0.11)))))

    for cx, cy, radius, rim in particles:
        d = np.hypot(xx - cx, yy - cy)
        absorb[d <= radius] += 0.20                       # body
        absorb[(d >= radius - rim) & (d <= radius)] += rng.uniform(0.55, 0.75)  # rim

    img = 232.0 * np.exp(-absorb)
    img += cv2.GaussianBlur(rng.normal(0, 6, (height, width)), (0, 0), 3.0) * 3
    img = cv2.GaussianBlur(img, (0, 0), 1.5) + rng.normal(0, 4.5, (height, width))
    img = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    cv2.imwrite(output_path, img)
    return [(cx, cy, r) for cx, cy, r, _ in particles]


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "test_tem_packed_hollow.png")
    truth = generate_packed_hollow_image(path)
    print(f"Generated {path} with {len(truth)} particles")
