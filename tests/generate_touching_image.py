"""Generate hollow particles arranged in touching clusters.

Real TEM samples rarely show tidy, well-separated particles: they dry into
chains and clumps where neighbours touch or overlap slightly. Those are the
particles a user still expects to be measured, so this fixture builds clusters
with a controlled amount of contact and reports the true geometry.
"""
import os

import cv2
import numpy as np


def generate_touching_image(output_path, width=1000, height=800,
                            num_particles=24, seed=11, max_overlap=0.15):
    """Draw hollow particles in touching clusters.

    ``max_overlap`` is the deepest allowed intrusion as a fraction of the
    smaller radius: 0 means merely tangent, 0.15 a slight overlap. Nothing is
    placed deeper than that, so every particle here is one the analyzer should
    find.
    """
    rng = np.random.RandomState(seed)

    bg = rng.randint(178, 196, (height, width), dtype=np.uint8)
    img = np.clip(bg.astype(np.int16) + rng.normal(0, 6, bg.shape), 0, 255).astype(np.uint8)

    placed = []
    margin = 30
    seeds_left = 4

    def fits(cx, cy, r):
        if not (margin + r <= cx <= width - margin - r):
            return False
        if not (margin + r <= cy <= height - margin - r):
            return False
        for px, py, pr in placed:
            d = np.hypot(cx - px, cy - py)
            if d < (r + pr) - max_overlap * min(r, pr):
                return False
        return True

    attempts = 0
    while len(placed) < num_particles and attempts < 40000:
        attempts += 1
        if not placed or (seeds_left > 0 and rng.random() < 0.04):
            r = int(rng.randint(45, 60))
            cx = int(rng.randint(margin + r, width - margin - r))
            cy = int(rng.randint(margin + r, height - margin - r))
            if fits(cx, cy, r):
                placed.append((cx, cy, r))
                seeds_left -= 1
            continue

        # Grow the cluster: attach a neighbour to an existing particle.
        px, py, pr = placed[rng.randint(len(placed))]
        r = int(rng.randint(45, 60))
        angle = rng.uniform(0, 2 * np.pi)
        gap = rng.uniform(-max_overlap * min(r, pr), 3)
        d = pr + r - gap
        cx = int(round(px + d * np.cos(angle)))
        cy = int(round(py + d * np.sin(angle)))
        if fits(cx, cy, r):
            placed.append((cx, cy, r))

    for cx, cy, r in placed:
        shell = max(6, int(r * rng.uniform(0.16, 0.24)))
        cv2.circle(img, (cx, cy), r, int(rng.randint(45, 70)), -1)
        cv2.circle(img, (cx, cy), r - shell, int(rng.randint(140, 170)), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)

    contacts = sum(
        1
        for i, (ax, ay, ar) in enumerate(placed)
        for bx, by, br in placed[i + 1:]
        if np.hypot(ax - bx, ay - by) < ar + br + 2
    )
    print(f"Generated touching-cluster image: {output_path}")
    print(f"  Particles: {len(placed)}   touching pairs: {contacts}")
    return placed


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    generate_touching_image(os.path.join(here, "test_tem_touching.png"))
