"""Hollow spheres on an open background, drawn from the projection physics.

Every other fixture here paints a ring and calls it a shell. This one computes
what a hollow sphere actually does to a beam, because the images it stands in
for are the ones where that matters: particles sitting apart on a support film,
each one darkening the beam by how much shell the ray passed through.

For a sphere of outer radius R and cavity radius r_in, a ray at distance b from
the centre crosses

    t(b) = 2 * ( sqrt(R^2 - b^2) - sqrt(max(0, r_in^2 - b^2)) )

of shell, and the image is I0 * exp(-mu * t). Three consequences decide what a
measuring program may and may not do with such an image, and all three are
properties of that formula rather than of any drawing choice:

  - t is largest at b = r_in, so the darkest ring is the *inner* radius. A
    boundary placed on the darkest line reports the cavity, not the particle.
  - t falls to zero only at b = R, so the outer radius is where the darkening
    ends - the outer flank of the dark band, which is what this fixture's
    ground truth records.
  - the interior is darker than the background, because a ray through the
    middle still crosses 2*(R - r_in) of shell. That is the opposite of a
    densely packed field, where the gaps between particles are darkened by the
    neighbours and the interiors come out brighter, and it is why both
    arrangements are kept: a boundary rule that only works on one of them is
    reading contrast rather than structure.

Returns (x, y, R, cored) per particle, R being the outer radius in pixels
and `cored` whether that particle carries a dense core.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_shelled_image(path, radius=60, shell_frac=0.18, count=40,
                           size=1024, seed=7, mu=0.020, gap=0.06, noise=6.0,
                           cored=0.0, core_frac=0.55, core_mu=2.5):
    """Sparse hollow spheres, `radius` px on average, `gap` apart at the least.

    ``cored`` is the fraction of particles carrying a dense core inside the
    cavity - yolk-shell, or simply ones the template was never removed from.
    Real preparations contain a minority of them and they are much darker than
    the rest, which is a problem for any step that decides what a particle
    should look like from the population: a minority that looks different is
    exactly what such a rule throws away. ``core_mu`` is a multiple of the
    shell's absorption, since the core is solid where the shell is porous.
    """
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    thickness = np.zeros((size, size), np.float32)

    placed = []
    for _ in range(20000):
        if len(placed) >= count:
            break
        r = radius * rng.uniform(0.75, 1.25)
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        if any(np.hypot(cx - px, cy - py) < (r + pr) * (1 + gap)
               for px, py, pr in placed):
            continue
        placed.append((cx, cy, r))

    has_core = rng.rand(len(placed)) < cored
    for (cx, cy, r), core in zip(placed, has_core):
        r_in = r * (1.0 - shell_frac)
        b2 = (xx - cx) ** 2 + (yy - cy) ** 2
        outer = np.sqrt(np.clip(r * r - b2, 0, None))
        inner = np.sqrt(np.clip(r_in * r_in - b2, 0, None))
        thickness += 2.0 * (outer - inner)
        if core:
            r_c = r_in * core_frac
            thickness += 2.0 * core_mu * np.sqrt(np.clip(r_c * r_c - b2, 0, None))

    img = 232.0 * np.exp(-mu * thickness)
    # The microscope's blur is what turns the outer edge from a step into a
    # flank, and it is the whole reason the boundary needs a sub-pixel rule.
    img = cv2.GaussianBlur(img, (0, 0), max(1.0, radius * 0.025))
    img = img + rng.normal(0, noise, (size, size))
    cv2.imwrite(path, cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                                   cv2.COLOR_GRAY2BGR))
    print(f"Generated shelled image: {path}")
    print(f"  Particles: {len(placed)}   outer radius ~{radius}px   "
          f"shell {shell_frac * 100:.0f}% of radius"
          + (f"   cored {int(has_core.sum())}" if cored else ""))
    return [(x, y, r, bool(c)) for (x, y, r), c in zip(placed, has_core)]


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench")
    os.makedirs(out, exist_ok=True)
    generate_shelled_image(os.path.join(out, "shelled.png"))
