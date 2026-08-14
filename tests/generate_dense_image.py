"""Generate a dense TEM-like image with ~80 tightly packed spherical particles."""
import cv2
import numpy as np
import os


def generate_dense_tem_image(output_path, width=2048, height=1800, target_count=85, seed=123):
    rng = np.random.RandomState(seed)

    bg = rng.randint(170, 195, (height + 120, width), dtype=np.uint8)
    grad_x = np.linspace(0.92, 1.08, width).reshape(1, -1)
    grad_y = np.linspace(0.95, 1.05, height + 120).reshape(-1, 1)
    bg = np.clip(bg * grad_x * grad_y, 0, 255).astype(np.uint8)
    noise = rng.normal(0, 10, bg.shape).astype(np.int16)
    img = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    particles = []
    margin = 40
    nm_per_px = 0.25

    size_groups = [
        (0.25, 12, 18),   # small
        (0.50, 18, 30),   # medium
        (0.20, 30, 45),   # large
        (0.05, 45, 55),   # very large
    ]

    attempts = 0
    max_attempts = target_count * 50

    while len(particles) < target_count and attempts < max_attempts:
        attempts += 1

        roll = rng.random()
        cumul = 0
        radius = 20
        for frac, rmin, rmax in size_groups:
            cumul += frac
            if roll < cumul:
                radius = rng.randint(rmin, rmax + 1)
                break

        cx = rng.randint(margin + radius, width - margin - radius)
        cy = rng.randint(margin + radius, height - margin - radius)

        min_gap = 3
        too_close = False
        for px, py, pr, _ in particles:
            dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if dist < (radius + pr + min_gap):
                too_close = True
                break

        if too_close:
            # Allow touching/slight overlap for ~20% of particles
            if rng.random() < 0.2:
                too_close = False
                for px, py, pr, _ in particles:
                    dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    if dist < (radius + pr) * 0.5:
                        too_close = True
                        break

        if too_close:
            continue

        brightness = rng.randint(30, 85)
        edge_dark = max(10, brightness - rng.randint(10, 25))

        cv2.circle(img, (cx, cy), radius, int(brightness), -1)
        cv2.circle(img, (cx, cy), radius, int(edge_dark), 2)

        y1 = max(0, cy - radius - 2)
        y2 = min(height, cy + radius + 2)
        x1 = max(0, cx - radius - 2)
        x2 = min(width, cx + radius + 2)
        if y2 > y1 and x2 > x1:
            patch = img[y1:y2, x1:x2].astype(np.int16)
            pnoise = rng.normal(0, 4, patch.shape).astype(np.int16)
            img[y1:y2, x1:x2] = np.clip(patch + pnoise, 0, 255).astype(np.uint8)

        # The rim is two pixels wide and straddles `radius`, so the particle
        # actually ends about a pixel further out than the number handed to
        # cv2.circle. The ground truth records where the particle ends, since
        # that is what the analyzer is asked for; recording the nominal radius
        # instead scored a correct measurement as 6.8% large.
        particles.append((cx, cy, radius + 1, brightness))

    # Scale bar
    bar_y = height + 30
    img[height:, :] = 0
    scale_bar_px = 800
    bar_x_start = width - scale_bar_px - 120
    bar_x_end = width - 120
    cv2.rectangle(img, (bar_x_start, bar_y), (bar_x_end, bar_y + 10), 255, -1)
    cv2.putText(img, "200 nm", (bar_x_start + scale_bar_px // 2 - 80, bar_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    cv2.putText(img, "TEM  200kV  x150k", (30, bar_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 200, 2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)

    diameters = [r * 2 * nm_per_px for _, _, r, _ in particles]
    print(f"Generated dense test image: {output_path}")
    print(f"  Size: {width}x{height + 120}")
    print(f"  Particles placed: {len(particles)}")
    print(f"  Scale: {nm_per_px} nm/px (200nm bar = {scale_bar_px}px)")
    print(f"  Diameter range: {min(diameters):.1f} - {max(diameters):.1f} nm")
    print(f"  Mean diameter: {np.mean(diameters):.1f} nm")

    overlaps = 0
    for i, (x1, y1, r1, _) in enumerate(particles):
        for j, (x2, y2, r2, _) in enumerate(particles):
            if i >= j:
                continue
            d = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if d < r1 + r2:
                overlaps += 1
    print(f"  Overlapping pairs: {overlaps}")

    return particles, nm_per_px


if __name__ == "__main__":
    output = os.path.join(os.path.dirname(__file__), "test_tem_dense.png")
    generate_dense_tem_image(output)
