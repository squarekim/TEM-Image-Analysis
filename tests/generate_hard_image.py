"""Generate a realistic TEM-like image with overlapping particles, edge cases, and noise."""
import cv2
import numpy as np
import os


def generate_hard_tem_image(output_path, width=1024, height=900, seed=42):
    rng = np.random.RandomState(seed)

    bg_base = rng.randint(160, 200, (height + 100, width), dtype=np.uint8)
    grad = np.linspace(0.9, 1.1, width).reshape(1, -1)
    bg = np.clip(bg_base * grad, 0, 255).astype(np.uint8)
    noise = rng.normal(0, 12, bg.shape).astype(np.int16)
    img = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    particles = []

    # 1) Overlapping cluster - 3 particles touching/overlapping
    cluster_cx, cluster_cy = 300, 300
    for dx, dy, r in [(-30, -25, 35), (25, -10, 30), (-5, 30, 28)]:
        cx, cy = cluster_cx + dx, cluster_cy + dy
        brightness = rng.randint(35, 75)
        cv2.circle(img, (cx, cy), r, int(brightness), -1)
        particles.append((cx, cy, r, "overlap"))

    # 2) Tightly packed pair
    cv2.circle(img, (600, 200), 32, 55, -1)
    cv2.circle(img, (658, 200), 30, 65, -1)
    particles.append((600, 200, 32, "pair"))
    particles.append((658, 200, 30, "pair"))

    # 3) Particles at edges (partially cut off)
    cv2.circle(img, (10, 500), 40, 50, -1)
    particles.append((10, 500, 40, "edge"))
    cv2.circle(img, (width - 5, 350), 35, 60, -1)
    particles.append((width - 5, 350, 35, "edge"))
    cv2.circle(img, (500, 5), 30, 45, -1)
    particles.append((500, 5, 30, "edge"))

    # 4) Very small particles
    for cx, cy, r in [(800, 500, 8), (830, 520, 7), (770, 480, 9), (850, 550, 6)]:
        cv2.circle(img, (cx, cy), r, rng.randint(40, 70), -1)
        particles.append((cx, cy, r, "small"))

    # 5) Normal isolated particles with varying contrast
    isolated = [
        (150, 600, 38, 40), (400, 500, 42, 80), (650, 600, 25, 50),
        (500, 400, 33, 60), (850, 300, 28, 45), (200, 150, 20, 70),
        (750, 750, 36, 55), (450, 700, 30, 65), (900, 700, 22, 50),
    ]
    for cx, cy, r, b in isolated:
        cv2.circle(img, (cx, cy), r, b, -1)
        particles.append((cx, cy, r, "normal"))

    # 6) Non-spherical debris (should be rejected)
    pts = np.array([[100, 750], [160, 730], [170, 780], [120, 800]], np.int32)
    cv2.fillPoly(img, [pts], 60)
    pts2 = np.array([[650, 100], [700, 90], [710, 130], [660, 120]], np.int32)
    cv2.fillPoly(img, [pts2], 55)

    # Add texture noise to particles
    for cx, cy, r, _ in particles:
        y1 = max(0, cy - r - 2)
        y2 = min(height, cy + r + 2)
        x1 = max(0, cx - r - 2)
        x2 = min(width, cx + r + 2)
        if y2 > y1 and x2 > x1:
            patch = img[y1:y2, x1:x2].astype(np.int16)
            pnoise = rng.normal(0, 5, patch.shape).astype(np.int16)
            img[y1:y2, x1:x2] = np.clip(patch + pnoise, 0, 255).astype(np.uint8)

    # Scale bar area
    bar_y = height + 20
    img[height:, :] = 0
    scale_bar_px = 400
    bar_x_start = width - scale_bar_px - 80
    bar_x_end = width - 80
    cv2.rectangle(img, (bar_x_start, bar_y), (bar_x_end, bar_y + 8), 255, -1)
    cv2.putText(img, "200 nm", (bar_x_start + scale_bar_px // 2 - 50, bar_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    cv2.putText(img, "TEM  200kV", (20, bar_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 200, 1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)

    nm_per_px = 0.5
    print(f"Generated hard test image: {output_path}")
    print(f"  Total particles placed: {len(particles)}")
    by_type = {}
    for *_, t in particles:
        by_type[t] = by_type.get(t, 0) + 1
    for t, n in by_type.items():
        print(f"    {t}: {n}")
    print(f"  Non-spherical debris: 2 (should be rejected)")
    print(f"  Scale: {nm_per_px} nm/px")
    return particles, nm_per_px


if __name__ == "__main__":
    output = os.path.join(os.path.dirname(__file__), "test_tem_hard.png")
    generate_hard_tem_image(output)
