"""Generate a TEM-like image matching the user's real image characteristics:
- ~15-20 large spherical particles
- Some overlapping
- 500 nm scale bar
- Particles are relatively large (200-500nm diameter range)
- Gray background with typical TEM contrast
"""
import cv2
import numpy as np
import os


def generate_real_like_tem(output_path, width=800, height=720, seed=99):
    rng = np.random.RandomState(seed)

    # Background similar to real TEM - medium gray with slight texture
    bg = rng.randint(140, 160, (height + 80, width), dtype=np.uint8)
    grad_x = np.linspace(0.95, 1.05, width).reshape(1, -1)
    grad_y = np.linspace(0.97, 1.03, height + 80).reshape(-1, 1)
    bg = np.clip(bg * grad_x * grad_y, 0, 255).astype(np.uint8)
    noise = rng.normal(0, 6, bg.shape).astype(np.int16)
    img = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    particles = []
    margin = 30
    nm_per_px = 500.0 / 200.0  # 500nm scale bar = 200px

    # Large particles similar to the real image
    target_particles = [
        # (cx, cy, radius) - positioned to mimic the real image layout
        (120, 120, 70), (280, 100, 75), (450, 90, 68),
        (600, 130, 72), (700, 250, 65),
        (80, 300, 73), (230, 270, 70), (400, 250, 76),
        (560, 300, 69), (150, 480, 74),
        (330, 440, 71), (500, 430, 67), (650, 420, 73),
        (200, 620, 70), (400, 600, 72), (580, 580, 68),
        (80, 560, 65),
        # Some extra overlapping ones
        (340, 310, 55), (490, 160, 60),
    ]

    for cx, cy, radius in target_particles:
        if cx - radius < 0 or cx + radius >= width or cy - radius < 0 or cy + radius >= height:
            continue

        # Darker than background, with slight internal texture
        brightness = rng.randint(85, 115)
        edge_dark = max(60, brightness - rng.randint(15, 30))

        # Draw filled circle with edge
        cv2.circle(img, (cx, cy), radius, int(brightness), -1)
        cv2.circle(img, (cx, cy), radius, int(edge_dark), 3)

        # Internal texture
        y1 = max(0, cy - radius - 2)
        y2 = min(height, cy + radius + 2)
        x1 = max(0, cx - radius - 2)
        x2 = min(width, cx + radius + 2)
        if y2 > y1 and x2 > x1:
            patch = img[y1:y2, x1:x2].astype(np.int16)
            pnoise = rng.normal(0, 5, patch.shape).astype(np.int16)
            img[y1:y2, x1:x2] = np.clip(patch + pnoise, 0, 255).astype(np.uint8)

        particles.append((cx, cy, radius))

    # Scale bar - 500nm
    bar_y_start = height + 10
    img[height:, :] = 0
    scale_bar_px = 200
    bar_x_start = width - scale_bar_px - 60
    bar_x_end = width - 60
    cv2.rectangle(img, (bar_x_start, bar_y_start + 20), (bar_x_end, bar_y_start + 28), 255, -1)
    cv2.putText(img, "500 nm", (bar_x_start + scale_bar_px // 2 - 50, bar_y_start + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)

    diameters = [r * 2 * nm_per_px for _, _, r in particles]
    print(f"Generated real-like TEM image: {output_path}")
    print(f"  Size: {width}x{height + 80}")
    print(f"  Particles: {len(particles)}")
    print(f"  Scale: {nm_per_px:.3f} nm/px")
    print(f"  Diameter range: {min(diameters):.1f} - {max(diameters):.1f} nm")
    print(f"  Mean diameter: {np.mean(diameters):.1f} nm")

    # Count overlaps
    overlaps = 0
    for i in range(len(particles)):
        for j in range(i+1, len(particles)):
            x1, y1, r1 = particles[i]
            x2, y2, r2 = particles[j]
            d = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if d < r1 + r2:
                overlaps += 1
    print(f"  Overlapping pairs: {overlaps}")

    return particles, nm_per_px


if __name__ == "__main__":
    output = os.path.join(os.path.dirname(__file__), "test_tem_real_like.png")
    generate_real_like_tem(output)
