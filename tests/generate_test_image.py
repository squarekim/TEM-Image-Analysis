"""Generate a synthetic TEM-like image with spherical particles and scale bar for testing."""
import cv2
import numpy as np
import os


def generate_tem_image(output_path, width=1024, height=900, num_particles=25, seed=42):
    rng = np.random.RandomState(seed)

    img = np.ones((height + 100, width), dtype=np.uint8) * 180
    noise = rng.normal(0, 8, img.shape[:2]).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    particle_info = []
    margin = 60
    nm_per_px = 0.5  # 200nm scale bar = 400px

    for _ in range(num_particles * 3):
        if len(particle_info) >= num_particles:
            break

        radius = rng.randint(15, 50)
        cx = rng.randint(margin + radius, width - margin - radius)
        cy = rng.randint(margin + radius, height - margin - radius)

        overlap = False
        for px, py, pr in particle_info:
            dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if dist < (radius + pr + 8):
                overlap = True
                break
        if overlap:
            continue

        brightness = rng.randint(40, 90)
        cv2.circle(img, (cx, cy), radius, int(brightness), -1)

        edge_noise = rng.normal(0, 2, (radius * 4, radius * 4)).astype(np.int16)
        y1 = max(0, cy - radius * 2)
        y2 = min(height, cy + radius * 2)
        x1 = max(0, cx - radius * 2)
        x2 = min(width, cx + radius * 2)
        eh, ew = y2 - y1, x2 - x1
        if eh > 0 and ew > 0:
            patch = img[y1:y2, x1:x2].astype(np.int16)
            noise_patch = edge_noise[:eh, :ew]
            img[y1:y2, x1:x2] = np.clip(patch + noise_patch, 0, 255).astype(np.uint8)

        particle_info.append((cx, cy, radius))

    bar_y = height + 20
    bar_region = img[height:, :]
    bar_region[:] = 0

    scale_bar_px = 400
    bar_x_start = width - scale_bar_px - 80
    bar_x_end = width - 80
    cv2.rectangle(img, (bar_x_start, bar_y), (bar_x_end, bar_y + 8), 255, -1)

    cv2.putText(img, "200 nm", (bar_x_start + scale_bar_px // 2 - 50, bar_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)

    cv2.putText(img, "TEM  200kV", (20, bar_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 200, 1)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, img)

    print(f"Generated test image: {output_path}")
    print(f"  Size: {width}x{height + 100}")
    print(f"  Particles: {len(particle_info)}")
    print(f"  Scale: {nm_per_px} nm/px (200nm bar = {scale_bar_px}px)")
    print(f"  Expected particle diameters: {[r*2*nm_per_px for _, _, r in particle_info]} nm")

    return particle_info, nm_per_px


if __name__ == "__main__":
    output = os.path.join(os.path.dirname(__file__), "test_tem_spherical.png")
    generate_tem_image(output)
