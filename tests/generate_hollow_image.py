"""Generate a synthetic TEM image of hollow silica nanoparticles."""
import cv2
import numpy as np
import os


def generate_hollow_tem_image(output_path, width=1600, height=1400, num_particles=50, seed=77):
    rng = np.random.RandomState(seed)

    bg = rng.randint(175, 200, (height + 100, width), dtype=np.uint8)
    grad = np.linspace(0.93, 1.07, width).reshape(1, -1)
    bg = np.clip(bg * grad, 0, 255).astype(np.uint8)
    noise = rng.normal(0, 8, bg.shape).astype(np.int16)
    img = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    particles = []
    margin = 50

    for _ in range(num_particles * 5):
        if len(particles) >= num_particles:
            break

        radius = rng.randint(20, 55)
        shell_thickness = max(3, int(radius * rng.uniform(0.15, 0.30)))
        cx = rng.randint(margin + radius, width - margin - radius)
        cy = rng.randint(margin + radius, height - margin - radius)

        overlap = False
        for px, py, pr, _, _ in particles:
            dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if dist < (radius + pr + 5):
                overlap = True
                break
            # Allow some touching pairs
            if dist < (radius + pr) and rng.random() < 0.15:
                overlap = False

        if overlap:
            continue

        shell_dark = rng.randint(50, 90)
        center_bright = rng.randint(150, 185)

        # Draw outer dark circle (shell)
        cv2.circle(img, (cx, cy), radius, int(shell_dark), -1)
        # Draw inner bright circle (hollow center)
        inner_r = radius - shell_thickness
        if inner_r > 3:
            cv2.circle(img, (cx, cy), inner_r, int(center_bright), -1)

        # Add slight noise texture
        y1, y2 = max(0, cy - radius - 2), min(height, cy + radius + 2)
        x1, x2 = max(0, cx - radius - 2), min(width, cx + radius + 2)
        if y2 > y1 and x2 > x1:
            patch = img[y1:y2, x1:x2].astype(np.int16)
            pnoise = rng.normal(0, 4, patch.shape).astype(np.int16)
            img[y1:y2, x1:x2] = np.clip(patch + pnoise, 0, 255).astype(np.uint8)

        particles.append((cx, cy, radius, shell_thickness, inner_r))

    # Scale bar
    bar_y = height + 20
    img[height:, :] = 0
    scale_bar_px = 500
    bar_x_start = width - scale_bar_px - 80
    bar_x_end = width - 80
    cv2.rectangle(img, (bar_x_start, bar_y), (bar_x_end, bar_y + 8), 255, -1)
    cv2.putText(img, "200 nm", (bar_x_start + scale_bar_px // 2 - 60, bar_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
    cv2.putText(img, "TEM  200kV", (20, bar_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 200, 1)

    nm_per_px = 200 / scale_bar_px  # 0.4 nm/px

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)

    diameters = [r * 2 * nm_per_px for _, _, r, _, _ in particles]
    print(f"Generated hollow silica TEM image: {output_path}")
    print(f"  Size: {width}x{height + 100}")
    print(f"  Particles: {len(particles)}")
    print(f"  Scale: {nm_per_px:.3f} nm/px")
    print(f"  Diameter range: {min(diameters):.1f} - {max(diameters):.1f} nm")
    print(f"  Mean diameter: {np.mean(diameters):.1f} nm")
    print(f"  Shell thickness: {min(t for _,_,_,t,_ in particles)}-{max(t for _,_,_,t,_ in particles)} px")

    return particles, nm_per_px


if __name__ == "__main__":
    output = os.path.join(os.path.dirname(__file__), "test_tem_hollow.png")
    generate_hollow_tem_image(output)
