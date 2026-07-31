"""Run particle analysis on the synthetic TEM image and print results."""
import cv2
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tem_analyzer.analyzer import ScaleBarDetector, ParticleAnalyzer

image_path = os.path.join(os.path.dirname(__file__), "test_tem_spherical.png")
image = cv2.imread(image_path)
print(f"Image loaded: {image.shape}")

detector = ScaleBarDetector()
nm_per_px, scale_text = detector.detect(image)
print(f"\nScale bar detection:")
print(f"  nm/px: {nm_per_px}")
print(f"  Scale text: {scale_text}")

if nm_per_px is None:
    print("  (OCR not available, using known scale: 0.5 nm/px)")
    nm_per_px = 0.5

analyzer = ParticleAnalyzer(nm_per_px=nm_per_px)
particles = analyzer.analyze(image, min_area_px=50)

print(f"\nDetected particles: {len(particles)}")
print(f"\n{'#':>3}  {'Diameter (nm)':>13}  {'Area (nm²)':>12}  {'Center':>12}")
print("-" * 50)
for i, p in enumerate(particles):
    print(f"{i+1:>3}  {p['diameter']:>13.2f}  {p['area']:>12.2f}  ({p['center_x']}, {p['center_y']})")

stats = ParticleAnalyzer.compute_statistics(particles)
if stats:
    print(f"\n--- Statistics ---")
    print(f"  Count: {stats['count']}")
    print(f"  Mean:  {stats['mean']:.2f} nm")
    print(f"  Std:   {stats['std']:.2f} nm")
    print(f"  Min:   {stats['min']:.2f} nm")
    print(f"  Max:   {stats['max']:.2f} nm")
    print(f"  D10:   {stats['d10']:.2f} nm")
    print(f"  D50:   {stats['d50']:.2f} nm")
    print(f"  D90:   {stats['d90']:.2f} nm")

result_img = image.copy()
for i, p in enumerate(particles):
    cx, cy = p["center_x"], p["center_y"]
    r = int(p["radius_px"])
    cv2.circle(result_img, (cx, cy), r, (0, 255, 0), 2)
    cv2.putText(result_img, str(i+1), (cx-10, cy-r-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

result_path = os.path.join(os.path.dirname(__file__), "test_result.png")
cv2.imwrite(result_path, result_img)
print(f"\nResult image saved: {result_path}")
