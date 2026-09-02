"""Regression coverage for separating overlapping solid particle clumps."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_analyzer.analyzer import ParticleAnalyzer, load_image  # noqa: E402
from tests.generate_hard_image import generate_hard_tem_image  # noqa: E402


def test_hard_fixture_overlapping_cluster_is_split(tmp_path):
    path = tmp_path / "hard_overlap.png"
    truth, _ = generate_hard_tem_image(str(path))
    expected = [(x, y, r) for x, y, r, kind in truth if kind == "overlap"]

    particles = ParticleAnalyzer(edge_level=0.95).analyze(
        load_image(str(path)), min_area_px=100, circularity_thresh=0.5, hollow=True
    )
    valid = [p for p in particles if not p.get("excluded")]

    matched = 0
    for tx, ty, tr in expected:
        if any(np.hypot(p["center_x"] - tx, p["center_y"] - ty) < tr * 0.6 for p in valid):
            matched += 1

    assert matched == len(expected)
