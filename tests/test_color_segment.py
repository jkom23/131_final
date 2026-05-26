"""Tests for HSV color segmentation on synthetic images."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import pytest

from segmentation.color_segment import segment_image
import config


def solid_bgr(bgr_color, shape=(300, 300)):
    img = np.zeros((*shape, 3), dtype=np.uint8)
    img[:] = bgr_color
    return img


def rect_image(rects, shape=(400, 400)):
    """Create image with colored rectangles: list of (bgr, x, y, w, h)."""
    img = np.ones((*shape, 3), dtype=np.uint8) * 200   # neutral gray background
    for bgr, x, y, w, h in rects:
        img[y:y+h, x:x+w] = bgr
    return img


def test_red_detected():
    # Pure saturated red in BGR
    img = rect_image([((0, 0, 200), 50, 50, 120, 120)])
    regions = segment_image(img)
    colors = [r.color_name for r in regions]
    assert "red" in colors, f"Expected red, got {colors}"


def test_blue_detected():
    img = rect_image([((200, 50, 0), 50, 50, 120, 120)])
    regions = segment_image(img)
    colors = [r.color_name for r in regions]
    assert "blue" in colors, f"Expected blue, got {colors}"


def test_yellow_detected():
    img = rect_image([((0, 220, 220), 50, 50, 120, 120)])
    regions = segment_image(img)
    colors = [r.color_name for r in regions]
    assert "yellow" in colors, f"Expected yellow, got {colors}"


def test_multi_color_image():
    rects = [
        ((0, 0, 200),   10,  10, 100, 100),   # red
        ((200, 50, 0), 200,  10, 100, 100),   # blue
        ((0, 220, 220), 10, 200, 100, 100),   # yellow
    ]
    img = rect_image(rects, shape=(400, 400))
    regions = segment_image(img)
    found = {r.color_name for r in regions}
    assert "red"    in found, f"Missing red, found {found}"
    assert "blue"   in found, f"Missing blue, found {found}"
    assert "yellow" in found, f"Missing yellow, found {found}"


def test_subthreshold_noise_ignored():
    # Large red rect + 2×2 noise pixel
    img = rect_image([
        ((0, 0, 200), 50, 50, 150, 150),
        ((0, 0, 200), 350, 350, 2, 2),
    ])
    regions = [r for r in segment_image(img) if r.color_name == "red"]
    # Only the large blob should survive
    assert len(regions) == 1
