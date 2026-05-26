"""Tests for region extraction from synthetic masks."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from segmentation.regions import extract_regions
import config


def make_mask(shape, blobs):
    """Create binary mask with rectangular blobs at given (x,y,w,h) locations."""
    mask = np.zeros(shape, dtype=np.uint8)
    for x, y, w, h in blobs:
        mask[y:y+h, x:x+w] = 255
    return mask


def test_single_blob_detected():
    mask = make_mask((200, 200), [(10, 10, 50, 50)])
    regions = extract_regions(mask, "red", 200*200)
    assert len(regions) == 1
    assert regions[0].color_name == "red"
    assert regions[0].area_px > 0


def test_multiple_blobs():
    mask = make_mask((400, 400), [(0, 0, 80, 80), (200, 200, 80, 80), (300, 300, 80, 80)])
    regions = extract_regions(mask, "blue", 400*400)
    assert len(regions) == 3


def test_subthreshold_blob_discarded():
    big   = (10, 10, 100, 100)
    tiny  = (250, 250, 3, 3)     # well below default threshold
    mask  = make_mask((400, 400), [big, tiny])
    orig_frac = config.MIN_REGION_AREA_FRACTION
    orig_px   = config.MIN_REGION_AREA_PX
    config.MIN_REGION_AREA_FRACTION = 0.001
    config.MIN_REGION_AREA_PX = 100
    try:
        regions = extract_regions(mask, "green", 400*400)
        assert len(regions) == 1
    finally:
        config.MIN_REGION_AREA_FRACTION = orig_frac
        config.MIN_REGION_AREA_PX = orig_px


def test_area_fraction():
    img_area = 1000 * 1000
    mask = make_mask((1000, 1000), [(0, 0, 100, 100)])
    regions = extract_regions(mask, "yellow", img_area)
    assert len(regions) == 1
    assert abs(regions[0].area_fraction - 0.01) < 0.001


def test_bbox_accuracy():
    mask = make_mask((300, 300), [(50, 60, 80, 90)])
    regions = extract_regions(mask, "orange", 300*300)
    assert len(regions) == 1
    x, y, w, h = regions[0].bbox
    assert x == 50
    assert y == 60
    assert abs(w - 80) <= 2
    assert abs(h - 90) <= 2
