from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from segmentation.regions import Region, extract_regions


def gray_world_balance(bgr: np.ndarray) -> np.ndarray:
    """Simple gray-world white balance correction."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    mean = (b.mean() + g.mean() + r.mean()) / 3.0
    b = np.clip(b * (mean / (b.mean() + 1e-6)), 0, 255)
    g = np.clip(g * (mean / (g.mean() + 1e-6)), 0, 255)
    r = np.clip(r * (mean / (r.mean() + 1e-6)), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def fill_interior_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill any interior voids in a binary mask that are enclosed and not reachable
    from the image boundary.  This fixes "inner parts" that HSV didn't classify
    but are clearly inside a color region (stud shadows, connector gaps, etc.).

    Algorithm:
      1. Pad mask with a guaranteed-background border (0).
      2. Invert: background→255, foreground→0.
      3. Flood-fill from the top-left corner of the padded image → all true
         background pixels become 0; isolated interior voids remain 255.
      4. Strip padding; OR result into original mask.
    """
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    inv = cv2.bitwise_not(padded)
    ff_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(inv, ff_mask, (0, 0), 0)
    interior = inv[1:-1, 1:-1]     # strip padding; interior holes are still 255
    return cv2.bitwise_or(mask, interior)


def assign_enclosed_unclassified(color_masks: Dict[str, np.ndarray],
                                 roi_shape: tuple) -> Dict[str, np.ndarray]:
    """
    Find pixels that no color claimed AND are enclosed within a single color's
    region, then assign them to that color.

    Logic:
      - Build a union of all classified pixels.
      - Find connected components of unclassified pixels.
      - For each unclassified component, dilate it slightly and measure overlap
        with each color mask.  If only ONE color overlaps → assign to it.
    """
    h, w = roi_shape
    classified = np.zeros((h, w), dtype=np.uint8)
    for m in color_masks.values():
        classified |= (m > 0).astype(np.uint8)

    unclassified = (classified == 0).astype(np.uint8) * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(unclassified, connectivity=8)
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for label_id in range(1, n):
        comp = (labels == label_id).astype(np.uint8) * 255
        # Dilate to probe immediate neighbours
        probed = cv2.dilate(comp, small_kernel, iterations=2)

        touching = {name for name, m in color_masks.items()
                    if np.any((probed > 0) & (m > 0))}

        if len(touching) == 1:
            owner = next(iter(touching))
            color_masks[owner] = cv2.bitwise_or(color_masks[owner], comp)

    return color_masks


def auto_detect_roi(bgr: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Find the tight bounding box of all LEGO-colored pixels by thresholding on
    saturation and value.  Background (white wall, gray carpet) has very low
    saturation and is excluded.  Trims ROI_BASEPLATE_TRIM_PX from the bottom
    to cut the flat baseplate rows.
    Returns (x_frac, y_frac, w_frac, h_frac).
    """
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    colored = cv2.inRange(
        hsv,
        np.array([0, config.ROI_SAT_MIN, config.ROI_VAL_MIN], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )

    k = max(3, config.MORPH_OPEN_SIZE)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    colored = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, kernel)

    pts = cv2.findNonZero(colored)
    if pts is None:
        return (0.0, 0.0, 1.0, 1.0)

    x, y, w, h = cv2.boundingRect(pts)

    # Trim flat baseplate rows from the bottom
    h = max(1, h - config.ROI_BASEPLATE_TRIM_PX)

    pad_x = max(1, int(0.01 * W))
    pad_y = max(1, int(0.01 * H))
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    x2 = min(W, x + w + 2 * pad_x)
    y2 = min(H, y + h + 2 * pad_y)

    return (x / W, y / H, (x2 - x) / W, (y2 - y) / H)


def resolve_roi(bgr: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    if config.ROI is None:
        return None
    if config.ROI == "auto":
        return auto_detect_roi(bgr)
    return config.ROI


def apply_roi(bgr: np.ndarray,
              roi: Optional[Tuple[float, float, float, float]]) -> tuple:
    if roi is None:
        return bgr, 0, 0
    h, w = bgr.shape[:2]
    xf, yf, wf, hf = roi
    x0 = int(xf * w)
    y0 = int(yf * h)
    x1 = min(w, x0 + int(wf * w))
    y1 = min(h, y0 + int(hf * h))
    return bgr[y0:y1, x0:x1], x0, y0


def segment_image(bgr: np.ndarray) -> Tuple[List[Region], Optional[tuple]]:
    """
    Full pipeline: BGR image → (list of color Regions, roi tuple).

    Steps:
      1. Optional Gaussian blur + gray-world WB.
      2. Resolve ROI (auto or manual); crop.
      3. Per-color HSV thresholding → morphological cleanup →
         fill interior holes → connected components.
      4. Post-process: assign enclosed unclassified pixels to surrounding color.
      5. Translate ROI-relative coords back to full-image coords.
    """
    full_h, full_w = bgr.shape[:2]
    full_area = full_h * full_w

    if config.GAUSSIAN_BLUR_KERNEL > 0:
        k = config.GAUSSIAN_BLUR_KERNEL
        bgr = cv2.GaussianBlur(bgr, (k, k), 0)

    if config.USE_GRAY_WORLD:
        bgr = gray_world_balance(bgr)

    roi = resolve_roi(bgr)
    roi_bgr, roi_ox, roi_oy = apply_roi(bgr, roi)
    roi_h, roi_w = roi_bgr.shape[:2]
    roi_area = roi_h * roi_w if roi_h * roi_w > 0 else 1

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    ko = config.MORPH_OPEN_SIZE
    kc = config.MORPH_CLOSE_SIZE
    open_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))

    # Build all color masks first (needed for enclosed-region assignment)
    color_masks: Dict[str, np.ndarray] = {}
    for color_name, ranges in config.HSV_PALETTE.items():
        combined = np.zeros((roi_h, roi_w), dtype=np.uint8)
        for rng in ranges:
            lo = np.array(rng["lower"], dtype=np.uint8)
            hi = np.array(rng["upper"], dtype=np.uint8)
            combined |= cv2.inRange(hsv, lo, hi)

        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  open_kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)
        combined = fill_interior_holes(combined)   # fill stud shadows / connector gaps
        color_masks[color_name] = combined

    # Assign enclosed unclassified pixels to their surrounding color
    color_masks = assign_enclosed_unclassified(color_masks, (roi_h, roi_w))

    all_regions: List[Region] = []
    for color_name, mask in color_masks.items():
        regions = extract_regions(mask, color_name, roi_area)

        if roi_ox > 0 or roi_oy > 0:
            for reg in regions:
                full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
                full_mask[roi_oy:roi_oy + roi_h, roi_ox:roi_ox + roi_w] = reg.mask
                reg.mask = full_mask

                bx, by, bw, bh = reg.bbox
                reg.bbox = (bx + roi_ox, by + roi_oy, bw, bh)
                cx, cy = reg.centroid
                reg.centroid = (cx + roi_ox, cy + roi_oy)
                reg.contour = reg.contour + np.array([roi_ox, roi_oy])
                reg.area_fraction = reg.area_px / full_area

        all_regions.extend(regions)

    return all_regions, roi


def print_area_fraction(bgr: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """Print area_fraction for a bbox to help calibrate MIN_REGION_AREA_FRACTION."""
    frac = (w * h) / (bgr.shape[0] * bgr.shape[1])
    print(f"bbox ({x},{y},{w},{h}) area={w*h}px  area_fraction={frac:.6f}")
