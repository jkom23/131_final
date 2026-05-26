from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config


@dataclass
class Region:
    color_name: str
    mask: np.ndarray                     # H×W binary uint8 (0 or 255)
    contour: np.ndarray                  # largest external cv2 contour
    bbox: Tuple[int, int, int, int]      # x, y, w, h
    area_px: int
    area_fraction: float
    centroid: Tuple[float, float]        # cx, cy
    texture_class: Optional[str] = None
    texture_confidence: Optional[float] = None


def extract_regions(color_mask: np.ndarray, color_name: str,
                    image_area: int) -> List[Region]:
    """
    Given a binary color mask, return one Region per connected component
    that exceeds the configured minimum size thresholds.
    """
    min_px = max(
        config.MIN_REGION_AREA_PX,
        int(config.MIN_REGION_AREA_FRACTION * image_area),
    )

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        color_mask, connectivity=8
    )

    regions: List[Region] = []
    for label_id in range(1, n_labels):   # 0 is the background label
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_px:
            continue

        component_mask = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_id]

        # FUTURE: split touching same-color bricks into per-brick instances

        regions.append(Region(
            color_name=color_name,
            mask=component_mask,
            contour=contour,
            bbox=(x, y, w, h),
            area_px=area,
            area_fraction=area / image_area,
            centroid=(float(cx), float(cy)),
        ))

    return regions
