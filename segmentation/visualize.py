from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from segmentation.regions import Region


def draw_regions(bgr: np.ndarray, regions: List[Region],
                 alpha: float = 0.35) -> np.ndarray:
    """
    Overlay each region on the image:
      - semi-transparent color fill
      - contour outline
      - text label (color, texture if available, area%)
    Returns annotated BGR image.
    """
    overlay = bgr.copy()

    # Draw solid fills on overlay layer
    for region in regions:
        color_bgr = config.PALETTE_BGR.get(region.color_name, (128, 128, 128))
        overlay[region.mask.astype(bool)] = color_bgr

    # Blend fill with original
    output = cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0)

    # Draw contours and labels on top of blend
    for region in regions:
        color_bgr = config.PALETTE_BGR.get(region.color_name, (128, 128, 128))
        cv2.drawContours(output, [region.contour], -1, color_bgr, 2)

        cx = int(region.centroid[0])
        cy = int(region.centroid[1])

        lines = [region.color_name]
        if region.texture_class:
            lines.append(region.texture_class)
        lines.append(f"{region.area_fraction * 100:.1f}%")

        for i, line in enumerate(lines):
            y_pos = cy + i * 15
            # Black outline then white text for readability
            cv2.putText(output, line, (cx, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output, line, (cx, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    return output


def draw_roi_box(bgr: np.ndarray,
                 roi: Optional[Tuple[float, float, float, float]]) -> np.ndarray:
    """Draw the active ROI bounding box on the image (dashed white rectangle)."""
    if roi is None:
        return bgr
    H, W = bgr.shape[:2]
    xf, yf, wf, hf = roi
    x0, y0 = int(xf * W), int(yf * H)
    x1, y1 = int((xf + wf) * W), int((yf + hf) * H)
    out = bgr.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 3)
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), 1)
    return out


def save_and_show(annotated: np.ndarray, out_path: str,
                  show: bool = False) -> None:
    cv2.imwrite(out_path, annotated)
    if show:
        import matplotlib.pyplot as plt
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(14, 9))
        plt.imshow(rgb)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
