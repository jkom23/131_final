#!/usr/bin/env python3
"""
Run HSV color segmentation on a LEGO wall image and write a debug overlay.

Usage:
    python scripts/run_segmentation.py --image IMG_1361.jpeg --out debug.png
    python scripts/run_segmentation.py --image IMG_1361.jpeg --out debug.png --show
    python scripts/run_segmentation.py --image IMG_1361.jpeg --out debug.png --no-grid
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from segmentation.color_segment import apply_roi, segment_image
from segmentation.stud_grid import build_stud_grid, draw_stud_grid, estimate_stud_pitch_px
from segmentation.visualize import draw_regions, draw_roi_box, save_and_show
import config


def draw_stud_test_grid(bgr: np.ndarray, roi, full_w: int, full_h: int) -> np.ndarray:
    """
    Draw a raw stud grid over the ROI so the user can verify cell size.
    Each cell is labeled with its (col,row) index.
    Alternating light/dark shading makes individual cells easy to count.
    """
    if roi is None:
        return bgr

    from segmentation.color_segment import apply_roi as _apply
    roi_bgr, roi_ox, roi_oy = _apply(bgr, roi)
    pitch = estimate_stud_pitch_px(roi_bgr, full_image_width=full_w)
    cw = max(1, int(round(pitch)))
    ch = max(1, int(round(pitch * config.BRICK_TO_STUD_RATIO)))

    xf, yf, wf, hf = roi
    x0 = int(xf * full_w)
    y0 = int(yf * full_h)
    roi_w = int(wf * full_w)
    roi_h = int(hf * full_h)

    n_cols = max(1, int(np.ceil(roi_w / cw)))
    n_rows = max(1, int(np.ceil(roi_h / ch)))

    out = bgr.copy()
    for row in range(n_rows):
        for col in range(n_cols):
            px0 = x0 + col * cw
            py0 = y0 + row * ch
            px1 = min(full_w, px0 + cw)
            py1 = min(full_h, py0 + ch)
            shade = 60 if (row + col) % 2 == 0 else 30
            overlay_cell = out[py0:py1, px0:px1].astype(np.float32)
            out[py0:py1, px0:px1] = np.clip(overlay_cell + shade, 0, 255).astype(np.uint8)
            cv2.rectangle(out, (px0, py0), (px1 - 1, py1 - 1), (255, 255, 255), 1)

    # Label dimensions in the top-left corner
    label = f"stud {cw}x{ch}px  grid {n_cols}x{n_rows}"
    cv2.putText(out, label, (x0 + 4, y0 + ch - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label, (x0 + 4, y0 + ch - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    print(f"Stud pitch: {pitch:.1f} px  cell {cw}×{ch} px  grid {n_cols}×{n_rows}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="LEGO wall color segmentation")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--out", default="debug.png", help="Output overlay path")
    parser.add_argument("--show", action="store_true", help="Show with matplotlib")
    parser.add_argument("--no-grid", action="store_true", help="Skip stud grid overlay")
    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        sys.exit(f"Error: could not read '{args.image}'")

    h, w = bgr.shape[:2]
    print(f"Image: {args.image}  ({w}×{h})")

    regions, roi = segment_image(bgr)

    if roi:
        xf, yf, wf, hf = roi
        print(f"Auto ROI: ({int(xf*w)},{int(yf*h)}) {int(wf*w)}×{int(hf*h)} px")

    regions_sorted = sorted(regions, key=lambda r: -r.area_fraction)
    print(f"\n{'#':<4} {'Color':<16} {'Area%':>6}  BBox (x,y,w,h)")
    print("─" * 60)
    for i, r in enumerate(regions_sorted):
        bx, by, bw, bh = r.bbox
        print(f"{i:<4} {r.color_name:<16} {r.area_fraction*100:>5.2f}%  "
              f"({bx},{by},{bw},{bh})")
    print(f"\nTotal: {len(regions)} regions")

    # Color region overlay
    annotated = draw_regions(bgr, regions)
    annotated = draw_roi_box(annotated, roi)

    # Stud test grid (always on by default so user can verify sizes)
    if not args.no_grid and roi:
        annotated = draw_stud_test_grid(annotated, roi, w, h)

    save_and_show(annotated, args.out, show=args.show)
    print(f"Overlay → {args.out}")


if __name__ == "__main__":
    main()
