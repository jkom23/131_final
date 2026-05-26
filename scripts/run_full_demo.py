#!/usr/bin/env python3
"""
Full pipeline: color segmentation + texture classification → annotated overlay.

Usage:
    python scripts/run_full_demo.py --image IMG_1361.jpeg
    python scripts/run_full_demo.py --image IMG_1361.jpeg --out full_demo.png --show
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

import config
from segmentation.color_segment import segment_image
from segmentation.visualize import draw_regions, save_and_show
from texture.infer import classify_texture


def main() -> None:
    parser = argparse.ArgumentParser(description="LEGO wall full demo")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="full_demo.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        sys.exit(f"Error: could not read '{args.image}'")

    regions, _ = segment_image(bgr)

    # Attach texture predictions
    has_model = os.path.exists(config.TEXTURE_CHECKPOINT)
    for region in regions:
        x, y, w, h = region.bbox
        crop = bgr[y:y+h, x:x+w]
        texture, conf = classify_texture(crop)
        region.texture_class = texture
        region.texture_confidence = conf

    # Print summary — this is the seam point for the projection layer
    print(f"\n{'#':<4} {'Color':<14} {'Texture':<10} {'Conf':>5}  {'Area%':>6}  BBox")
    print("─" * 75)
    for i, r in enumerate(sorted(regions, key=lambda x: -x.area_fraction)):
        bx, by, bw, bh = r.bbox
        conf_str = f"{r.texture_confidence:.2f}" if r.texture_confidence is not None else "  n/a"
        print(f"{i:<4} {r.color_name:<14} {r.texture_class:<10} {conf_str:>5}  "
              f"{r.area_fraction*100:>5.2f}%  ({bx},{by},{bw},{bh})")

    if not has_model:
        print("\n[Note] No texture checkpoint found — texture shows 'other'.")
        print(f"       Run texture/train.py after labeling crops to enable real predictions.")

    annotated = draw_regions(bgr, regions)
    save_and_show(annotated, args.out, show=args.show)
    print(f"\nOverlay saved → {args.out}")


if __name__ == "__main__":
    main()
