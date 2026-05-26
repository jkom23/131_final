#!/usr/bin/env python3
"""
Export per-region image crops for manual texture labeling.

Usage:
    python scripts/export_crops.py --image IMG_1361.jpeg
    python scripts/export_crops.py --image IMG_1361.jpeg --out-dir data/region_crops
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

import config
from segmentation.color_segment import segment_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Export region crops for labeling")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--out-dir", default=config.REGION_CROPS_DIR,
                        help="Directory to write crop images")
    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        sys.exit(f"Error: could not read '{args.image}'")

    os.makedirs(args.out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.image))[0]
    regions, _ = segment_image(bgr)
    regions_sorted = sorted(regions, key=lambda r: -r.area_fraction)

    written = []
    for idx, region in enumerate(regions_sorted):
        x, y, w, h = region.bbox
        crop = bgr[y:y+h, x:x+w]
        filename = f"{base}_{region.color_name}_{idx:03d}.jpg"
        path = os.path.join(args.out_dir, filename)
        cv2.imwrite(path, crop)
        written.append((filename, region.color_name, region.area_fraction))
        print(f"  {filename}  ({region.color_name}, {region.area_fraction*100:.2f}%)")

    print(f"\nExported {len(written)} crops → {args.out_dir}")
    print(f"Next: label them in {config.LABELS_CSV}")
    print( "  Format: crop_filename,texture_class")
    print( "  Classes: flat | stud | curved | other")


if __name__ == "__main__":
    main()
