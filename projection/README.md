# Projection Layer — Future Work

This module is **not yet implemented**. It will take the perception output from
`run_full_demo.py` and project animated content back onto the LEGO wall.

## Inputs it will consume

| Input | Type | Source |
|-------|------|--------|
| `regions` | `List[Region]` | `segmentation.color_segment.segment_image()` |
| `asset_dir` | `str` | `assets/` directory |
| `homography` | `np.ndarray` (3×3) | Camera→projector calibration (separate step) |

## Per-frame algorithm (pseudocode)

```python
for region in regions:
    asset_path = pick_asset(region.color_name, region.texture_class)
    frame = read_current_frame(asset_path)
    masked = apply_contour_mask(frame, region.contour)
    warped = cv2.warpPerspective(masked, homography, projector_resolution)
    composite = blend(composite, warped)
project_fullscreen(composite)
```

## Homography calibration

A separate calibration script will:
1. Project a known pattern (checkerboard / ArUco markers).
2. Capture it with the camera.
3. Compute the camera→projector homography via `cv2.findHomography`.
4. Save the 3×3 matrix to `calibration/homography.npy`.
