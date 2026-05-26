# Animation Assets

Place user-supplied animation files here following this naming convention.
The projection layer (future work) will pick assets using the selection rule below.

## Directory layout

```
assets/
  base/
    red.mp4          # base animation for the red region
    orange.mp4
    yellow.mp4
    lime_green.mp4
    dark_green.mp4
    blue.mp4
    dark_teal.mp4
    pink.mp4
    white.mp4
    gray.mp4
  texture/
    flat.mp4         # texture-override animation
    stud.mp4
    curved.mp4
    other.mp4
```

Accepted formats: `.mp4`, `.gif`, or a directory of sequentially-numbered frames
(e.g. `red/frame_0001.png`, `red/frame_0002.png`, …).

## Selection rule

For each region the projection layer will:
1. Check `assets/texture/<texture_class>.mp4` — use it if the file exists.
2. Otherwise fall back to `assets/base/<color_name>.mp4`.

## What the projection layer will consume (FUTURE)

Per-region tuple: `(contour, color_name, texture_class)` from `run_full_demo.py`.
It will mask the chosen asset frame to the contour, warp it through the
camera→projector homography matrix, composite all regions, and display fullscreen.
See `projection/README.md` for details.
