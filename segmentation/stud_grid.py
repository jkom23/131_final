"""
Stud-grid representation of the LEGO wall.

Each cell of the grid corresponds to one stud footprint (8 mm × 8 mm for
standard LEGO; the same code works for DUPLO at 16 mm by adjusting config).

Stud pitch estimation:
  From a side-on photo, each row of bricks produces a periodic horizontal
  stripe pattern.  The stripe period equals the brick height, which is
  1.2 × stud_pitch for both standard LEGO and DUPLO.
  We find this period via autocorrelation of the column-averaged saturation
  profile within the ROI, then derive stud_pitch_px = brick_period / 1.2.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from segmentation.regions import Region


# ── Pitch estimation ─────────────────────────────────────────────────────────

def _row_saturation_profile(roi_bgr: np.ndarray) -> np.ndarray:
    """Average HSV saturation per row (1-D signal, length = ROI height)."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    return sat.mean(axis=1)


def _autocorr_period(signal: np.ndarray,
                     min_period: int = 60,
                     max_period_frac: float = 0.45) -> Optional[int]:
    """
    Find the dominant period of a 1-D signal via normalized autocorrelation.
    Returns period in samples, or None if no clear peak is found.
    """
    n = len(signal)
    sig = signal - signal.mean()
    norm = np.dot(sig, sig) + 1e-8

    max_lag = int(n * max_period_frac)
    acorr = np.array([np.dot(sig[:n - lag], sig[lag:]) / norm
                      for lag in range(min_period, max_lag)])

    if len(acorr) == 0:
        return None

    # First local maximum after the zero crossing
    for i in range(1, len(acorr) - 1):
        if acorr[i] > acorr[i - 1] and acorr[i] > acorr[i + 1] and acorr[i] > 0.1:
            return min_period + i

    # Fallback: global maximum
    peak = int(np.argmax(acorr))
    return min_period + peak if acorr[peak] > 0.05 else None


def estimate_stud_pitch_px(roi_bgr: np.ndarray,
                           full_image_width: Optional[int] = None) -> float:
    """
    Estimate stud pitch in pixels for the current image.

    Priority:
      1. STUD_PITCH_PX override (absolute px value, set directly in config).
      2. STUD_CALIBRATION ratio: scales (calib_stud_px / calib_image_width)
         to the current full image width.
      3. Autocorrelation of the row-saturation profile in the ROI.
      4. Heuristic fallback (~6 brick rows span the ROI height).
    """
    if config.STUD_PITCH_PX is not None:
        return float(config.STUD_PITCH_PX)

    if config.STUD_CALIBRATION is not None and full_image_width is not None:
        calib_stud_px, calib_image_w = config.STUD_CALIBRATION
        scale = getattr(config, "STUD_SCALE_FACTOR", 1.0)
        return full_image_width * (calib_stud_px / calib_image_w) * scale

    roi_h = roi_bgr.shape[0]
    min_p = max(60, roi_h // 10)
    profile = _row_saturation_profile(roi_bgr)
    period = _autocorr_period(profile, min_period=min_p)

    if period is not None:
        return period / config.BRICK_TO_STUD_RATIO

    return roi_h / (6 * config.BRICK_TO_STUD_RATIO)


# ── Grid construction ─────────────────────────────────────────────────────────

def build_stud_grid(
    regions: List[Region],
    roi: Tuple[float, float, float, float],
    image_shape: Tuple[int, int],
    roi_bgr: np.ndarray,
) -> Dict:
    """
    Divide the ROI into a regular grid of stud cells and assign each cell
    the dominant color label.

    Returns a dict with keys:
      grid_colors  : 2-D array of color_name strings (shape rows × cols)
      cell_h_px    : cell height in pixels
      cell_w_px    : cell width in pixels (= stud_pitch_px)
      roi_x0, roi_y0 : top-left pixel of ROI in full-image coords
      stud_pitch_px  : estimated stud pitch
    """
    full_h, full_w = image_shape
    xf, yf, wf, hf = roi
    roi_x0 = int(xf * full_w)
    roi_y0 = int(yf * full_h)
    roi_w  = int(wf * full_w)
    roi_h  = int(hf * full_h)

    stud_pitch = estimate_stud_pitch_px(roi_bgr, full_image_width=full_w)
    cell_w = max(1, int(round(stud_pitch)))
    cell_h = max(1, int(round(stud_pitch * config.BRICK_TO_STUD_RATIO)))

    n_cols = max(1, int(np.ceil(roi_w / cell_w)))
    n_rows = max(1, int(np.ceil(roi_h / cell_h)))

    # Build a label image: each pixel gets its region's color index (or -1)
    label_img = np.full((full_h, full_w), -1, dtype=np.int32)
    for idx, region in enumerate(regions):
        label_img[region.mask > 0] = idx

    # For each cell, find the dominant (most frequent) color label
    grid_colors = np.full((n_rows, n_cols), "", dtype=object)

    for row in range(n_rows):
        for col in range(n_cols):
            px0 = roi_x0 + col * cell_w
            py0 = roi_y0 + row * cell_h
            px1 = min(full_w, px0 + cell_w)
            py1 = min(full_h, py0 + cell_h)

            cell_labels = label_img[py0:py1, px0:px1].ravel()
            valid = cell_labels[cell_labels >= 0]
            if len(valid) == 0:
                grid_colors[row, col] = ""
                continue

            vals, counts = np.unique(valid, return_counts=True)
            dominant_idx = vals[np.argmax(counts)]
            grid_colors[row, col] = regions[dominant_idx].color_name

    return {
        "grid_colors":   grid_colors,
        "cell_h_px":     cell_h,
        "cell_w_px":     cell_w,
        "roi_x0":        roi_x0,
        "roi_y0":        roi_y0,
        "stud_pitch_px": stud_pitch,
        "n_rows":        n_rows,
        "n_cols":        n_cols,
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def draw_stud_grid(bgr: np.ndarray, grid_info: Dict,
                   alpha: float = 0.25) -> np.ndarray:
    """
    Overlay the stud grid on the image:
      - Each occupied cell gets a semi-transparent color fill.
      - Grid lines are drawn as thin white lines.
    Returns annotated BGR image.
    """
    output  = bgr.copy()
    overlay = bgr.copy()

    grid_colors = grid_info["grid_colors"]
    ch = grid_info["cell_h_px"]
    cw = grid_info["cell_w_px"]
    x0 = grid_info["roi_x0"]
    y0 = grid_info["roi_y0"]
    n_rows, n_cols = grid_colors.shape

    for row in range(n_rows):
        for col in range(n_cols):
            color_name = grid_colors[row, col]
            if not color_name:
                continue
            bgr_color = config.PALETTE_BGR.get(color_name, (100, 100, 100))
            px0 = x0 + col * cw
            py0 = y0 + row * ch
            px1 = min(bgr.shape[1], px0 + cw)
            py1 = min(bgr.shape[0], py0 + ch)
            cv2.rectangle(overlay, (px0, py0), (px1, py1), bgr_color, -1)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # Grid lines
    for row in range(n_rows + 1):
        py = y0 + row * ch
        cv2.line(output, (x0, py), (x0 + n_cols * cw, py), (200, 200, 200), 1)
    for col in range(n_cols + 1):
        px = x0 + col * cw
        cv2.line(output, (px, y0), (px, y0 + n_rows * ch), (200, 200, 200), 1)

    return output
