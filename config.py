# All tunable parameters for the LEGO wall perception pipeline.
# HSV ranges MUST be re-tuned per lighting setup.

# --- Image processing ---
GAUSSIAN_BLUR_KERNEL = 3    # kernel size for pre-blur; 0 to disable
USE_GRAY_WORLD = False       # toggle gray-world white-balance normalization
MORPH_OPEN_SIZE  = 5         # open kernel: removes speckle noise (keep small)
# At 5712px wide, LEGO stud-row gaps are ~100px; close kernel must be ≥ 2× that gap.
# Scale this proportionally if using lower-resolution captures.
MORPH_CLOSE_SIZE = 121       # close kernel: fills stud-row gaps (scale to image res)

# --- Region filtering ---
MIN_REGION_AREA_FRACTION = 0.0008   # primary: fraction of total image area
MIN_REGION_AREA_PX = 400            # secondary absolute pixel floor

# --- Region of Interest ---
# "auto"  → auto-detect from image content (recommended)
# None    → full image
# (x_frac, y_frac, w_frac, h_frac) → manual fractions of image dimensions
ROI = "auto"

# Saturation/value floor used by auto-ROI detection to distinguish LEGO from background.
# White wall ≈ S 0-20, gray carpet ≈ S 0-25; LEGO bricks (even light ones) ≥ ~60.
ROI_SAT_MIN = 80    # pixels with S < this are treated as background (wall/carpet)
ROI_VAL_MIN = 55    # pixels with V < this are also treated as background

# After auto-detection, trim this many pixels from the ROI bottom to cut the
# flat baseplate rows (they sit below the colored towers).
# Increase if baseplates are still visible; 0 to keep them.
ROI_BASEPLATE_TRIM_PX = 280

# --- Texture model ---
TEXTURE_MODEL = "tiny"           # "tiny" | "mobilenet"
TEXTURE_INPUT_SIZE = 96          # crop resize (px) for texture CNN
TEXTURE_CHECKPOINT = "texture/checkpoints/best.pt"

# --- Paths ---
RAW_IMAGES_DIR = "data/raw_images"
REGION_CROPS_DIR = "data/region_crops"
LABELS_CSV = "data/labels/texture_labels.csv"

# ---------------------------------------------------------------------------
# HSV Color Palette
# OpenCV HSV: H 0-179, S 0-255, V 0-255
# Each color maps to a list of {"lower": (H,S,V), "upper": (H,S,V)} dicts.
# Multiple ranges per color are OR-ed (needed for red, which wraps H=0/179).
# TUNE THESE to your lighting before each capture session.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 10-color LEGO palette.
# Key separators:
#   blue vs teal   → V threshold (~95): blue is bright, teal is dark
#   green vs forest → V threshold (~85): forest is very dark
#   lime vs green   → H boundary (~60): lime is more yellow, green is purer
# Tune S/V bounds per your lighting.
# ---------------------------------------------------------------------------
HSV_PALETTE = {
    "red": [
        {"lower": (0,   110,  60), "upper": (9,   255, 255)},
        {"lower": (168, 110,  60), "upper": (179, 255, 255)},
    ],
    "orange": [
        {"lower": (9,  140, 100), "upper": (22, 255, 255)},
    ],
    "hazmat_yellow": [
        {"lower": (22, 100, 120), "upper": (34, 255, 255)},
    ],
    "lime": [
        # bright apple/lime green — yellowish, high V
        {"lower": (34,  80, 110), "upper": (60, 255, 255)},
    ],
    "green": [
        # medium standard LEGO green — purer hue, medium brightness
        {"lower": (58,  55,  85), "upper": (85, 255, 200)},
    ],
    "forest": [
        # very dark green — same hue range as green but low V
        {"lower": (48,  55,  15), "upper": (90, 255,  85)},
    ],
    "teal": [
        # very dark blue-green connector rows only (shadow/grooves between bricks)
        {"lower": (82,  55,  15), "upper": (115, 255,  62)},
    ],
    "blue": [
        # LEGO blue — threshold at V=60 so shadowed blue bricks still count
        {"lower": (95,  90,  60), "upper": (128, 255, 255)},
    ],
    "pink": [
        {"lower": (140,  40, 100), "upper": (168, 255, 255)},
    ],
    "white": [
        # slight saturation floor excludes the neutral white wall
        {"lower": (0,   8,  190), "upper": (179,  40, 255)},
    ],
}

# BGR draw colors for visualization overlays
PALETTE_BGR = {
    "red":           (  0,   0, 220),
    "orange":        (  0, 140, 255),
    "hazmat_yellow": (  0, 215, 215),
    "lime":          ( 40, 215,  40),
    "green":         ( 10, 140,  10),
    "forest":        (  5,  70,   5),
    "teal":          ( 90, 110,  20),
    "blue":          (210,  70,  20),
    "pink":          (170,  80, 220),
    "white":         (230, 230, 230),
}

TEXTURE_CLASSES = ["flat", "stud", "curved", "other"]

# ---------------------------------------------------------------------------
# Stud grid
# ---------------------------------------------------------------------------
STUD_REAL_MM = 8.0          # physical stud pitch (8 mm for standard LEGO)

# Calibration reference: (stud_px, image_width_px) measured in any capture.
# The pipeline scales the stud size to the current image width automatically.
# Example below: in a 1086-px-wide reference image, one stud = 60 px.
STUD_CALIBRATION = (60, 1086)   # (stud_px, reference_image_width_px)

# Scale factor applied on top of the calibration result.
# Use < 1.0 if the test grid cells look too large relative to real studs.
# Example: 0.5 halves the cell size, doubling the grid density.
STUD_SCALE_FACTOR = 0.3

# Override: set a fixed pixel pitch directly (bypasses calibration + autocorr).
STUD_PITCH_PX = None

# Brick height / stud pitch ratio (same for LEGO and DUPLO: 9.6/8 = 1.2)
BRICK_TO_STUD_RATIO = 1.2
