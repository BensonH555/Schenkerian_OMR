"""Preprocess new sheet music images to match MUSCIMA++ datasets_r_staff format.

MUSCIMA++ training images are:
- 1-bit: white symbols on BLACK background
- Staff lines REMOVED (only symbols remain)

This script uses a multi-stage approach:
1. Grayscale intensity separation (ink is dark, staff lines are lighter gray)
2. Row projection to locate staff line positions
3. Targeted removal at detected staff positions
4. Morphological cleanup for remnants
5. Connected component filtering for noise

Usage:
    python preprocess_new_images.py --input_dir /path/to/images --output_dir /path/to/output
"""

import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import cv2
from scipy.ndimage import gaussian_filter1d


def detect_staff_rows(binary, w):
    """Use row projection to find rows that are likely staff lines.

    Staff lines have high horizontal ink density compared to surrounding rows.
    Returns a boolean mask of staff-line rows.
    """
    row_sum = np.sum(binary > 0, axis=1).astype(np.float64)
    row_smooth = gaussian_filter1d(row_sum, sigma=1)
    h = len(row_sum)

    # Local median baseline (window of 40 rows)
    window = 40
    local_median = np.array([
        np.median(row_smooth[max(0, y - window):min(h, y + window)])
        for y in range(h)
    ])

    # Staff rows: significantly above local median AND have minimum density
    min_density = w * 0.08  # at least 8% of width
    is_staff_row = (row_smooth > local_median * 1.8) & (row_smooth > min_density)

    # Expand each detected row by ±1 to catch full line thickness
    staff_rows = np.zeros(h, dtype=bool)
    for y in range(h):
        if is_staff_row[y]:
            for dy in range(-1, 2):
                if 0 <= y + dy < h:
                    staff_rows[y + dy] = True

    return staff_rows


def remove_staff_lines(img_gray, ink_threshold=120):
    """Multi-stage staff line removal.

    Stage 1: Grayscale intensity separation
      - Pixels darker than ink_threshold → ink (kept)
      - Pixels between ink_threshold and paper → potential staff lines
    Stage 2: Row projection on the "potential staff" pixels
      - Identify rows with high horizontal density → staff line rows
    Stage 3: At staff line rows, only keep pixels that are clearly ink
      - Use a stricter (lower) threshold at staff positions
    Stage 4: Morphological cleanup
      - Short horizontal opening to catch any remaining staff fragments
      - Aspect ratio filter for thin horizontal remnants
    """
    h, w = img_gray.shape

    # --- Stage 1: Intensity-based separation ---
    # Dark ink (hand-drawn): clearly below threshold
    ink_mask = img_gray < ink_threshold
    # Medium gray (staff lines + some ink): between threshold and paper
    paper_threshold = 200
    medium_mask = (img_gray >= ink_threshold) & (img_gray < paper_threshold)

    # Start with all ink pixels
    result = np.zeros((h, w), dtype=np.uint8)
    result[ink_mask] = 255

    # --- Stage 2: Also include medium-gray pixels that are NOT on staff rows ---
    # First, find staff line positions using row projection on medium+ink pixels
    all_foreground = np.zeros((h, w), dtype=np.uint8)
    all_foreground[img_gray < paper_threshold] = 255

    staff_rows = detect_staff_rows(all_foreground, w)

    # Medium-gray pixels NOT on staff rows are likely smudged ink → keep them
    medium_not_staff = medium_mask & ~staff_rows[:, np.newaxis]
    result[medium_not_staff] = 255

    # --- Stage 3: At staff row positions, be more aggressive ---
    # On staff rows, only keep very dark ink (stricter threshold)
    strict_threshold = ink_threshold - 20
    for y in range(h):
        if staff_rows[y]:
            for x in range(w):
                if result[y, x] > 0 and img_gray[y, x] >= strict_threshold:
                    # Check vertical connectivity: if this pixel connects
                    # to non-staff-row ink above AND below, keep it (it's a stem)
                    has_above = False
                    has_below = False
                    for dy in range(1, 8):
                        if y - dy >= 0 and not staff_rows[y - dy] and result[y - dy, x] > 0:
                            has_above = True
                            break
                    for dy in range(1, 8):
                        if y + dy < h and not staff_rows[y + dy] and result[y + dy, x] > 0:
                            has_below = True
                            break
                    if not (has_above and has_below):
                        result[y, x] = 0

    # --- Stage 4: Morphological cleanup ---
    # Short horizontal opening to catch remaining horizontal fragments
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horiz = cv2.morphologyEx(result, cv2.MORPH_OPEN, hk)
    # Protect vertical structures
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vert = cv2.morphologyEx(result, cv2.MORPH_OPEN, vk)
    staff_only = horiz.copy()
    staff_only[vert > 0] = 0
    result[staff_only > 0] = 0

    # --- Stage 5: Aspect ratio filter ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    for label_id in range(1, num_labels):
        comp_w = stats[label_id, cv2.CC_STAT_WIDTH]
        comp_h = stats[label_id, cv2.CC_STAT_HEIGHT]
        comp_area = stats[label_id, cv2.CC_STAT_AREA]
        # Thin horizontal remnants
        if comp_h <= 5 and comp_w > 40 and comp_w / max(comp_h, 1) > 8:
            result[labels == label_id] = 0
        elif comp_h <= 8 and comp_w > 80 and comp_w / max(comp_h, 1) > 12:
            fill = comp_area / (comp_w * comp_h)
            if fill < 0.5:
                result[labels == label_id] = 0

    return result, staff_rows


def preprocess_single(img_path, output_path, debug_dir=None):
    """Preprocess a single image to MUSCIMA++ datasets_r_staff format."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")

    h_orig, w_orig = img.shape
    fname = os.path.basename(img_path)
    print(f"  Original size: {w_orig}x{h_orig}")

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, fname.replace('.jpg', '_0_gray.png')), img)

    # Stage 1-4: Multi-stage staff removal (operates on grayscale directly)
    no_staff, staff_rows = remove_staff_lines(img, ink_threshold=120)

    if debug_dir:
        # Save staff row visualization
        staff_vis = img.copy()
        for y in range(h_orig):
            if staff_rows[y]:
                staff_vis[y, :] = staff_vis[y, :] // 2  # darken staff rows
        cv2.imwrite(os.path.join(debug_dir, fname.replace('.jpg', '_1_staff_rows.png')), staff_vis)
        cv2.imwrite(os.path.join(debug_dir, fname.replace('.jpg', '_2_no_staff.png')), no_staff)

    # Content region detection: remove noise in blank areas
    blurred = cv2.GaussianBlur(img, (51, 51), 0)
    _, content_mask = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)
    content_mask = cv2.dilate(content_mask, np.ones((30, 30), np.uint8), iterations=1)
    no_staff[content_mask == 0] = 0

    # Remove small noise components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(no_staff, connectivity=8)
    min_area = 20
    clean = np.zeros_like(no_staff)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label_id] = 255

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, fname.replace('.jpg', '_3_clean.png')), clean)

    # Save as 1-bit PNG (white on black = MUSCIMA++ format)
    pil_img = Image.fromarray(clean, mode='L')
    pil_img = pil_img.point(lambda x: 255 if x > 0 else 0, '1')
    pil_img.save(output_path)

    remaining = np.sum(clean > 0)
    print(f"  Result: {remaining} foreground pixels")

    return clean


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory with input images')
    parser.add_argument('--output_dir', required=True, help='Output directory for preprocessed images')
    parser.add_argument('--debug', action='store_true', help='Save intermediate processing steps')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    debug_dir = None
    if args.debug:
        debug_dir = os.path.join(args.output_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(args.input_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Processing {len(img_files)} images...")
    for fname in img_files:
        print(f"\n--- {fname} ---")
        in_path = os.path.join(args.input_dir, fname)
        out_name = os.path.splitext(fname)[0] + '.png'
        out_path = os.path.join(args.output_dir, out_name)
        try:
            preprocess_single(in_path, out_path, debug_dir)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\nDone! Preprocessed images saved to {args.output_dir}")
