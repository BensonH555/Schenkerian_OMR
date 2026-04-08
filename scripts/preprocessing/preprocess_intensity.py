"""Preprocess using grayscale intensity separation (Method 1).

Key idea: Staff lines are pre-printed LIGHT GRAY, handwritten ink is DARK.
By thresholding on intensity BEFORE binarization, we can separate them cleanly.

Also removes non-music content (title text, page numbers, margin notes).

Usage:
    python preprocess_intensity.py --input_dir /path/to/images --output_dir /path/to/output
"""

import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import cv2


def preprocess_single(img_path, output_path, ink_threshold=130, debug_dir=None):
    """Preprocess using intensity-based staff/ink separation."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")

    h, w = img.shape
    fname = os.path.basename(img_path)
    print(f"  Original size: {w}x{h}")

    # === Step 1: Intensity-based ink extraction ===
    # Ink is dark (< ink_threshold), staff lines are lighter, paper is bright
    # This separates ink from staff BEFORE binarization
    ink_mask = (img < ink_threshold).astype(np.uint8) * 255

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, fname.replace('.jpg', '_1_ink_only.png')), ink_mask)

    ink_pixels_before = np.sum(ink_mask > 0)
    print(f"  Ink pixels (threshold={ink_threshold}): {ink_pixels_before}")

    # === Step 2: Remove title/header and footer ===
    # Music staves have sustained dense ink rows; title text is sparse/isolated
    row_ink = np.sum(ink_mask > 0, axis=1)
    window = 30
    smoothed = np.convolve(row_ink, np.ones(window)/window, mode='same')

    music_threshold = w * 0.02
    first_music_row = 0
    for y in range(h):
        if smoothed[y] > music_threshold:
            if y + 50 < h and np.mean(smoothed[y:y+50]) > music_threshold * 0.5:
                first_music_row = max(0, y - 20)
                break

    last_music_row = h
    for y in range(h - 1, 0, -1):
        if smoothed[y] > music_threshold:
            if y - 50 >= 0 and np.mean(smoothed[y-50:y]) > music_threshold * 0.5:
                last_music_row = min(h, y + 20)
                break

    print(f"  Music region: rows {first_music_row}-{last_music_row} (of {h})")

    ink_mask[:first_music_row] = 0
    ink_mask[last_music_row:] = 0

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, fname.replace('.jpg', '_2_region.png')), ink_mask)

    # === Step 3: Morphological cleanup ===
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, noise_kernel)

    # === Step 4: Connected component filtering ===
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    clean = np.zeros_like(ink_mask)
    for label_id in range(1, num_labels):
        comp_w = stats[label_id, cv2.CC_STAT_WIDTH]
        comp_h = stats[label_id, cv2.CC_STAT_HEIGHT]
        comp_area = stats[label_id, cv2.CC_STAT_AREA]

        if comp_area < 15:
            continue
        if comp_h <= 4 and comp_w > 80 and comp_w / max(comp_h, 1) > 10:
            continue
        clean[labels == label_id] = 255

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, fname.replace('.jpg', '_3_clean.png')), clean)

    # === Step 5: Save as 1-bit PNG ===
    pil_img = Image.fromarray(clean, mode='L')
    pil_img = pil_img.point(lambda x: 255 if x > 0 else 0, '1')
    pil_img.save(output_path)

    remaining = np.sum(clean > 0)
    print(f"  Final: {remaining} foreground pixels (removed {ink_pixels_before - remaining})")
    return clean


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory with input images')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--ink_threshold', type=int, default=130,
                        help='Grayscale threshold for ink (pixels darker than this are ink)')
    parser.add_argument('--debug', action='store_true', help='Save intermediate steps')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    debug_dir = None
    if args.debug:
        debug_dir = os.path.join(args.output_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(args.input_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Processing {len(img_files)} images (ink_threshold={args.ink_threshold})...")
    for fname in img_files:
        print(f"\n--- {fname} ---")
        in_path = os.path.join(args.input_dir, fname)
        out_name = os.path.splitext(fname)[0] + '.png'
        out_path = os.path.join(args.output_dir, out_name)
        try:
            preprocess_single(in_path, out_path, args.ink_threshold, debug_dir)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\nDone! Results saved to {args.output_dir}")
