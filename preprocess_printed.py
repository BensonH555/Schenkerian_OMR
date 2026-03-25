"""Preprocess PRINTED sheet music images for OMR.

For printed scores, staff lines have the same ink darkness as symbols,
so grayscale intensity separation doesn't work. Instead, we exploit that
printed staff lines are perfectly straight horizontal lines.

Strategy:
1. Auto-rotate portrait images to landscape
2. Binarize
3. Detect staff line y-positions using horizontal run-length analysis
4. Remove entire rows at staff line positions, preserving vertical crossings
5. Morphological + aspect ratio cleanup
6. Clean noise

Usage:
    python preprocess_printed.py --input_dir /path/to/images --output_dir /path/to/output
"""

import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import cv2


def find_staff_line_rows(binary):
    """Find staff line row positions using horizontal run-length analysis.

    Staff lines are rows where foreground pixels form long horizontal runs.
    Uses vectorized approach for speed.
    """
    h, w = binary.shape
    long_run_score = np.zeros(h)
    min_run = 30  # minimum run length to count as "long"

    for y in range(h):
        row = binary[y]
        fg = np.sum(row > 0)
        if fg == 0:
            continue

        # Vectorized run-length encoding
        padded = np.concatenate([[0], row, [0]])
        diffs = np.diff((padded > 0).astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        run_lengths = ends - starts

        total_fg = np.sum(run_lengths)
        long_fg = np.sum(run_lengths[run_lengths > min_run])

        if total_fg > 0:
            long_run_score[y] = long_fg / total_fg

    return long_run_score


def detect_staff_bands(long_run_score, min_score=0.4):
    """Detect staff line bands from run-length scores.

    Returns list of (start_row, end_row, center, thickness).
    More lenient than before - accepts any thin band with high score.
    """
    h = len(long_run_score)
    candidates = np.where(long_run_score > min_score)[0]

    if len(candidates) == 0:
        return []

    # Group consecutive candidate rows into bands
    bands = []
    start = candidates[0]
    for i in range(1, len(candidates)):
        if candidates[i] - candidates[i-1] > 3:
            center = (start + candidates[i-1]) // 2
            thickness = candidates[i-1] - start + 1
            bands.append((start, candidates[i-1], center, thickness))
            start = candidates[i]
    center = (start + candidates[-1]) // 2
    thickness = candidates[-1] - start + 1
    bands.append((start, candidates[-1], center, thickness))

    # Staff lines are thin: keep only bands with thickness <= 8
    bands = [b for b in bands if b[3] <= 8]

    # Try to validate by grouping into 5-line systems
    validated = []
    used = set()

    for i in range(len(bands)):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(bands)):
            if j in used or len(group) >= 5:
                break

            prev_center = bands[group[-1]][2]
            curr_center = bands[j][2]
            spacing = curr_center - prev_center

            if len(group) == 1:
                if 10 <= spacing <= 120:
                    group.append(j)
            else:
                avg_sp = (bands[group[-1]][2] - bands[group[0]][2]) / (len(group) - 1)
                if abs(spacing - avg_sp) < avg_sp * 0.35:
                    group.append(j)

        # Accept groups of 3+ lines as valid staff
        if len(group) >= 3:
            for idx in group:
                used.add(idx)
                validated.append(bands[idx])

    # Also keep ungrouped bands that have very high score (likely staff lines
    # that couldn't be grouped due to missing neighbors)
    for i, b in enumerate(bands):
        if i not in used:
            avg_score = np.mean(long_run_score[b[0]:b[1]+1])
            if avg_score > 0.6 and b[3] <= 5:
                validated.append(b)

    return validated


def remove_at_staff_rows(binary, staff_bands):
    """Remove pixels at staff line positions, preserving vertical crossings.

    Vectorized version for much better performance.
    """
    result = binary.copy()
    h, w = binary.shape

    if not staff_bands:
        return result

    # Build staff row mask (with ±1 expansion for full coverage)
    staff_row_mask = np.zeros(h, dtype=bool)
    for start, end, center, thickness in staff_bands:
        y0 = max(0, start - 1)
        y1 = min(h, end + 2)
        staff_row_mask[y0:y1] = True

    # Build non-staff foreground mask
    non_staff_fg = (binary > 0) & ~staff_row_mask[:, np.newaxis]

    # For each staff row, check vertical connectivity
    search_range = 8

    for y in range(h):
        if not staff_row_mask[y]:
            continue

        for x in range(w):
            if result[y, x] == 0:
                continue

            # Check above: any non-staff foreground pixel within search_range
            has_above = False
            for dy in range(1, search_range + 1):
                yy = y - dy
                if yy >= 0 and non_staff_fg[yy, x]:
                    has_above = True
                    break

            if not has_above:
                result[y, x] = 0
                continue

            has_below = False
            for dy in range(1, search_range + 1):
                yy = y + dy
                if yy < h and non_staff_fg[yy, x]:
                    has_below = True
                    break

            if not has_below:
                result[y, x] = 0

    return result


def preprocess_printed(img_path, output_path, debug_dir=None):
    """Preprocess a printed score image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")

    h, w = img.shape
    fname = os.path.basename(img_path)

    # Auto-rotate portrait images to landscape
    rotated = False
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = img.shape
        rotated = True
        print(f"  Rotated to landscape: {w}x{h}")
    else:
        print(f"  Size: {w}x{h}")

    # Step 1: Binarize (adaptive threshold)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=51, C=15
    )
    noise_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, noise_k)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, fname.replace('.jpg', '_1_binary.png').replace('.png', '_1_binary.png')), binary)

    # Step 2: Find staff line rows
    long_run_score = find_staff_line_rows(binary)

    if debug_dir:
        score_img = np.zeros((h, 300), dtype=np.uint8)
        for y in range(h):
            x_val = int(long_run_score[y] * 299)
            score_img[y, :x_val] = 255
        cv2.imwrite(os.path.join(debug_dir, fname.replace('.jpg', '_2_runlength.png').replace('.png', '_2_runlength.png')), score_img)

    # Step 3: Detect staff line bands
    staff_bands = detect_staff_bands(long_run_score, min_score=0.4)
    print(f"  Detected {len(staff_bands)} staff line bands")

    if debug_dir:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for start, end, center, thickness in staff_bands:
            cv2.line(vis, (0, center), (w, center), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(debug_dir, fname.replace('.jpg', '_3_detected.png').replace('.png', '_3_detected.png')), vis)

    # Step 4: Remove staff line pixels
    no_staff = remove_at_staff_rows(binary, staff_bands)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, fname.replace('.jpg', '_4_no_staff.png').replace('.png', '_4_no_staff.png')), no_staff)

    # Step 5: Morphological cleanup
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horiz_rem = cv2.morphologyEx(no_staff, cv2.MORPH_OPEN, hk)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vert = cv2.morphologyEx(no_staff, cv2.MORPH_OPEN, vk)
    staff_rem = horiz_rem.copy()
    staff_rem[vert > 0] = 0
    no_staff[staff_rem > 0] = 0

    # Aspect ratio filter for thin horizontal remnants
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(no_staff, connectivity=8)
    for lid in range(1, num_labels):
        cw = stats[lid, cv2.CC_STAT_WIDTH]
        ch = stats[lid, cv2.CC_STAT_HEIGHT]
        if ch <= 4 and cw > 30 and cw / max(ch, 1) > 8:
            no_staff[labels == lid] = 0

    # Step 6: Remove small noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(no_staff, connectivity=8)
    clean = np.zeros_like(no_staff)
    for lid in range(1, num_labels):
        if stats[lid, cv2.CC_STAT_AREA] >= 15:
            clean[labels == lid] = 255

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, fname.replace('.jpg', '_5_clean.png').replace('.png', '_5_clean.png')), clean)

    # Save as 1-bit PNG
    pil_img = Image.fromarray(clean, mode='L')
    pil_img = pil_img.point(lambda x: 255 if x > 0 else 0, '1')
    pil_img.save(output_path)

    remaining = np.sum(clean > 0)
    print(f"  Result: {remaining} foreground pixels")
    return clean


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--debug', action='store_true')
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
            preprocess_printed(in_path, out_path, debug_dir)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\nDone! Results saved to {args.output_dir}")
