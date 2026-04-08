"""Preprocess Burkhart Schenkerian analysis images: staff line removal.

Detects groups of 5 parallel horizontal lines (staff systems) using
morphological horizontal opening to isolate line structures, then removes
them while preserving vertical structures (stems, barlines, etc.).

Pipeline:
1. Auto-rotate portrait -> landscape (clockwise)
2. Adaptive binarization
3. Morphological horizontal opening to extract long horizontal lines
4. Find line positions from horizontal projection profile
5. Group lines into 5-line staff systems with consistent spacing
6. Remove staff lines band-by-band, restoring vertical structures
7. Cleanup residual fragments
8. Save as 1-bit white-on-black PNG (MUSCIMA++ format)

Usage:
    python preprocess_burkhart.py [--debug]
"""

import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_DIR, "outputs/preprocessed_raw_graphs/images/Charles Burkhart")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs/preprocessed_raw_graphs/preprocessed/Charles Burkhart")


def find_horizontal_lines(binary):
    """Detect horizontal line positions using morphological opening + run-length.

    Two-stage approach:
    1. Morphological opening with moderate kernel to extract horizontal segments
    2. Run-length analysis on the original binary as fallback for thin/broken lines

    Both signals are combined to find the strongest horizontal line candidates.
    """
    h, w = binary.shape

    # Stage 1: Morphological opening with moderate kernel
    # Use w//15 (~220px) to handle handwritten lines with gaps
    horiz_len = max(w // 15, 80)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    horiz_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)

    # Horizontal projection from morph mask
    morph_proj = np.sum(horiz_mask > 0, axis=1).astype(np.float64)

    # Stage 2: Run-length analysis on original binary
    # For each row, sum the length of horizontal runs >= min_run
    min_run = max(w // 50, 20)
    run_proj = np.zeros(h)
    for y in range(h):
        row = binary[y]
        fg_total = np.sum(row > 0)
        if fg_total == 0:
            continue
        padded = np.concatenate([[0], row, [0]])
        diffs = np.diff((padded > 0).astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        run_lengths = ends - starts
        long_runs = run_lengths[run_lengths >= min_run]
        if len(long_runs) > 0:
            run_proj[y] = np.sum(long_runs)

    # Combine: normalize both to [0,1] and take max
    morph_norm = morph_proj / max(np.max(morph_proj), 1)
    run_norm = run_proj / max(np.max(run_proj), 1)
    combined = np.maximum(morph_norm, run_norm * 0.8)
    combined_smooth = gaussian_filter1d(combined, sigma=1.5)

    # Find peaks: lines should have strong horizontal signal
    peaks, _ = find_peaks(combined_smooth, height=0.15, distance=5,
                          prominence=0.03)

    # Additional filter: the row should have significant horizontal coverage
    # (at least 5% of image width in long horizontal runs)
    min_coverage = w * 0.05
    valid = []
    for p in peaks:
        if morph_proj[p] >= min_coverage or run_proj[p] >= min_coverage:
            valid.append(p)
    peaks = np.array(valid) if valid else np.array([], dtype=int)

    # Merge very close peaks (within 4px) keeping the stronger one
    if len(peaks) > 1:
        merged = []
        group = [peaks[0]]
        for i in range(1, len(peaks)):
            if peaks[i] - group[-1] <= 4:
                group.append(peaks[i])
            else:
                merged.append(group[np.argmax(combined_smooth[group])])
                group = [peaks[i]]
        merged.append(group[np.argmax(combined_smooth[group])])
        peaks = np.array(merged)

    return peaks, combined_smooth, horiz_mask


def group_into_staff_systems(line_rows, h, min_spacing=20, max_spacing=90):
    """Group detected line rows into 5-line staff systems.

    Uses a pair-based search: for each pair (i, j) of candidate lines,
    treats their spacing as the reference, then greedily extends to find
    3 more lines at that spacing. Filters by spacing consistency (CV).
    """
    if len(line_rows) < 5:
        return []

    lines = np.sort(line_rows)
    n = len(lines)
    candidates = []

    for i in range(n - 4):
        for j in range(i + 1, n - 3):
            s1 = lines[j] - lines[i]
            if s1 < min_spacing or s1 > max_spacing:
                continue

            group = [i, j]
            target = s1

            for k in range(j + 1, min(n, j + 20)):
                expected = lines[group[-1]] + target
                actual = lines[k]
                if abs(actual - expected) <= target * 0.25:
                    group.append(k)
                    target = (lines[group[-1]] - lines[group[0]]) / (len(group) - 1)
                    if len(group) == 5:
                        break
                elif actual > expected + target * 0.4:
                    break

            if len(group) == 5:
                positions = lines[group]
                spacings = np.diff(positions)
                cv = np.std(spacings) / np.mean(spacings)
                if cv < 0.15:
                    candidates.append((cv, positions.copy()))

    # Sort by quality (lowest CV first)
    candidates.sort(key=lambda x: x[0])

    # Greedily select non-overlapping systems (no shared rows within ±15px)
    systems = []
    used = set()
    for cv, positions in candidates:
        if any(abs(p - u) < 15 for p in positions for u in used):
            continue
        systems.append(positions)
        for p in positions:
            used.add(p)

    systems.sort(key=lambda s: s[0])
    return systems


def _estimate_thickness(binary, center_row):
    """Estimate staff line thickness at given row by sampling vertical extents."""
    h, w = binary.shape
    if center_row < 0 or center_row >= h:
        return 2
    fg_cols = np.where(binary[center_row, :] > 0)[0]
    if len(fg_cols) == 0:
        return 2
    sample = fg_cols[::max(1, len(fg_cols) // 30)]
    thicknesses = []
    for x in sample:
        y_up = center_row
        while y_up > 0 and binary[y_up - 1, x] > 0:
            y_up -= 1
        y_down = center_row
        while y_down < h - 1 and binary[y_down + 1, x] > 0:
            y_down += 1
        thicknesses.append(y_down - y_up + 1)
    return int(np.median(thicknesses)) if thicknesses else 2


def remove_staff_lines(binary, systems):
    """Remove staff lines: blank each line band, restore vertical structures.

    For each detected staff line, erases all pixels in a band around the
    line center, then restores any vertical structures (stems, barlines)
    that were also erased.
    """
    result = binary.copy()
    h, w = binary.shape

    # Pre-compute vertical structure mask
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)

    for system in systems:
        for center in system:
            thickness = _estimate_thickness(binary, center)
            half = max(thickness // 2 + 1, 2)
            y0 = max(0, center - half)
            y1 = min(h, center + half + 1)

            # Blank the band
            result[y0:y1, :] = 0
            # Restore vertical structures
            result[y0:y1, :] = cv2.bitwise_or(result[y0:y1, :],
                                               vertical_mask[y0:y1, :])

    # Global pass: catch remaining long horizontal structures not in systems
    horiz_len = max(w // 30, 50)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    remaining_horiz = cv2.morphologyEx(result, cv2.MORPH_OPEN, hk)

    expand_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    remaining_horiz = cv2.dilate(remaining_horiz, expand_k, iterations=1)
    remaining_horiz = cv2.bitwise_and(remaining_horiz, result)
    remaining_horiz[vertical_mask > 0] = 0

    result = cv2.subtract(result, remaining_horiz)
    return result


def cleanup_remnants(no_staff):
    """Remove residual staff line fragments via morphology + connected components."""
    h, w = no_staff.shape

    for klen in [10, 18, 30]:
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        horiz = cv2.morphologyEx(no_staff, cv2.MORPH_OPEN, hk)
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
        vert = cv2.morphologyEx(no_staff, cv2.MORPH_OPEN, vk)
        horiz[vert > 0] = 0
        no_staff[horiz > 0] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(no_staff, 8)
    for lid in range(1, num_labels):
        cw = stats[lid, cv2.CC_STAT_WIDTH]
        ch = stats[lid, cv2.CC_STAT_HEIGHT]
        area = stats[lid, cv2.CC_STAT_AREA]
        aspect = cw / max(ch, 1)
        if ch <= 3 and cw > 12 and aspect > 4:
            no_staff[labels == lid] = 0
        elif ch <= 5 and cw > 20 and aspect > 6:
            no_staff[labels == lid] = 0
        elif ch <= 8 and cw > 40 and aspect > 8:
            if area / (cw * ch) < 0.4:
                no_staff[labels == lid] = 0
    return no_staff


def remove_noise(image, min_area=15):
    """Remove small connected components."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, 8)
    clean = np.zeros_like(image)
    for lid in range(1, num_labels):
        if stats[lid, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == lid] = 255
    return clean


def preprocess_single(img_path, output_path, debug_dir=None):
    """Full preprocessing pipeline for one image."""
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Cannot load image: {img_path}")

    fname = os.path.splitext(os.path.basename(img_path))[0]
    h, w = img_gray.shape

    # Auto-rotate portrait -> landscape
    if h > w:
        img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)
        h, w = img_gray.shape
        print(f"  Rotated to landscape: {w}x{h}")
    else:
        print(f"  Size: {w}x{h}")

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{fname}_0_gray.png"), img_gray)

    # Step 1: Adaptive binarization
    binary = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=51, C=15
    )
    noise_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, noise_k)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{fname}_1_binary.png"), binary)

    # Step 2: Find horizontal lines via morphological opening
    line_rows, projection, horiz_mask = find_horizontal_lines(binary)
    print(f"  Found {len(line_rows)} horizontal lines")

    if debug_dir:
        # Save horizontal mask
        cv2.imwrite(os.path.join(debug_dir, f"{fname}_2_horizmask.png"), horiz_mask)
        # Save projection profile
        profile_img = np.zeros((h, 300), dtype=np.uint8)
        max_proj = max(np.max(projection), 1)
        for y in range(h):
            x_val = int(projection[y] / max_proj * 299)
            profile_img[y, :x_val] = 128
        for p in line_rows:
            x_val = int(projection[p] / max_proj * 299)
            profile_img[max(0, p - 1):p + 2, :max(x_val, 1)] = 255
        cv2.imwrite(os.path.join(debug_dir, f"{fname}_2_projection.png"), profile_img)

    # Step 3: Group into 5-line staff systems
    systems = group_into_staff_systems(line_rows, h)
    total_lines = sum(len(s) for s in systems)
    print(f"  Detected {len(systems)} staff systems ({total_lines} lines)")
    for si, system in enumerate(systems):
        spacings = np.diff(system)
        print(f"    System {si+1}: rows {system}, spacings {spacings}, "
              f"avg={np.mean(spacings):.0f}")

    if debug_dir:
        vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
                  (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255)]
        for si, system in enumerate(systems):
            color = colors[si % len(colors)]
            for center in system:
                cv2.line(vis, (0, center), (w, center), color, 2)
            cv2.putText(vis, f"S{si+1}", (10, system[2]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        matched = set()
        for s in systems:
            for c in s:
                matched.add(c)
        for p in line_rows:
            if p not in matched:
                cv2.circle(vis, (w - 30, p), 6, (128, 128, 128), -1)
        cv2.imwrite(os.path.join(debug_dir, f"{fname}_3_systems.png"), vis)

    # Step 4: Remove staff lines
    no_staff = remove_staff_lines(binary, systems)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{fname}_4_removed.png"), no_staff)

    # Step 5: Cleanup remnants
    no_staff = cleanup_remnants(no_staff)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{fname}_5_cleanup.png"), no_staff)

    # Step 6: Remove noise
    clean = remove_noise(no_staff, min_area=15)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{fname}_6_final.png"), clean)

    # Save as 1-bit PNG (white on black = MUSCIMA++ format)
    pil_img = Image.fromarray(clean, mode='L')
    pil_img = pil_img.point(lambda x: 255 if x > 0 else 0, '1')
    pil_img.save(output_path)

    remaining = np.sum(clean > 0)
    print(f"  Result: {remaining} foreground pixels")
    return clean


def main():
    parser = ArgumentParser(description="Preprocess Burkhart Schenkerian images")
    parser.add_argument('--images_dir', default=IMAGES_DIR)
    parser.add_argument('--output_dir', default=OUTPUT_DIR)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    debug_dir = None
    if args.debug:
        debug_dir = os.path.join(args.output_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)

    img_files = sorted([
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    print(f"Processing {len(img_files)} Burkhart images...")
    print(f"Input:  {args.images_dir}")
    print(f"Output: {args.output_dir}")
    print()

    for fname in img_files:
        print(f"--- {fname} ---")
        in_path = os.path.join(args.images_dir, fname)
        out_name = os.path.splitext(fname)[0] + '.png'
        out_path = os.path.join(args.output_dir, out_name)
        try:
            preprocess_single(in_path, out_path, debug_dir)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
        print()

    print("Done!")


if __name__ == "__main__":
    main()
