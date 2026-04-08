"""Preprocess Edward Laufer Schenkerian analysis images.

Complete pipeline:
1. Convert PDFs to images (300 DPI)
2. Auto-orient: ensure staff lines are horizontal
3. Remove staff lines using two methods:
   - Method A: Morphological (detect 5-line groups, erase by position)
   - Method B: Intensity-based (staff lines lighter than handwritten ink)
4. Save results

Output structure:
  outputs/preprocessed/Edward Laufer/method_a/  (morphological)
  outputs/preprocessed/Edward Laufer/method_b/  (intensity-based)
"""

import os
import glob
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_DIR, "outputs/raw_images/Edward Laufer")
OUT_A = os.path.join(PROJECT_DIR, "outputs/preprocessed/Edward Laufer/method_a")
OUT_B = os.path.join(PROJECT_DIR, "outputs/preprocessed/Edward Laufer/method_b")
DEBUG_DIR = os.path.join(PROJECT_DIR, "outputs/preprocessed/Edward Laufer/debug")


# ─── Step 1: PDF to images ───

def convert_all_pdfs(raw_dir):
    """Convert all PDFs to PNG, remove PDFs."""
    pdfs = glob.glob(os.path.join(raw_dir, "**/*.pdf"), recursive=True) + \
           glob.glob(os.path.join(raw_dir, "**/*.PDF"), recursive=True)
    for pdf in sorted(set(pdfs)):
        print(f"  Converting PDF: {os.path.basename(pdf)}")
        pages = convert_from_path(pdf, dpi=300)
        base = os.path.splitext(pdf)[0]
        for i, page in enumerate(pages):
            suffix = f"-{i+1:02d}" if len(pages) > 1 else ""
            out = f"{base}{suffix}.png"
            page.save(out, "PNG")
        os.remove(pdf)


# ─── Step 2: Orientation fix ───

def horiz_line_score(gray):
    """Score horizontal line strength via morphological opening."""
    h, w = gray.shape
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=51, C=15)
    klen = max(w // 12, 80)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    return np.sum(horiz > 0)


def fix_orientation(img_path):
    """Ensure staff lines are horizontal. Returns corrected grayscale image.

    Strategy: test current orientation and 90° CW rotation.
    Pick the one with stronger horizontal lines.
    Then check 180° flip for upside-down.
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Downscale for fast scoring
    h, w = gray.shape
    scale = min(1500 / max(h, w), 1.0)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)))

    # Score at 0° and 90° CW
    score_0 = horiz_line_score(small)
    small_90 = cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)
    score_90 = horiz_line_score(small_90)

    if score_90 > score_0 * 1.2:
        # 90° rotation is better, but which direction?
        # Try CW and CCW, pick the one where content is in upper half (right-side-up)
        gray_cw = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        gray_ccw = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img_ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Use text density in top 20% to decide direction
        def top_density(g):
            hh = g.shape[0]
            top = g[:int(hh * 0.2), :]
            return np.sum(top < 180)  # dark pixels = content

        if top_density(gray_cw) >= top_density(gray_ccw):
            return img_cw, gray_cw, "rot90cw"
        else:
            return img_ccw, gray_ccw, "rot90ccw"
    else:
        # Already landscape-ish, check for upside-down
        small_180 = cv2.rotate(small, cv2.ROTATE_180)

        def top_text_score(g):
            hh = g.shape[0]
            top = g[:int(hh * 0.18), :]
            binary = cv2.adaptiveThreshold(
                top, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, blockSize=51, C=15)
            n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            text_count = 0
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                ch = stats[i, cv2.CC_STAT_HEIGHT]
                cw = stats[i, cv2.CC_STAT_WIDTH]
                if 30 < area < 8000 and ch < 120 and cw < 300:
                    text_count += 1
            return text_count

        score_up = top_text_score(small)
        score_flip = top_text_score(small_180)

        if score_flip > score_up * 1.5 and score_flip > 3:
            gray = cv2.rotate(gray, cv2.ROTATE_180)
            img = cv2.rotate(img, cv2.ROTATE_180)
            return img, gray, "rot180"

        return img, gray, "ok"


# ─── Step 3a: Morphological staff line removal ───

def find_staff_lines(binary):
    """Detect horizontal line positions using morphological + run-length."""
    h, w = binary.shape

    horiz_len = max(w // 15, 80)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    horiz_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    morph_proj = np.sum(horiz_mask > 0, axis=1).astype(np.float64)

    # Run-length
    min_run = max(w // 50, 20)
    run_proj = np.zeros(h)
    for y in range(h):
        row = binary[y]
        if np.sum(row > 0) == 0:
            continue
        padded = np.concatenate([[0], row, [0]])
        diffs = np.diff((padded > 0).astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        runs = ends - starts
        long = runs[runs >= min_run]
        if len(long) > 0:
            run_proj[y] = np.sum(long)

    morph_norm = morph_proj / max(np.max(morph_proj), 1)
    run_norm = run_proj / max(np.max(run_proj), 1)
    combined = np.maximum(morph_norm, run_norm * 0.8)
    combined = gaussian_filter1d(combined, sigma=1.5)

    peaks, _ = find_peaks(combined, height=0.12, distance=5, prominence=0.025)

    min_cov = w * 0.04
    valid = [p for p in peaks if morph_proj[p] >= min_cov or run_proj[p] >= min_cov]
    peaks = np.array(valid) if valid else np.array([], dtype=int)

    # Merge close peaks
    if len(peaks) > 1:
        merged = []
        group = [peaks[0]]
        for i in range(1, len(peaks)):
            if peaks[i] - group[-1] <= 4:
                group.append(peaks[i])
            else:
                merged.append(group[np.argmax(combined[group])])
                group = [peaks[i]]
        merged.append(group[np.argmax(combined[group])])
        peaks = np.array(merged)

    return peaks, horiz_mask


def group_staff_systems(line_rows, h, min_sp=15, max_sp=100):
    """Group lines into 5-line staff systems."""
    if len(line_rows) < 5:
        return []
    lines = np.sort(line_rows)
    n = len(lines)
    candidates = []

    for i in range(n - 4):
        for j in range(i + 1, n - 3):
            s1 = lines[j] - lines[i]
            if s1 < min_sp or s1 > max_sp:
                continue
            group = [i, j]
            target = s1
            for k in range(j + 1, min(n, j + 25)):
                expected = lines[group[-1]] + target
                actual = lines[k]
                if abs(actual - expected) <= target * 0.3:
                    group.append(k)
                    target = (lines[group[-1]] - lines[group[0]]) / (len(group) - 1)
                    if len(group) == 5:
                        break
                elif actual > expected + target * 0.5:
                    break
            if len(group) == 5:
                pos = lines[group]
                spacings = np.diff(pos)
                cv = np.std(spacings) / np.mean(spacings)
                if cv < 0.2:
                    candidates.append((cv, pos.copy()))

    candidates.sort(key=lambda x: x[0])
    systems = []
    used = set()
    for cv, pos in candidates:
        if any(abs(p - u) < 12 for p in pos for u in used):
            continue
        systems.append(pos)
        for p in pos:
            used.add(p)
    systems.sort(key=lambda s: s[0])
    return systems


def estimate_thickness(binary, row):
    """Estimate staff line thickness at given row."""
    h, w = binary.shape
    if row < 0 or row >= h:
        return 2
    fg = np.where(binary[row, :] > 0)[0]
    if len(fg) == 0:
        return 2
    sample = fg[::max(1, len(fg) // 30)]
    thicknesses = []
    for x in sample:
        yu = row
        while yu > 0 and binary[yu - 1, x] > 0:
            yu -= 1
        yd = row
        while yd < h - 1 and binary[yd + 1, x] > 0:
            yd += 1
        thicknesses.append(yd - yu + 1)
    return int(np.median(thicknesses)) if thicknesses else 2


def remove_staff_morph(binary, systems):
    """Method A: Remove staff lines by position, restore vertical structures."""
    result = binary.copy()
    h, w = binary.shape

    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    vert_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)

    for system in systems:
        for center in system:
            thick = estimate_thickness(binary, center)
            half = max(thick // 2 + 1, 2)
            y0 = max(0, center - half)
            y1 = min(h, center + half + 1)
            result[y0:y1, :] = 0
            result[y0:y1, :] = cv2.bitwise_or(result[y0:y1, :],
                                               vert_mask[y0:y1, :])

    # Catch remaining horizontal lines not in systems
    hlen = max(w // 30, 50)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1))
    remaining = cv2.morphologyEx(result, cv2.MORPH_OPEN, hk)
    ek = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    remaining = cv2.dilate(remaining, ek, iterations=1)
    remaining = cv2.bitwise_and(remaining, result)
    remaining[vert_mask > 0] = 0
    result = cv2.subtract(result, remaining)
    return result


# ─── Step 3b: Intensity-based staff line removal ───

def remove_staff_intensity(gray, binary):
    """Method B: Use grayscale intensity to separate staff from ink.

    Staff lines (pre-printed) tend to be lighter gray; handwritten ink is darker.
    Vectorized implementation for speed.
    """
    h, w = gray.shape

    # Ink: very dark pixels (< 110)
    ink_mask = (gray < 110).astype(np.uint8) * 255

    # Staff candidate: medium gray (110-190)
    staff_cand = ((gray >= 110) & (gray < 190)).astype(np.uint8) * 255

    # Find staff-heavy rows from projection
    row_proj = np.sum(staff_cand > 0, axis=1)
    threshold = np.percentile(row_proj[row_proj > 0], 70) if np.any(row_proj > 0) else w * 0.1
    staff_rows_mask = row_proj > threshold

    result = binary.copy()

    # Build masks for vertical connectivity check (vectorized)
    # For each pixel, check if ink exists above/below within 10px in non-staff rows
    non_staff_ink = np.zeros_like(ink_mask)
    for y in range(h):
        if not staff_rows_mask[y]:
            non_staff_ink[y] = ink_mask[y]

    # Dilate non_staff_ink vertically to create "reachable" mask
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    reachable_above = cv2.dilate(non_staff_ink, vk, iterations=1)
    reachable_below = cv2.dilate(non_staff_ink, vk, iterations=1)

    # Pixels to remove: on staff rows, not dark ink, not connecting to ink above AND below
    for y in range(h):
        if not staff_rows_mask[y]:
            continue
        # Mask: pixels that are foreground, not dark ink, on a staff row
        removable = (result[y] > 0) & (gray[y] >= 110)
        # Keep if connected to non-staff ink both above and below
        connected = (reachable_above[y] > 0) & (reachable_below[y] > 0)
        remove = removable & ~connected
        result[y, remove] = 0

    return result


# ─── Cleanup ───

def cleanup(image, min_area=15):
    """Remove small noise and thin horizontal fragments."""
    # Morphological cleanup
    for klen in [10, 18, 30]:
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        horiz = cv2.morphologyEx(image, cv2.MORPH_OPEN, hk)
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
        vert = cv2.morphologyEx(image, cv2.MORPH_OPEN, vk)
        horiz[vert > 0] = 0
        image[horiz > 0] = 0

    # Connected component filtering
    n, labels, stats, _ = cv2.connectedComponentsWithStats(image, 8)
    clean = np.zeros_like(image)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = cw / max(ch, 1)
        if area < min_area:
            continue
        if ch <= 3 and cw > 12 and aspect > 4:
            continue
        if ch <= 5 and cw > 20 and aspect > 6:
            continue
        clean[labels == i] = 255
    return clean


def save_1bit(image, path):
    """Save as 1-bit white-on-black PNG."""
    pil = Image.fromarray(image, mode='L')
    pil = pil.point(lambda x: 255 if x > 0 else 0, '1')
    pil.save(path)


# ─── Main pipeline ───

def preprocess_single(img_path, out_a_path, out_b_path, debug_dir=None):
    """Full pipeline for one image."""
    fname = os.path.splitext(os.path.basename(img_path))[0]

    # Orient
    img_color, gray, orient_status = fix_orientation(img_path)
    h, w = gray.shape
    print(f"    Orient: {orient_status} -> {w}x{h}")

    # Binarize
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=51, C=15)
    nk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, nk)

    # Find staff lines
    line_rows, horiz_mask = find_staff_lines(binary)
    systems = group_staff_systems(line_rows, h)
    total_lines = sum(len(s) for s in systems)
    print(f"    Found {len(line_rows)} lines -> {len(systems)} staff systems ({total_lines} lines)")

    if debug_dir:
        # Save oriented gray (downscaled for debug)
        scale = min(1800 / max(h, w), 1.0)
        dbg_gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
        cv2.imwrite(os.path.join(debug_dir, f"{fname}_0_oriented.png"), dbg_gray)

        # Save systems visualization
        vis = cv2.resize(img_color, (int(w * scale), int(h * scale)))
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
                  (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255)]
        for si, system in enumerate(systems):
            color = colors[si % len(colors)]
            for center in system:
                y_scaled = int(center * scale)
                cv2.line(vis, (0, y_scaled), (vis.shape[1], y_scaled), color, 1)
        cv2.imwrite(os.path.join(debug_dir, f"{fname}_1_systems.png"), vis)

    # Method A: morphological removal
    result_a = remove_staff_morph(binary, systems)
    result_a = cleanup(result_a)
    save_1bit(result_a, out_a_path)

    # Method B: intensity-based removal
    result_b = remove_staff_intensity(gray, binary)
    result_b = cleanup(result_b)
    save_1bit(result_b, out_b_path)

    print(f"    Method A: {np.sum(result_a > 0)} fg pixels")
    print(f"    Method B: {np.sum(result_b > 0)} fg pixels")


def main():
    print("=== Preprocessing Edward Laufer ===\n")

    # Collect all images (PDFs already converted)
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(RAW_DIR, "**", ext), recursive=True))
    images = sorted(set(images))
    print(f"\nStep 2-4: Processing {len(images)} images...\n")

    os.makedirs(OUT_A, exist_ok=True)
    os.makedirs(OUT_B, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    for i, img_path in enumerate(images):
        rel = os.path.relpath(img_path, RAW_DIR)
        fname = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        print(f"[{i+1}/{len(images)}] {rel}")

        # Flatten subdirectory structure in output names
        flat_name = rel.replace(os.sep, "_")
        flat_name = os.path.splitext(flat_name)[0] + ".png"

        out_a = os.path.join(OUT_A, flat_name)
        out_b = os.path.join(OUT_B, flat_name)

        try:
            preprocess_single(img_path, out_a, out_b, DEBUG_DIR)
        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
        print()

    print("Done!")


if __name__ == "__main__":
    main()
