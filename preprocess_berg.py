"""Preprocess Berg images with tuned parameters.

Based on preprocess_new_images.py (Rachmaninov style) but with gentler
thresholds to preserve more handwritten content.

Method 6: Gentler intensity separation (ink_threshold=140, strict-20)
Method 7: Even gentler (ink_threshold=150, strict-15, wider vertical check)
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_new_images import detect_staff_rows


def remove_staff_gentle(img_gray, ink_threshold=140, strict_gap=20, vert_range=10):
    """Gentler staff line removal that preserves more ink content."""
    h, w = img_gray.shape

    ink_mask = img_gray < ink_threshold
    paper_threshold = 200
    medium_mask = (img_gray >= ink_threshold) & (img_gray < paper_threshold)

    result = np.zeros((h, w), dtype=np.uint8)
    result[ink_mask] = 255

    all_foreground = np.zeros((h, w), dtype=np.uint8)
    all_foreground[img_gray < paper_threshold] = 255
    staff_rows = detect_staff_rows(all_foreground, w)

    # Keep medium-gray pixels NOT on staff rows
    medium_not_staff = medium_mask & ~staff_rows[:, np.newaxis]
    result[medium_not_staff] = 255

    # At staff rows, use a gentler strict threshold
    strict_threshold = ink_threshold - strict_gap

    # Pre-compute vertical structure mask for fast connectivity check
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_range))
    vert_mask = cv2.morphologyEx(result, cv2.MORPH_OPEN, vk)

    for y in range(h):
        if not staff_rows[y]:
            continue
        removable = (result[y] > 0) & (img_gray[y] >= strict_threshold)
        keep = vert_mask[y] > 0
        result[y, removable & ~keep] = 0

    # Lighter morphological cleanup
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horiz = cv2.morphologyEx(result, cv2.MORPH_OPEN, hk)
    vk2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vert2 = cv2.morphologyEx(result, cv2.MORPH_OPEN, vk2)
    staff_only = horiz.copy()
    staff_only[vert2 > 0] = 0
    result[staff_only > 0] = 0

    # Gentler aspect ratio filter
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, 8)
    for lid in range(1, num_labels):
        cw = stats[lid, cv2.CC_STAT_WIDTH]
        ch = stats[lid, cv2.CC_STAT_HEIGHT]
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < 15:
            result[labels == lid] = 0
        elif ch <= 3 and cw > 60 and cw / max(ch, 1) > 15:
            result[labels == lid] = 0

    return result


def preprocess_single(img_path, output_path, ink_threshold=140, strict_gap=20, vert_range=10):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    no_staff = remove_staff_gentle(img, ink_threshold, strict_gap, vert_range)

    # Content region mask
    blurred = cv2.GaussianBlur(img, (51, 51), 0)
    _, content_mask = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)
    content_mask = cv2.dilate(content_mask, np.ones((30, 30), np.uint8))
    no_staff[content_mask == 0] = 0

    # Remove small noise
    n, labels, stats, _ = cv2.connectedComponentsWithStats(no_staff, 8)
    clean = np.zeros_like(no_staff)
    for lid in range(1, n):
        if stats[lid, cv2.CC_STAT_AREA] >= 15:
            clean[labels == lid] = 255

    pil = Image.fromarray(clean, mode='L')
    pil = pil.point(lambda x: 255 if x > 0 else 0, '1')
    pil.save(output_path)
    return np.sum(clean > 0)


if __name__ == "__main__":
    RAW = "outputs/raw_images/Timothy Jackson/Berg"
    BASE = "outputs/preprocessed/Timothy Jackson/Berg"

    configs = {
        'method6_gentle': {'ink_threshold': 140, 'strict_gap': 20, 'vert_range': 10},
        'method7_softer': {'ink_threshold': 150, 'strict_gap': 15, 'vert_range': 12},
    }

    files = sorted([f for f in os.listdir(RAW) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    for method_name, params in configs.items():
        out_dir = os.path.join(BASE, method_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {method_name} (thresh={params['ink_threshold']}, gap={params['strict_gap']}) ===", flush=True)

        for i, f in enumerate(files):
            src = os.path.join(RAW, f)
            name = os.path.splitext(f)[0] + '.png'
            dst = os.path.join(out_dir, name)
            fg = preprocess_single(src, dst, **params)
            print(f"  [{i+1}/{len(files)}] {f[:50]}: {fg} fg", flush=True)

    print("\nDone!")
