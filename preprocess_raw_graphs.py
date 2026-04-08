"""Preprocess all raw graph images from the Analysis Dataset.

Steps:
1. Convert PDFs to individual page images
2. Apply handwritten score preprocessing (staff line removal + binarization)
3. Maintain the same directory structure as Raw Graphs

Usage:
    python preprocess_raw_graphs.py
"""

import os
import sys
from argparse import ArgumentParser
from pdf2image import convert_from_path
from preprocess_new_images import preprocess_single

RAW_DIR = "Schenkerian_Data/Analysis Dataset/Raw Graphs"
OUTPUT_DIR = "outputs/preprocessed_raw_graphs"


def convert_pdfs_to_images(raw_dir, images_dir):
    """Convert all PDFs to images, preserving directory structure."""
    converted = 0
    for root, dirs, files in os.walk(raw_dir):
        rel_path = os.path.relpath(root, raw_dir)
        out_subdir = os.path.join(images_dir, rel_path)

        for fname in sorted(files):
            if not fname.lower().endswith('.pdf'):
                continue

            pdf_path = os.path.join(root, fname)
            base = os.path.splitext(fname)[0]
            os.makedirs(out_subdir, exist_ok=True)

            # Check if already converted
            existing = [f for f in os.listdir(out_subdir) if f.startswith(base + '-')]
            if existing:
                print(f"  [skip] {os.path.join(rel_path, fname)} ({len(existing)} pages already exist)")
                converted += len(existing)
                continue

            print(f"  Converting: {os.path.join(rel_path, fname)}")
            try:
                pages = convert_from_path(pdf_path, dpi=300)
                for i, page in enumerate(pages, 1):
                    out_name = f"{base}-{i:02d}.png"
                    out_path = os.path.join(out_subdir, out_name)
                    page.save(out_path, 'PNG')
                print(f"    → {len(pages)} pages")
                converted += len(pages)
            except Exception as e:
                print(f"    ERROR: {e}")

    return converted


def copy_images(raw_dir, images_dir):
    """Copy/link existing JPG/PNG images, preserving directory structure."""
    copied = 0
    for root, dirs, files in os.walk(raw_dir):
        rel_path = os.path.relpath(root, raw_dir)
        out_subdir = os.path.join(images_dir, rel_path)

        for fname in sorted(files):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            src = os.path.join(root, fname)
            os.makedirs(out_subdir, exist_ok=True)
            # Convert to PNG name
            out_name = os.path.splitext(fname)[0] + '.png'
            dst = os.path.join(out_subdir, out_name)

            if os.path.exists(dst):
                print(f"  [skip] {os.path.join(rel_path, fname)}")
                copied += 1
                continue

            # Copy as-is (preprocessing happens later)
            from PIL import Image
            img = Image.open(src)
            img.save(dst)
            print(f"  Copied: {os.path.join(rel_path, fname)}")
            copied += 1

    return copied


def preprocess_all(images_dir, output_dir):
    """Run handwritten preprocessing on all images."""
    processed = 0
    errors = 0
    for root, dirs, files in os.walk(images_dir):
        rel_path = os.path.relpath(root, images_dir)
        out_subdir = os.path.join(output_dir, rel_path)

        img_files = sorted([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not img_files:
            continue

        os.makedirs(out_subdir, exist_ok=True)

        for fname in img_files:
            in_path = os.path.join(root, fname)
            out_name = os.path.splitext(fname)[0] + '.png'
            out_path = os.path.join(out_subdir, out_name)

            if os.path.exists(out_path):
                processed += 1
                continue

            print(f"  Preprocessing: {os.path.join(rel_path, fname)}")
            try:
                preprocess_single(in_path, out_path)
                processed += 1
            except Exception as e:
                print(f"    ERROR: {e}")
                errors += 1

    return processed, errors


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--raw_dir', default=RAW_DIR)
    parser.add_argument('--output_dir', default=OUTPUT_DIR)
    parser.add_argument('--step', choices=['pdf', 'preprocess', 'all'], default='all',
                        help='Which step to run: pdf (convert PDFs only), preprocess, or all')
    args = parser.parse_args()

    # Intermediate dir for converted PDF pages + copied images
    images_dir = os.path.join(args.output_dir, 'images')

    if args.step in ['pdf', 'all']:
        print("=" * 60)
        print("Step 1: Converting PDFs to images")
        print("=" * 60)
        n = convert_pdfs_to_images(args.raw_dir, images_dir)
        print(f"\nConverted/found {n} PDF pages")

        print("\n" + "=" * 60)
        print("Step 2: Copying existing images")
        print("=" * 60)
        n = copy_images(args.raw_dir, images_dir)
        print(f"\nCopied/found {n} images")

    if args.step in ['preprocess', 'all']:
        print("\n" + "=" * 60)
        print("Step 3: Preprocessing all images")
        print("=" * 60)
        processed, errors = preprocess_all(images_dir, os.path.join(args.output_dir, 'preprocessed'))
        print(f"\nDone! Processed: {processed}, Errors: {errors}")
        print(f"Output: {os.path.join(args.output_dir, 'preprocessed')}")
