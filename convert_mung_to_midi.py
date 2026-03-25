#!/usr/bin/env python
"""Script to test mung2midi conversion on ground truth and predicted MuNG files."""
import os
import sys
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Create output dir
output_dir = "/home/users/yh477/lab/Schenkerian_OMR/midi_output"
os.makedirs(output_dir, exist_ok=True)

# Check imports
print("=" * 60)
print("Step 1: Checking imports")
print("=" * 60)

try:
    from mung.io import read_nodes_from_file
    print("  [OK] mung.io.read_nodes_from_file")
except ImportError as e:
    print(f"  [FAIL] mung.io: {e}")
    sys.exit(1)

try:
    from mung2midi.run_inference import convert_mung_to_midi, build_midi
    print("  [OK] mung2midi.run_inference.convert_mung_to_midi")
except ImportError as e:
    print(f"  [FAIL] mung2midi: {e}")
    sys.exit(1)

try:
    from midiutil import MIDIFile
    print("  [OK] midiutil.MIDIFile")
except ImportError as e:
    print(f"  [FAIL] midiutil: {e}")
    sys.exit(1)


def save_midi(midi_file, output_path):
    """Save a MIDIFile object to disk."""
    with open(output_path, 'wb') as f:
        midi_file.writeFile(f)
    fsize = os.path.getsize(output_path)
    print(f"  Saved MIDI: {output_path} ({fsize} bytes)")


def try_convert(xml_path, label):
    """Try to convert a MuNG XML to MIDI, returning success/failure info."""
    print(f"\n{'=' * 60}")
    print(f"Converting: {label}")
    print(f"  File: {xml_path}")
    print(f"{'=' * 60}")

    if not os.path.exists(xml_path):
        print(f"  [SKIP] File does not exist")
        return False

    # Read nodes
    try:
        nodes = read_nodes_from_file(xml_path)
        print(f"  [OK] Read {len(nodes)} nodes")
    except Exception as e:
        print(f"  [FAIL] Could not read nodes: {e}")
        return False

    # Print class summary
    class_counts = {}
    for n in nodes:
        cn = n.class_name
        class_counts[cn] = class_counts.get(cn, 0) + 1

    # Show key classes
    key_classes = ['staff', 'staffLine', 'staffSpace', 'staffGrouping',
                   'noteheadFull', 'noteheadHalf', 'noteheadWhole',
                   'gClef', 'fClef', 'cClef',
                   'keySignature', 'timeSignature',
                   'stem', 'beam', 'flag8thUp', 'flag8thDown',
                   'measureSeparator', 'thinBarline']
    print(f"  Key class counts:")
    for cls in key_classes:
        if cls in class_counts:
            print(f"    {cls}: {class_counts[cls]}")

    # Check for staff/staffLine (critical for pitch inference)
    has_staff = 'staff' in class_counts
    has_staffLine = 'staffLine' in class_counts
    has_staffSpace = 'staffSpace' in class_counts
    print(f"  Has staff: {has_staff}, staffLine: {has_staffLine}, staffSpace: {has_staffSpace}")

    # Try conversion
    try:
        midi_file = convert_mung_to_midi(nodes)
        if midi_file is None:
            print(f"  [FAIL] convert_mung_to_midi returned None (inference failed)")
            return False
        print(f"  [OK] MIDI conversion succeeded")

        # Save
        basename = os.path.splitext(os.path.basename(xml_path))[0]
        midi_path = os.path.join(output_dir, f"{label}_{basename}.mid")
        save_midi(midi_file, midi_path)
        return True

    except Exception as e:
        print(f"  [FAIL] Conversion error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ============================================================
# Step 2: Convert a ground truth file
# ============================================================
print("\n\n" + "#" * 60)
print("Step 2: Ground Truth Conversion")
print("#" * 60)

gt_file = "/home/users/yh477/lab/Schenkerian_OMR/data/MUSCIMA++/v2.0/data/annotations/CVC-MUSCIMA_W-01_N-10_D-ideal.xml"
gt_success = try_convert(gt_file, "gt")

# ============================================================
# Step 3: Convert a model-predicted file (same document)
# ============================================================
print("\n\n" + "#" * 60)
print("Step 3: Model-Predicted Conversion")
print("#" * 60)

pred_file = "/home/users/yh477/lab/Schenkerian_OMR/data/MUSCIMA++/v2.0_gen/data/annotations/CVC-MUSCIMA_W-01_N-10_D-ideal.xml"
if os.path.exists(pred_file):
    pred_success = try_convert(pred_file, "pred")
else:
    # Try a test-split file that exists in v2.0_gen
    print(f"  Predicted file {pred_file} not found, trying another...")
    # Try files from test split
    test_files = [
        "CVC-MUSCIMA_W-01_N-10_D-ideal",
        "CVC-MUSCIMA_W-01_N-14_D-ideal",
        "CVC-MUSCIMA_W-03_N-07_D-ideal",
        "CVC-MUSCIMA_W-22_N-10_D-ideal",
    ]
    pred_success = False
    for tf in test_files:
        pred_file = f"/home/users/yh477/lab/Schenkerian_OMR/data/MUSCIMA++/v2.0_gen/data/annotations/{tf}.xml"
        if os.path.exists(pred_file):
            pred_success = try_convert(pred_file, "pred")
            break
    if not pred_success:
        # Just try the first available file
        import glob
        pred_files = sorted(glob.glob("/home/users/yh477/lab/Schenkerian_OMR/data/MUSCIMA++/v2.0_gen/data/annotations/*.xml"))
        if pred_files:
            pred_success = try_convert(pred_files[0], "pred")


# ============================================================
# Summary
# ============================================================
print("\n\n" + "#" * 60)
print("SUMMARY")
print("#" * 60)
print(f"  Ground truth conversion: {'SUCCESS' if gt_success else 'FAILED'}")
print(f"  Predicted conversion:    {'SUCCESS' if pred_success else 'FAILED'}")
print(f"  Output directory: {output_dir}")
if os.path.exists(output_dir):
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        print(f"    {f} ({os.path.getsize(fpath)} bytes)")
