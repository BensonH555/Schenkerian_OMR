"""Merge GT staff nodes with completeOMR predictions, then convert to MIDI.

Fixes the 'attached to more than one staffline/staffspace' issue by ensuring
each notehead links to only its NEAREST staffLine/staffSpace.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys
import yaml
import json
import copy
import numpy as np
import logging

logging.disable(logging.WARNING)

from mung.io import read_nodes_from_file
from mung.node import Node
from mung2midi.inference import PitchInferenceEngine, OnsetsInferenceEngine
from mung2midi.run_inference import convert_mung_to_midi

# Paths
PRED_ANN = '/home/users/yh477/lab/Schenkerian_OMR/data/MUSCIMA++/v2.0_gen/data/annotations'
GT_ANN = '/home/users/yh477/lab/Schenkerian_OMR/data/MUSCIMA++/v2.0/data/annotations'
SPLIT_FILE = '/home/users/yh477/lab/Schenkerian_OMR/splits/mob_split.yaml'
OUT_DIR = '/home/users/yh477/lab/Schenkerian_OMR/midi_output/pred_merged_v2'
GT_OUT_DIR = '/home/users/yh477/lab/Schenkerian_OMR/midi_output/gt_v2'

STAFF_CLASSES = {'staff', 'staffLine', 'staffSpace'}
# mung2midi's actual notehead class names (from _CONST.NOTEHEAD_CLASS_NAMES)
NOTEHEAD_CLASSES = {'noteheadFull', 'noteheadHalf', 'noteheadFullSmall',
                    'noteheadHalfSmall', 'noteheadWhole',
                    'notehead-full', 'notehead-empty'}

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GT_OUT_DIR, exist_ok=True)

with open(SPLIT_FILE) as f:
    split = yaml.safe_load(f)


def node_center(node):
    return ((node.top + node.bottom) / 2.0, (node.left + node.right) / 2.0)


def distance(n1, n2):
    c1 = node_center(n1)
    c2 = node_center(n2)
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5


def merge_staff_nodes(pred_nodes, gt_nodes):
    """Merge GT staff/staffLine/staffSpace nodes directly into predicted node list.

    No ID remapping — GT staff IDs are typically already above pred max ID.
    Each notehead gets outlink to exactly ONE nearest staffLine/staffSpace.
    """
    # Collect GT staff-related nodes (keep original IDs and internal links)
    staff_nodes = [n for n in gt_nodes if n.class_name in STAFF_CLASSES]
    if not staff_nodes:
        return pred_nodes

    # Check for ID conflicts and rebuild staff nodes with new IDs if needed
    pred_ids = {n.id for n in pred_nodes}
    staff_ids_set = {n.id for n in staff_nodes}
    conflicts = pred_ids & staff_ids_set
    if conflicts:
        max_id = max(pred_ids)
        id_remap = {}
        for sn in staff_nodes:
            if sn.id in conflicts:
                max_id += 1
                id_remap[sn.id] = max_id
        # Rebuild staff nodes with remapped IDs
        new_staff = []
        for sn in staff_nodes:
            new_id = id_remap.get(sn.id, sn.id)
            new_out = [id_remap.get(oid, oid) for oid in sn.outlinks]
            new_in = [id_remap.get(iid, iid) for iid in sn.inlinks]
            new_staff.append(Node(
                id_=new_id, class_name=sn.class_name,
                top=sn.top, left=sn.left,
                width=sn.width, height=sn.height,
                outlinks=new_out, inlinks=new_in,
            ))
        staff_nodes = new_staff

    # Separate staff types
    staff_group_nodes = [n for n in staff_nodes if n.class_name == 'staff']
    staffline_nodes = [n for n in staff_nodes if n.class_name in ('staffLine', 'staffSpace')]

    # Build staff -> children (staffLine/staffSpace) mapping
    staff_id_to_node = {n.id: n for n in staff_nodes}
    staff_children = {}  # staff_id -> list of staffLine/staffSpace nodes
    for sg in staff_group_nodes:
        children = []
        for oid in sg.outlinks:
            child = staff_id_to_node.get(oid)
            if child and child.class_name in ('staffLine', 'staffSpace'):
                children.append(child)
        # Also check inlinks (some graphs use inlinks for staff->staffLine)
        for iid in sg.inlinks:
            child = staff_id_to_node.get(iid)
            if child and child.class_name in ('staffLine', 'staffSpace') and child not in children:
                children.append(child)
        # If no explicit links, find staffLines by spatial overlap
        if not children:
            for sl in staffline_nodes:
                if (sl.top >= sg.top - 5 and sl.bottom <= sg.bottom + 5 and
                    abs((sl.left + sl.right)/2 - (sg.left + sg.right)/2) < sg.width):
                    children.append(sl)
        staff_children[sg.id] = children

    # For each predicted node, add links to nearest staff and staffLine
    for pn in pred_nodes:
        # Link all symbols to nearest staff group
        if staff_group_nodes:
            best_staff = min(staff_group_nodes, key=lambda s: distance(pn, s))
            if best_staff.id not in pn.outlinks:
                pn.outlinks.append(best_staff.id)

            # For noteheads: link to nearest staffLine WITHIN the assigned staff
            if pn.class_name in NOTEHEAD_CLASSES:
                # Remove any existing staffLine/staffSpace links
                sl_ids = {n.id for n in staffline_nodes}
                pn.outlinks = [oid for oid in pn.outlinks if oid not in sl_ids]

                # Get staffLines belonging to this staff
                children = staff_children.get(best_staff.id, [])
                if children:
                    nearest_sl = min(children, key=lambda s: distance(pn, s))
                    pn.outlinks.append(nearest_sl.id)

    merged = list(pred_nodes) + staff_nodes

    # Clean broken references: remove outlinks/inlinks pointing to non-existent nodes
    all_ids = {n.id for n in merged}
    for n in merged:
        n.outlinks = [oid for oid in n.outlinks if oid in all_ids]
        n.inlinks = [iid for iid in n.inlinks if iid in all_ids]

    return merged


def try_convert(nodes, name, out_dir, suffix=''):
    """Try converting nodes to MIDI. Returns True on success."""
    try:
        from midiutil import MIDIFile

        pie = PitchInferenceEngine()
        pitches, pitch_names = pie.infer_pitches(nodes, with_names=True)

        oie = OnsetsInferenceEngine(nodes)
        durations = oie.durations(nodes)
        onsets = oie.onsets(nodes)

        # Build MIDI
        midi = MIDIFile(1, deinterleave=False)
        midi.addTempo(0, 0, 120)

        mapping = {}
        for node_id, pitch in pitches.items():
            if node_id in onsets and node_id in durations:
                onset = onsets[node_id]
                dur = durations[node_id]
                if isinstance(dur, list):
                    dur = dur[0] if dur else 1.0
                midi.addNote(0, 0, int(pitch), float(onset), float(dur), 100)
                mapping[str(node_id)] = {
                    'pitch': int(pitch),
                    'pitch_name': pitch_names.get(node_id, ''),
                    'onset': float(onset),
                    'duration': float(dur)
                }

        mid_path = os.path.join(out_dir, f'{name}{suffix}.mid')
        with open(mid_path, 'wb') as f:
            midi.writeFile(f)

        json_path = os.path.join(out_dir, f'{name}{suffix}_mapping.json')
        with open(json_path, 'w') as f:
            json.dump(mapping, f, indent=2)

        return True
    except Exception as e:
        print(f'  FAIL {name}: {type(e).__name__}: {e}')
        return False


# Run on test split
print("=" * 60)
print("Part 1: GT with edge case fixes")
print("=" * 60)
gt_success = 0
for name in split['test']:
    gt_xml = os.path.join(GT_ANN, f'{name}.xml')
    if not os.path.exists(gt_xml):
        continue
    nodes = read_nodes_from_file(gt_xml)
    if try_convert(nodes, name, GT_OUT_DIR, '_gt'):
        gt_success += 1

print(f"\nGT: {gt_success}/{len(split['test'])} success")

print("\n" + "=" * 60)
print("Part 2: Predicted + GT staff merge")
print("=" * 60)
pred_success = 0
pred_total = 0
for name in split['test']:
    pred_xml = os.path.join(PRED_ANN, f'{name}.xml')
    gt_xml = os.path.join(GT_ANN, f'{name}.xml')
    if not os.path.exists(pred_xml) or not os.path.exists(gt_xml):
        print(f'  SKIP {name}: missing files')
        continue

    pred_total += 1
    pred_nodes = read_nodes_from_file(pred_xml)
    gt_nodes = read_nodes_from_file(gt_xml)

    merged = merge_staff_nodes(pred_nodes, gt_nodes)
    if try_convert(merged, name, OUT_DIR, '_pred'):
        pred_success += 1

print(f"\nPredicted+Staff: {pred_success}/{pred_total} success")
print(f"\nDone! GT results in {GT_OUT_DIR}")
print(f"Predicted results in {OUT_DIR}")
