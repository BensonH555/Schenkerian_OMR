"""Run completeOMR inference on new images (no ground truth needed).

Steps:
1. Preprocess: convert to 1-bit grayscale PNG, resize if needed
2. YOLOv8 detection: tiled inference → detected symbols
3. MLP assembler: pairwise edge prediction → final MuNG XML
4. Visualization: draw bounding boxes and edges on images

Usage:
    python infer_new_images.py --input_dir /path/to/images --output_dir /path/to/output
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mung.io import write_nodes_to_file
from mung.node import Node
from tqdm import tqdm
from argparse import ArgumentParser

# Add completeOMR to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import YOLOSoft, MLP, MLPwithSoftClass
from utils.constants import get_classlist_and_classdict
from configs.assembler.default import get_cfg_defaults
from ultralytics.utils.plotting import Colors

import yaml

# ---- Tiling parameters (same as detector_infer.py) ----
PATCH_SIZE = 1216
MARGIN = 128
STEP_SIZE = PATCH_SIZE - 2 * MARGIN


def bbox_iop(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    inter = max(min(b1_x2, b2_x2) - max(b1_x1, b2_x1), 0) * max(
        min(b1_y2, b2_y2) - max(b1_y1, b2_y1), 0)
    return inter / (w1 * h1)


def preprocess_image(img_path, output_path):
    """Convert image to format suitable for completeOMR detection.

    Uses the full preprocessing pipeline from preprocess_new_images.py:
    binarization, staff line removal, and noise cleanup.
    """
    from preprocess_new_images import preprocess_single
    preprocess_single(img_path, output_path)
    return Image.open(output_path)


def run_detection(model, images_dir, batch_size=8):
    """Run tiled YOLOv8 detection on all images in directory."""
    images = []
    indices = []

    img_files = sorted([f for f in os.listdir(images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for img_name in tqdm(img_files, desc='Splitting images into tiles'):
        img = Image.open(os.path.join(images_dir, img_name))
        W, H = img.size
        x_steps = max(1, (W - 1) // STEP_SIZE + 1)
        y_steps = max(1, (H - 1) // STEP_SIZE + 1)
        for x_id in range(x_steps):
            for y_id in range(y_steps):
                offset = (x_id * STEP_SIZE - MARGIN, y_id * STEP_SIZE - MARGIN)
                subimage = img.crop((offset[0], offset[1],
                                     offset[0] + PATCH_SIZE, offset[1] + PATCH_SIZE))
                images.append(subimage)
                indices.append((img_name, x_id, y_id))

    # Run detection in batches
    results = []
    for idx in tqdm(range(0, len(images), batch_size), desc='Running YOLOv8 detection'):
        batch_imgs = images[idx:min(idx + batch_size, len(images))]
        results.extend(model(batch_imgs, verbose=False))

    # Merge overlapping detections
    imgname2preds = {}
    for index, result in zip(indices, results):
        img_name, x_id, y_id = index
        offset = (x_id * STEP_SIZE - MARGIN, y_id * STEP_SIZE - MARGIN)
        if img_name not in imgname2preds:
            imgname2preds[img_name] = {}
        preds_dict = imgname2preds[img_name]
        preds_list = []
        preds_dict[(x_id, y_id)] = preds_list

        def remove_truncated(plist, true_box, prob):
            to_remove = None
            for idx, (box_pre, prob_pre) in enumerate(plist):
                if bbox_iop(box_pre, true_box) * np.dot(prob.data, prob_pre.data) > 0.20:
                    to_remove = idx
                    break
            if to_remove:
                del plist[to_remove]

        for box, prob in zip(result.boxes, result.probs):
            bbox = box.xyxy.squeeze().cpu().numpy()
            probdata = prob.cpu().numpy()
            left, top, right, bottom = bbox
            if (right <= 2 * MARGIN and x_id > 0) or (bottom <= 2 * MARGIN and y_id > 0):
                continue
            true_box = bbox + np.array([offset[0], offset[1], offset[0], offset[1]])
            if x_id > 0 and left < 2 * MARGIN:
                if (x_id - 1, y_id) in preds_dict:
                    remove_truncated(preds_dict[(x_id - 1, y_id)], true_box, probdata)
            if y_id > 0 and top < 2 * MARGIN:
                if (x_id, y_id - 1) in preds_dict:
                    remove_truncated(preds_dict[(x_id, y_id - 1)], true_box, probdata)
            preds_list.append((true_box, probdata))

    return imgname2preds


def run_assembler(nodes, class_probs, assembler_model, cfg, class_dict, device, threshold_dist=200, edge_threshold=0.5):
    """Run MLP edge prediction on detected nodes."""
    if len(nodes) < 2:
        return nodes

    # Build candidate pairs within distance threshold
    pairs = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
            ci = ((nodes[i].top + nodes[i].bottom) / 2, (nodes[i].left + nodes[i].right) / 2)
            cj = ((nodes[j].top + nodes[j].bottom) / 2, (nodes[j].left + nodes[j].right) / 2)
            dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
            if dist <= threshold_dist:
                pairs.append((i, j))

    if not pairs:
        return nodes

    # Get image dimensions for normalization (use bounding box extent)
    all_rights = [n.right for n in nodes]
    all_bottoms = [n.bottom for n in nodes]
    img_w = max(all_rights) + 1
    img_h = max(all_bottoms) + 1

    # Predict edges in batches
    assembler_model.eval()
    edge_predictions = []

    batch_size = cfg.EVAL.BATCH_SIZE if hasattr(cfg.EVAL, 'BATCH_SIZE') else 4096

    for batch_start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[batch_start:batch_start + batch_size]

        source_ids = torch.tensor([p[0] for p in batch_pairs], dtype=torch.long).to(device)
        target_ids = torch.tensor([p[1] for p in batch_pairs], dtype=torch.long).to(device)

        # Normalized bounding boxes
        source_boxes = torch.tensor([
            [nodes[p[0]].left / img_w, nodes[p[0]].top / img_h,
             nodes[p[0]].right / img_w, nodes[p[0]].bottom / img_h]
            for p in batch_pairs
        ], dtype=torch.float32).to(device)

        target_boxes = torch.tensor([
            [nodes[p[1]].left / img_w, nodes[p[1]].top / img_h,
             nodes[p[1]].right / img_w, nodes[p[1]].bottom / img_h]
            for p in batch_pairs
        ], dtype=torch.float32).to(device)

        # Class features
        if cfg.MODEL.MODE == "MLP":
            # Hard label (embedding)
            source_class = torch.tensor(
                [class_dict.get(nodes[p[0]].class_name, 0) for p in batch_pairs],
                dtype=torch.long).to(device)
            target_class = torch.tensor(
                [class_dict.get(nodes[p[1]].class_name, 0) for p in batch_pairs],
                dtype=torch.long).to(device)
        else:
            # Soft probability
            source_class = torch.tensor(
                [class_probs[p[0]] for p in batch_pairs],
                dtype=torch.float32).to(device)
            target_class = torch.tensor(
                [class_probs[p[1]] for p in batch_pairs],
                dtype=torch.float32).to(device)

        batch_data = {
            'source_id': source_ids,
            'target_id': target_ids,
            'source_bbox': source_boxes,
            'target_bbox': target_boxes,
            'source_class': source_class,
            'target_class': target_class,
        }

        with torch.no_grad():
            output = assembler_model(batch_data)
            probs = torch.sigmoid(output).cpu().numpy()

        for idx, (i, j) in enumerate(batch_pairs):
            if probs[idx] > edge_threshold:
                edge_predictions.append((i, j))

    # Add edges to nodes
    for i, j in edge_predictions:
        if j not in nodes[i].outlinks:
            nodes[i].outlinks.append(j)
        if i not in nodes[j].inlinks:
            nodes[j].inlinks.append(i)

    return nodes


def visualize_results(img_path, nodes, output_path, class_dict):
    """Draw bounding boxes and edges on image."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    color_palette = Colors()

    # Draw edges first (underneath boxes)
    for i, node in enumerate(nodes):
        cx1 = (node.left + node.right) / 2
        cy1 = (node.top + node.bottom) / 2
        for j in node.outlinks:
            if j < len(nodes):
                cx2 = (nodes[j].left + nodes[j].right) / 2
                cy2 = (nodes[j].top + nodes[j].bottom) / 2
                draw.line([(cx1, cy1), (cx2, cy2)], fill=(0, 200, 0, 128), width=1)

    # Draw bounding boxes
    rev_dict = {v: k for k, v in class_dict.items()}
    for i, node in enumerate(nodes):
        cls_idx = class_dict.get(node.class_name, 0)
        color = color_palette(cls_idx)
        draw.rectangle([node.left, node.top, node.right, node.bottom],
                       outline=color, width=2)
        # Label
        label = node.class_name[:15]
        draw.text((node.left, max(0, node.top - 12)), label, fill=color)

    img.save(output_path)
    return img


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory with input images')
    parser.add_argument('--output_dir', default='outputs/new_inference', help='Output directory')
    parser.add_argument('--detector_model',
                       default='runs/detect/outputs/detection_v8l_b8_i640/train2/weights/best.pt',
                       help='YOLOv8 model path')
    parser.add_argument('--assembler_dir',
                       default='outputs/assembly_on_gen',
                       help='Assembler checkpoint directory')
    parser.add_argument('--skip_preprocess', action='store_true',
                       help='Skip preprocessing (if images are already binarized PNGs)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--edge_threshold', type=float, default=0.5,
                        help='Edge confidence threshold (default: 0.5)')
    parser.add_argument('--threshold_dist', type=int, default=200,
                       help='Max distance for edge candidate pairs')
    args = parser.parse_args()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    preproc_dir = os.path.join(args.output_dir, 'preprocessed')
    ann_dir = os.path.join(args.output_dir, 'annotations')
    vis_dir = os.path.join(args.output_dir, 'visualization')
    os.makedirs(preproc_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- Step 1: Preprocess ----
    if not args.skip_preprocess:
        print("\n=== Step 1: Preprocessing images ===")
        img_files = [f for f in os.listdir(args.input_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for fname in tqdm(img_files, desc='Preprocessing'):
            in_path = os.path.join(args.input_dir, fname)
            out_name = os.path.splitext(fname)[0] + '.png'
            out_path = os.path.join(preproc_dir, out_name)
            preprocess_image(in_path, out_path)
        detect_dir = preproc_dir
    else:
        detect_dir = args.input_dir

    # ---- Step 2: YOLOv8 Detection ----
    print("\n=== Step 2: Running YOLOv8 detection ===")
    print(f"Loading detector from {args.detector_model}...")
    detector = YOLOSoft(args.detector_model)
    imgname2preds = run_detection(detector, detect_dir, args.batch_size)
    del detector  # free GPU memory
    torch.cuda.empty_cache()

    class_list, class_dict = get_classlist_and_classdict('essential')
    id2clsname = {idx: clsname for clsname, idx in class_dict.items()}

    # Convert detections to nodes
    print(f"\nDetected symbols in {len(imgname2preds)} images:")
    imgname2nodes = {}
    imgname2probs = {}
    for img_name, preds_dict in imgname2preds.items():
        nodes = []
        soft_classes = []
        for preds_list in preds_dict.values():
            for box, prob in preds_list:
                idx = len(nodes)
                nodes.append(Node(idx,
                    class_name=id2clsname[int(prob.top1)],
                    top=round(box[1].item()), left=round(box[0].item()),
                    height=round((box[3] - box[1]).item()),
                    width=round((box[2] - box[0]).item()),
                    inlinks=[], outlinks=[]))
                soft_classes.append(prob.data)
        imgname2nodes[img_name] = nodes
        imgname2probs[img_name] = soft_classes
        print(f"  {img_name}: {len(nodes)} symbols detected")

    # ---- Step 3: MLP Edge Prediction ----
    print("\n=== Step 3: Running MLP edge prediction ===")
    assembler_cfg = get_cfg_defaults()
    assembler_config_path = os.path.join(args.assembler_dir, 'config.yaml')
    if os.path.exists(assembler_config_path):
        assembler_cfg.merge_from_file(assembler_config_path)

    # Build and load assembler model
    if assembler_cfg.MODEL.MODE == "MLP":
        assembler = MLP(assembler_cfg)
    else:
        assembler = MLPwithSoftClass(assembler_cfg)

    # Find best checkpoint
    ckpt_path = os.path.join(args.assembler_dir, 'model_final.pth')
    if not os.path.exists(ckpt_path):
        # Try to find latest epoch checkpoint
        import glob
        ckpts = glob.glob(os.path.join(args.assembler_dir, 'model_ep*.pth'))
        if ckpts:
            ckpts.sort(key=lambda x: int(x.split('ep')[-1].split('.')[0]))
            ckpt_path = ckpts[-1]
        else:
            raise FileNotFoundError(f"No checkpoint found in {args.assembler_dir}")

    print(f"Loading assembler from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    assembler.load_state_dict(checkpoint['model'])
    assembler.to(device)
    assembler.eval()

    for img_name in tqdm(imgname2nodes, desc='Predicting edges'):
        nodes = imgname2nodes[img_name]
        probs = imgname2probs[img_name]
        nodes = run_assembler(nodes, probs, assembler, assembler_cfg,
                            class_dict, device, args.threshold_dist, args.edge_threshold)

        # Count edges
        total_edges = sum(len(n.outlinks) for n in nodes)
        print(f"  {img_name}: {total_edges} edges predicted")

        # Save MuNG XML
        doc_name = os.path.splitext(img_name)[0]
        xml_path = os.path.join(ann_dir, f"{doc_name}.xml")
        write_nodes_to_file(nodes, xml_path, doc_name, "Schenkerian_OMR")

        # Fix Mask="None" issue (causes mung-studio error)
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
        modified = False
        for mask in root.iter("Mask"):
            if mask.text and mask.text.strip() == "None":
                mask.text = None
                modified = True
        if modified:
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        # Save soft class probabilities
        npy_path = os.path.join(ann_dir, f"{doc_name}.npy")
        np.save(npy_path, np.array(probs))

    # ---- Step 4: Visualization ----
    print("\n=== Step 4: Generating visualizations ===")
    for img_name in tqdm(imgname2nodes, desc='Visualizing'):
        # Use preprocessed image for visualization
        orig_path = os.path.join(detect_dir, img_name)

        vis_path = os.path.join(vis_dir, os.path.splitext(img_name)[0] + '_annotated.png')
        visualize_results(orig_path, imgname2nodes[img_name], vis_path, class_dict)

    print(f"\n=== Done! ===")
    print(f"Preprocessed images: {preproc_dir}")
    print(f"MuNG XML annotations: {ann_dir}")
    print(f"Visualizations: {vis_dir}")
