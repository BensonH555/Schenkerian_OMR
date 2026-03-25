"""Visualize detected symbols AND predicted edges on sheet music images."""
import os
import torch
import numpy as np
import glob
import yaml
from PIL import Image, ImageDraw
from mung.io import read_nodes_from_file
from mung.graph import NotationGraph
from utils.constants import get_classlist_and_classdict
from configs.assembler.default import get_cfg_defaults
from model import MLP, MLPwithSoftClass
from ultralytics.utils.plotting import Colors
from argparse import ArgumentParser
from tqdm import tqdm


def get_node_center(node):
    cx = (node.left + node.right) / 2
    cy = (node.top + node.bottom) / 2
    return cx, cy


def main():
    parser = ArgumentParser()
    parser.add_argument('--mung_root', default='data/MUSCIMA++/v2.0_gen/data',
                        help='Root dir with annotations/ and images/')
    parser.add_argument('--assembler_ckpt', default='outputs/assembly_on_gen/model_final.pth',
                        help='Assembler checkpoint')
    parser.add_argument('--split_file', default='splits/mob_split.yaml')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--save_dir', default='quick_results_with_edges')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Edge prediction threshold')
    parser.add_argument('--max_distance', type=int, default=200,
                        help='Max distance between nodes to consider as edge candidate')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config and model
    cfg = get_cfg_defaults()
    saved_cfg_path = os.path.join(os.path.dirname(args.assembler_ckpt), 'config.yaml')
    if os.path.exists(saved_cfg_path):
        cfg.merge_from_file(saved_cfg_path)

    if cfg.MODEL.MODE == "MLP":
        model = MLP(cfg)
    else:
        model = MLPwithSoftClass(cfg)
    ckpt = torch.load(args.assembler_ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    # Load split
    with open(args.split_file) as f:
        split = yaml.safe_load(f)
    doc_names = split[args.split]

    class_list, class_dict = get_classlist_and_classdict('essential')
    color_palette = Colors()

    ann_dir = os.path.join(args.mung_root, 'annotations')
    img_dir = os.path.join(args.mung_root, 'images')

    for doc_name in tqdm(doc_names, desc="Visualizing"):
        xml_path = os.path.join(ann_dir, f"{doc_name}.xml")
        npy_path = os.path.join(ann_dir, f"{doc_name}.npy")
        img_path = os.path.join(img_dir, f"{doc_name}.png")

        if not os.path.exists(xml_path) or not os.path.exists(img_path):
            print(f"Skipping {doc_name}: missing files")
            continue

        nodes = read_nodes_from_file(xml_path)
        if len(nodes) == 0:
            continue

        # Load image
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        image_arr = np.array(Image.open(img_path).convert('1')).astype('uint8')
        h, w = image_arr.shape

        # Load soft class probabilities
        if os.path.exists(npy_path):
            soft_classes = np.load(npy_path)
        else:
            soft_classes = None

        # Draw bounding boxes
        for i, node in enumerate(nodes):
            cls_id = class_dict.get(node.class_name, 0)
            color = color_palette(cls_id)
            draw.rectangle([node.left, node.top, node.right, node.bottom],
                           outline=color, width=2)

        # Build candidate pairs (within max_distance)
        pairs = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist = nodes[i].distance_to(nodes[j])
                if dist < args.max_distance:
                    pairs.append((i, j))

        if len(pairs) == 0:
            img.save(os.path.join(args.save_dir, f"{doc_name}.png"))
            continue

        # Normalize bounding boxes
        reshape_weight = torch.tensor([2.0 / h, 2.0 / h, 2.0 / h, 2.0 / h])
        reshape_bias = torch.tensor([-1.0, -w / h, -1.0, -w / h])

        # Predict edges in batches
        predicted_edges = []
        batch_size = cfg.EVAL.BATCH_SIZE
        for batch_start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[batch_start:batch_start + batch_size]
            source_bboxes, target_bboxes = [], []
            source_classes, target_classes = [], []

            for i, j in batch_pairs:
                s_bbox = torch.tensor(nodes[i].bounding_box, dtype=torch.float) * reshape_weight + reshape_bias
                t_bbox = torch.tensor(nodes[j].bounding_box, dtype=torch.float) * reshape_weight + reshape_bias
                source_bboxes.append(s_bbox)
                target_bboxes.append(t_bbox)

                if cfg.MODEL.MODE == "MLP":
                    source_classes.append(torch.tensor(class_dict.get(nodes[i].class_name, 0)))
                    target_classes.append(torch.tensor(class_dict.get(nodes[j].class_name, 0)))
                elif soft_classes is not None:
                    source_classes.append(torch.tensor(soft_classes[i], dtype=torch.float))
                    target_classes.append(torch.tensor(soft_classes[j], dtype=torch.float))
                else:
                    s_cls = torch.zeros(177)
                    s_cls[class_dict.get(nodes[i].class_name, 0)] = 1.0
                    t_cls = torch.zeros(177)
                    t_cls[class_dict.get(nodes[j].class_name, 0)] = 1.0
                    source_classes.append(s_cls)
                    target_classes.append(t_cls)

            batch = {
                'source_bbox': torch.stack(source_bboxes).to(device),
                'target_bbox': torch.stack(target_bboxes).to(device),
                'source_class': torch.stack(source_classes).to(device),
                'target_class': torch.stack(target_classes).to(device),
            }

            with torch.no_grad():
                logits = model(batch)
                probs = torch.sigmoid(logits).squeeze(-1).cpu()

            for k, (i, j) in enumerate(batch_pairs):
                if probs[k].item() > args.threshold:
                    predicted_edges.append((i, j, probs[k].item()))

        # Draw predicted edges as lines
        for i, j, prob in predicted_edges:
            cx1, cy1 = get_node_center(nodes[i])
            cx2, cy2 = get_node_center(nodes[j])
            # Color: green for high confidence, yellow for lower
            green = int(255 * prob)
            red = int(255 * (1 - prob))
            color = (red, green, 0)
            draw.line([(cx1, cy1), (cx2, cy2)], fill=color, width=2)

        img.save(os.path.join(args.save_dir, f"{doc_name}.png"))

    print(f"Done! Results saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
