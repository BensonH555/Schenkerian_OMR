"""Train edge prediction model v2 - with 3-fold cross-validation and improved params.

Changes from v1:
- Max edge distance: 200 -> 800 (covers 95%+ of real edges)
- Limit negative samples per object to 10 (prevent extreme imbalance)
- pos_weight: 3 -> 5 (more positive emphasis with larger distance range)
- Epochs: 100 -> 200
- 3-fold CV: each page takes a turn as validation
"""

import os
import sys
import shutil
import yaml
import numpy as np
import torch
from PIL import Image
from mung.io import read_nodes_from_file, write_nodes_to_file
from mung.node import Node as MungNode
from mung.graph import NotationGraph
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from utils.constants import get_classlist_and_classdict
from utils.data_pool import PairwiseMungoDataPool, load_config, config2data_pool_dict
from configs.assembler.default import get_cfg_defaults
from model import MLP

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPLETE_ANN_DIR = os.path.join(PROJECT_DIR, "Complete Annotations")
PREPROC_DIR = os.path.join(PROJECT_DIR, "outputs/preprocessed/Rachmaninov")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs/assembly_schenkerian_v2")

PAGES = [
    ("rachmaninov_bells_mvt2_p2", "Rachmaninov The Bells Second Mvmt 002"),
    ("rachmaninov_bells_mvt2_p6", "Rachmaninov The Bells Second Mvmt 006"),
    ("rachmaninov_bells_mvt2_p7", "Rachmaninov The Bells Second Mvmt 007"),
]


def prepare_data():
    """Prepare data for all 3 pages."""
    data_dir = os.path.join(OUTPUT_DIR, "data")
    ann_dir = os.path.join(data_dir, "annotations")
    img_dir = os.path.join(data_dir, "images")
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # Build class dict from all annotations
    all_classes = set()
    for folder_name, _ in PAGES:
        src_xml = os.path.join(COMPLETE_ANN_DIR, folder_name, "mung.xml")
        for n in read_nodes_from_file(src_xml):
            all_classes.add(n.class_name)
    class_list = sorted(all_classes)
    class_dict = {cls: idx for idx, cls in enumerate(class_list)}
    print(f"  {len(class_list)} classes")

    doc_names = []
    for folder_name, doc_name in PAGES:
        src_xml = os.path.join(COMPLETE_ANN_DIR, folder_name, "mung.xml")
        nodes = read_nodes_from_file(src_xml)

        # Re-index
        old_to_new = {n.id: i for i, n in enumerate(nodes)}
        new_nodes = []
        for i, n in enumerate(nodes):
            new_nodes.append(MungNode(i, class_name=n.class_name,
                top=n.top, left=n.left, height=n.height, width=n.width,
                inlinks=[old_to_new[l] for l in n.inlinks if l in old_to_new],
                outlinks=[old_to_new[l] for l in n.outlinks if l in old_to_new]))

        dst_xml = os.path.join(ann_dir, f"{doc_name}.xml")
        write_nodes_to_file(new_nodes, dst_xml, doc_name, "Schenkerian_OMR")
        # Remove Mask elements
        tree = ET.parse(dst_xml)
        for ne in tree.getroot().findall(".//Node"):
            for m in ne.findall("Mask"):
                ne.remove(m)
        tree.write(dst_xml, encoding="utf-8", xml_declaration=True)

        # Generate one-hot npy
        npy_data = np.array([
            np.eye(len(class_dict), dtype=np.float32)[class_dict[n.class_name]]
            for n in new_nodes
        ])
        np.save(os.path.join(ann_dir, f"{doc_name}.npy"), npy_data)

        # Copy preprocessed image
        src_img = os.path.join(PREPROC_DIR, f"{doc_name}.png")
        shutil.copy2(src_img, os.path.join(img_dir, f"{doc_name}.png"))

        edges = sum(len(n.outlinks) for n in new_nodes)
        doc_names.append(doc_name)
        print(f"  {doc_name}: {len(new_nodes)} nodes, {edges} edges")

    return doc_names, class_list, class_dict


def load_mungs(names, ann_dir, img_dir):
    mungs, images = [], []
    for name in names:
        nodes = read_nodes_from_file(os.path.join(ann_dir, f"{name}.xml"))
        mung = NotationGraph(nodes)
        mungs.append(mung)
        image = np.array(Image.open(os.path.join(img_dir, f"{name}.png")).convert('1')).astype('uint8')
        images.append(image)
    return mungs, images


def train_fold(fold_idx, train_names, val_names, class_list, class_dict, cfg):
    print(f"\n{'='*50}")
    print(f"Fold {fold_idx+1}: train={train_names}, val={val_names}")
    print(f"{'='*50}")

    ann_dir = os.path.join(OUTPUT_DIR, "data", "annotations")
    img_dir = os.path.join(OUTPUT_DIR, "data", "images")

    # Custom data config with larger distance
    data_config = load_config("configs/muscima_nogrammar.yaml")
    data_config['mode'] = cfg.MODEL.MODE
    data_config['THRESHOLD_NEGATIVE_DISTANCE'] = 800
    data_config['MAX_NEGATIVE_EXAMPLES_PER_OBJECT'] = 10
    data_pool_dict = config2data_pool_dict(data_config)

    tr_mungs, tr_images = load_mungs(train_names, ann_dir, img_dir)
    val_mungs, val_images = load_mungs(val_names, ann_dir, img_dir)

    training_pool = PairwiseMungoDataPool(
        mungs=tr_mungs, images=tr_images,
        max_edge_length=800, max_negative_samples=3,
        patch_size=(256, 512), zoom=1.0,
        class_list=class_list, class_dict=class_dict,
        mode=cfg.MODEL.MODE)
    validation_pool = PairwiseMungoDataPool(
        mungs=val_mungs, images=val_images,
        max_edge_length=800, max_negative_samples=3,
        patch_size=(256, 512), zoom=1.0,
        class_list=class_list, class_dict=class_dict,
        mode=cfg.MODEL.MODE)

    print(f"  Train samples: {len(training_pool)}, Val samples: {len(validation_pool)}")

    # Count positive ratio
    pos_count = sum(1 for i in range(len(training_pool)) if training_pool[i]['label'].item() == 1)
    neg_count = len(training_pool) - pos_count
    print(f"  Positive: {pos_count}, Negative: {neg_count}, Ratio: 1:{neg_count/max(pos_count,1):.1f}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    model = MLP(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    pos_weight = torch.tensor([cfg.TRAIN.POS_WEIGHT], dtype=torch.float32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = torch.utils.data.DataLoader(
        training_pool, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        validation_pool, batch_size=cfg.EVAL.BATCH_SIZE, shuffle=False, num_workers=0)

    best_acc = 0
    best_state = None
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output = model(batch)
            loss = criterion(output.squeeze(-1), batch['label'].float().squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            model.eval()
            correct, total, tp, fp, fn = 0, 0, 0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    output = model(batch)
                    preds = (torch.sigmoid(output.squeeze(-1)) > 0.5).long()
                    labels = batch['label'].squeeze(-1).long()
                    correct += (preds == labels).sum().item()
                    total += len(preds)
                    tp += ((preds == 1) & (labels == 1)).sum().item()
                    fp += ((preds == 1) & (labels == 0)).sum().item()
                    fn += ((preds == 0) & (labels == 1)).sum().item()
            acc = correct / max(total, 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
                  f"acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}", flush=True)
            if f1 > best_acc:
                best_acc = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc


def main():
    print("=== Training Schenkerian Edge Predictor v2 ===\n")

    print("Step 1: Preparing data...")
    doc_names, class_list, class_dict = prepare_data()

    cfg = get_cfg_defaults()
    cfg.MODEL.MODE = "MLP"
    cfg.MODEL.MLP_CONFIG = [64, 64]  # wider network
    cfg.MODEL.EMBEDDING_DIM = 32
    cfg.MODEL.VOCAB_DIM = len(class_dict)
    cfg.TRAIN.NUM_EPOCHS = 100
    cfg.TRAIN.BATCH_SIZE = 256
    cfg.TRAIN.LEARNING_RATE = 1e-3
    cfg.TRAIN.POS_WEIGHT = 5
    cfg.EVAL.BATCH_SIZE = 4096
    cfg.SYSTEM.NUM_GPUS = 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save config and class dict
    with open(os.path.join(OUTPUT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    with open(os.path.join(OUTPUT_DIR, 'class_dict.yaml'), 'w') as f:
        yaml.dump(class_dict, f)

    # 3-fold cross-validation
    print("\nStep 2: 3-fold cross-validation...")
    fold_results = []
    best_model = None
    best_f1 = 0

    for fold in range(3):
        val_names = [doc_names[fold]]
        train_names = [doc_names[i] for i in range(3) if i != fold]
        model, f1 = train_fold(fold, train_names, val_names, class_list, class_dict, cfg)
        fold_results.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    print(f"\n{'='*50}")
    print(f"3-fold CV results: {[f'{f:.4f}' for f in fold_results]}")
    print(f"Mean F1: {np.mean(fold_results):.4f}")
    print(f"Best fold F1: {best_f1:.4f}")

    # Save best model
    final_path = os.path.join(OUTPUT_DIR, "model_final.pth")
    torch.save({'epoch': cfg.TRAIN.NUM_EPOCHS, 'model': best_model.state_dict()}, final_path)
    print(f"Best model saved to {final_path}")

    # Also train on all 3 pages for final model
    print("\nStep 3: Training final model on all data...")
    model_all, _ = train_fold(-1, doc_names, doc_names, class_list, class_dict, cfg)
    all_path = os.path.join(OUTPUT_DIR, "model_all.pth")
    torch.save({'epoch': cfg.TRAIN.NUM_EPOCHS, 'model': model_all.state_dict()}, all_path)
    print(f"All-data model saved to {all_path}")


if __name__ == "__main__":
    main()
