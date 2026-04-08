"""Train edge prediction model on Schenkerian Complete Annotations.

Uses Stephen's 3 annotated pages (p2, p6, p7) to train an MLP assembler
that predicts edges between detected symbols.

Steps:
1. Prepare data: convert Complete Annotations to the format expected by
   the training pipeline (annotations/, images/, split.yaml)
2. Generate .npy soft class probability files (one-hot from class names)
3. Train the MLP model
"""

import os
import sys
import shutil
import yaml
import numpy as np
import torch
from PIL import Image
from mung.io import read_nodes_from_file, write_nodes_to_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from utils.constants import get_classlist_and_classdict
from utils.data_pool import load_munglinker_data
from configs.assembler.default import get_cfg_defaults
from model import MLP

# ─── Config ───
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPLETE_ANN_DIR = os.path.join(PROJECT_DIR, "Complete Annotations")
PREPROC_DIR = os.path.join(PROJECT_DIR, "outputs/preprocessed/Rachmaninov")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs/assembly_schenkerian")

PAGES = {
    "rachmaninov_bells_mvt2_p2": "Rachmaninov The Bells Second Mvmt 002",
    "rachmaninov_bells_mvt2_p6": "Rachmaninov The Bells Second Mvmt 006",
    "rachmaninov_bells_mvt2_p7": "Rachmaninov The Bells Second Mvmt 007",
}


def prepare_data():
    """Convert Complete Annotations to training format."""
    data_dir = os.path.join(OUTPUT_DIR, "data")
    ann_dir = os.path.join(data_dir, "annotations")
    img_dir = os.path.join(data_dir, "images")
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # Build class dict from actual annotations
    all_classes = set()
    for folder_name in PAGES:
        src_xml = os.path.join(COMPLETE_ANN_DIR, folder_name, "mung.xml")
        if os.path.exists(src_xml):
            for n in read_nodes_from_file(src_xml):
                all_classes.add(n.class_name)
    class_list = sorted(all_classes)
    class_dict = {cls: idx for idx, cls in enumerate(class_list)}
    print(f"  {len(class_list)} unique classes")
    doc_names = []

    for folder_name, doc_name in PAGES.items():
        src_xml = os.path.join(COMPLETE_ANN_DIR, folder_name, "mung.xml")
        if not os.path.exists(src_xml):
            print(f"  WARNING: {src_xml} not found, skipping")
            continue

        # Copy and rename XML
        nodes = read_nodes_from_file(src_xml)
        # Map unknown classes to nearest known class, keep all nodes to preserve edges
        from mung.node import Node as MungNode
        new_nodes = []
        old_to_new = {}
        for new_idx, node in enumerate(nodes):
            old_to_new[node.id] = new_idx
        for new_idx, node in enumerate(nodes):
            # If class not in dict, map to 'other' or keep as-is
            cls = node.class_name if node.class_name in class_dict else 'noteheadBlack'
            new_inlinks = [old_to_new[l] for l in node.inlinks if l in old_to_new]
            new_outlinks = [old_to_new[l] for l in node.outlinks if l in old_to_new]
            new_nodes.append(MungNode(new_idx, class_name=cls,
                top=node.top, left=node.left,
                height=node.height, width=node.width,
                inlinks=new_inlinks, outlinks=new_outlinks))
        nodes = new_nodes

        dst_xml = os.path.join(ann_dir, f"{doc_name}.xml")
        write_nodes_to_file(nodes, dst_xml, doc_name, "Schenkerian_OMR")

        # Remove all Mask elements (they cause parsing errors)
        import xml.etree.ElementTree as ET
        tree = ET.parse(dst_xml)
        root = tree.getroot()
        for node_elem in root.findall(".//Node"):
            for mask in node_elem.findall("Mask"):
                node_elem.remove(mask)
        tree.write(dst_xml, encoding="utf-8", xml_declaration=True)

        # Generate one-hot .npy
        npy_data = []
        for node in nodes:
            one_hot = np.zeros(len(class_dict), dtype=np.float32)
            one_hot[class_dict[node.class_name]] = 1.0
            npy_data.append(one_hot)
        np.save(os.path.join(ann_dir, f"{doc_name}.npy"), np.array(npy_data))

        # Copy preprocessed image
        src_img = os.path.join(PREPROC_DIR, f"{doc_name}.png")
        dst_img = os.path.join(img_dir, f"{doc_name}.png")
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        else:
            print(f"  WARNING: {src_img} not found")

        doc_names.append(doc_name)
        total_edges = sum(len(n.outlinks) for n in nodes)
        print(f"  {doc_name}: {len(nodes)} nodes, {total_edges} edges")

    # Create split file: use p2+p6 for train, p7 for validation
    split = {
        "train": doc_names[:2],  # p2, p6
        "valid": doc_names[2:],  # p7
        "test": doc_names[2:],   # p7
    }
    split_path = os.path.join(OUTPUT_DIR, "split.yaml")
    with open(split_path, 'w') as f:
        yaml.dump(split, f)
    print(f"\n  Split: train={split['train']}, valid={split['valid']}")
    return split_path


def train_model(split_path):
    """Train MLP assembler on the prepared data."""
    cfg = get_cfg_defaults()
    # Use hard labels (MLP mode with embeddings)
    cfg.MODEL.MODE = "MLP"
    cfg.MODEL.MLP_CONFIG = [32, 32]
    cfg.MODEL.EMBEDDING_DIM = 32
    cfg.TRAIN.NUM_EPOCHS = 100
    cfg.TRAIN.BATCH_SIZE = 256
    cfg.TRAIN.EVAL_FREQUENCY = 10
    cfg.TRAIN.SAVE_FREQUENCY = 20
    cfg.TRAIN.LEARNING_RATE = 1e-3
    cfg.TRAIN.POS_WEIGHT = 3  # upweight positive edges (fewer than negatives)
    cfg.EVAL.BATCH_SIZE = 1024
    cfg.SYSTEM.NUM_GPUS = 0  # CPU

    data_dir = os.path.join(OUTPUT_DIR, "data")
    # Rebuild class dict from prepared data
    all_classes = set()
    ann_dir = os.path.join(OUTPUT_DIR, "data", "annotations")
    for f in os.listdir(ann_dir):
        if f.endswith('.xml'):
            for n in read_nodes_from_file(os.path.join(ann_dir, f)):
                all_classes.add(n.class_name)
    class_list = sorted(all_classes)
    class_dict = {cls: idx for idx, cls in enumerate(class_list)}
    print(f"  {len(class_list)} classes")

    # Update model vocab dim
    cfg.MODEL.VOCAB_DIM = len(class_dict)

    # Load data config
    from utils.data_pool import load_split, load_config, PairwiseMungoDataPool
    from utils.data_pool import config2data_pool_dict
    import copy

    data_config = load_config("configs/muscima_nogrammar.yaml")
    data_config['mode'] = cfg.MODEL.MODE
    data_pool_dict = config2data_pool_dict(data_config)
    split = load_split(split_path)

    exclude_classes = [cn for cn in class_dict if cn not in class_list]

    # Load training data
    print("\nLoading training data...")
    from glob import glob
    from mung.graph import NotationGraph
    from tqdm import tqdm

    def load_mungs_and_images(names, ann_dir, img_dir, mode='MLP'):
        mungs, images = [], []
        for name in names:
            xml_path = os.path.join(ann_dir, f"{name}.xml")
            img_path = os.path.join(img_dir, f"{name}.png")
            npy_path = os.path.join(ann_dir, f"{name}.npy")

            nodes = read_nodes_from_file(xml_path)
            mung = NotationGraph(nodes)

            if mode == "MLPwithSoftClass":
                npy = np.load(npy_path)
                for v_idx, v in enumerate(mung.vertices):
                    v.class_prob = npy[v_idx]

            mungs.append(mung)
            image = np.array(Image.open(img_path).convert('1')).astype('uint8')
            images.append(image)
            print(f"  Loaded {name}: {len(nodes)} nodes")
        return mungs, images

    ann_dir = os.path.join(data_dir, "annotations")
    img_dir = os.path.join(data_dir, "images")

    tr_mungs, tr_images = load_mungs_and_images(split['train'], ann_dir, img_dir, cfg.MODEL.MODE)
    val_mungs, val_images = load_mungs_and_images(split['valid'], ann_dir, img_dir, cfg.MODEL.MODE)

    training_pool = PairwiseMungoDataPool(
        mungs=tr_mungs, images=tr_images,
        max_edge_length=data_pool_dict['max_edge_length'],
        max_negative_samples=data_pool_dict['max_negative_samples'],
        patch_size=data_pool_dict['patch_size'],
        zoom=data_pool_dict['zoom'],
        class_list=class_list, class_dict=class_dict,
        mode=cfg.MODEL.MODE
    )
    validation_pool = PairwiseMungoDataPool(
        mungs=val_mungs, images=val_images,
        max_edge_length=data_pool_dict['max_edge_length'],
        max_negative_samples=data_pool_dict['max_negative_samples'],
        patch_size=data_pool_dict['patch_size'],
        zoom=data_pool_dict['zoom'],
        class_list=class_list, class_dict=class_dict,
        mode=cfg.MODEL.MODE
    )

    print(f"\nTraining samples: {len(training_pool)}")
    print(f"Validation samples: {len(validation_pool)}")

    # Build model
    model = MLP(cfg)
    device = "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    pos_weight = torch.tensor([cfg.TRAIN.POS_WEIGHT], dtype=torch.float32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = torch.utils.data.DataLoader(
        training_pool, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        validation_pool, batch_size=cfg.EVAL.BATCH_SIZE, shuffle=False, num_workers=0)

    # Save config
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())

    # Training loop
    print(f"\nTraining for {cfg.TRAIN.NUM_EPOCHS} epochs...")
    best_acc = 0
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            output = model(batch)
            loss = criterion(output.squeeze(-1), batch['label'].float().squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % cfg.TRAIN.EVAL_FREQUENCY == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    for k in batch:
                        if isinstance(batch[k], torch.Tensor):
                            batch[k] = batch[k].to(device)
                    output = model(batch)
                    preds = (torch.sigmoid(output.squeeze(-1)) > 0.5).long()
                    correct += (preds == batch['label'].squeeze(-1)).sum().item()
                    total += len(preds)
            acc = correct / max(total, 1)
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, val_acc={acc:.4f}", flush=True)
            if acc > best_acc:
                best_acc = acc

        if (epoch + 1) % cfg.TRAIN.SAVE_FREQUENCY == 0:
            save_path = os.path.join(OUTPUT_DIR, f"model_ep{epoch+1}.pth")
            torch.save({'epoch': epoch+1, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()}, save_path)

    # Save final
    final_path = os.path.join(OUTPUT_DIR, "model_final.pth")
    torch.save({'epoch': cfg.TRAIN.NUM_EPOCHS, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, final_path)
    print(f"\nDone! Best val accuracy: {best_acc:.4f}")
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    print("=== Training Schenkerian Edge Predictor ===\n")
    print("Step 1: Preparing data...")
    split_path = prepare_data()
    print("\nStep 2: Training model...")
    train_model(split_path)
