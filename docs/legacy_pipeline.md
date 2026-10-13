# Historical pipeline guide (April 2026)

This is the repository README preserved from commit `cdcc23933d2073a0a5c39ef5f382518bd0819abf`. It describes the earlier staff-removal, detector, assembler, and MIDI experiments. Commands, file locations, model recommendations, and reported metrics below are historical and have not been revalidated for the current MuNGStudio workflow. They do not describe the current deployed system or its performance.

For the current annotation snapshot and format documentation, start with the [repository README](../README.md).

---

# Schenkerian OMR

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-yellow)

An Optical Music Recognition (OMR) pipeline for music score analysis. The system detects music symbols using YOLO-based object detection, predicts relationships (edges) between symbols using an MLP assembler, and outputs Music Notation Graph (MuNG) XML files that can be converted to MIDI.

## Pipeline Overview

```
Input Image → Preprocessing (staff line removal) → Symbol Detection (YOLO) → Edge Prediction (MLP) → MuNG XML → MIDI
```

1. **Preprocessing**: Remove staff lines from score images, producing binary images (white symbols on black background) matching the MUSCIMA++ `datasets_r_staff` format.
2. **Symbol Detection**: YOLO-based detector identifies 67 essential music symbol classes with bounding boxes and soft class probabilities.
3. **Edge Prediction (Notation Assembly)**: An MLP predicts pairwise relationships between detected symbols (e.g., notehead→stem, note→staff).
4. **Output**: MuNG XML annotations + visualizations. Optionally convert to MIDI via `mung2midi`.

## Trained Models

Three pre-trained detector models are included in `trained_models/`, all fine-tuned on MUSCIMA++ (84 training images, 500 epochs):

| Model | File | mAP@50 | mAP@50-95 |
|-------|------|--------|-----------|
| YOLOv8-L | `yolov8l_muscima.pt` | 89.0% | 71.0% |
| YOLO11-L | `yolo11l_muscima.pt` | 88.9% | 72.5% |
| RT-DETR-L | `rtdetr_l_muscima.pt` | 74.7% | 60.1% |

**Recommendation**: YOLOv8 and YOLO11 perform comparably. YOLO11 has slightly higher recall. RT-DETR underperforms on this small dataset (Transformer-based models need more training data).

## Installation

```bash
pip install ultralytics mung mung2midi midiutil torch torchvision numpy opencv-python Pillow scipy scikit-learn pyyaml yacs tqdm pandas
```

## Quick Start: Inference on New Images

The main inference script `infer_new_images.py` runs the full pipeline on any directory of score images.

### For handwritten scores

```bash
python infer_new_images.py \
    --input_dir /path/to/handwritten/images \
    --output_dir outputs/my_results \
    --detector_model trained_models/yolov8l_muscima.pt
```

### For printed scores

Printed scores require a different preprocessing method. Run preprocessing separately first, then inference with `--skip_preprocess`:

```bash
# Step 1: Preprocess (removes staff lines from printed scores)
python preprocess_printed.py \
    --input_dir /path/to/printed/images \
    --output_dir outputs/preprocessed_printed

# Step 2: Run inference on preprocessed images
python infer_new_images.py \
    --input_dir outputs/preprocessed_printed \
    --output_dir outputs/printed_results \
    --detector_model trained_models/yolov8l_muscima.pt \
    --skip_preprocess
```

### Output structure

```
outputs/my_results/
├── preprocessed/     # Binarized images with staff lines removed (only when preprocessing is run)
├── annotations/      # MuNG XML files + soft class probabilities (.npy)
└── visualization/    # Annotated images with bounding boxes and edges
```

When using `--skip_preprocess` (with pre-processed input images), the `preprocessed/` folder is not created.

## Inference Options

```bash
python infer_new_images.py \
    --input_dir DIR              # (required) Input image directory
    --output_dir DIR             # Output directory (default: outputs/new_inference)
    --detector_model PATH        # YOLO model path (default: runs/detect/.../best.pt)
    --assembler_dir DIR          # MLP assembler checkpoint dir (default: outputs/assembly_on_gen)
    --edge_threshold FLOAT       # Edge confidence threshold, 0.0-1.0 (default: 0.5)
    --threshold_dist INT         # Max pixel distance for edge candidates (default: 200)
    --batch_size INT             # YOLO batch size (default: 8)
    --skip_preprocess            # Skip preprocessing (use if images are already binarized)
```

### Edge threshold

The `--edge_threshold` parameter controls how many edges are predicted:
- **0.5** (default): Dense edges (~3000+ per image). Good for capturing all relationships.
- **0.9**: Moderate density. Balanced between precision and recall.
- **0.95-0.99**: Sparse edges. Higher precision, fewer false positives.
- **1.0**: No edges at all (symbol detection only).

Example with high threshold:
```bash
python infer_new_images.py \
    --input_dir /path/to/images \
    --output_dir outputs/sparse_edges \
    --detector_model trained_models/yolo11l_muscima.pt \
    --edge_threshold 0.95
```

## Preprocessing

### Handwritten scores (`preprocess_new_images.py`)

Uses multi-stage staff line removal based on grayscale intensity separation (handwritten ink is darker than printed staff lines):

1. Grayscale intensity separation (ink < 120 vs. staff lines 120-200)
2. Row projection to locate staff line positions
3. Targeted removal at staff positions, preserving vertical crossings (stems, barlines)
4. Morphological cleanup for remaining horizontal fragments
5. Connected component filtering for noise

```bash
python preprocess_new_images.py \
    --input_dir /path/to/images \
    --output_dir /path/to/output \
    --debug    # Optional: save intermediate processing steps
```

### Printed scores (`preprocess_printed.py`)

For printed scores, staff lines have the same ink darkness as symbols, so intensity-based methods don't work. Instead uses horizontal run-length analysis to detect straight staff lines:

1. Auto-rotates portrait images to landscape
2. Adaptive binarization
3. Horizontal run-length analysis to detect staff line rows
4. Groups detected lines into 5-line staff systems for validation
5. Removes staff line pixels while preserving vertical crossings (stems, barlines)
6. Morphological + aspect ratio cleanup

```bash
python preprocess_printed.py \
    --input_dir /path/to/images \
    --output_dir /path/to/output \
    --debug    # Optional: save intermediate processing steps
```

## Training

### Data Preparation

Download and set up the MUSCIMA++ v2.0 dataset:

```bash
mkdir -p data
wget -O data/MUSCIMA++.zip https://github.com/OMR-Research/muscima-pp/releases/download/v2.0/MUSCIMA-pp_v2.0.zip
unzip data/MUSCIMA++.zip -d data/MUSCIMA++
```

Download the CVC-MUSCIMA staff-removed images and place them so the directory structure is:

```
data/
└── MUSCIMA++
    ├── datasets_r_staff/
    │   └── images/
    └── v2.0/
        ├── data/
        │   └── annotations/
        └── specifications/
```

### Generate detection dataset

```bash
# Essential classes (67 classes, recommended)
python utils/detector_data.py

# All classes
python utils/detector_data.py --classes all \
    --save_dir data/MUSCIMA++/datasets_r_staff_all_crop \
    --save_config data_staff_removed_all_crop.yaml
```

### Train a detector

```bash
# YOLOv8
python train_detector.py \
    --config data/MUSCIMA++/datasets_r_staff_essn_crop/data_staff_removed_essn_crop.yaml \
    --model weights/yolov8l.pt \
    --output_dir outputs/detection_v8l

# YOLO11
python train_detector.py \
    --config data/MUSCIMA++/datasets_r_staff_essn_crop/data_staff_removed_essn_crop.yaml \
    --model weights/yolo11l.pt \
    --output_dir outputs/detection_yolo11l

# RT-DETR
python train_detector.py \
    --config data/MUSCIMA++/datasets_r_staff_essn_crop/data_staff_removed_essn_crop.yaml \
    --model weights/rtdetr-l.pt \
    --output_dir outputs/detection_rtdetr_l
```

All three model types use the same Ultralytics API. Default: 500 epochs, batch size 8, image size 640.

### Run detection on full dataset

```bash
python detector_infer.py \
    --model outputs/detection_v8l/train/weights/best.pt \
    --save_dir data/MUSCIMA++/v2.0_gen \
    --visualize --grids --links
```

### Train the assembler (edge prediction)

```bash
python train_assembler.py \
    -m data/MUSCIMA++/v2.0_gen/data \
    --model_config configs/assembler/on_gen/MLP32_on_gen.yaml \
    --exp_name assembly_on_gen
```

### Test the assembler

```bash
python train_assembler.py \
    -m data/MUSCIMA++/v2.0_gen/data \
    --model_config configs/assembler/on_gen/MLP32_on_gen.yaml \
    --exp_name assembly_on_gen \
    --test_only --load_epochs 180
```

## MIDI Conversion

To convert MuNG XML output to MIDI, install `mung2midi` and `midiutil`:

```bash
pip install mung2midi midiutil
```

Use `convert_mung_to_midi.py` to test conversion:

```bash
python convert_mung_to_midi.py
```

Note: MIDI conversion via `mung2midi` requires staff/staffLine/staffSpace nodes for pitch inference. The detector does not predict these classes, so for predicted outputs you need to merge ground-truth staff annotations with predicted symbols. See `run_pred_midi.py` for an example of this merge process.

## Project Structure

```
Schenkerian_OMR/
├── trained_models/           # Pre-trained YOLO detector weights
│   ├── yolov8l_muscima.pt
│   ├── yolo11l_muscima.pt
│   └── rtdetr_l_muscima.pt
├── infer_new_images.py       # Main inference script (full pipeline)
├── preprocess_new_images.py  # Staff removal for handwritten scores
├── preprocess_printed.py     # Staff removal for printed scores
├── train_detector.py         # Train YOLO detector
├── train_assembler.py        # Train MLP edge predictor
├── detector_infer.py         # Run detector on full dataset
├── convert_mung_to_midi.py   # MuNG XML → MIDI conversion
├── run_pred_midi.py          # Merge predicted + GT staff for MIDI
├── visualize_edges.py        # Edge visualization utility
├── model/                    # Model definitions (YOLO wrapper, MLP)
├── utils/                    # Data generation, constants, utilities
├── configs/                  # Assembler model configs
└── splits/                   # Train/valid/test split definitions
```
