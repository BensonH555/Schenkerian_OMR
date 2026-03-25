# CompleteOMR: Toward a More Complete OMR Solution

## Project Summary

This document provides a comprehensive overview of the **completeOMR** project, which accompanies the paper *"Toward a More Complete OMR Solution"*. The system performs end-to-end **Optical Music Recognition (OMR)** — automatically detecting music symbols in handwritten score images and assembling them into a structured music notation graph.

---

## 1. Problem Statement

Optical Music Recognition aims to convert images of music scores into machine-readable symbolic music representations. Traditional OMR systems treat this as a pipeline of independent subtasks, but often fail to connect detected symbols into meaningful musical structures (e.g., linking a notehead to its stem, beam, or flag).

This project addresses the **notation assembly** problem: given detected music symbols, predict the semantic relationships (edges) between them to construct a complete **Music Notation Graph (MuNG)**.

---

## 2. Input Data

### 2.1 Primary Dataset: MUSCIMA++ v2.0

- **Source**: [MUSCIMA++ v2.0 on GitHub](https://github.com/OMR-Research/muscima-pp/releases/tag/v2.0)
- **Content**: 140 handwritten music score images with symbol-level annotations
- **Annotations per image**: XML files containing bounding boxes, class labels, pixel masks, and directed semantic edges (outlinks/inlinks) between symbols
- **Total annotations**: 91,254 annotated music objects
- **Class vocabulary**: 163 distinct symbol classes (noteheads, stems, beams, rests, clefs, accidentals, dynamics, etc.)
- **Writers**: 50 different human writers, each writing the same 20 musical excerpts
- **Annotation format**: MuNG XML — each `<Node>` element contains `Id`, `ClassName`, `Top`, `Left`, `Width`, `Height`, `Outlinks`, `Inlinks`, and optional mask data

### 2.2 Score Images: CVC-MUSCIMA (Staff-Removed)

- **Source**: CVC-MUSCIMA dataset, staff-line-removed variant
- **Content**: The same 140 music score images with staff lines algorithmically removed
- **Purpose**: Removing staff lines reduces visual noise and allows the object detector to focus on the music symbols themselves. Staff line positions can be recovered separately and are not needed for symbol detection.
- **Format**: Binary PNG images (black symbols on white background)
- **Resolution**: Approximately 3000×2000 pixels per image

### 2.3 Data Split

Defined in `splits/mob_split.yaml`, following a **writer-independent** protocol:

| Split      | Images | Purpose                      |
|------------|--------|------------------------------|
| Training   | 84     | Train both detector and assembler |
| Validation | 28     | Hyperparameter tuning, early stopping |
| Test       | 28     | Final evaluation             |

No writer appears in more than one split, preventing the model from memorizing handwriting styles rather than learning musical structure.

---

## 3. System Architecture

The pipeline consists of two main stages plus a data preparation step:

```
Raw Score Images
       │
       ▼
┌─────────────────────┐
│  Stage 1: Detection │  (YOLOv8l)
│  - Detect symbols   │
│  - Output bboxes    │
│  - Output soft class│
│    probabilities    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────┐
│  Pseudo-Edge Generation │  (Hungarian Matching)
│  - Match predictions to │
│    ground truth          │
│  - Transfer edge labels │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Stage 2: Assembly      │  (MLP Binary Classifier)
│  - Predict edges between│
│    detected symbol pairs │
│  - Build notation graph │
└─────────┬───────────────┘
          │
          ▼
   Music Notation Graph
```

---

## 4. Stage 1: Music Symbol Detection

### 4.1 Detection Dataset Generation

**Script**: `utils/detector_data.py`

The raw MUSCIMA++ annotations are converted into YOLO-format training data:

1. **Class filtering**: From 163 total classes, 67 "essential" classes are selected. These cover all musically meaningful symbols (noteheads, stems, beams, flags, rests, clefs, key/time signatures, accidentals, dynamics, slurs, ties) while excluding non-essential categories.

2. **Random cropping**: Full-resolution images (~3000×2000) are too large for efficient YOLO training at 640×640 input size. Each training/validation image is randomly cropped into 14 patches, producing:
   - **1,176 training patches** (84 images × 14 crops)
   - **392 validation patches** (28 images × 14 crops)
   - **28 test images** (kept at full resolution for realistic evaluation)

3. **Label conversion**: Bounding boxes are converted from absolute pixel coordinates to YOLO's normalized format: `(center_x, center_y, width, height)` relative to patch dimensions.

4. **Output**: A standard YOLO directory structure (`train/images/`, `train/labels/`, `valid/images/`, `valid/labels/`) and a `data_staff_removed_essn_crop.yaml` config file.

### 4.2 Detector Training

**Script**: `train_detector.py`

- **Model**: YOLOv8l (Large variant) — a single-stage anchor-free object detector
- **Backbone**: CSPDarknet with cross-stage partial connections for multi-scale feature extraction
- **Neck**: PANet (Path Aggregation Network) for feature pyramid fusion
- **Head**: Decoupled classification and regression heads
- **Initialization**: Pretrained on COCO (`yolov8l.pt`), then fine-tuned on the music symbol dataset
- **Training configuration**:
  - Max epochs: 500 (with early stopping, patience=100)
  - Batch size: 8
  - Input resolution: 640×640
  - Loss: CIoU (regression) + BCE (classification) + DFL (distribution focal loss)

### 4.3 Detector Inference (Tiled)

**Script**: `detector_infer.py`

Since training uses 640×640 crops but inference must handle full-resolution images (~3000×2000), a **tiled inference** strategy is used:

1. **Image tiling**: Each full image is split into overlapping 1216×1216 patches with 128-pixel margins (step size = 960 pixels)
2. **Per-patch inference**: YOLOv8 runs on each patch independently
3. **Result merging**: Predictions from all patches are combined. Overlapping detections from adjacent patches are resolved by computing Intersection-over-Prediction (IoP) weighted by class probability similarity — if a detection in the overlap zone duplicates one from an already-processed patch, the duplicate is removed.

### 4.4 Soft Class Probabilities (Key Innovation)

Standard YOLO outputs only the top-1 predicted class and its confidence score. This project introduces a **custom predictor** (`model/predictor.py: DetectionSoftPredictor`) that modifies the NMS post-processing to preserve the **full class probability distribution** over all classes for each detection.

**How it works**:
- The standard NMS pipeline computes `argmax` over class scores to get a single class label
- `DetectionSoftPredictor` retains the complete probability vector (shape `(num_classes,)`) for every surviving detection after NMS
- The output tensor per detection becomes `[x1, y1, x2, y2, confidence, class, prob_class_0, prob_class_1, ..., prob_class_N]`

**Why this matters**: A notehead that the detector is 60% confident is a "full notehead" and 30% confident is a "half notehead" carries useful information — the assembler can leverage this ambiguity rather than committing to a potentially wrong hard decision.

The soft probabilities are saved as `.npy` files alongside the XML annotations and are used as input features for the notation assembly model.

---

## 5. Pseudo-Edge Generation

**Location**: `detector_infer.py` (activated with `--links` flag)

To train the assembly model, we need edge labels between *detected* symbols (not ground truth symbols). Since detections don't perfectly align with ground truth, this project uses **Hungarian matching** to transfer edge annotations:

1. **Cost matrix construction**: For each image, a cost matrix `C` of shape `(num_detections, num_gt_nodes)` is computed as:
   ```
   C[i, j] = IoU(detection_i, gt_node_j) × P(detection_i, class_of_gt_node_j)
   ```
   where `IoU` is bounding box Intersection-over-Union and `P` is the soft class probability of detection `i` for the class of ground truth node `j`.

2. **Optimal matching**: `scipy.optimize.linear_sum_assignment` solves the bipartite matching problem to find the one-to-one assignment that maximizes total cost. Matches with cost < 0.05 are rejected.

3. **Edge transfer**: For each matched pair `(detection_i → gt_node_j)`, the outgoing edges of `gt_node_j` are copied to `detection_i`, but only if the target of each edge is also successfully matched to a detection. These become **pseudo-edges** stored in the XML `Outlinks` field.

---

## 6. Stage 2: Notation Assembly

### 6.1 Task Definition

Given a set of detected music symbols with their bounding boxes and soft class probabilities, predict which pairs of symbols are semantically connected (i.e., share an edge in the notation graph).

This is formulated as **pairwise binary classification**: for every candidate pair of symbols `(A, B)`, predict probability `P(edge exists | A, B)`.

### 6.2 Candidate Pair Selection

Not all symbol pairs are considered — the number of pairs grows quadratically. A **distance threshold** (default: 200 pixels) filters out pairs whose bounding boxes are too far apart to plausibly share an edge. This dramatically reduces the number of candidates while preserving nearly all true edges.

### 6.3 Model Architecture

**Script**: `train_assembler.py`, **Model**: `model/model.py: MLPwithSoftClass`

For each candidate pair `(A, B)`, the input features are:

| Feature | Dimension | Description |
|---------|-----------|-------------|
| `source_bbox` | 4 | Normalized bounding box of symbol A: `(top, left, bottom, right)`, normalized to `[-1, 1]` relative to image dimensions |
| `target_bbox` | 4 | Normalized bounding box of symbol B (same format) |
| `source_class` | 177 | Soft class probability vector of symbol A (from detector) |
| `target_class` | 177 | Soft class probability vector of symbol B (from detector) |

**Architecture**:
```
source_class (177-d) ──→ Linear(177, 32) ──→ source_cls_embed (32-d)
source_bbox  (4-d)   ──────────────────────→ source_bbox     (4-d)
                                              ├─ concat ─→ source_embed (36-d)
target_class (177-d) ──→ Linear(177, 32) ──→ target_cls_embed (32-d)
target_bbox  (4-d)   ──────────────────────→ target_bbox     (4-d)
                                              ├─ concat ─→ target_embed (36-d)

source_embed + target_embed ──→ concat (72-d)
       │
       ▼
   Linear(72, 32) → ReLU
       │
   Linear(32, 32) → ReLU
       │
   Linear(32, 1) → output logit
       │
   sigmoid → P(edge)
```

- **Embedding**: Soft class vectors are projected from 177-d to 32-d via a learned linear layer
- **MLP**: Two hidden layers of 32 units each with ReLU activations
- **Output**: Single logit passed through sigmoid to get edge probability
- **Loss**: Binary Cross-Entropy with logits (`BCEWithLogitsLoss`)
- **Optimizer**: Adam, lr=1e-3

### 6.4 Training Configuration

Defined in `configs/assembler/on_gen/MLP32_on_gen.yaml`:

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch size | 256 |
| Evaluation frequency | Every 10 epochs |
| Checkpoint frequency | Every 20 epochs |
| Positive weight | 1.0 (equal weighting) |
| Seed | 95 |

---

## 7. Evaluation

### 7.1 Match+AUC Metric

**Script**: `utils/metrics.py: compute_matching_score()`

Standard detection metrics (mAP) only measure symbol detection quality. This project introduces a **Match+AUC** metric that evaluates the full pipeline end-to-end:

1. **Symbol matching**: Hungarian matching aligns predicted symbols to ground truth symbols using the same IoU × class probability cost matrix as in pseudo-edge generation.

2. **Edge comparison**: After matching, for every pair of predicted symbols `(i, j)`:
   - **Ground truth label**: True if the matched ground truth nodes share an edge, False otherwise. If either predicted symbol has no match, the label is False.
   - **Predicted score**: The edge probability output by the assembly model (0 if the pair was not evaluated).

3. **Metric computation**:
   - **AUC**: Area under the Precision-Recall curve — measures ranking quality across all thresholds
   - **F1**: Harmonic mean of Precision and Recall at threshold 0.5
   - **Precision**: Fraction of predicted edges that are correct
   - **Recall**: Fraction of true edges that are predicted

### 7.2 Reproduction Results

| Metric    | Value |
|-----------|-------|
| **AUC**       | 0.920 |
| **F1**        | 0.895 |
| **Precision** | 0.853 |
| **Recall**    | 0.947 |

---

## 8. Output Description

### 8.1 Detection Stage Outputs

For each of the 140 images, the detector produces:

1. **MuNG XML file** (`data/MUSCIMA++/v2.0_gen/data/annotations/<doc_name>.xml`):
   - One `<Node>` element per detected symbol
   - Fields: `Id`, `ClassName` (top-1 predicted class), `Top`, `Left`, `Width`, `Height`, `Outlinks` (pseudo-edges to other detected symbols), `Inlinks`
   - This is a standard MuNG-format file compatible with the `mung` Python library

2. **Soft class probability file** (`data/MUSCIMA++/v2.0_gen/data/annotations/<doc_name>.npy`):
   - NumPy array of shape `(N, num_classes)` where `N` is the number of detected symbols
   - Each row is the full class probability distribution for one detection
   - Row order matches the node order in the corresponding XML file

3. **Visualization images** (optional, with `--visualize` flag):
   - The original image with colored bounding boxes drawn for each detection
   - Color-coded by predicted class
   - Optionally shows the tiling grid overlay

### 8.2 Assembly Stage Outputs

1. **Trained model checkpoints** (`outputs/assembly_on_gen/`):
   - `model_ep20.pth`, `model_ep40.pth`, ..., `model_final.pth`
   - Each checkpoint contains model weights, optimizer state, and epoch number

2. **Edge predictions** (computed during evaluation):
   - For each pair of detected symbols within distance threshold: a probability score in `[0, 1]` indicating whether an edge exists
   - These form the predicted **Music Notation Graph**

### 8.3 Combined Output: Music Notation Graph

The final output of the complete pipeline is a **Music Notation Graph (MuNG)** for each input score image. This graph consists of:
- **Nodes**: Detected music symbols with positions (bounding boxes) and class labels
- **Edges**: Predicted semantic relationships between symbols (e.g., notehead→stem, stem→beam, note→slur)

---

## 9. Downstream Applications

The Music Notation Graph output can be utilized in several ways:

### 9.1 Conversion to Symbolic Music Formats
The notation graph can be post-processed into standard music encoding formats:
- **MusicXML**: The most widely supported interchange format for music notation software (Finale, Sibelius, MuseScore)
- **MEI (Music Encoding Initiative)**: A scholarly music encoding standard
- **MIDI**: For audio playback and music information retrieval
- **Humdrum/kern**: For computational musicology

### 9.2 Music Information Retrieval (MIR)
- **Score search**: Enable content-based retrieval of music scores from large archives
- **Music similarity**: Compare musical content across different handwritten manuscripts
- **Melodic/harmonic analysis**: Extract pitch sequences, chord progressions, and rhythmic patterns from scanned scores

### 9.3 Digital Music Library Applications
- **Digitization of historical manuscripts**: Convert handwritten scores in libraries and archives into searchable, editable digital formats
- **Accessibility**: Make printed/handwritten music available to visually impaired musicians via screen readers or Braille music notation
- **Preservation**: Create structured digital copies of deteriorating physical manuscripts

### 9.4 Music Education and Practice
- **Automatic score following**: Align audio performances to detected notation for practice feedback
- **Transposition and arrangement**: Programmatically modify detected scores for different instruments or keys
- **Score annotation**: Overlay analytical markings on recognized notation

### 9.5 Research in Computational Musicology
- **Large-scale corpus analysis**: Apply music-theoretic analyses across thousands of digitized scores
- **Style classification**: Use the structured notation graph for composer or period identification
- **Music generation**: Use recognized scores as training data for generative music models

---

## 10. Project Structure

```
completeOMR/
├── README.md                          # Original setup and usage instructions
├── PROJECT_SUMMARY.md                 # This document
├── train_detector.py                  # YOLOv8 training script
├── detector_infer.py                  # Tiled inference + pseudo-edge generation
├── train_assembler.py                 # Notation assembly training and evaluation
├── model/
│   ├── __init__.py                    # Exports model classes
│   ├── model.py                       # YOLOSoft, MLP, MLPwithSoftClass, MLPwithSoftClassExtraMLP
│   └── predictor.py                   # DetectionSoftPredictor (custom NMS preserving soft probs)
├── utils/
│   ├── __init__.py                    # Package init
│   ├── constants.py                   # Class lists and dictionaries (67/20/163 classes)
│   ├── data_pool.py                   # PairwiseMungoDataPool dataset for assembly training
│   ├── detector_data.py               # YOLO dataset generation from MUSCIMA++ annotations
│   ├── metrics.py                     # Match+AUC evaluation metric
│   └── utility.py                     # Seed setting and miscellaneous utilities
├── configs/
│   ├── assembler/
│   │   ├── default.py                 # YACS default configuration
│   │   └── on_gen/
│   │       └── MLP32_on_gen.yaml      # Assembly model hyperparameters
│   └── muscima_nogrammar.yaml         # Data pool configuration (bbox mode, distance threshold)
├── splits/
│   └── mob_split.yaml                 # Train/valid/test split definition
└── data/                              # Data directory (created during setup)
    └── MUSCIMA++/
        ├── v2.0/                      # Ground truth annotations
        │   ├── data/annotations/      # 140 XML annotation files
        │   └── specifications/        # Class specification files
        ├── datasets_r_staff/          # Staff-removed score images
        │   └── images/                # 140 PNG files
        ├── datasets_r_staff_essn_crop/ # Generated YOLO training dataset
        │   ├── train/                 # 1176 cropped training images + labels
        │   └── valid/                 # 392 cropped validation images + labels
        └── v2.0_gen/                  # Detector output (generated annotations)
            └── data/
                ├── annotations/       # 140 XML + 140 NPY files
                └── images/            # Symlink to staff-removed images
```

---

## 11. Key Technical Contributions

1. **Soft class probabilities**: Preserving the full class distribution from the detector (rather than hard top-1 labels) provides richer features for the assembly model, improving edge prediction accuracy.

2. **Pseudo-edge generation via Hungarian matching**: Automatically transfers ground truth edge annotations to imperfectly detected symbols, enabling end-to-end training without requiring manual annotation of detected outputs.

3. **Tiled inference with overlap resolution**: Enables processing of full-resolution score images using a detector trained on small patches, with a principled strategy for merging predictions across tile boundaries.

4. **Match+AUC evaluation**: A novel end-to-end evaluation metric that accounts for both detection quality and edge prediction quality in a unified framework.

5. **Decoupled two-stage design**: Separating detection from assembly allows each component to be independently optimized and replaced, providing flexibility for future improvements.

---

## 12. Dependencies

| Package       | Purpose                                              |
|---------------|------------------------------------------------------|
| `ultralytics` | YOLOv8 framework (training, inference, NMS)          |
| `torch`       | Deep learning framework (MLP training, tensor ops)   |
| `torchvision` | Box IoU computation, NMS operations                  |
| `mung`        | MUSCIMA++ data parsing (read/write MuNG XML files)   |
| `yacs`        | Configuration system for assembler model             |
| `scipy`       | Hungarian matching (`linear_sum_assignment`)          |
| `scikit-learn` | Precision-recall curve and AUC computation           |
| `Pillow`      | Image loading and visualization                      |
| `numpy`       | Array operations, soft probability storage           |
| `PyYAML`      | Configuration and split file parsing                 |
| `tqdm`        | Progress bars                                        |

---

## 13. References

- **Paper**: "Toward a More Complete OMR Solution"
- **MUSCIMA++ v2.0**: Hajič Jr., J., et al. (2018). *The MUSCIMA++ Dataset for Handwritten Optical Music Recognition.*
- **CVC-MUSCIMA**: Fornés, A., et al. (2012). *CVC-MUSCIMA: A Ground-truth of Handwritten Music Score Images for Writer Identification and Staff Removal.*
- **YOLOv8**: Ultralytics (2023). *YOLOv8: A State-of-the-Art Real-Time Object Detection System.*
- **MuNG / MuNGLinker**: Hajič Jr., J., et al. (2018). *Understanding Optical Music Recognition.*
