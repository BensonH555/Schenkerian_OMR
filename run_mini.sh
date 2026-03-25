#!/bin/bash
# Quick test run of the full completeOMR pipeline (~20 minutes)
# Uses minimal data split, nano model, and few epochs

set -e
cd "$(dirname "$0")"
# Using 3 GPUs via DDP, data sized to avoid DDP stalling

echo "========== Step 1: Generate mini detection dataset =========="
python utils/detector_data.py \
    --split_file splits/mini_split.yaml \
    --crop_times 4 \
    --save_dir data/MUSCIMA++/datasets_r_staff_essn_mini \
    --save_config data_staff_removed_essn_mini.yaml

echo "========== Step 2: Train YOLOv8n detector (10 epochs) =========="
python train_detector.py \
    --config data/MUSCIMA++/datasets_r_staff_essn_mini/data_staff_removed_essn_mini.yaml \
    --model yolov8n.pt \
    --epochs 10 \
    --batch_size 16 \
    --output_dir outputs/detection_mini

echo "========== Step 3: Run detector inference on test images =========="
# Find the best.pt path (ultralytics may add subdirectories)
BEST_PT=$(find outputs/detection_mini -name "best.pt" | head -1)
echo "Using model: $BEST_PT"

python detector_infer.py \
    --model "$BEST_PT" \
    --save_dir data/MUSCIMA++/v2.0_gen_mini \
    --links

echo "========== Step 4: Train assembler (10 epochs) =========="
python train_assembler.py \
    -m data/MUSCIMA++/v2.0_gen_mini/data \
    -s splits/mini_split.yaml \
    --model_config configs/assembler/on_gen/MLP32_on_gen.yaml \
    --exp_name assembly_mini \
    --opts TRAIN.NUM_EPOCHS 10 TRAIN.EVAL_FREQUENCY 5 TRAIN.SAVE_FREQUENCY 5 SYSTEM.NUM_WORKERS 2

echo "========== Step 5: Test assembler =========="
python train_assembler.py \
    -m data/MUSCIMA++/v2.0_gen_mini/data \
    -s splits/mini_split.yaml \
    --model_config configs/assembler/on_gen/MLP32_on_gen.yaml \
    --exp_name assembly_mini \
    --test_only --load_epochs 10

echo "========== DONE =========="
