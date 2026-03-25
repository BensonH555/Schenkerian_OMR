#!/bin/bash
# Full completeOMR pipeline run on RTX 5000 Ada (32GB)
set -e
cd "$(dirname "$0")"

LOG_DIR="mini_run_logs"
LOG_FILE="$LOG_DIR/full_run_log.txt"
mkdir -p "$LOG_DIR"

# Use only the RTX 5000 Ada (32GB) to avoid OOM on 2080 Ti
export CUDA_VISIBLE_DEVICES=1

log() {
    echo "$1" | tee -a "$LOG_FILE"
}

elapsed() {
    local t=$1
    printf '%dh %dm %ds' $((t/3600)) $((t%3600/60)) $((t%60))
}

TOTAL_START=$SECONDS
log "=========================================="
log "Full Pipeline Run — $(date)"
log "GPU: RTX 5000 Ada (32GB), CUDA_VISIBLE_DEVICES=1"
log "=========================================="

# Step 1: Detection dataset already generated
log ""
log "========== Step 1: Detection dataset already exists, skipping =========="

# Step 2: Train YOLOv8l detector (500 epochs, early stopping)
log ""
log "========== Step 2: Train YOLOv8l detector (500 epochs) =========="
STEP_START=$SECONDS
python train_detector.py \
    --config data/MUSCIMA++/datasets_r_staff_essn_full/data_staff_removed_essn_full.yaml \
    --model weights/yolov8l.pt \
    --epochs 500 \
    --batch_size 8 \
    --output_dir outputs/detection_full \
    2>&1 | tee -a "$LOG_FILE"
STEP_TIME=$((SECONDS - STEP_START))
log ">>> Step 2 completed in $(elapsed $STEP_TIME)"

# Step 3: Detector inference
log ""
log "========== Step 3: Detector inference on all images =========="
BEST_PT=$(find outputs/detection_full -name "best.pt" | head -1)
log "Using model: $BEST_PT"
STEP_START=$SECONDS
python detector_infer.py \
    --model "$BEST_PT" \
    --save_dir data/MUSCIMA++/v2.0_gen_full \
    --visualize --grids --links \
    2>&1 | tee -a "$LOG_FILE"
STEP_TIME=$((SECONDS - STEP_START))
log ">>> Step 3 completed in $(elapsed $STEP_TIME)"

# Step 4: Train assembler (200 epochs)
log ""
log "========== Step 4: Train assembler (200 epochs) =========="
STEP_START=$SECONDS
python train_assembler.py \
    -m data/MUSCIMA++/v2.0_gen_full/data \
    -s splits/mob_split.yaml \
    --model_config configs/assembler/on_gen/MLP32_on_gen.yaml \
    --exp_name assembly_full \
    --opts SYSTEM.NUM_WORKERS 2 \
    2>&1 | tee -a "$LOG_FILE"
STEP_TIME=$((SECONDS - STEP_START))
log ">>> Step 4 completed in $(elapsed $STEP_TIME)"

# Step 5: Test assembler
log ""
log "========== Step 5: Test assembler =========="
STEP_START=$SECONDS
python train_assembler.py \
    -m data/MUSCIMA++/v2.0_gen_full/data \
    -s splits/mob_split.yaml \
    --model_config configs/assembler/on_gen/MLP32_on_gen.yaml \
    --exp_name assembly_full \
    --test_only --load_epochs 180 \
    2>&1 | tee -a "$LOG_FILE"
STEP_TIME=$((SECONDS - STEP_START))
log ">>> Step 5 completed in $(elapsed $STEP_TIME)"

TOTAL_TIME=$((SECONDS - TOTAL_START))
log ""
log "=========================================="
log "ALL DONE — Total time: $(elapsed $TOTAL_TIME)"
log "=========================================="
