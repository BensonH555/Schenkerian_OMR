#!/bin/bash
# Quick pipeline: train detector on mini data, run inference, save visual results
# Target: ~30 minutes total
set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=1

LOG_DIR="mini_run_logs"
LOG_FILE="$LOG_DIR/quick_run_log.txt"
RESULT_DIR="quick_results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

log() { echo "$1" | tee -a "$LOG_FILE"; }
elapsed() { printf '%dm %ds' $((($1)%3600/60)) $(($1%60)); }

> "$LOG_FILE"
TOTAL_START=$SECONDS
log "=========================================="
log "Quick Pipeline Run — $(date)"
log "GPU: RTX 5000 Ada (32GB)"
log "=========================================="

# Step 1: Generate mini detection dataset
log ""
log "========== Step 1: Generate mini detection dataset =========="
STEP_START=$SECONDS
python utils/detector_data.py \
    --split_file splits/mini_split.yaml \
    --crop_times 4 \
    --save_dir data/MUSCIMA++/datasets_r_staff_essn_quick \
    --save_config data_staff_removed_essn_quick.yaml \
    2>&1 | tee -a "$LOG_FILE"
STEP_TIME=$((SECONDS - STEP_START))
log ">>> Step 1 completed in $(elapsed $STEP_TIME)"

# Step 2: Train YOLOv8n detector (20 epochs, fast)
log ""
log "========== Step 2: Train YOLOv8n detector (20 epochs) =========="
STEP_START=$SECONDS
python train_detector.py \
    --config data/MUSCIMA++/datasets_r_staff_essn_quick/data_staff_removed_essn_quick.yaml \
    --model yolov8n.pt \
    --epochs 20 \
    --batch_size 16 \
    --output_dir outputs/detection_quick \
    2>&1 | tee -a "$LOG_FILE"
STEP_TIME=$((SECONDS - STEP_START))
log ">>> Step 2 completed in $(elapsed $STEP_TIME)"

# Step 3: Run detector inference on test images with visualization
log ""
log "========== Step 3: Detector inference + visualization =========="
BEST_PT=$(find outputs/detection_quick -name "best.pt" | head -1)
log "Using model: $BEST_PT"
STEP_START=$SECONDS
python detector_infer.py \
    --model "$BEST_PT" \
    --save_dir data/MUSCIMA++/v2.0_gen_quick \
    --visualize --grids --links \
    2>&1 | tee -a "$LOG_FILE"
STEP_TIME=$((SECONDS - STEP_START))
log ">>> Step 3 completed in $(elapsed $STEP_TIME)"

# Step 4: Copy visual results to quick_results folder
log ""
log "========== Step 4: Collecting results =========="
# Copy visualization images
cp -r data/MUSCIMA++/v2.0_gen_quick/data/visualization/* "$RESULT_DIR/" 2>/dev/null || true
# Copy a few annotation XMLs as samples
mkdir -p "$RESULT_DIR/sample_annotations"
ls data/MUSCIMA++/v2.0_gen_quick/data/annotations/*.xml | head -4 | xargs -I{} cp {} "$RESULT_DIR/sample_annotations/"

NUM_VIS=$(ls "$RESULT_DIR"/*.png 2>/dev/null | wc -l)
NUM_XML=$(ls "$RESULT_DIR/sample_annotations/"*.xml 2>/dev/null | wc -l)
log "Copied $NUM_VIS visualization images and $NUM_XML sample annotations to $RESULT_DIR/"

TOTAL_TIME=$((SECONDS - TOTAL_START))
log ""
log "=========================================="
log "ALL DONE — Total time: $(elapsed $TOTAL_TIME)"
log "Results in: $(pwd)/$RESULT_DIR/"
log "=========================================="
