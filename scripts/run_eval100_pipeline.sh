#!/usr/bin/env bash
set -euo pipefail

BASE=/inspire/ssd/project/exploration-topic/public/MedGen3D-git
OUT="$BASE/analysis/task_specific"
LOG="$OUT/logs"
mkdir -p "$LOG"
cd "$BASE"

echo "$(date -Is) pipeline watcher started" | tee -a "$LOG/pipeline.log"

while pgrep -f "task_specific_ct_baseline.py train" >/dev/null; do
  sleep 60
done

python scripts/task_specific_ct_baseline.py eval --model redcnn \
  --root data/AbdomenAtlas1.0Mini --checkpoint "$OUT/redcnn/best.pt" \
  --cases analysis/eval100_protocol/case_ids.txt --output "$OUT/redcnn/eval100.json" \
  > "$LOG/redcnn_eval100.log" 2>&1
python scripts/task_specific_ct_baseline.py eval --model fbpconvnet \
  --root data/AbdomenAtlas1.0Mini --checkpoint "$OUT/fbpconvnet/best.pt" \
  --cases analysis/eval100_protocol/case_ids.txt --output "$OUT/fbpconvnet/eval100.json" \
  > "$LOG/fbpconvnet_eval100.log" 2>&1
touch "$OUT/CT_BASELINES_COMPLETE"

while pgrep -f "nnUNetv2_plan_and_preprocess" >/dev/null; do
  sleep 60
done
export nnUNet_raw="$OUT/nnunet_raw"
export nnUNet_preprocessed="$OUT/nnunet_preprocessed"
export nnUNet_results="$OUT/nnunet_results"

nnUNetv2_train 501 3d_fullres 0 -device cuda > "$LOG/nnunet_train.log" 2>&1
nnUNetv2_predict -i "$nnUNet_raw/Dataset501_MedGen3DAbdomen9/imagesTs" \
  -o "$OUT/nnunet_predictions" -d 501 -c 3d_fullres -f 0 -device cuda \
  > "$LOG/nnunet_predict.log" 2>&1
python scripts/evaluate_nnunet_aorta.py --pred "$OUT/nnunet_predictions" \
  --gt "$nnUNet_raw/Dataset501_MedGen3DAbdomen9/labelsTs" \
  --cases analysis/eval100_protocol/case_ids.txt --output "$OUT/nnunet_eval100.json" \
  > "$LOG/nnunet_eval100.log" 2>&1
touch "$OUT/NNUNET_COMPLETE"

# The checkpoint directory contains weights only; the importable Wan source is
# maintained separately in the shared project tree.
PYTHONPATH="src:/inspire/ssd/project/exploration-topic/public/Wan2.2-official" python scripts/evaluate_medgen3d.py \
  --config configs/experiments/multitask.yaml \
  --checkpoint outputs/medgen3d_joint_v1_formal_20260811/step_00029000.pt \
  --checkpoint-dir "$BASE/Wan2.2-TI2V-5B" --samples-per-task 100 \
  --sampling-steps 30 --seed 20260812 --output-dir "$OUT/medgen3d_eval100" \
  > "$LOG/medgen3d_eval100.log" 2>&1
touch "$OUT/MEDGEN3D_COMPLETE"
python scripts/summarize_eval100_comparison.py --root "$OUT" \
  > "$LOG/comparison_summary.log" 2>&1
echo "$(date -Is) pipeline complete" | tee -a "$LOG/pipeline.log"
