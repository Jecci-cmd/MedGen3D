#!/usr/bin/env bash
set -Eeuo pipefail

repo=/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/MedGen3D-main5task
main_root=/inspire/qb-ilm/project/video-generation/public/lijiaxi/MedGen3D-main
log_root="$main_root/logs"
status_file="$main_root/pipeline_status.txt"
env_root=${MEDGEN3D_ENV_ROOT:-/inspire/qb-ilm/project/video-generation/public/lijiaxi/.envs/medgen3d-v3}
abdomen=/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/data/AbdomenAtlas1.0Mini
brats_download=/inspire/qb-ilm/project/video-generation/public/lijiaxi/MedGen3D-synthesis/datasets/BraTS2021/download
brats_root="$main_root/data/brats2021_t1_t2"
ctrate_download="$main_root/downloads/ctrate1300"
ctrate_root="$main_root/data/ctrate_report_ct"
recon_root="$main_root/data/reconstruction_main18_v1"
wan_repo=/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/Wan2.2-official
wan_weights=/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/Wan2.2-TI2V-5B
projection_workers=${MEDGEN3D_PROJECTION_WORKERS:-32}
config=${MEDGEN3D_CONFIG:-$repo/configs/experiments/main5task_feedforward_h200x8.yaml}
export MEDGEN3D_CONFIG="$config"

mkdir -p "$log_root" "$main_root/data"

stage() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$status_file"
}

wait_for_count() {
  local label=$1 expected=$2 command=$3
  while true; do
    local observed
    observed=$(eval "$command")
    stage "waiting $label $observed/$expected"
    [[ "$observed" -ge "$expected" ]] && break
    sleep 60
  done
}

source "$env_root/bin/activate"
export PYTHONPATH="$repo/src:$repo:$wan_repo${PYTHONPATH:+:$PYTHONPATH}"
export MEDGEN3D_FORCE_FLASH_ATTN2=1
export HF_HOME=/inspire/qb-ilm/project/video-generation/public/lijiaxi/.cache/huggingface
export TORCH_HOME=/inspire/qb-ilm/project/video-generation/public/lijiaxi/.cache/torch

stage "supervisor_started commit=$(git -C "$repo" rev-parse --short HEAD)"

keeper_handoff=0
start_keeper() {
  [[ "$keeper_handoff" -eq 1 ]] || return 0
  stage "starting_emergency_gpu_keeper"
  for gpu in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$gpu" nohup python -u -c '
import torch
a=torch.randn((8192,8192),device="cuda",dtype=torch.float16)
b=torch.randn_like(a)
while True:
    c=a@b
    torch.cuda.synchronize()
' >> "$log_root/main5task_keeper_gpu${gpu}.log" 2>&1 &
  done
}
trap start_keeper EXIT

wait_for_count abdomenatlas_files 600 \
  "find '$abdomen/raw/extracted' -type f \( -name ct.nii.gz -o -name combined_labels.nii.gz \) | wc -l"
stage "abdomenatlas_preprocessing_started"
python "$repo/scripts/freeze_abdomenatlas1300_split.py" --root "$abdomen"
python "$repo/scripts/prepare_abdomenatlas.py" --config "$repo/configs/data/abdomenatlas1300_prepare.yaml" inventory
# Every downstream task is derived for all splits, so all 1300 source cases
# must first exist in the same canonical grid.  The command is idempotent and
# skips already valid outputs from earlier pilot runs.
python "$repo/scripts/prepare_abdomenatlas.py" --config "$repo/configs/data/abdomenatlas1300_prepare.yaml" canonicalize --split all
python "$repo/scripts/prepare_abdomenatlas.py" --config "$repo/configs/data/abdomenatlas1300_prepare.yaml" derive --split all --task restoration --workers "$projection_workers"
python "$repo/scripts/prepare_abdomenatlas.py" --config "$repo/configs/data/abdomenatlas1300_prepare.yaml" sdf-volumes --split all --refresh-stale
python "$repo/scripts/prepare_reconstruction_multiview.py" --dataset-root "$abdomen" --output-root "$recon_root" --split all --workers "$projection_workers" --skip-write-manifests
python "$repo/scripts/prepare_abdomenatlas.py" --config "$repo/configs/data/abdomenatlas1300_prepare.yaml" case-manifests --split all
python "$repo/scripts/prepare_reconstruction_multiview.py" --dataset-root "$abdomen" --output-root "$recon_root" --finalize
stage "abdomenatlas_preprocessing_complete"

while [[ ! -f "$brats_download/KAGGLE_DOWNLOAD_DONE" ]]; do
  bytes=$(stat -c %s "$brats_download/BraTS2021_Training_Data.tar" 2>/dev/null || echo 0)
  stage "waiting brats_archive bytes=$bytes"
  sleep 60
done
tar -tf "$brats_download/BraTS2021_Training_Data.tar" >/dev/null
stage "brats_extraction_started"
brats_stage="$brats_download/extracted_main"
mkdir -p "$brats_stage"
if [[ ! -f "$brats_stage/_EXTRACTED" ]]; then
  tar -xf "$brats_download/BraTS2021_Training_Data.tar" -C "$brats_stage"
  find "$brats_stage" -type f -name 'BraTS2021_*.tar' -print0 | \
    xargs -0 -r -P 16 -I '{}' sh -c 'tar -xf "$1" -C "$2"' sh '{}' "$brats_stage"
  date -Is > "$brats_stage/_EXTRACTED"
fi
python "$repo/scripts/prepare_brats2021_main.py" --input-root "$brats_stage" --output-root "$brats_root"
stage "brats_preprocessing_complete"

wait_for_count ctrate_volumes 1300 \
  "find '$ctrate_download/files' -type f -name '*.nii.gz' | wc -l"
stage "ctrate_preprocessing_started"
python "$repo/scripts/prepare_ctrate_main.py" \
  --files-root "$ctrate_download/files" \
  --train-reports "$ctrate_download/selected_train_reports.csv" \
  --valid-reports "$ctrate_download/selected_valid_reports.csv" \
  --output-root "$ctrate_root"
stage "ctrate_preprocessing_complete"

cd "$repo"
python - <<'PY'
from medgen3d.config import load_experiment_config
from medgen3d.data import audit_multitask_splits
import os
c = load_experiment_config(os.environ.get(
    'MEDGEN3D_CONFIG', 'configs/experiments/main5task_feedforward_h200x8.yaml'
))
audit_multitask_splits(c['data'], c['experiment']['tasks'])
print('five-task manifest audit passed')
PY
stage "five_task_manifest_audit_complete"

if [[ "${MEDGEN3D_DATA_ONLY:-0}" == "1" ]]; then
  stage "data_only_pipeline_complete"
  exit 0
fi

# Stop only the eight known keeper children in this notebook immediately
# before the CUDA probe. The supervising shell itself stays alive.
for pid in $(pgrep -P 131845 2>/dev/null || true); do kill "$pid" 2>/dev/null || true; done
sleep 5
keeper_handoff=1
stage "gpu_probe_started"
torchrun --standalone --nproc_per_node=8 "$repo/scripts/train_medgen3d.py" \
  --config "$config" \
  --checkpoint-dir "$wan_weights" --max-steps 2 --limit-train-cases 5 \
  --run-name medgen3d_main5task_feedforward_probe
stage "formal_training_started"
torchrun --standalone --nproc_per_node=8 "$repo/scripts/train_medgen3d.py" \
  --config "$config" \
  --checkpoint-dir "$wan_weights"
stage "formal_training_complete"
