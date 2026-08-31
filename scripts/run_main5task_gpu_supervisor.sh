#!/usr/bin/env bash
set -Eeuo pipefail

repo=/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/MedGen3D-main5task
main_root=/inspire/qb-ilm/project/video-generation/public/lijiaxi/MedGen3D-main
log_root="$main_root/logs"
cpu_status="$main_root/pipeline_status.txt"
gpu_status="$main_root/gpu_pipeline_status.txt"
training_lock="$main_root/main5task_training_owner.lock"
env_root=${MEDGEN3D_ENV_ROOT:-/inspire/qb-ilm/project/video-generation/public/lijiaxi/.envs/medgen3d-v3}
wan_repo=/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/Wan2.2-official
wan_weights=/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/Wan2.2-TI2V-5B
config=${MEDGEN3D_CONFIG:-$repo/configs/experiments/main5task_feedforward_h200x8.yaml}
gpu_count=${MEDGEN3D_GPU_COUNT:-$(nvidia-smi -L | wc -l)}
if [[ "$gpu_count" != "4" && "$gpu_count" != "8" ]]; then
  echo "Expected 4 or 8 H200 GPUs, found $gpu_count" >&2
  exit 1
fi

mkdir -p "$log_root"
stage() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$gpu_status"; }

keeper_pids=()
start_keepers() {
  ((${#keeper_pids[@]} == 0)) || return 0
  stage "gpu_keepers_started"
  for ((gpu=0; gpu<gpu_count; gpu++)); do
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$env_root/bin/python" -u -c '
import torch
a=torch.randn((8192,8192), device="cuda", dtype=torch.float16)
b=torch.randn_like(a)
while True:
    torch.mm(a, b)
    torch.cuda.synchronize()
' >> "$log_root/main5task_keeper_gpu${gpu}.log" 2>&1 &
    keeper_pids+=("$!")
  done
}

stop_keepers() {
  ((${#keeper_pids[@]} > 0)) || return 0
  stage "gpu_keepers_stopping"
  kill "${keeper_pids[@]}" 2>/dev/null || true
  wait "${keeper_pids[@]}" 2>/dev/null || true
  keeper_pids=()
}

hold_after_failure() {
  rc=$?
  stage "gpu_pipeline_failed exit_code=$rc; restoring_keepers"
  start_keepers
  while true; do sleep 300; done
}
trap hold_after_failure ERR

source "$env_root/bin/activate"
export PYTHONPATH="$repo/src:$repo:$wan_repo${PYTHONPATH:+:$PYTHONPATH}"
export MEDGEN3D_FORCE_FLASH_ATTN2=1
export HF_HOME=/inspire/qb-ilm/project/video-generation/public/lijiaxi/.cache/huggingface
export TORCH_HOME=/inspire/qb-ilm/project/video-generation/public/lijiaxi/.cache/torch

# A queued H200 notebook may arrive before CPU preparation finishes.  Keep all
# eight GPUs visibly active until the authoritative five-task audit sentinel is
# present, so the notebook is not reclaimed while waiting.
start_keepers
stage "waiting_for_data_only_pipeline_complete"
until grep -q 'data_only_pipeline_complete' "$cpu_status" 2>/dev/null; do
  sleep 60
done
stage "data_completion_observed"

# Multiple 4/8-GPU candidates may be queued to minimize scheduling latency.
# Claim a shared atomic directory before touching training outputs so only the
# first RUNNING notebook launches the process group.
if ! mkdir "$training_lock" 2>/dev/null; then
  owner=$(cat "$training_lock/owner" 2>/dev/null || echo unknown)
  stage "training_owner_exists owner=$owner; stopping_keepers_and_exiting"
  stop_keepers
  exit 0
fi
printf 'host=%s pid=%s gpu_count=%s started=%s\n' \
  "$(hostname)" "$$" "$gpu_count" "$(date -Is)" > "$training_lock/owner"
stage "training_owner_acquired"

stop_keepers
stage "preflight_started commit=$(git -C "$repo" rev-parse --short HEAD) gpu_count=$gpu_count config=$config"
"$env_root/bin/python" "$repo/scripts/preflight_training.py" \
  --config "$config" --checkpoint-dir "$wan_weights" --wan-repo "$wan_repo"
stage "gpu_probe_started"
torchrun --standalone --nproc_per_node="$gpu_count" "$repo/scripts/train_medgen3d.py" \
  --config "$config" --checkpoint-dir "$wan_weights" --max-steps 2 \
  --limit-train-cases 5 --run-name medgen3d_main5task_feedforward_probe
stage "formal_training_started"
torchrun --standalone --nproc_per_node="$gpu_count" "$repo/scripts/train_medgen3d.py" \
  --config "$config" --checkpoint-dir "$wan_weights"
stage "formal_training_complete"

# Preserve the allocated machine for checkpoint selection and evaluation.
start_keepers
stage "training_complete_keepers_active"
while true; do sleep 300; done
