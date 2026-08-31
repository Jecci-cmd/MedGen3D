#!/usr/bin/env bash
set -Eeuo pipefail

repo=/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/MedGen3D-xy256-z65-git
main_root=/inspire/qb-ilm/project/video-generation/public/lijiaxi/MedGen3D-main
env_root=/inspire/qb-ilm/project/video-generation/public/lijiaxi/.envs/medgen3d-v3
wan_repo=/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/Wan2.2-official
wan_weights=/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/Wan2.2-TI2V-5B
config="$repo/configs/experiments/main5task_feedforward_lora_all_xy256_z65_h200x8.yaml"
log="$main_root/logs/train_xy256_z65_h200x8.log"
status="$main_root/train_xy256_z65_status.txt"
brats="$main_root/data/brats2021_t1_t2_xy256"
ctrate="$main_root/data/ctrate_report_ct_xy256"

mkdir -p "$main_root/logs"
stage() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$status"; }

stage "waiting_for_xy256_preprocessing"
until [[ -f "$brats/PREP_DONE" && -f "$ctrate/PREP_DONE" ]]; do
  stage "preprocess_counts brats=$(find "$brats/volumes" -type f -name '*.npy' 2>/dev/null | wc -l) ctrate=$(find "$ctrate/volumes" -type f -name '*.npy' 2>/dev/null | wc -l)"
  sleep 60
done

source "$env_root/bin/activate"
export PYTHONPATH="$repo/src:$repo:$wan_repo${PYTHONPATH:+:$PYTHONPATH}"
export MEDGEN3D_FORCE_FLASH_ATTN2=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/inspire/qb-ilm/project/video-generation/public/lijiaxi/.cache/huggingface
export TORCH_HOME=/inspire/qb-ilm/project/video-generation/public/lijiaxi/.cache/torch

cd "$repo"
python - <<'PY' | tee -a "$log"
from medgen3d.config import load_experiment_config
from medgen3d.data import audit_multitask_splits, build_task_dataset

path = "configs/experiments/main5task_feedforward_lora_all_xy256_z65_h200x8.yaml"
config = load_experiment_config(path)
audit_multitask_splits(config["data"], config["experiment"]["tasks"])
for task in config["experiment"]["tasks"]:
    dataset = build_task_dataset(
        config["data"], task, "train", seed=config["train"]["seed"], num_samples=1
    )
    sample = dataset[0]
    shape = tuple(sample["condition"].shape)
    if shape != (1, 65, 256, 256) or tuple(sample["target"].shape) != shape:
        raise RuntimeError(f"{task} produced invalid shapes: {shape}, {sample['target'].shape}")
    print(task, shape, sample["metadata"].get("sliding_window_start_z"), flush=True)
print("five-task XY256/Z65 data probe passed", flush=True)
PY
stage "five_task_data_probe_passed"

stage "waiting_for_existing_gpu_workloads"
while pgrep -f 'evaluate_medgen3d_final.py|train_medgen3d.py|torch.distributed.run' >/dev/null; do
  sleep 60
done

python - <<'PY' | tee -a "$log"
import torch
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
for index in range(8):
    free, total = torch.cuda.mem_get_info(index)
    if total - free > 2 * 1024**3:
        raise RuntimeError(f"GPU {index} is not free: {(total-free)/1024**3:.2f} GiB used")
print(torch.__version__, torch.version.cuda, "GPUs", torch.cuda.device_count())
import flash_attn
print("flash_attn", flash_attn.__version__)
PY
stage "gpu_environment_probe_passed"

probe_name=medgen3d_main5task_lora_all_xy256_z65_h200x8_probe
stage "two_step_probe_started"
"$env_root/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$repo/scripts/train_medgen3d.py" --config "$config" \
  --checkpoint-dir "$wan_weights" --max-steps 2 --limit-train-cases 16 \
  --run-name "$probe_name" 2>&1 | tee -a "$log"
stage "two_step_probe_passed"

stage "formal_training_started commit=$(git rev-parse HEAD)"
exec "$env_root/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$repo/scripts/train_medgen3d.py" --config "$config" \
  --checkpoint-dir "$wan_weights" 2>&1 | tee -a "$log"
