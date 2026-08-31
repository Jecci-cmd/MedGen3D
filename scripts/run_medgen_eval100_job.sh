#!/usr/bin/env bash
set -euo pipefail

BASE=/inspire/ssd/project/exploration-topic/public/MedGen3D-git
WAN_SRC=/inspire/ssd/project/exploration-topic/public/Wan2.2-official
PIP_INDEX=http://nexus.sii.shaipower.online/repository/pypi/simple/
LOG="$BASE/analysis/task_specific/logs/medgen3d_eval100_job.log"
cd "$BASE"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "$(date -Is) job environment preparation"
python -m pip install easydict ftfy wcwidth nibabel scipy scikit-image matplotlib pyyaml \
  'transformers==4.51.3' 'diffusers>=0.31.0' 'accelerate>=1.1.1' \
  tokenizers safetensors sentencepiece protobuf opencv-python-headless imageio imageio-ffmpeg \
  --index-url "$PIP_INDEX" --trusted-host nexus.sii.shaipower.online
if ! python -c 'import flash_attn' >/dev/null 2>&1; then
  MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation \
    --index-url "$PIP_INDEX" --trusted-host nexus.sii.shaipower.online
fi

export PYTHONPATH="src:$WAN_SRC"
python - <<'PY'
import torch, torchvision, wan
from wan.modules import attention
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("wan", wan.__file__)
print("FA2", attention.FLASH_ATTN_2_AVAILABLE, "FA3", attention.FLASH_ATTN_3_AVAILABLE)
assert torch.cuda.is_available()
assert attention.FLASH_ATTN_2_AVAILABLE
PY

COMMON=(
  --config configs/experiments/multitask.yaml
  --checkpoint outputs/medgen3d_joint_v1_formal_20260811/step_00029000.pt
  --checkpoint-dir Wan2.2-TI2V-5B
  --sampling-steps 30
  --seed 20260812
)

echo "$(date -Is) one-case reconstruction smoke test"
python scripts/evaluate_medgen3d.py "${COMMON[@]}" \
  --samples-per-task 1 --tasks reconstruction \
  --output-dir analysis/task_specific/medgen3d_smoke_job
touch analysis/task_specific/MEDGEN3D_SMOKE_COMPLETE

echo "$(date -Is) resumable 100-case, three-task evaluation"
python scripts/evaluate_medgen3d.py "${COMMON[@]}" \
  --samples-per-task 100 \
  --output-dir analysis/task_specific/medgen3d_eval100_official_compare
touch analysis/task_specific/MEDGEN3D_COMPLETE
echo "$(date -Is) MedGen3D eval100 complete"
