#!/usr/bin/env bash
# Evaluate a trained three-task MVG checkpoint on the frozen 200-case ID and
# OOD cohorts. ID training data supplies the fixed in-context support example
# in both settings; no OOD target is ever used as support.
set -euo pipefail

REPO=${REPO:-/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/MedGen3D}
MVG=${MVG:-/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/MedGen3D-main/baselines/generalist/MVG}
PY=${PY:-/inspire/qb-ilm/project/video-generation/public/lijiaxi/.envs/medgen3d-v3/bin/python}
OUT=${OUT:-/inspire/qb-ilm/project/video-generation/public/lijiaxi/MedGen3D-main/baselines/generalist/MVG/evaluation/final_id_ood}
CKPT=${1:?"usage: $0 /path/to/model-099.pt"}

CT_ID=/inspire/ssd/project/video-generation/public/lijiaxi/medicalmodel/data/AbdomenAtlas1.0Mini
SYNTH_ID=/inspire/qb-ilm/project/video-generation/public/lijiaxi/MedGen3D-main/data/brats2021_t1_t2_xy256
OOD=/inspire/qb-ilm/project/video-generation/public/lijiaxi/MedGen3D-OOD/processed
mkdir -p "$OUT"
export PYTHONPATH="$REPO/src:$MVG${PYTHONPATH:+:$PYTHONPATH}"

"$PY" "$REPO/scripts/baselines/evaluate_mvg.py" \
  --mvg-root "$MVG" --checkpoint "$CKPT" --output "$OUT/id_metrics.json" --batch-size 12 \
  --ct-root "$CT_ID" --ct-test-manifest "$CT_ID/processed/manifests/test.jsonl" --ct-train-manifest "$CT_ID/processed/manifests/train.jsonl" \
  --synth-root "$SYNTH_ID" --synth-test-manifest "$SYNTH_ID/manifests/test.jsonl" --synth-train-manifest "$SYNTH_ID/manifests/train.jsonl"

"$PY" "$REPO/scripts/baselines/evaluate_mvg.py" \
  --mvg-root "$MVG" --checkpoint "$CKPT" --output "$OUT/ood_metrics.json" --batch-size 12 --seg-nsd-tolerance-mm 1.0 \
  --ct-root "$OOD/dap_atlas_ood" --ct-train-root "$CT_ID" \
  --ct-test-manifest "$OOD/dap_atlas_ood/manifests/restoration_ood_test.jsonl" \
  --seg-test-manifest "$OOD/dap_atlas_ood/manifests/segmentation_ood_test.jsonl" \
  --restoration-test-manifest "$OOD/dap_atlas_ood/manifests/restoration_ood_test.jsonl" --ct-train-manifest "$CT_ID/processed/manifests/train.jsonl" \
  --synth-root "$OOD/brats_peds_t1n_t2w" --synth-train-root "$SYNTH_ID" \
  --synth-test-manifest "$OOD/brats_peds_t1n_t2w/manifests/ood_test.jsonl" --synth-train-manifest "$SYNTH_ID/manifests/train.jsonl"

"$PY" "$REPO/scripts/baselines/summarize_mvg_metrics.py" \
  --id "$OUT/id_metrics.json" --ood "$OUT/ood_metrics.json" --output "$OUT/paper_rows.tex" --label "MVG"
