# MedGen3D

MedGen3D repurposes a pretrained **Wan 2.2 video VAE and DiT** as one
feed-forward model for five 3D medical imaging task families:

- anatomical segmentation (mask targets are represented as signed distance
  fields);
- low-dose CT restoration;
- sparse-view CT reconstruction;
- T1-to-T2 MRI synthesis; and
- report-conditioned chest CT generation.

All tasks use the same latent interface, DiT backbone, text-conditioned
volume-to-volume prediction interface, and masked L2 regression objective. The
current main experiment performs direct target-latent prediction in one
Transformer pass rather than iterative flow-matching inference.

> **Repository status.** This repository contains the research training,
> preprocessing, and evaluation code. Dataset files, pretrained Wan weights,
> trained MedGen3D checkpoints, and generated outputs are not committed.

## Method at a glance

```text
task input volume + task prompt
              |
              v
       frozen Wan video VAE
              |
              v
 shared Wan DiT + task conditioning
              |
       one-pass latent prediction
              |
              v
       frozen Wan video VAE
              |
              v
        target 3D volume
```

The canonical released configuration freezes the pretrained Wan weights and
adds LoRA adapters to every DiT linear layer:

```text
configs/experiments/main5task_feedforward_lora_all_xy256_z65_ctrate_v2.yaml
```

LoRA checkpoints contain the adapters and MedGen3D conditioning modules only;
the official Wan checkpoint must also be available at inference time.

## Installation

The formal experiments use Linux, Python 3.10+, CUDA GPUs, PyTorch 2.4+, and
eight H200 GPUs. Smaller setups are useful for tests and debugging but are not
the paper's training setting.

```bash
git clone https://github.com/Jecci-cmd/MedGen3D.git
cd MedGen3D

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Install the official Wan 2.2 code and obtain the `Wan2.2-TI2V-5B` checkpoint
separately. Add the Wan repository to `PYTHONPATH` when running training or
evaluation:

```bash
export WAN_REPO=/path/to/Wan2.2
export WAN_WEIGHTS=/path/to/Wan2.2-TI2V-5B
export PYTHONPATH="$PWD/src:$WAN_REPO:$PYTHONPATH"
```

## Data and configuration

The five-task protocol uses AbdomenAtlas, BraTS 2021, and CT-RATE. Downloading
these datasets requires accepting their respective licenses and access terms.
Edit the paths in `configs/data/main5task_xy256_z65.yaml` and the preparation config
before running the scripts.

Main preparation entry points are:

```text
scripts/prepare_abdomenatlas.py
scripts/reconstruction/prepare_reconstruction_multiview.py
scripts/synthesis/prepare_brats2021_main.py
scripts/generation/prepare_ctrate_v2.py
```

### Canonical CT-RATE V2 for full-volume generation

For full-volume generation, prepare CT-RATE on a fixed physical grid:

```bash
python scripts/prepare_ctrate_v2.py \
  --files-root /path/to/ctrate/files \
  --train-reports /path/to/train_reports.csv \
  --valid-reports /path/to/valid_reports.csv \
  --output-root /path/to/ctrate_report_ct_v2 \
  --target-spacing-mm 1.5 --xy-shape 256 256 --workers 8
```

V2 reorients each scan to RAS, resamples to `1.5 mm` isotropic voxels, uses a
fixed `384 mm × 384 mm` in-plane field of view, and retains complete z extent.
Train it with
`configs/experiments/main5task_feedforward_lora_all_xy256_z65_ctrate_v2.yaml`.
Generation still uses 65-slice windows for memory efficiency, but now receives
normalized window position and extent; inference covers the complete z axis
with overlap blending.

The formal split sizes are:

| Dataset / task | Train | Validation | Test |
|---|---:|---:|---:|
| AbdomenAtlas (segmentation, restoration, reconstruction) | 1,000 | 100 | 200 |
| BraTS T1-to-T2 synthesis | 1,000 | 50 | 200 |
| CT-RATE report-to-CT generation | 1,000 | 100 | 200 |

Reconstruction is trained and validated at 18 views and evaluated at 10, 18,
and 20 views. The split and preprocessing scripts are deterministic under the
seeds stored in the configuration files.

## Sanity checks

The CPU mock path checks the data-to-loss plumbing without loading Wan weights:

```bash
python scripts/train_medgen3d.py \
  --config configs/experiments/main5task_feedforward_lora_all_xy256_z65_ctrate_v2.yaml \
  --mock
```

Before a formal run, validate the environment, configuration, and checkpoint:

```bash
python scripts/preflight_training.py \
  --config configs/experiments/main5task_feedforward_lora_all_xy256_z65_ctrate_v2.yaml \
  --checkpoint-dir "$WAN_WEIGHTS" \
  --wan-repo "$WAN_REPO"
```

## Training

Launch the all-layer LoRA experiment on eight GPUs:

```bash
torchrun --standalone --nproc_per_node=8 scripts/train_medgen3d.py \
  --config configs/experiments/main5task_feedforward_lora_all_xy256_z65_ctrate_v2.yaml \
  --checkpoint-dir "$WAN_WEIGHTS"
```

Resume from a MedGen3D checkpoint with `--resume /path/to/checkpoint.pt`. Use
`--export-portable /path/to/output.pt` to export a portable adapter checkpoint.

## Evaluation

Evaluate a checkpoint on one or more tasks:

```bash
python scripts/evaluate_medgen3d.py \
  --config configs/experiments/main5task_feedforward_lora_all_xy256_z65_ctrate_v2.yaml \
  --checkpoint /path/to/medgen3d-checkpoint.pt \
  --checkpoint-dir "$WAN_WEIGHTS" \
  --samples-per-task 200 \
  --tasks segmentation restoration reconstruction synthesis generation \
  --output-dir outputs/evaluation
```

Evaluation can be sharded with `--num-shards N --shard-index I`; merge completed
shards using `scripts/merge_evaluation_shards.py`. Metrics configured for the
five tasks include Dice/NSD/HD95/ASSD, MAE/RMSE/PSNR/SSIM, and volumetric
generation metrics. Generation first writes canonical NIfTI volumes.  The
frozen generation protocol is four complementary metrics: MAISI 2.5D FID
(RadImageNet features; lower is better), CT-CLIP Fréchet (whole-volume embedding
distance; lower is better), CT-CLIP T2I (report-to-generated-volume;
higher is better), and CT-CLIP I2I (generated-to-reference-volume; higher is
better).

The evaluation assets are local-only and are never downloaded by an evaluator.
Set `MEDGEN3D_EVAL_ASSETS` once to a directory with this layout:

```text
evaluation_assets/
├── maisi/radimagenet-models/
│   └── weights/RadImageNet-ResNet50_notop.pth
└── ctclip/
    ├── CT-CLIP/
    ├── CT-CLIP_v2.pt
    └── BiomedVLP-CXR-BERT-specialized/
```

For external baseline predictions whose NIfTI grid differs from the reference
CT, first create a strict `results.json` with
`scripts/generation/build_generation_results.py`, then pass
`--prediction-spacing header` to the CT-CLIP evaluator.  MedGen3D fused
outputs use the reference physical grid and retain the default `target` mode.

Compute MAISI FID with the official MAISI preprocessing protocol:

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/generation/evaluate_maisi_fid_2p5d.py \
  --results outputs/evaluation/results.json \
  --output outputs/evaluation/maisi_fid.json
```

Then compute the CT-CLIP metrics with the CT-RATE metadata CSV:

```bash
python scripts/generation/evaluate_ctclip_metrics.py \
  --results outputs/evaluation/results.json \
  --metadata-csv /path/to/ct_rate_validation_metadata.csv \
  --output outputs/evaluation/ctclip_metrics.json
```

MAISI FID uses RAS orientation, 1-mm resampling, `512 x 512 x 512` centre
crop/pad, the central 40% of XY/YZ/ZX slices, and a RadImageNet ResNet-50.
CT-CLIP uses its physical grid `(1.5, 0.75, 0.75)`, centre-cropped/padded to
`480 x 480 x 240`, and `[-1000, 1000]` HU clipping.  Do not compare numbers
from the removed StyleGAN-V/I3D/torchmetrics protocol with these values.

## Experiment protocol

- Hardware: 8 x H200.
- Optimizer updates: 100,000.
- Task sampling: uniform across the five tasks.
- Segmentation prompts: exact round-robin organ balancing, with organ-centred
  foreground/surface sampling and targeted zoom augmentation for small or thin
  structures.
- Inputs and targets: encoded and decoded by the same frozen video VAE.
- Adaptation variants: full-parameter training and all-layer LoRA.

## Repository layout

```text
configs/       data, model, training, and experiment configurations
scripts/       preparation, auditing, training, and evaluation entry points
src/medgen3d/  model, trainer, inference, metrics, and configuration
src/medicalmodel_data/  volumetric data and degradation pipeline
analysis/      optional frozen-cohort analysis utilities
```

## Reproducibility notes

- Keep the dataset manifests and split seeds fixed when comparing methods.
- Record both the MedGen3D commit and the official Wan checkpoint revision.
- Do not compare partial evaluation shards with complete baselines.
- HU-based CT metrics require the same clipping and inverse-normalization
  convention for predictions and references.
- The repository deliberately excludes cluster-specific launch wrappers; use
  the portable commands above on your scheduler of choice.
