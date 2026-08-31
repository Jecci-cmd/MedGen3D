# Frozen 100-case comparison protocol

Frozen on 2026-08-13 from the existing patient-disjoint `test` split. These
100 patients are evaluation-only and must not be used for fitting, checkpoint
selection, early stopping, architecture selection, or preprocessing tuning.

## Immutable cohort

- `case_ids.txt`: 100 ordered patient IDs.
- `manifest.jsonl`: exact source manifest snapshot for the same 100 patients.
- SHA-256 `case_ids.txt`: `7d670e418e5b298a09dcf9c1ee21ecc0952b3110eda52e1c58a0844106fd08c4`
- SHA-256 `manifest.jsonl`: `f9d9a5c92f9206299d353782fa17682adccc01889479cb18fddf6fc3543b09b4`

## Task contracts

All models consume the same canonical 1.5 mm isotropic CT data and are scored
case-by-case over the same valid voxels.

| Task | Input | Target | Metrics |
| --- | --- | --- | --- |
| Segmentation | canonical clean CT | requested organ mask | Dice, NSD@1 mm, HD95 mm, ASSD mm |
| Restoration | fixed synthetic low-dose CT (`i0=1e4`, 720 views) | canonical clean CT | MAE HU, RMSE HU, PSNR, SSIM |
| Reconstruction | fixed 90-view parallel-beam FBP CT | canonical clean CT | MAE HU, RMSE HU, PSNR, SSIM |

The primary segmentation comparison must use the same requested organ(s) for
both systems. Empty-ground-truth cases are reported separately rather than
silently averaged with non-empty cases. CT metrics retain paired per-patient
rows and report mean, median, standard deviation, improvement fraction, and a
patient-level bootstrap 95% confidence interval.

## Task-specific comparators

- Segmentation: nnU-Net v2 3D full-resolution, trained only on the frozen
  500-case training split and selected with the 50-case validation split.
- Restoration: RED-CNN, trained on the same paired low-dose/clean training
  volumes.
- Reconstruction: FBPConvNet, trained on the same paired 90-view FBP/clean
  training volumes.

No comparator may use the 100 evaluation patients for training or model
selection. Any departure from this protocol must be recorded beside the raw
results.
