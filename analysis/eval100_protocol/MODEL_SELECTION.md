# Task-specific comparator selection

## Segmentation — nnU-Net v2 (3D full resolution)

Official implementation: https://github.com/MIC-DKFZ/nnUNet

Reference: Isensee et al., *nnU-Net: a self-configuring method for deep
learning-based biomedical image segmentation*, Nature Methods, 2021.

Rationale: it is a supervised, task-specific 3D medical segmentation system
that configures preprocessing and architecture from the training set. We train
it from scratch on the same 500 patients, use the frozen 50 validation patients
for model selection, and report the requested organ on the frozen 100 patients.

## Low-dose restoration — RED-CNN

Reference: Chen et al., *Low-Dose CT with a Residual Encoder-Decoder
Convolutional Neural Network*, IEEE TMI, 2017; arXiv:1702.00288.

Rationale: RED-CNN is specifically designed for image-domain low-dose CT
denoising, using an encoder-decoder with deconvolution and shortcut
connections. It is trained on the exact synthetic low-dose/clean pairs used by
MedGen3D.

## Sparse-view reconstruction — FBPConvNet

Reference: Jin et al., *Deep Convolutional Neural Network for Inverse Problems
in Imaging*, IEEE TIP, 2017; arXiv:1611.03679.

Rationale: FBPConvNet is the canonical learned post-processing baseline for CT
inverse problems: a multiresolution residual U-Net maps FBP reconstructions to
clean CT. It is trained specifically on the exact 90-view FBP/clean pairs in
this benchmark.

## Fairness constraints

- Identical patient-disjoint 500/50/100 split for every learned system.
- Identical 100-case manifest and input volumes at test time.
- No public pretrained weights trained on overlapping AbdomenAtlas patients.
- Validation-only checkpoint selection; no test-set tuning.
- Identical HU clipping, valid-voxel crop, metric code, and patient-level
  aggregation.
- Model parameter counts, training steps, inference time, and failures are
  recorded alongside accuracy metrics.
