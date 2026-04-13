# SegMoE v2

SegMoE v2 rebuild for 3D prostate mpMRI with anatomy priors, strict OOF
calibration, FP-bank hard negatives, lesion-only refinement, and lesion-only
gating.

## Environment

You can continue with your existing `segmoe` environment. A fresh env is still
supported, but it is not required.

### Use Existing `segmoe` Env

```powershell
conda activate segmoe
pip install -r requirements\v2-runtime.txt
pip install -e . --no-deps
```

Optional test tools:

```powershell
pip install -r requirements\v2-dev.txt
pytest -q
```

Quick sanity check:

```powershell
python -c "import torch, nibabel, batchgenerators, batchgeneratorsv2, acvl_utils, dynamic_network_architectures, SimpleITK, monai; print('deps ok')"
python -c "import segmoe_v2; from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor; print('segmoe_v2 + nnUNet ok')"
```

## Data Foundation CLI

The first stable workflow layer is the manifest and backend data-prep layer.
Run these commands before wiring anatomy, layer1, layer2, or gate training.

### 1. Build Manifest And Backend Splits

`build-manifest` writes:

- `manifest.jsonl`
- `manifest_summary.csv`
- `nnUNet splits_final.json`
- `nnFormer splits_final.pkl`

Example:

```powershell
$env:PYTHONPATH='src'; 
python -m segmoe_v2.cli.main build-manifest --roots E:\NJMU_prostate_data\njmu_2012_2019_nca_zscore E:\NJMU_prostate_data\njmu_2012_2019_pca_zscore E:\NJMU_prostate_data\njmu_2020_2023_nca_zscore E:\NJMU_prostate_data\njmu_2020_2023_pca_zscore --manifest-out data\manifest\cases.jsonl --nnunet-splits-out data\manifest\nnunet_splits_final.json --nnformer-splits-out data\manifest\nnformer_splits_final.pkl
```

Optional:

- `--summary-out` to override the default `manifest_summary.csv` location
- `--patient-map` with a CSV containing `case_id,patient_id`

### 2. Audit Manifest

```powershell
python -m segmoe_v2.cli.main audit-manifest `
  --manifest data\manifest\cases.jsonl `
  --nnunet-splits data\manifest\nnunet_splits_final.json `
  --nnformer-splits data\manifest\nnformer_splits_final.pkl
```

The audit checks:

- duplicate cases
- missing modality or label files
- patient leakage
- per-fold train/val consistency
- nnUNet/nnFormer split alignment
- PCA/NCA balance in test and each validation fold

### 3. Export Backend Data

The export layer is task-aware.

- `anatomy` keeps the raw `0/1/2/3` labels and relies on
  `nnUNetTrainerSegMoEAnatomy` for `WG/PZ/TZ` target adaptation and masked
  loss.
- `lesion` exports binary lesion labels where `PCA: label==3 -> 1` and
  `NCA -> 0`.

nnUNet:

```powershell
python -m segmoe_v2.cli.main export-nnunet-task `
  --manifest data\manifest\cases.jsonl `
  --task-root data\exports\nnunet `
  --dataset-id 501 `
  --dataset-name ProstateAnatomy `
  --task anatomy
```

nnFormer:

```powershell
python -m segmoe_v2.cli.main export-nnformer-task `
  --manifest data\manifest\cases.jsonl `
  --task-root data\exports\nnformer `
  --dataset-id 502 `
  --dataset-name ProstateLesion `
  --task lesion
```

SwinUNETR:

```powershell
python -m segmoe_v2.cli.main prepare-swinunetr-data `
  --manifest data\manifest\cases.jsonl `
  --output-dir data\exports\swinunetr `
  --task anatomy
```

This writes dataset index files and fold-specific train/val/test lists for the
later supervised SwinUNETR runner.

## What Next After Manifest

The immediate next step is not `layer1/layer2/gate`. The first real model loop
is:

1. export anatomy task for nnUNet
2. run nnUNet planning and preprocessing
3. train anatomy `fold 0`
4. let validation export `P_WG`, `P_PZ`, `P_TZ` probabilities

### Anatomy nnUNet First Loop

Set the nnUNet workspace variables once in your current shell:

```powershell
$root = (Resolve-Path ".").Path
$env:PYTHONPATH="src;src\segmoe_v2\nnU-Net"
$env:nnUNet_raw=Join-Path $root "nnUNet_raw"
$env:nnUNet_preprocessed=Join-Path $root "nnUNet_preprocessed"
$env:nnUNet_results=Join-Path $root "nnUNet_results"
New-Item -ItemType Directory -Force $env:nnUNet_raw, $env:nnUNet_preprocessed, $env:nnUNet_results | Out-Null
```

These three folders are expected directly under the repository root so they are
easy to inspect:

- [nnUNet_raw](/e:/Seg-MoE-v2/nnUNet_raw)
- [nnUNet_preprocessed](/e:/Seg-MoE-v2/nnUNet_preprocessed)
- [nnUNet_results](/e:/Seg-MoE-v2/nnUNet_results)

Export the anatomy task directly into `nnUNet_raw`:

```powershell
python -m segmoe_v2.cli.main export-nnunet-task `
  --manifest data\manifest\cases.jsonl `
  --task-root $env:nnUNet_raw `
  --dataset-id 501 `
  --dataset-name ProstateAnatomy `
  --task anatomy
```

Run planning and preprocessing for `3d_fullres`:

```powershell
python -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints `
  -d 501 `
  --verify_dataset_integrity `
  --clean `
  -pl nnUNetPlannerResEncM `
  -c 3d_fullres `
  -npfp 8 `
  -np 4
```

This is important: the training command below uses `-p nnUNetResEncUNetMPlans`, so
you must generate the matching plans file with `-pl nnUNetPlannerResEncM`.
If you only run the default planner, you will get `nnUNetPlans.json` but not
`nnUNetResEncUNetMPlans.json`, and training will fail with `FileNotFoundError`.

Train anatomy `fold 0` with the custom trainer:

```powershell
python -m nnunetv2.run.run_training `
  501 `
  3d_fullres `
  0 `
  -tr nnUNetTrainerSegMoEAnatomy `
  -p nnUNetResEncUNetMPlans `
  --npz
```

The anatomy trainer now applies the full SegMoE-v2 anatomy rule set:

- inputs are `T2W + ADC + DWI`
- `WG/PZ/TZ` are implemented as three independent sigmoid heads
- `T2` is never dropped
- `ADC/DWI` use independent modality dropout with default `p=0.35`
- a `T2-only` branch is trained with probability-space consistency regularization
- `lambda_consistency=0.2` with a `10 epoch` linear warm-up
- consistency is masked out on `label==3` voxels
- exported probabilities are postprocessed with hierarchy consistency:
  - `P_PZ <= P_WG`
  - `P_TZ <= P_WG`
  - `P_PZ + P_TZ <= P_WG`

This validation path is probability-first. It writes anatomy probabilities, not
hard masks, into the fold validation folder:

- channel order: `P_WG`, `P_PZ`, `P_TZ`
- main artifact: `.npz`
- sidecar metadata: `prediction_manifest.jsonl`
- exported probabilities already include hierarchy-consistency postprocessing

If you later want to rerun only fold-0 validation probability export:

```powershell
python -m nnunetv2.run.run_training `
  501 `
  3d_fullres `
  0 `
  -tr nnUNetTrainerSegMoEAnatomy `
  -p nnUNetResEncUNetMPlans `
  --val `
  --npz
```

### Why Anatomy First

Anatomy is currently the first supported closed loop because:

- the `WG/PZ/TZ` semantics already live in the custom trainer, not in the
  nnUNet label schema
- validation exports `3-channel probability` bundles in the exact format needed
  for later lesion priors
- `SwinUNETR` and `nnFormer` data prep are ready, but their supervised anatomy
  training loops are not the first blocking step

## Geometry Audit Before Training

If `nnUNet plan_and_preprocess` warns that modalities do not share the same
direction or origin, run the geometry audit on the full manifest before
training.

```powershell
python -m segmoe_v2.cli.main audit-geometry `
  --manifest data\manifest\cases.jsonl
```

By default this writes:

- `data\manifest\geometry_audit.csv`
- `data\manifest\geometry_audit_summary.json`

The CSV is one row per case and compares `T2W / ADC / DWI / label` geometry
using:

- `shape`
- `spacing`
- `origin`
- `direction`
- `axcodes`
- `affine`

The final recommendation per case is one of:

- `no_action`
- `header_harmonize_recommended`
- `resample_required`

Interpretation:

- `no_action`: geometry differences are negligible; training can proceed
- `header_harmonize_recommended`: likely small header drift only; inspect before training, and consider rewriting geometry to a common reference
- `resample_required`: there is a substantive mismatch in shape, orientation, spacing, or direction; fix the case before training

If you want to tighten or relax the thresholds:

```powershell
python -m segmoe_v2.cli.main audit-geometry `
  --manifest data\manifest\cases.jsonl `
  --soft-origin-mm 0.005 `
  --hard-origin-mm 0.5 `
  --soft-direction 1e-5 `
  --hard-direction 1e-4
```

This step is useful to decide whether you can train directly, only need mild
header harmonization, or need real pre-training resampling/alignment.

## Anatomy Visual QC

Before promoting anatomy priors into lesion `layer1`, generate a small visual
QC pack from exported anatomy probabilities.

Example:

```powershell
python -m segmoe_v2.cli.main visualize-anatomy-qc `
  --manifest data\manifest\cases.geometry_fixed.jsonl `
  --prediction-manifest nnUNet_results\Dataset501_ProstateAnatomy\nnUNetTrainerSegMoEAnatomy__nnUNetResEncUNetMPlans__3d_fullres\fold_0\validation\prediction_manifest.jsonl `
  --output-dir data\qc\anatomy_fold0
```

Default sampling is:

- `5` normal trainval cases
- `3` lesion cases
- `2` geometry-fixed cases

Outputs:

- overlay PNGs with `T2W + P_WG/P_PZ/P_TZ`
- `selection_summary.json`

Use this QC step to verify:

- `PZ/TZ` do not obviously leak outside gland
- `WG` is not hollowed out by lesion voxels
- lesion-adjacent `PZ/TZ` are still anatomically plausible

## Geometry Fix To T2

If `geometry_audit_summary.json` reports flagged cases, you can repair only
those cases and write a patched manifest that points to the fixed files.

The repair policy is:

- use `T2W` as the geometry reference
- keep `T2W` as-is
- `ADC/DWI` are either copied, header-harmonized, or resampled to `T2W`
- labels use nearest-neighbor resampling when resample is required
- unflagged cases remain unchanged in the patched manifest

Run the repair:

```powershell
python -m segmoe_v2.cli.main fix-geometry-to-t2 `
  --manifest data\manifest\cases.jsonl `
  --audit-csv data\manifest\geometry_audit.csv
```

By default this writes:

- `data\geometry_fixed\...` repaired image bundles
- `data\manifest\cases.geometry_fixed.jsonl`
- `data\geometry_fixed\geometry_fix_report.csv`
- `data\geometry_fixed\geometry_fix_report.json`

If you only want to repair the stronger mismatches:

```powershell
python -m segmoe_v2.cli.main fix-geometry-to-t2 `
  --manifest data\manifest\cases.jsonl `
  --audit-csv data\manifest\geometry_audit.csv `
  --include-recommendations resample_required
```

If you rerun the command and want to overwrite previous repaired outputs:

```powershell
python -m segmoe_v2.cli.main fix-geometry-to-t2 `
  --manifest data\manifest\cases.jsonl `
  --audit-csv data\manifest\geometry_audit.csv `
  --overwrite
```

After that, switch later export/training commands to the patched manifest:

```powershell
python -m segmoe_v2.cli.main export-nnunet-task `
  --manifest data\manifest\cases.geometry_fixed.jsonl `
  --task-root $env:nnUNet_raw `
  --dataset-id 501 `
  --dataset-name ProstateAnatomy `
  --task anatomy
```

## Vendored Backends

The upstream backends are kept as vendored source trees under:

- `src/segmoe_v2/nnU-Net`
- `src/segmoe_v2/nnFormer`
- `src/segmoe_v2/SwinUNETR`

They are integrated in "source-tree mode". The SegMoE runners resolve these
roots automatically and inject the required `PYTHONPATH` and environment
variables, so they do not need to be repackaged as `segmoe_v2` subpackages.
