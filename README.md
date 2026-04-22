# SegMoE v2

SegMoE v2 is a 3D prostate mpMRI segmentation workspace. The current Linux
layout is aligned around three vendored backends:

- `external/nnU-Net`: active anatomy closed loop.
- `external/MedNeXt-main`: nnU-Net v1 style MedNeXt backend.
- `external/SegMamba`: SegMamba source tree and model import path.

The first production training loop is still nnU-Net anatomy. MedNeXt data export
and runner wiring are aligned. SegMamba uses the SegMoE adapter
`python -m segmoe_v2.segmamba_adapter` for Layer1 lesion+mimic training instead
of the upstream random split scripts.

## Environment

Create the conda environment from Linux/WSL:

```bash
conda env create -f environment.segmoe-v2.yml
conda activate segmoe
python -m pip install --upgrade pip
```

Install PyTorch separately so the wheel matches your GPU driver/CUDA stack. On
the current RTX 5090 WSL setup, the working build was CUDA 13:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Then install the project runtime and editable package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Optional editable installs for backend entry points:

```bash
python -m pip install -e external/MedNeXt-main --no-deps
python -m pip install -e external/SegMamba/causal-conv1d
python -m pip install -e external/SegMamba/mamba
```

Notes:

- `environment.segmoe-v2.yml` includes Linux C/C++ compiler wrappers and sets
  `CC` / `CXX`. This is required by `torch.compile` / Triton.
- Do not let `pip install -e .` install dependencies again; use `--no-deps` so
  your selected PyTorch build is not replaced.
- SegMamba's `mamba` and `causal-conv1d` are CUDA extensions. They are not
  needed for the nnU-Net anatomy loop.

Sanity check:

```bash
python - <<'PY'
from pathlib import Path
import torch
from segmoe_v2.backend_data import resolve_vendored_backend_root

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("nnUNet:", resolve_vendored_backend_root("nnunet"))
print("MedNeXt:", resolve_vendored_backend_root("mednext"))
print("SegMamba:", resolve_vendored_backend_root("segmamba"))
PY
```

## Shell Setup

For direct backend commands, set the source paths and workspaces:

```bash
export PYTHONPATH="$PWD/src:$PWD/external/nnU-Net:$PWD/external/MedNeXt-main:$PWD/external/SegMamba:$PWD/external/SegMamba/mamba:$PWD/external/SegMamba/causal-conv1d"

export nnUNet_raw="$PWD/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnUNet_results"

export nnUNet_raw_data_base="$PWD/MedNeXt_raw_data_base"
export RESULTS_FOLDER="$PWD/MedNeXt_results"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
mkdir -p "$nnUNet_raw_data_base/nnUNet_raw_data" "$nnUNet_preprocessed" "$RESULTS_FOLDER"
```

## Data Manifest

Build one canonical case manifest and matching backend splits:

```bash
python -m segmoe_v2.cli.main build-manifest \
  --roots /path/to/njmu_2012_2019_nca_zscore /path/to/njmu_2012_2019_pca_zscore /path/to/njmu_2020_2023_nca_zscore /path/to/njmu_2020_2023_pca_zscore \
  --manifest-out data/manifest/cases.jsonl \
  --summary-out data/manifest/manifest_summary.csv \
  --nnunet-splits-out data/manifest/nnunet_splits_final.json \
  --mednext-splits-out data/manifest/mednext_splits_final.pkl \
  --segmamba-splits-out data/manifest/segmamba_splits_final.json
```

Audit the manifest:

```bash
python -m segmoe_v2.cli.main audit-manifest \
  --manifest data/manifest/cases.jsonl \
  --nnunet-splits data/manifest/nnunet_splits_final.json \
  --mednext-splits data/manifest/mednext_splits_final.pkl \
  --segmamba-splits data/manifest/segmamba_splits_final.json
```

If geometry warnings appear during planning, audit and repair before export:

```bash
python -m segmoe_v2.cli.main audit-geometry \
  --manifest data/manifest/cases.jsonl

python -m segmoe_v2.cli.main fix-geometry-to-t2 \
  --manifest data/manifest/cases.jsonl \
  --audit-csv data/manifest/geometry_audit.csv
```

Use `data/manifest/cases.geometry_fixed.jsonl` for later export commands if you
ran the repair step.

## nnU-Net Anatomy Loop

Export the anatomy task into the nnU-Net v2 raw workspace:

```bash
python -m segmoe_v2.cli.main export-nnunet-task \
  --manifest data/manifest/cases.geometry_fixed.jsonl \
  --task-root "$nnUNet_raw" \
  --dataset-id 501 \
  --dataset-name ProstateAnatomy \
  --task anatomy
```

Plan and preprocess with the same planner name used by training:

```bash
python -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints \
  -d 501 \
  --verify_dataset_integrity \
  --clean \
  -pl nnUNetPlannerResEncM \
  -c 3d_fullres \
  -npfp 8 \
  -np 4
```

Train fold 0 on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m nnunetv2.run.run_training \
  501 3d_fullres 0 \
  -tr nnUNetTrainerSegMoEAnatomy \
  -p nnUNetResEncUNetMPlans \
  -num_gpus 2 \
  --npz
```

You do not install a separate DDP package. `-num_gpus 2` makes nnU-Net spawn two
PyTorch distributed processes and use NCCL through `torch.distributed`.

`torch.compile` is enabled by nnU-Net unless `nnUNet_compile` disables it. If
compiler/Triton issues block progress, run the same command with:

```bash
nnUNet_compile=f CUDA_VISIBLE_DEVICES=0,1 python -m nnunetv2.run.run_training \
  501 3d_fullres 0 \
  -tr nnUNetTrainerSegMoEAnatomy \
  -p nnUNetResEncUNetMPlans \
  -num_gpus 2 \
  --npz
```

The first compiled epoch can spend several minutes in Triton compilation. That
is normal. The C compiler error means `CC` / `CXX` is not visible inside the
activated conda environment.

## Anatomy Outputs

The custom anatomy trainer uses:

- input channels: `T2W`, `ADC`, `DWI`
- output heads: `P_WG`, `P_PZ`, `P_TZ`
- masked anatomy loss around lesion label `3`
- T2-only consistency regularization
- hierarchy postprocessing: `P_PZ <= P_WG`, `P_TZ <= P_WG`,
  `P_PZ + P_TZ <= P_WG`

Validation probability exports are written under:

```text
nnUNet_results/Dataset501_ProstateAnatomy/nnUNetTrainerSegMoEAnatomy__nnUNetResEncUNetMPlans__3d_fullres/fold_<fold>/validation
```

The primary anatomy cache is one `.npz` per case with `probabilities` shaped
`(3, Z, Y, X)` and `channel_names = ["P_WG", "P_PZ", "P_TZ"]`. The validation
manifest writes `split = "val_<fold>"`. Hard masks are only optional QC sidecars,
not downstream inputs.

Generate a visual QC pack after validation:

```bash
python -m segmoe_v2.cli.main visualize-anatomy-qc \
  --manifest data/manifest/cases.geometry_fixed.jsonl \
  --prediction-manifest nnUNet_results/Dataset501_ProstateAnatomy/nnUNetTrainerSegMoEAnatomy__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation/prediction_manifest.jsonl \
  --output-dir data/qc/anatomy_fold0
```

## MedNeXt Backend

MedNeXt uses the old nnU-Net v1 workspace layout. Export into
`$nnUNet_raw_data_base/nnUNet_raw_data`, not directly into
`$nnUNet_raw_data_base`:

```bash
python -m segmoe_v2.cli.main export-mednext-task \
  --manifest data/manifest/cases.geometry_fixed.jsonl \
  --task-root "$nnUNet_raw_data_base/nnUNet_raw_data" \
  --dataset-id 502 \
  --dataset-name ProstateLesion \
  --task lesion
```

Plan and preprocess:

```bash
python -m nnunet_mednext.experiment_planning.nnUNet_plan_and_preprocess \
  -t 502 \
  --verify_dataset_integrity \
  -tf 8 \
  -tl 8
```

Train one fold:

```bash
CUDA_VISIBLE_DEVICES=0 python -m nnunet_mednext.run.run_training \
  3d_fullres \
  nnUNetTrainerV2_MedNeXt_S_kernel3 \
  502 \
  0 \
  -p nnUNetPlansv2.1_trgSp_1x1x1 \
  --npz
```

The `MedNeXtRunner` now resolves `external/MedNeXt-main` automatically and
uses `python -m nnunet_mednext...` by default, so editable install is useful but
not required for runner subprocesses.

## SegMamba Backend

Layer1 SegMamba is wired through the SegMoE adapter, not the upstream BraTS
`3_train.py`/`4_predict.py` defaults. Build the Stage A+ gland ROI manifest
from anatomy probabilities first:

```bash
python -m segmoe_v2.cli.main build-gland-crop-manifest \
  --manifest data/manifest/cases.geometry_fixed.jsonl \
  --anatomy-predictions data/exports/anatomy_oof/prediction_manifest.jsonl \
  --output data/exports/layer1/gland_crop_manifest.jsonl \
  --min-crop-size-zyx 24 192 192
```

Layer1 gland crops are candidate-first ROIs. The crop is based on the largest
`P_WG >= 0.35` component with a 12 mm margin, then expanded to at least
`(24, 192, 192)` in Z/Y/X when native image size allows. Crop-space predictions
store `bbox_zyx` and `native_shape_zyx`; use the shared ROI helper to reinflate
them for full-image QC, component mining, FP-bank visualization, and Layer2
statistics.

Prepare SegMamba Layer1 ROI data and split/index files:

```bash
python -m segmoe_v2.cli.main prepare-segmamba-data \
  --manifest data/manifest/cases.geometry_fixed.jsonl \
  --output-dir data/exports/segmamba \
  --task lesion \
  --anatomy-predictions data/exports/anatomy_oof/prediction_manifest.jsonl \
  --crop-manifest data/exports/layer1/gland_crop_manifest.jsonl
```

This writes:

- `arrays/<case_id>.npz` with `data` shape `(6,Z,Y,X)` and tri-state `seg`
- `dataset_index.jsonl`
- `fold_<k>_train.jsonl`
- `fold_<k>_val.jsonl`
- `test.jsonl`
- `split_metadata.json`
- `segmamba_config.json`

Layer1 lesion labels are high-recall candidate labels:

- `0=background`
- `1=PCA lesion`
- `2=NCA mimic`

The Layer1 target treats `{1,2}` as positive. Layer2 later treats NCA mimic and
Layer1 FP components as hard negatives for refinement.

Layer1 exports are source-aware:

- `label==1` PCA lesion remains positive with BCE weight `1.25`
- `label==2` NCA mimic remains positive with BCE weight `0.75`
- background uses weight `1.0`

SegMamba `.npz` arrays contain `seg_source`, `seg_target`, and `voxel_weight`.
nnU-Net and MedNeXt lesion exports write binary candidate labels plus sidecar
`sourceLabels*` and `weights*` folders.

To prepare all three Layer1 experts from the same manifest/prior/crop contract:

```bash
python -m segmoe_v2.cli.main prepare-layer1-moe \
  --manifest data/manifest/cases.geometry_fixed.jsonl \
  --anatomy-predictions data/exports/anatomy_oof/prediction_manifest.jsonl \
  --crop-manifest data/exports/layer1/gland_crop_manifest.jsonl \
  --config-out data/exports/layer1/layer1_moe_config.json \
  --nnunet-task-root nnUNet_raw \
  --nnunet-dataset-id 502 \
  --nnunet-dataset-name ProstateLayer1 \
  --mednext-task-root MedNeXt_raw_data_base/nnUNet_raw_data \
  --mednext-dataset-id 502 \
  --mednext-dataset-name ProstateLayer1 \
  --segmamba-output-dir data/exports/segmamba
```

The generated `layer1_moe_config.json` defines the three Layer1 experts:

- nnU-Net: local boundary expert
- MedNeXt: large-kernel/multiscale context expert
- SegMamba: long-range ROI context expert

The `SegMambaRunner` resolves `external/SegMamba` and injects:

- `external/SegMamba`
- `external/SegMamba/mamba`
- `external/SegMamba/causal-conv1d`

into `PYTHONPATH`. Runner calls with `config=data/exports/segmamba/segmamba_config.json`
use `python -m segmoe_v2.segmamba_adapter` by default and export crop-space
`.npz` logits as the primary Layer1 prediction artifact.

## Tests

Run the repo tests that cover manifest/export and runner alignment:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
pytest tests/test_cli.py tests/test_manifest.py tests/test_runners.py -q
```

## Troubleshooting

If conda cannot resolve `repo.anaconda.com`, keep using a conda-forge mirror and
`nodefaults`. This environment file does not require Anaconda defaults.

If `torch.compile` fails with `Failed to find C compiler`, check:

```bash
echo "$CC"
echo "$CXX"
which "$CC"
which "$CXX"
```

If compilation is too slow during smoke testing, set `nnUNet_compile=f`. Use the
compiled path again for real timing comparisons once the environment is stable.

If background augmentation workers crash with `fft_conv_pytorch` memory errors,
the vendored nnU-Net trainer has Gaussian blur benchmarking disabled for the
default augmentation path.
