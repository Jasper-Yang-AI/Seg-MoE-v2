# Layer1 current model statistics

Generated from Seg-MoE-v2 current local artifacts. Core model is nnUNetResEnc Layer1 source-aware candidate model. Task definition: one sigmoid candidate head; PCA lesion label 1 and NCA mimic label 2 are both treated as positives.

Files:
- layer1_current_fold_summary.csv: one row per fold with training and validation summary.
- layer1_current_cohort_summary.csv: fold/cohort and all-fold cohort aggregation.
- layer1_current_case_metrics.csv: per validation case metrics at threshold 0.5.
- layer1_current_threshold_sweep.csv: threshold sweep from saved validation probabilities.
- layer1_current_backend_status.csv: non-nnUNet backend status; MedNeXt is not a completed expert yet.
