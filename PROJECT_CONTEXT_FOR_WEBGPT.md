# Seg-MoE-v2 工程上下文文档（供网页 GPT 使用）

更新时间：2026-04-22

适用对象：
- 需要在网页 GPT（如 ChatGPT 网页版、Claude 网页版等）中快速让模型理解本工程
- 希望获得可执行的训练、调试、排障建议和指令

---

## 1. 项目一句话定义

Seg-MoE v2 是一个前列腺 mpMRI 3D 分割工程，核心目标是：

1) 用 nnU-Net v2 跑解剖三头（WG/PZ/TZ）概率图；
2) 将解剖先验用于后续病灶高召回（Layer1）与多专家融合链路；
3) 提供统一的 manifest / split / 导出管线，兼容 nnU-Net、MedNeXt、SegMamba 三类后端。

---

## 2. 当前仓库状态（基于当前工作区实况）

### 2.1 数据规模与分层

来自 [data/manifest/manifest_summary.csv](data/manifest/manifest_summary.csv)：

- 总病例数：3632
- nca：1479
- pca：2153
- test：545
- trainval：3087
- val 每折：618/618/617/617/617

### 2.2 几何审计与修复

来自 [data/manifest/geometry_audit_summary.json](data/manifest/geometry_audit_summary.json)：

- no_action：3570
- header_harmonize_recommended：52
- resample_required：10
- needs_preprocessing_count：62

来自 [data/geometry_fixed/geometry_fix_report.json](data/geometry_fixed/geometry_fix_report.json)：

- fixed_case_count：62
- adc/dwi 以 header_harmonize 为主，少量 resample
- label 绝大多数 copy，少量 header_harmonize

### 2.3 nnU-Net 解剖训练产物

可见路径：
- [nnUNet_results/Dataset501_ProstateAnatomy/nnUNetTrainerSegMoEAnatomy__nnUNetResEncUNetMPlans__3d_fullres](nnUNet_results/Dataset501_ProstateAnatomy/nnUNetTrainerSegMoEAnatomy__nnUNetResEncUNetMPlans__3d_fullres)

已有 fold 目录：
- fold_0, fold_1, fold_2, fold_3

fold_0 验证目录存在大量 npz/pkl，并有清单：
- [nnUNet_results/Dataset501_ProstateAnatomy/nnUNetTrainerSegMoEAnatomy__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation/prediction_manifest.jsonl](nnUNet_results/Dataset501_ProstateAnatomy/nnUNetTrainerSegMoEAnatomy__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation/prediction_manifest.jsonl)

实测概率包键：
- keys = channel_names, probabilities
- probabilities 形状示例 = (3, 22, 550, 576)
- channel_names = P_WG, P_PZ, P_TZ

### 2.4 当前 SegMamba 导出状态（关键）

当前 [data/exports/segmamba/segmamba_config.json](data/exports/segmamba/segmamba_config.json) 为：

- input_channels = 3
- modalities = T2W, ADC, DWI
- 无 positive_label_values 字段
- 目录下没有 arrays 子目录

这表示当前导出并非 adapter 期望的 6 通道 npz 训练形态（见后文“已知风险/阻塞”）。

---

## 3. 目录与核心入口

### 3.1 根级关键文件

- [README.md](README.md): 项目运行说明与命令流
- [pyproject.toml](pyproject.toml): 包配置、入口脚本、基础依赖
- [requirements.txt](requirements.txt): 运行依赖
- [environment.segmoe-v2.yml](environment.segmoe-v2.yml): conda 环境、编译器变量

### 3.2 Python 包入口

- [src/segmoe_v2/__main__.py](src/segmoe_v2/__main__.py)
- [src/segmoe_v2/cli/main.py](src/segmoe_v2/cli/main.py)

命令入口脚本名（pyproject）：
- segmoe-v2 -> segmoe_v2.cli.main:main

### 3.3 后端仓库（vendored）

- [external/nnU-Net](external/nnU-Net)
- [external/MedNeXt-main](external/MedNeXt-main)
- [external/SegMamba](external/SegMamba)

---

## 4. 架构总览

### 4.1 模块分层

1) 契约与通用 IO
- [src/segmoe_v2/contracts.py](src/segmoe_v2/contracts.py)
- [src/segmoe_v2/io_utils.py](src/segmoe_v2/io_utils.py)

2) 数据发现/审计/修复
- [src/segmoe_v2/manifest.py](src/segmoe_v2/manifest.py)
- [src/segmoe_v2/geometry_audit.py](src/segmoe_v2/geometry_audit.py)
- [src/segmoe_v2/geometry_fix.py](src/segmoe_v2/geometry_fix.py)

3) 标签语义与任务映射
- [src/segmoe_v2/labels.py](src/segmoe_v2/labels.py)

4) 后端数据导出
- [src/segmoe_v2/backend_data.py](src/segmoe_v2/backend_data.py)

5) nnU-Net anatomy 支撑
- [src/segmoe_v2/nnunet_anatomy.py](src/segmoe_v2/nnunet_anatomy.py)
- [src/segmoe_v2/nnunet_anatomy_predict.py](src/segmoe_v2/nnunet_anatomy_predict.py)

6) SegMamba 适配
- [src/segmoe_v2/segmamba_adapter.py](src/segmoe_v2/segmamba_adapter.py)

7) 训练数据集/采样/特征/融合/校准
- [src/segmoe_v2/datasets.py](src/segmoe_v2/datasets.py)
- [src/segmoe_v2/sampling.py](src/segmoe_v2/sampling.py)
- [src/segmoe_v2/features.py](src/segmoe_v2/features.py)
- [src/segmoe_v2/fp_bank.py](src/segmoe_v2/fp_bank.py)
- [src/segmoe_v2/fusion.py](src/segmoe_v2/fusion.py)
- [src/segmoe_v2/calibration.py](src/segmoe_v2/calibration.py)
- [src/segmoe_v2/gate.py](src/segmoe_v2/gate.py)
- [src/segmoe_v2/oof.py](src/segmoe_v2/oof.py)

8) Runner 层
- [src/segmoe_v2/runners/base.py](src/segmoe_v2/runners/base.py)
- [src/segmoe_v2/runners/nnunet.py](src/segmoe_v2/runners/nnunet.py)
- [src/segmoe_v2/runners/mednext.py](src/segmoe_v2/runners/mednext.py)
- [src/segmoe_v2/runners/segmamba.py](src/segmoe_v2/runners/segmamba.py)
- [src/segmoe_v2/runners/utils.py](src/segmoe_v2/runners/utils.py)

---

## 5. 数据契约（核心字段）

定义位置：[src/segmoe_v2/contracts.py](src/segmoe_v2/contracts.py)

### 5.1 CaseManifestRow

关键字段：
- case_id, patient_id
- era_bin, cohort_type
- has_lesion_label3, label_unique_values
- fixed_split, val_fold
- t2w_path, adc_path, dwi_path, label_path
- spacing, image_shape, affine_hash
- metadata

语义：
- cohort_type 为 pca / nca
- fixed_split 为 trainval / test
- val_fold 仅对 trainval 生效

### 5.2 TaskSpec

- anatomy: 输入 T2W/ADC/DWI，输出 WG/PZ/TZ
- lesion: 默认输入可含解剖先验 P_WG/P_PZ/P_TZ，输出 lesion

### 5.3 PredictionRecord

关键字段：
- task, stage, model_name
- fold, split, case_id, predictor_fold
- prob_path / logit_path
- channel_names
- source_manifest_hash

### 5.4 CalibrationRecord / FPComponentRecord

- CalibrationRecord：记录温度标定参数与样本统计
- FPComponentRecord：记录假阳性组件几何/统计特征

---

## 6. 关键文件格式

### 6.1 manifest（jsonl）

文件：
- [data/manifest/cases.jsonl](data/manifest/cases.jsonl)
- [data/manifest/cases.geometry_fixed.jsonl](data/manifest/cases.geometry_fixed.jsonl)
- [data/manifest/cases.geometry_fixed.localpaths.jsonl](data/manifest/cases.geometry_fixed.localpaths.jsonl)

说明：
- cases.jsonl 中可以包含 Windows 源路径
- localpaths 版本映射到当前 Linux 工作区路径，更利于本地运行

### 6.2 split 文件

- nnUNet split: json
- MedNeXt split: pkl
- SegMamba split: json

对应文件：
- [data/manifest/nnunet_splits_final.json](data/manifest/nnunet_splits_final.json)
- [data/manifest/mednext_splits_final.pkl](data/manifest/mednext_splits_final.pkl)
- [data/manifest/segmamba_splits_final.json](data/manifest/segmamba_splits_final.json)

### 6.3 几何审计与修复

- [data/manifest/geometry_audit.csv](data/manifest/geometry_audit.csv)
- [data/manifest/geometry_audit_summary.json](data/manifest/geometry_audit_summary.json)
- [data/geometry_fixed/geometry_fix_report.csv](data/geometry_fixed/geometry_fix_report.csv)
- [data/geometry_fixed/geometry_fix_report.json](data/geometry_fixed/geometry_fix_report.json)

### 6.4 nnU-Net anatomy 概率包

示例目录：
- [nnUNet_results/Dataset501_ProstateAnatomy/nnUNetTrainerSegMoEAnatomy__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation](nnUNet_results/Dataset501_ProstateAnatomy/nnUNetTrainerSegMoEAnatomy__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation)

npz 常见键：
- probabilities: [3, Z, Y, X], float32
- channel_names: P_WG, P_PZ, P_TZ

预测清单：
- [nnUNet_results/Dataset501_ProstateAnatomy/nnUNetTrainerSegMoEAnatomy__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation/prediction_manifest.jsonl](nnUNet_results/Dataset501_ProstateAnatomy/nnUNetTrainerSegMoEAnatomy__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation/prediction_manifest.jsonl)

### 6.5 SegMamba 导出物

目录：
- [data/exports/segmamba](data/exports/segmamba)

当前实际包含：
- dataset_index.jsonl
- fold_k_train/val.jsonl
- test.jsonl
- split_metadata.json
- segmamba_config.json

注意：当前没有 arrays 目录（无 npz 训练包）。

---

## 7. 三条主工作流

### 7.1 数据发现 -> 审计 -> 几何修复

1) build-manifest
- 扫描根目录，构建清单与分层 K 折

2) audit-manifest
- 校验 split 一致性、patient 泄漏、cohort 平衡

3) audit-geometry
- 生成 recommendation: no_action / header_harmonize_recommended / resample_required

4) fix-geometry-to-t2
- 以 T2 为参考修复 ADC/DWI/label 几何

代码位置：
- [src/segmoe_v2/manifest.py](src/segmoe_v2/manifest.py)
- [src/segmoe_v2/geometry_audit.py](src/segmoe_v2/geometry_audit.py)
- [src/segmoe_v2/geometry_fix.py](src/segmoe_v2/geometry_fix.py)

### 7.2 nnU-Net anatomy 训练与概率导出

1) export-nnunet-task --task anatomy

2) nnUNet plan_and_preprocess + run_training

3) 训练中可用 --npz 落盘验证概率包（见 repo 记忆说明）

4) 也可通过 segmoe_v2.nnunet_anatomy_predict 独立导出概率与 prediction_manifest

代码位置：
- [src/segmoe_v2/backend_data.py](src/segmoe_v2/backend_data.py)
- [src/segmoe_v2/nnunet_anatomy.py](src/segmoe_v2/nnunet_anatomy.py)
- [src/segmoe_v2/nnunet_anatomy_predict.py](src/segmoe_v2/nnunet_anatomy_predict.py)

### 7.3 SegMamba Layer1 数据准备与训练

标准目标链路：

1) build-gland-crop-manifest
- 从 P_WG 概率生成 ROI bbox

2) prepare-segmamba-data（带 anatomy-predictions + crop-manifest）
- 期望生成 arrays/case.npz，6 通道输入

3) segmoe_v2.segmamba_adapter train/predict

当前仓库状态提示：
- 当前导出未生成 arrays，config 为 3 通道；直接使用 adapter 训练会触发输入格式不匹配风险。

代码位置：
- [src/segmoe_v2/gland_crop.py](src/segmoe_v2/gland_crop.py)
- [src/segmoe_v2/backend_data.py](src/segmoe_v2/backend_data.py)
- [src/segmoe_v2/segmamba_adapter.py](src/segmoe_v2/segmamba_adapter.py)

---

## 8. 标签语义与任务语义（非常关键）

定义位置：[src/segmoe_v2/labels.py](src/segmoe_v2/labels.py)

### 8.1 原始标签中的关键语义

- 0: background
- 1: PZ
- 2: TZ
- 3: 病灶位点（在 PCA 表示 lesion，在 NCA 作为 mimic 来源）

### 8.2 anatomy 三头目标

- WG = 1/2/3 的并集
- PZ = 1
- TZ = 2
- PZ/TZ 在 label==3 位置使用 ignore 策略（valid_mask=False）

### 8.3 Layer1 高召回目标

- 先构建 source tri-state:
  - pca: label==3 -> 1
  - nca: label==3 -> 2
- 再 high recall 二值化：source in {1,2} 视为正类

---

## 9. CLI 命令面

定义位置：[src/segmoe_v2/cli/main.py](src/segmoe_v2/cli/main.py)

当前主要子命令：

1) build-manifest
2) audit-manifest
3) build-gland-crop-manifest
4) export-nnunet-task
5) export-mednext-task
6) prepare-segmamba-data
7) audit-geometry
8) fix-geometry-to-t2
9) visualize-anatomy-qc

---

## 10. Runner 子系统与环境变量策略

### 10.1 Runner 抽象

定义位置：[src/segmoe_v2/runners/base.py](src/segmoe_v2/runners/base.py)

接口：
- train_fold
- predict_fold
- export_probabilities

### 10.2 nnU-Net Runner

定义位置：[src/segmoe_v2/runners/nnunet.py](src/segmoe_v2/runners/nnunet.py)

默认环境变量：
- nnUNet_raw
- nnUNet_preprocessed
- nnUNet_results

anatomy 预测默认走：
- python -m segmoe_v2.nnunet_anatomy_predict

### 10.3 MedNeXt Runner

定义位置：[src/segmoe_v2/runners/mednext.py](src/segmoe_v2/runners/mednext.py)

默认环境变量：
- nnUNet_raw_data_base
- nnUNet_preprocessed
- RESULTS_FOLDER

### 10.4 SegMamba Runner

定义位置：[src/segmoe_v2/runners/segmamba.py](src/segmoe_v2/runners/segmamba.py)

环境变量：
- SEGMAMBA_DATA_DIR
- SEGMAMBA_LOGDIR
- SEGMAMBA_PREDICTION_DIR
- PYTHONPATH 注入 mamba 与 causal-conv1d

### 10.5 vendored backend 路径解析

定义位置：[src/segmoe_v2/backend_data.py](src/segmoe_v2/backend_data.py)

优先级：
1) 显式传入路径
2) 环境变量（SEGMOE_NNUNET_ROOT / SEGMOE_MEDNEXT_ROOT / SEGMOE_SEGMAMBA_ROOT）
3) external 下默认目录

---

## 11. 测试覆盖概况

测试目录：[tests](tests)

重点测试：

- manifest 与 split 审计
  - [tests/test_manifest.py](tests/test_manifest.py)
  - [tests/test_cli.py](tests/test_cli.py)

- geometry 审计与修复
  - [tests/test_geometry_audit.py](tests/test_geometry_audit.py)
  - [tests/test_geometry_fix.py](tests/test_geometry_fix.py)

- 标签/解剖损失/概率导出
  - [tests/test_labels.py](tests/test_labels.py)
  - [tests/test_nnunet_anatomy.py](tests/test_nnunet_anatomy.py)
  - [tests/test_nnunet_predictor.py](tests/test_nnunet_predictor.py)

- 后端导出与 SegMamba 适配
  - [tests/test_backend_layer1_exports.py](tests/test_backend_layer1_exports.py)
  - [tests/test_segmamba_adapter.py](tests/test_segmamba_adapter.py)

- runners
  - [tests/test_runners.py](tests/test_runners.py)

- 其它功能
  - [tests/test_sampling.py](tests/test_sampling.py)
  - [tests/test_fp_bank.py](tests/test_fp_bank.py)
  - [tests/test_fusion.py](tests/test_fusion.py)
  - [tests/test_calibration.py](tests/test_calibration.py)
  - [tests/test_anatomy_visual_qc.py](tests/test_anatomy_visual_qc.py)

---

## 12. 已知风险与易踩坑

### 12.1 路径体系混用（Windows 与 Linux）

- [data/manifest/cases.jsonl](data/manifest/cases.jsonl) 中是 Windows 路径
- Linux 训练建议优先使用 localpaths 或在导出时重写

### 12.2 SegMamba adapter 输入格式要求

- adapter 的数据集读取逻辑要求记录指向 .npz（segmamba_npz 或 image 为 .npz）
- 当前 [data/exports/segmamba/dataset_index.jsonl](data/exports/segmamba/dataset_index.jsonl) 为 NIfTI 列表，且无 arrays
- 需要先具备 anatomy 预测清单并重跑 prepare-segmamba-data 生成 arrays

### 12.3 几何修复输出中的反斜杠

- report 中可见 output_root 为 data\\geometry_fixed 形式
- 在 Linux 命令中需统一为正斜杠路径

### 12.4 nnUNet validation 概率导出行为

根据仓库记忆（nnunet_validation.md）：

- run_training 默认验证只写 summary，不一定写逐例概率
- 通过 --npz 可写出 validation/*.npz 与 prediction_manifest
- 已训练后可 --val --npz 补导出

### 12.5 SegMamba CUDA 扩展安装

项目上下文显示历史命令中存在参数拼写错误（--no-depsd-isolation），会导致安装失败。
需要修正 pip 参数并确保 CUDA/toolchain 可用。

---

## 13. 推荐给网页 GPT 的“上下文提示词”模板

下面这段可以直接复制给网页 GPT。

---

你现在是本项目的技术顾问，请基于以下事实给出可执行建议。

【项目】Seg-MoE-v2，前列腺 mpMRI 3D 分割。

【核心链路】
1) manifest 构建与审计
2) geometry audit + fix（T2 参考）
3) nnU-Net anatomy 训练（WG/PZ/TZ）与概率导出
4) SegMamba Layer1 准备与训练（可用 anatomy 先验）

【仓库关键位置】
- CLI: src/segmoe_v2/cli/main.py
- 数据契约: src/segmoe_v2/contracts.py
- 导出逻辑: src/segmoe_v2/backend_data.py
- 几何审计/修复: src/segmoe_v2/geometry_audit.py + geometry_fix.py
- nnU-Net anatomy: src/segmoe_v2/nnunet_anatomy.py + nnunet_anatomy_predict.py
- SegMamba adapter: src/segmoe_v2/segmamba_adapter.py
- runners: src/segmoe_v2/runners/

【当前状态】
- manifest 总病例 3632，test 545，trainval 3087
- 几何需处理 62 例（52 header harmonize，10 resample）
- nnUNet fold_0 验证目录已有大量 anatomy 概率 npz 与 prediction_manifest
- 当前 data/exports/segmamba 为 3 通道配置，尚无 arrays 子目录

【关键约束】
- 标签语义：0背景，1PZ，2TZ，3病灶位点（PCA lesion / NCA mimic 来源）
- Layer1 高召回正类 = {1,2}（由 source tri-state 转换）
- SegMamba adapter 训练期望 .npz 记录，不是原始 nifti 列表

【你要输出】
1) 先给“最短可执行路径”（按命令顺序）
2) 再给“风险检查清单”（每步如何验收）
3) 如果发现当前状态与目标不一致，明确指出阻塞点与修复命令
4) 输出命令时区分 Linux shell 与 Python 命令

---

## 14. 推荐向网页 GPT 提问的任务类型

1) 环境排障
- 例如：SegMamba CUDA 扩展编译失败如何定位（gcc/nvcc/torch/cuda 兼容矩阵）

2) 训练计划制定
- 例如：在 2 GPU 条件下，nnU-Net anatomy + SegMamba Layer1 的最短闭环计划

3) 数据健康检查
- 例如：如何自动检查 manifest 路径有效性、split 完整性、几何修复覆盖率

4) 结果评估与迭代
- 例如：如何基于 anatomy 概率和 Layer1 日志制定下一轮超参策略

---

## 15. 给网页 GPT 的回答风格约束（建议一起贴）

请按以下格式回复：

1) 先给“立即执行版”命令（不超过 10 条）
2) 每条命令附带 1 行验收标准
3) 标注潜在破坏性操作（删除/覆盖）
4) 若信息不足，先提最多 5 个关键澄清问题

---

## 16. 附：关键源码导航（便于模型定位）

- [src/segmoe_v2/cli/main.py](src/segmoe_v2/cli/main.py)
- [src/segmoe_v2/contracts.py](src/segmoe_v2/contracts.py)
- [src/segmoe_v2/manifest.py](src/segmoe_v2/manifest.py)
- [src/segmoe_v2/backend_data.py](src/segmoe_v2/backend_data.py)
- [src/segmoe_v2/labels.py](src/segmoe_v2/labels.py)
- [src/segmoe_v2/geometry_audit.py](src/segmoe_v2/geometry_audit.py)
- [src/segmoe_v2/geometry_fix.py](src/segmoe_v2/geometry_fix.py)
- [src/segmoe_v2/gland_crop.py](src/segmoe_v2/gland_crop.py)
- [src/segmoe_v2/nnunet_anatomy.py](src/segmoe_v2/nnunet_anatomy.py)
- [src/segmoe_v2/nnunet_anatomy_predict.py](src/segmoe_v2/nnunet_anatomy_predict.py)
- [src/segmoe_v2/segmamba_adapter.py](src/segmoe_v2/segmamba_adapter.py)
- [src/segmoe_v2/datasets.py](src/segmoe_v2/datasets.py)
- [src/segmoe_v2/sampling.py](src/segmoe_v2/sampling.py)
- [src/segmoe_v2/fp_bank.py](src/segmoe_v2/fp_bank.py)
- [src/segmoe_v2/fusion.py](src/segmoe_v2/fusion.py)
- [src/segmoe_v2/calibration.py](src/segmoe_v2/calibration.py)
- [src/segmoe_v2/gate.py](src/segmoe_v2/gate.py)
- [src/segmoe_v2/oof.py](src/segmoe_v2/oof.py)
- [src/segmoe_v2/runners/base.py](src/segmoe_v2/runners/base.py)
- [src/segmoe_v2/runners/nnunet.py](src/segmoe_v2/runners/nnunet.py)
- [src/segmoe_v2/runners/mednext.py](src/segmoe_v2/runners/mednext.py)
- [src/segmoe_v2/runners/segmamba.py](src/segmoe_v2/runners/segmamba.py)
- [src/segmoe_v2/runners/utils.py](src/segmoe_v2/runners/utils.py)

---

如果你希望，我可以继续基于这个文档生成两份衍生版本：

1) “超短版（1 页）”给网页 GPT 快速理解
2) “执行清单版（逐步命令 + 验收项）”给你直接在终端跑
