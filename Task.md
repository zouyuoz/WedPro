# AI-Claim Audit Project Plan & Automation Blueprint
**Target Audience:** Autonomous Execution Agent / LLM-Developer
**Objective:** Verify or debunk 5 specific hypotheses regarding Model Robustness, Style Shifts, and Non-training Remediations using ImageNet-V1 and ImageNet-R across four target models.

---

## Phase 1: Environment Setup & Dataset Provisioning
### [x] [TODO 1.1] Environment Initialization
- [x] Create a Python 3.10+ virtual environment.
- [x] Install essential libraries via `pip`: `torch`, `torchvision`, `datasets`, `pandas`, `tqdm`, `requests`, `transformers` (using Hugging Face `transformers` instead of direct `CLIP.git` for stability).

### [x] [TODO 1.2] Dataset Acquisition & Structuring
- [x] Download **ImageNet-R** (Rendition / Shifted dataset) via HF Streaming.
- [x] Create a class mapping dictionary (`metadata/imagenet_class_index.json` and `data_setup.py`) to align the selected classes with ImageNet-1K.

### [x] [TODO 1.3] Metadata Definition
- [x] Define target styles (`metadata/renditions.json`): 16 official ImageNet-R styles used instead of hardcoded 15.

## Phase 2: Style Annotation via Zero-Shot CLIP
### [x] [TODO 2.1] Implement Style Annotation (Merged into `evaluate.py`)
- [x] Load the pre-trained `CLIP` model (`openai/clip-vit-base-patch32`).
- [x] Generate prompt templates for the styles (`"a {} rendition"`).
- [x] Iterate through all images in **ImageNet-R**:
	- [x] Extract features and compute Softmax to get confidence scores.
	- [x] Assign the style with the highest probability.
- [x] Save the results seamlessly into the unified JSON metrics stream (`results/metrics.json`).

## Phase 3: Model Inference & Unified Metrics Logging
### [x] [TODO 3.1] Implement Inference (`evaluate.py`)
- [x] Initialize the four target models with exact pre-trained weights:
    1. ResNeXt-101: `IMAGENET1K_V1`
    2. MobileNet-V3: `IMAGENET1K_V1`
    3. ViT-b-16 (V1): `IMAGENET1K_V1`
    4. ViT-b-16 (SWAG): `IMAGENET1K_SWAG_E2E_V1`
- [x] Set all models to `.eval()` mode and wrap with `torch.no_grad()`.
- [x] Define standard validation transforms.
- [x] Execute inference on  **ImageNet-R**.
- [x] Log raw outputs into a unified JSON schema `results/metrics.json`.

## Phase 4: Hypothesis Verification Suite (Audit Scripts)
### [x] [TODO 4.1] Audit Script 1: Overall Domain Shift Performance (Merged into `style_analysis.py`)
- [x] **Objective**: Verify if `ViT-b-16 (SWAG)` > `ViT-b-16 (V1)` > `ResNeXt-101` > `MobileNet`.
- [x] **Logic**: Load JSON, calculate Accuracy Drop, rank models.
- [x] **Output**: Saved in `results.csv` (C1) and summary table.

### [x] [TODO 4.2] Audit Script 2: Style-Specific Accuracy Drop (Merged into `style_analysis.py`)
- [x] **Objective**: Verify style drops (CNN vs ViT).
- [x] **Logic**: Group by style, compute drop, identify max drop.
- [x] **Output**: Saved in `results/style_error_analysis.csv` and `results.csv` (C2, C4).

### [x] [TODO 4.3] Audit Script 3: Non-Training Remediation Simulator (`audit_h3.py`)
- [x] **Objective**: Test if Cross-Architecture Logit Ensemble combined with Test-Time Augmentation (TTA) yields the highest boost.
- [x] **Logic**:
    - [x] **Simulate Logit Ensemble**: Combine logits of `ResNeXt-101` and `ViT-b-16 (SWAG)` via simple average: $\text{Logits}_{\text{ensemble}} = 0.5 \cdot \text{Logits}_{\text{ResNeXt}} + 0.5 \cdot \text{Logits}_{\text{ViT\_SWAG}}$. Calculate new Accuracy.
    - [x] **Simulate TTA (if raw images are re-evaluated)**: Run a sub-inference using horizontal flips and slight Gaussian blurs, averaging predictions. *(Skipped: requires re-running raw image inferences)*
    - [x] **Simulate Model Agreement Check / Confidence Rejection**: Reject predictions where ResNeXt and ViT-b-16 (SWAG) disagree, or filter out samples below a threshold. Check if accuracy on the accepted subset improves.
    - [x] Compare the accuracy gains across all non-training strategies.
- [x] **Output**: Save optimization metrics to `results/audit_h3_results.csv`.

### [x] [TODO 4.4] Audit Script 4: Failure Overlap Metrics (`audit_h4.py`)
- [x] **Objective**: Verify if Architecture dictates Failure Overlap more than Data Scale.
- [x] **Logic**:
    - [x] For each style, extract indices of misclassified images (`is_correct == False`).
    - [x] Compute the **Jaccard Similarity** of misclassified sets between pairs:
    - [x] ViT Pair: `(ViT-b-16 V1, ViT-b-16 SWAG)` -> Shared Architecture, Different Scale.
    - [x] High-Perf Cross Pair: `(ResNeXt-101, ViT-b-16 V1)` -> Different Architecture, Similar Training Scale.
    - [x] For overlapping failures, check if the incorrect `prediction` label matches between models (Wrong Prediction Match Rate).
- [x] **Output**: Generate a pair-wise correlation/overlap matrix saved to `results/audit_h4_results.json`.

### [x] [TODO 4.5] Audit Script 5: CLIP Confidence vs. Wrong Confidence Correlation (`audit_h5.py`)
- [x] **Objective**: Verify if higher CLIP style confidence correlates with higher CNN error rates and severe Wrong Confidence.
- [x] **Logic**:
    - [x] Bin ImageNet-R samples by CLIP `style_confidence` (e.g., bins: `[0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]`).
    - [x] For each bin, calculate the Error Rate (`1 - Accuracy`) of all models.
    - [x] Filter misclassified data points (`is_correct == False`), plot or calculate the correlation between CLIP `style_confidence` and the models' own `confidence`.
    - [x] Calculate the Pearson Correlation Coefficient ($r$) between CLIP confidence and Model Misclassification Rate.
- [x] **Output**: Save statistical test values and correlation coefficients to `results/audit_h5_results.json`.

## Phase 5: Consolidated Summary & Audit Report
### [x] [TODO 5.1] Execution of Report Generation (`generate_report.py`)
- [x] Consolidate results from Phase 4 scripts.
- [x] Format a final HTML/PDF reporting hypotheses (`report.pdf`).
- [ ] *Action Required: Update `generate_report.py` to include H3, H4, and H5 once implemented.*

1. 把 `ResNeXt-101`, `MobileNet-V3` 的 pretrain-weight 改成 IMAGENET1K-V1，可以直接看到他在 IMAGENET1K-V1 上的 acc:
	```Python
	    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 5483032,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 74.042,
                    "acc@5": 91.340,
                }
            },
            "_ops": 0.217,
            "_file_size": 21.114,
            "_docs": """These weights were trained from scratch by using a simple training recipe.""",
        },
    )
	```
	因此在 `evaluate.py` 中，就也不需要計算 clean 了，把跑 clean 相關的程式碼都刪除。
2. 我將會自己改動各種 .md 文件、報告中，對於 clean 資料集的改動 (ImageNet v1->v2)，這你不用改，主要改 evaluate.py 的程式碼就好了