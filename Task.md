# AI-Claim Audit Project Plan & Automation Blueprint
**Target Audience:** Autonomous Execution Agent / LLM-Developer
**Objective:** Verify or debunk 5 specific hypotheses regarding Model Robustness, Style Shifts, and Non-training Remediations using ImageNet-V2 and ImageNet-R across four target models.

---

## Phase 1: Environment Setup & Dataset Provisioning
### [TODO 1.1] Environment Initialization
- [ ] Create a Python 3.10+ virtual environment.
- [ ] Install essential libraries via `pip`:
  ```bash
  pip install torch torchvision timm ftfy regex tqdm pandas numpy scikit-learn requests
  pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)
  ```

### [TODO 1.2] Dataset Acquisition & Structuring
- [ ] Download **ImageNet-V2** (Clean baseline). Ensure it contains the standard 1000 classes mapped to ImageNet labels.
- [ ] Download **ImageNet-R** (Rendition / Shifted dataset). Note: ImageNet-R only contains 200 out of the 1000 ImageNet classes.
- [ ] Create a class mapping dictionary (`imagenet_r_to_v2.json`) to align the 200 classes of ImageNet-R with the standard 1000-class indices of ImageNet-1K for fair evaluation.

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
### [TODO 3.1] Implement `run_inference.py`
- [ ] Initialize the four target models with exact pre-trained weights:
    1. ResNeXt-101: `torchvision.models.resnext101_32x8d(weights="IMAGENET1K_V2")`
    2. MobileNet-V3: `torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2")`
    3. ViT-b-16 (V1): `torchvision.models.vit_b_16(weights="IMAGENET1K_V1")`
    4. ViT-b-16 (SWAG): `torchvision.models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1")`
- [ ] Set all models to `.eval()` mode and wrap with `torch.no_grad()`.
- [ ] Define a standard validation DataLoader for ImageNet-V2 and ImageNet-R (using appropriate `IMAGENET1K_V2` transforms: Resize 232/256, CenterCrop 224, Normalize).
- [ ] Execute inference on **ImageNet-V2** and **ImageNet-R** for all 4 models.
- [ ] Log raw outputs into a unified JSON schema metrics_raw.json:
    ```JSON
    {
        "model_name": {
            "imagenet_v2": [
                {"image_id": "img_01", "target": 195, "prediction": 195, "confidence": 0.92, "is_correct": true}
            ],
            "imagenet_r": [
                {
                    "image_id": "img_99", 
                    "target": 195, 
                    "prediction": 917, 
                    "confidence": 0.2767, 
                    "is_correct": false,
                    "style": "painting",
                    "style_confidence": 0.82,
                    "logits": [0.01, -0.5, 3.2, ...] // Optional: save top-5 logits if space permits
                }
            ]
        }
    }
    ```
## Phase 4: Hypothesis Verification Suite (Audit Scripts)
### [TODO 4.1] Audit Script 1: Overall Domain Shift Performance (`audit_h1.py`)
- [ ] **Objective**: Verify if `ViT-b-16 (SWAG)` > `ViT-b-16 (V1)` > `ResNeXt-101` > `MobileNet`.
- [ ] **Logic**:
    - [ ] Load `metrics_raw.json`.
    - [ ] Calculate baseline Accuracy on ImageNet-V2 for each model (filtered by the subset of 200 classes shared with ImageNet-R to ensure a controlled baseline).
    - [ ] Calculate Accuracy on ImageNet-R.
    - [ ] Compute `Accuracy Drop = Accuracy(V2_subset) - Accuracy(ImageNet-R)`.
    - [ ] Rank models by absolute accuracy on ImageNet-R and by minimal Accuracy Drop.
- [ ] **Output**: Save ranking dataframe to `audit_h1_results.csv`.

### [TODO 4.2] Audit Script 2: Style-Specific Accuracy Drop (`audit_h2.py`)
- [ ] **Objective**: Verify if CNNs drop most on `"line drawing"`, while ViTs drop most on `"pattern"`/`"embroidery"`.
- [ ] **Logic**:
    - [ ] Group ImageNet-R samples by their assigned `style`.
    - [ ] For each model, compute accuracy within each of the 15 style groups.
    - [ ] Calculate the specific drop compared to the clean baseline.
    - [ ] Identify the style that triggers the maximum drop for the CNN group (`ResNeXt`, `MobileNet`) and the ViT group (`V1`, `SWAG`).
- [ ] **Output**: Save pivot table (Styles $\times$ Models Accuracy) to `audit_h2_results.csv`.

### [TODO 4.3] Audit Script 3: Non-Training Remediation Simulator (`audit_h3.py`)
- [ ] **Objective**: Test if Cross-Architecture Logit Ensemble combined with Test-Time Augmentation (TTA) yields the highest boost.
- [ ] **Logic**:
    - [ ] **Simulate Logit Ensemble**: Combine logits of `ResNeXt-101` and `ViT-b-16 (SWAG)` via simple average: $\text{Logits}_{\text{ensemble}} = 0.5 \cdot \text{Logits}_{\text{ResNeXt}} + 0.5 \cdot \text{Logits}_{\text{ViT\_SWAG}}$. Calculate new Accuracy.
    - [ ] **Simulate TTA (if raw images are re-evaluated)**: Run a sub-inference using horizontal flips and slight Gaussian blurs, averaging predictions.
    - [ ] **Simulate Model Agreement Check / Confidence Rejection**: Reject predictions where ResNeXt and ViT-b-16 (SWAG) disagree, or filter out samples below a threshold. Check if accuracy on the accepted subset improves.
    - [ ] Compare the accuracy gains across all non-training strategies.
- [ ] **Output**: Save optimization metrics to `audit_h3_results.csv`.

### [TODO 4.4] Audit Script 4: Failure Overlap Metrics (`audit_h4.py`)
- [ ] **Objective**: Verify if Architecture dictates Failure Overlap more than Data Scale.
- [ ] **Logic**:
    - [ ] For each style, extract indices of misclassified images (`is_correct == False`).
    - [ ] Compute the **Jaccard Similarity** of misclassified sets between pairs:
    - [ ] ViT Pair: `(ViT-b-16 V1, ViT-b-16 SWAG)` -> Shared Architecture, Different Scale.
    - [ ] High-Perf Cross Pair: `(ResNeXt-101, ViT-b-16 V1)` -> Different Architecture, Similar Training Scale.
    - [ ] For overlapping failures, check if the incorrect `prediction` label matches between models (Wrong Prediction Match Rate).
- [ ] **Output**: Generate a pair-wise correlation/overlap matrix saved to `audit_h4_results.json`.

### [TODO 4.5] Audit Script 5: CLIP Confidence vs. Wrong Confidence Correlation (`audit_h5.py`)
- [ ] **Objective**: Verify if higher CLIP style confidence correlates with higher CNN error rates and severe Wrong Confidence.
- [ ] **Logic**:
    - [ ] Bin ImageNet-R samples by CLIP `style_confidence` (e.g., bins: `[0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]`).
    - [ ] For each bin, calculate the Error Rate (`1 - Accuracy`) of all models.
    - [ ] Filter misclassified data points (`is_correct == False`), plot or calculate the correlation between CLIP `style_confidence` and the models' own `confidence`.
    - [ ] Calculate the Pearson Correlation Coefficient ($r$) between CLIP confidence and Model Misclassification Rate.
- [ ] **Output**: Save statistical test values and correlation coefficients to `audit_h5_results.json`.

## Phase 5: Consolidated Summary & Audit Report
### [x] [TODO 5.1] Execution of Report Generation (`generate_report.py`)
- [x] Consolidate results from Phase 4 scripts.
- [x] Format a final HTML/PDF reporting hypotheses (`report.pdf`).
- [ ] *Action Required: Update `generate_report.py` to include H3, H4, and H5 once implemented.*