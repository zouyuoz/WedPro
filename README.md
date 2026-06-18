# AI-Claim Audit for Robust Vision (MILS 作業)

本專案旨在執行 **AI-Claim Audit**，透過實驗證據審核 AI 對模型穩健性（Robustness）的預測與解釋。我們針對 `ResNeXt-101` (大型架構) 與 `MobileNet-V3 Large` (輕量架構) 兩個模型，在 `ImageNet-R` (Rendition) 資料集上進行風格偏移（Style Shift）的穩健性分析。

## 核心任務
- **資料集**: 使用 `ImageNet1K-V1` 作為基準（Clean），`ImageNet-R` 作為偏移資料（Shifted）。
- **模型比較**: `ResNeXt-101` (Heavy) vs. `MobileNet-V3` (Lightweight)。
- **預訓練權重**: 兩者皆使用官方 `IMAGENET1K_V1` 版本。
- **主要指標**: Accuracy Drop, Wrong Confidence, Failure Overlap。
- **AI 審核**: 驗證 5 個關於模型穩健性的 AI 假說。

## 實驗環境
- **作業系統**: Linux (Ubuntu 24.04 核心)
- **硬體**: 
  - GPU: **NVIDIA GeForce RTX 5090** (32,607 MiB VRAM)
  - 架構: Blackwell (**sm_120**)
- **軟體版本**:
  - Python: 3.12.3
  - PyTorch: 2.11.0+cu128
  - CUDA Driver: 570.211.01 (系統回報 CUDA 12.8 相容)
- **執行說明**: 本專案已成功設定並支援在 **NVIDIA RTX 5090 (CUDA)** 上執行，利用 GPU 加速推論過程。

## 專案結構
- `data_setup.py`: 初始化環境，選定 20 個共同類別並準備 Metadata。
- `evaluate.py`: 執行模型推論，計算各項穩健性指標（支援 CPU/CUDA）。
- `analyze.py`: 分析評估結果，完成 AI Claim Audit 並產出最終表格。
- `results.csv`: 最終的 AI Claim Audit 審核結果表。
- `failure_cases/`: 自動儲存代表性的模型失敗案例影像。
- `metadata/`: 儲存實驗選定的類別資訊。

## 快速開始

### 1. 環境設定
建議使用 Python 3.10+，並建立虛擬環境：
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
*(註：若無 `requirements.txt`，請安裝 `torch`, `torchvision`, `datasets`, `pandas`, `tqdm`, `requests`, `matplotlib`, `transformers`, `weasyprint`)*

### 2. 資料集存取權限 (Hugging Face Token)
本專案使用 Hugging Face 串流讀取資料集。為確保能順利存取受限資料集（如 ImageNet-1K 相關變體）並避免 API 限制，請執行以下步驟：
1. 在 [Hugging Face 官網](https://huggingface.co/settings/tokens) 申請 Access Token。
2. 在專案根目錄建立一個名為 `HF_TOKEN.env` 的檔案。
3. 將你的 Token 寫入該檔案，格式如下：
   ```text
   HF_TOKEN=你的_token_內容
   ```
*(註：`HF_TOKEN.env` 已加入 `.gitignore`，不會被上傳至 GitHub)*

### 3. 執行流程與產出
專案分為多個階段執行，請依序執行以下腳本：

#### Step 1: 資料初始化
```bash
python data_setup.py
```
*   **產出**: `metadata/selected_classes.json` (從 `wala.csv` 映射並確認可用的 ImageNet 類別子集)。

#### Step 2: 基礎模型推論與風格標註
```bash
python evaluate_origin.py
```
*   **功能**: 執行基礎模型的推論。同時使用 CLIP 模型對每張圖片進行 Zero-shot 風格辨識與信心度計算。
*   **產出**: 
    *   `results/metrics.json` (包含所有基礎模型的預測結果與 CLIP 風格標註)。
    *   `failure_cases/` (自動儲存每個模型前幾次預測失敗的影像)。

#### Step 3: 免訓練補救措施 (TTA) 評估
```bash
python evaluate.py
```
*   **功能**: 針對特定的模型執行 Test-Time Augmentation (TTA) 多視角推論。
*   **產出**: 更新 `results/metrics.json`，將 TTA 變體的實驗數據附加至現有檔案中。

#### Step 4: 假說驗證與錯誤分析 (Audit Scripts)
依序執行以下分析腳本來驗證不同的 AI 假說：
```bash
python audit_h3.py
python audit_h4.py
python audit_h5.py
python misclassification_analysis.py
python style_analysis.py
```
*   **產出**:
    *   `results/audit_h3_results.csv`: Logit Ensemble 與 Confidence Rejection 的效益分析 (H3)。
    *   `results/audit_h4_results.json`: 失敗案例的 Jaccard 重疊率 (H4)。
    *   `results/audit_h5_results.json`: CLIP 風格信心度與模型錯誤率的相關性 (H5)。
    *   `results/misclassify_analysis_result.json`: 各模型在不同風格下最常誤判的類別統計。
    *   `results/style_error_analysis.csv`: 各風格的準確度下降報表。
    *   `results.csv`: 最終的 5 項 AI 假說審核總表 (Supported/Refuted)。

#### Step 5: 產生視覺化報告
```bash
python generate_report.py
python build_presentation.py
```
*   **產出**: 
    *   `report.pdf` (完整的綜合數據分析報告)。
    *   `presentation/index.html` (用於口頭報告的互動式投影片)。

---
*本專案為 NYCU MILS 課程作業實作。*
