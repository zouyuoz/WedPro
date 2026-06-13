# AI-Claim Audit for Robust Vision (MILS 作業)

本專案旨在執行 **AI-Claim Audit**，透過實驗證據審核 AI 對模型穩健性（Robustness）的預測與解釋。我們針對 `VGG-19` 與 `ViT-B/16` 兩個模型，在 `ImageNet-R` (Rendition) 資料集上進行風格偏移（Style Shift）的穩健性分析。

## 核心任務
- **資料集**: 使用 `ImageNet-V2` 作為基準（Clean），`ImageNet-R` 作為偏移資料（Shifted）。
- **模型比較**: `VGG-19` (CNN 架構) vs. `ViT-B/16` (Transformer 架構)。
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
*(註：若無 `requirements.txt`，請安裝 `torch`, `torchvision`, `datasets`, `pandas`, `tqdm`, `requests`, `matplotlib`, `transformers`)*

### 2. 資料集存取權限 (Hugging Face Token)
本專案使用 Hugging Face 串流讀取資料集。為確保能順利存取受限資料集（如 ImageNet-1K 相關變體）並避免 API 限制，請執行以下步驟：
1. 在 [Hugging Face 官網](https://huggingface.co/settings/tokens) 申請 Access Token。
2. 在專案根目錄建立一個名為 `HF_TOKEN.env` 的檔案。
3. 將你的 Token 寫入該檔案，格式如下：
   ```text
   HF_TOKEN=你的_token_內容
   ```
*(註：`HF_TOKEN.env` 已加入 `.gitignore`，不會被上傳至 GitHub)*

### 3. 執行流程
專案分為三個階段執行：

1.  **資料初始化**:
    ```bash
    python data_setup.py
    ```
2.  **執行模型評估**:
    （此步驟會透過 Hugging Face Streaming 模式讀取資料，不需下載完整 ImageNet 資料集）
    ```bash
    python evaluate.py
    ```
3.  **產出分析報告**:
    ```bash
    python analyze.py
    ```

## 實驗結果摘要

| 模型 | Clean Acc | Shifted Acc | Accuracy Drop |
| :--- | :--- | :--- | :--- |
| **VGG-19** | 75.50% | 17.00% | 0.5850 |
| **ViT-B/16** | 84.00% | 28.50% | 0.5550 |

### 主要發現 (Audit Decisions)
- **C1 (Supported)**: ViT 在風格偏移下的準確度下降（Accuracy Drop）確實比 VGG-19 小。
- **C3 (Refuted)**: 修正了 AI 的預測，實驗發現 ViT 在答錯時反而擁有更高的平均信心值，存在更嚴重的過度自信問題。
- **C5 (Refuted)**: 儘管架構不同，VGG 與 ViT 的失敗案例重疊率高達 **80.17%**。

## 提交 GitHub 說明
1. 初始化 Git 倉庫: `git init`
2. 新增所有檔案: `git add .`
3. 提交變更: `git commit -m "Complete MILS AI-Claim Audit Assignment"`
4. 推送至遠端: `git push origin main`

---
*本專案為 NYCU MILS 課程作業實作。*
