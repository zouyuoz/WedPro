# AI-Claim Audit for Robust Vision
MILS Assignment Report
Testing AI Explanations of Model Robustness with Evidence

## 1. 實驗設定 (Problem Setting)
### 1.1
比較不同模型架構在面對分佈偏移（Distribution Shift）時的穩健性。
- Clean 資料集: ImageNet-1K (直接使用 `torchvision` 提供的 `acc@1`)
- Shifted 資料集: ImageNet-R (包含 15 種風格變體)
- 受測模型: ResNeXt-101, MobileNet-V3, ViT-B16-V1, ViT-SWAG

### 1.2
說明 ImageNet-R 是有 15 種不同風格的物件的資料集
<圖片1>

### 1.3
使用 AI: Gemini 網頁版 (AI claims、其他), Gemini CLI (程式碼開發)
一開始是先用 plan mode 將所有要求完成，但效果不好，就從輸出結果去"誘達" AI，產生第二版的 AI claims，再去驗證，得到本版本。
<圖片2>

## 2. AI 假說審核總覽
prompt:
> 
```
# AI-Claim Audit for Robust Vision (MILS 作業)
本專案旨在執行 **AI-Claim Audit**，透過實驗證據審核 AI 對模型穩健性（Robustness）的預測與解釋。我們針對 `ResNeXt-101` (大型架構) 與 `MobileNet-V3 Large` (輕量架構) 兩個模型，在 `ImageNet-R` (Rendition) 資料集上進行風格偏移（Style Shift）的穩健性分析。

## 核心任務
- **資料集**: 使用 `ImageNet-V2` 作為基準（Clean），`ImageNet-R` 作為偏移資料（Shifted）。
- **模型比較**: `ResNeXt-101` (Heavy) vs. `MobileNet-V3` (Lightweight)。
- **預訓練權重**: 兩者皆使用官方 `IMAGENET1K_V2` 版本。
- **主要指標**: Accuracy Drop, Wrong Confidence, Failure Overlap。
- **AI 審核**: 驗證 5 個關於模型穩健性的 AI 假說。
- **風格**: ImageNet-R 有以下風格，由於資料集沒有風格標籤，將使用預訓練 CLIP 模型來判斷風格
["art","cartoon","graffiti","embroidery","graphics","origami","painting","pattern","plastic object","plush object","sculpture","line drawing","tattoo","toy","video game"
]
以上是這個 task 的要求，現在我要來問你，幾個問題
```

| ID | 假說內容 (Claim) | 證據 (Evidence) | 審核結果 |
|---|---|---|---|
| H1 | Overall Domain Shift: ViT-b-16 (SWAG) > ViT-b-16 (V1) > ResNeXt-101 > MobileNet | Accuracy Drop sorting reveals ResNeXt-101 drops less than ViT-V1. | 50-50 |
| H2 | Style-Specific Drop: CNNs drop most on 'line drawing', ViTs on 'pattern'/'embroidery'. | Both CNNs and ViTs have maximum drops on 'graffiti' and 'tattoo'. | Refuted |
| H3 | Non-Training Remediation: Cross-Architecture Logit Ensemble yields the highest boost. | Ensemble decreased acc by 5.75%. Agreement Rejection increased acc by 14.23%. | 50-50 |
| H4 | Failure Overlap: Architecture dictates Failure Overlap more than Data Scale. | Cross-Arch Overlap: 83.61%, Same-Arch Overlap: 63.58% | Refuted |
| H5 | CLIP Confidence vs Error: Higher style confidence correlates with higher CNN error rates. | Pearson r between Style Confidence and Error Rate is very weak (~0.1). | Refuted |

## 3. 各項 AI claim
### 3.1: 整體 Domain Shift 表現
問：這兩個模型，誰在進行 domain shift 之後表現會比較好，為甚麼？
> 預測排名：ViT-b-16 (SWAG) > ViT-b-16 (V1) $\ge$ ResNeXt-101 > MobileNet原因分析：ViT-b-16 (SWAG) 表現最優：該模型核心優勢在於其採用的 IMAGENET1K_SWAG_E2E_V1 權重。SWAG 策略在含有數十億張影像的弱監督資料集（IG-3B）上進行了預訓練。海量且多樣化的數據讓模型建立起極其強韌的基底，實質上打破了常規 ImageNet-1K 訓練帶來的特徵侷限，對各種未知的 Domain Shift（包含風格扭曲）具備極強的 Generalization 能力。Architecture Bias（架構偏誤）的釋放：標準的 ViT-b-16 (V1) 雖然只在 ImageNet-1K 上訓練，但因為 ViT 架構本質上依賴 Self-Attention 來捕捉全域語義，不像 CNN 受到局部局部卷積核的限制，因此 ViT 天生具備更高的 Shape Bias。在應對 ImageNet-R 這種結構完整、僅表面紋理改變的資料集時，ViT 的架構優勢使其抗跌幅能力通常優於或至少持平於重型 CNN（ResNeXt-101）。MobileNet 依舊墊底：受限於模型容量（Capacity）過小，為了效率犧牲了提取不變性特徵的能力，極度依賴局部像素紋理，因此最容易崩潰。

驗證:
| Model | Clean Acc | Shifted Acc | Acc Drop |
|---|---|---|---|
| ResNeXt-101 | 79.31% | 27.43% | 0.5189 |
| MobileNet-V3 | 74.04% | 21.30% | 0.5274 |
| ViT-B16-V1 | 81.07% | 28.07% | 0.5299 |
| ViT-B16-SWAG | 85.30% | 50.30% | 0.3500 |
- 發現：SWAG 預訓練大幅降低了 Accuracy Drop。而標準訓練的 ViT-B16-V1 的 Accuracy Drop (0.5299) 甚至大於兩個傳統 CNN 模型，顯示其對 Domain Shift 最為敏感。
- 以 shifted Acc 來看，AI 是對的。
- 但若以 Acc drop 來看，AI 則低估了標準 ViT 架構的脆弱性，其跌幅高過 ResNeXt-101 與 MobileNet-V3。

# 3.2: Style-Specific 脆弱點
問：哪種風格的acc drop 會最大，為甚麼？可以分為兩個模型分別回答，也可以說兩種模型都對一種風格有最大的 acc drop
> 這裡必須將模型分為 CNN 陣營 與 ViT 陣營 來進行分析，因為兩者的特徵提取機制截然不同。
CNN 陣營 (ResNeXt-101, MobileNet)："line drawing"（線條畫）的跌幅最大。
原因：CNN 的淺層網絡高度依賴邊緣、顏色與局部紋理的組合。線條畫將影像內部所有的色彩、幾何陰影全部濾除，僅剩下極度抽象且稀疏的線條。這會導致 CNN 的特徵活化地圖（Activation Map）出現嚴重的稀疏化，使其無法提取出有效的特徵進行分類。
ViT 陣營 (ViT-b-16 V1, ViT-b-16 SWAG)："pattern"（重複圖案）或 "embroidery"（刺繡）的跌幅最大。
原因：ViT 的第一步是將影像切分成固定大小的 Patch（例如 16x16 像素）並轉換為 Token。當面對 "pattern" 或 "embroidery" 時，全圖佈滿了強烈且高頻的重複幾何紋理或針線噪聲。在單個 Patch 內部，原本連續的物體邊緣被這些密集噪聲切斷，導致 Patch Embedding 階段就引入了錯誤的特徵編碼，進而影響後續 Self-Attention 的全域關聯計算。

驗證:
根據 `style_error_analysis.csv` 的實驗數據，AI 的預測被完全推翻（Refuted）：
1. **CNN 陣營**：ResNeXt-101 與 MobileNet-V3 跌幅最大的風格皆為 **"graffiti" (塗鴉)** 與 **"tattoo" (刺青)**，而非預測的 "line drawing"。
2. **ViT 陣營**：ViT-B16-V1 跌幅最大的同樣是 **"tattoo"** 與 **"graffiti"**；ViT-B16-SWAG 跌幅最大的則是 **"graffiti"** 與 **"video game"**。並非預測的 "pattern" 或 "embroidery"。
這顯示不論是 CNN 還是 ViT，面對 ImageNet-R 時，破壞物體邊界與局部紋理的 "graffiti" 與 "tattoo" 才是共同的最嚴重脆弱點。此外，AI 預測的 "line drawing" 在 CNN 的跌幅排名中僅居中，顯示 AI 過度強調了線條抽象化的威脅。

# 3.3: Style-Specific 脆弱點
問：對於資料進行甚麼非訓練的處理 (包含**但不限**test-time augmentation、simple preprocessing、confidence rejection、model ensemble、prompt ensemble、model agreement check 等等)，能夠對於domain shift後的表現有最大的提升
> 推薦策略：Cross-Architecture Logit Ensemble（跨架構 Logit 融合） + Test-Time Resolution Scaling（測試期解析度縮放）
具體操作與提升原因：
跨架構的 Model Ensemble：你現在擁有傳統 CNN（ResNeXt）與新式 ViT 兩種完全不同 Inductive Bias（歸納偏誤）的架構。CNN 關注局部紋理與連續變換，ViT 關注全域結構與 Patch 關聯。將 ResNeXt-101 與 ViT-b-16 (SWAG) 的輸出 Logits 進行加權平均，可以實現特徵互補，最大程度抹平單一架構在特定風格上的盲點。
針對 ViT 的 Resolution Scaling：ViT 對輸入尺寸非常敏感。在進行測試時，稍微提升輸入解析度（例如從 224x224 提升到 384x384），並對 Position Embedding 進行雙線性插值。這能讓 ViT 在面對 embroidery 或 pattern 等高頻風格噪聲時，在 Patch 內部保留更多原本物體的相對結構，被證實能顯著提升 ViT 在 OOD（Out-of-Distribution）資料上的精準度。

驗證:
1. Model Ensemble: table (audit_h3_results.csv):

    | Strategy | Coverage | Acc on Accepted | Gain vs SWAG |
    |---|---|---|---|
    | Baseline (ResNeXt-101) | 100.00% | 27.43% | -22.88% |
    | Baseline (ViT-B16-SWAG) | 100.00% | 50.30% | +0.00% |
    | Ensemble (Confidence-Weighted) | 100.00% | 44.55% | -5.75% |
    | Agreement Rejection | 37.85% | 64.53% | +14.23% |
- **單純 Model Ensemble 反而導致準確率下降 (-5.75%)**，因為表現較差的模型 (ResNeXt-101) 拖累了表現較好的模型 (ViT-B16-SWAG)，兩者的特徵並未如 AI 預期般完美互補。
- 一致性拒絕 (Agreement Rejection) 才能顯著提升留存樣本準確度 (+14.23%)，但代價是覆蓋率大幅縮減至 37.85%。

2. TTA (Test time augmentation):

    | Metric | Shifted Accuracy | TTA Shifted Accuracy | $\delta$ Acc |
    |---|---|---|---|
    | ResNeXt-101-TTA | 0.27425 | 0.28900 | +0.01475 |
    | MobileNet-V3-TTA | 0.21300 | 0.21025 | -0.00275 |
    | ViT-B16-V1-TTA | 0.28075 | 0.29175 | +0.01100 |
    | ViT-B16-SWAG-TTA | 0.50300 | 0.51375 | +0.01075 |
- TTA 帶來的提升非常微小（約 1%），MobileNet 甚至出現微幅下降。

綜合來看，AI 的核心建議（Logit Ensemble）在實驗中宣告失敗，甚至產生負面影響。然而，由於 TTA 確實有微幅正向收益，且非訓練處理（如拒絕機制）確實能提升準確率，故給予 **50-50** 的評價。

# 3.4: Style-Specific 脆弱點
問：在 Style Shift 下，ResNeXt-101 與 MobileNet-V3 發生錯誤的資料點，其不合常理的預測類別（Wrong Predictions）是否會高度重合？重合度在哪些風格最高？
> 假設性回答：網路架構（Architecture）對 Failure Overlap 的決定性，顯著高於預訓練數據量。
推測原因：
預期實驗結果會顯示 ViT-b-16 (V1) 與 ViT-b-16 (SWAG) 即使準確率差異巨大，但只要兩者同時出錯時，其預測出的錯誤標籤（Wrong Predictions）重合度會極高。這是因為它們共享了 Tokenization 與 Global Attention 的底層數學邏輯，對於風格影像的「誤判視角」是一致的。相反地，ResNeXt-101 與 ViT-b-16 (V1) 即使在某些風格上的絕對準確率接近，它們發生錯誤的資料點與錯誤類別也會高度發散，因為 CNN 與 ViT 的失敗機制（Failure Mechanism）本質上相互獨立。

驗證:
| Model Pair | Jaccard Similarity | Wrong Prediction Match Rate |
|---|---|---|
| ResNeXt-101 vs MobileNet-V3 | 0.8314 | 28.80% |
| ResNeXt-101 vs ViT-B16-V1 | 0.8361 | 32.64% |
| ResNeXt-101 vs ViT-B16-SWAG | 0.6179 | 28.75% |
| MobileNet-V3 vs ViT-B16-V1 | 0.8313 | 30.13% |
| MobileNet-V3 vs ViT-B16-SWAG | 0.5901 | 22.46% |
| ViT-B16-V1 vs ViT-B16-SWAG | 0.6358 | 31.94% |

- 原本 AI 預測「相同架構（兩個 ViT）的錯誤重合度會最高」，但數據顯示 **ViT-B16-V1 vs ViT-B16-SWAG** 的 Jaccard Similarity 僅有 **0.6358**。
- 相反地，傳統 CNN (ResNeXt-101) 與標準 ViT (ViT-B16-V1) 在同樣都只用 ImageNet-1K 訓練的情況下，Jaccard Similarity 高達 **0.8361**。這有力地證實了 **預訓練數據規模與分佈 (Data Scale/Distribution) 對於模型錯誤交集的決定性，遠大於網路架構 (Architecture) 本身**。

- SWAG 確實孤立了錯誤分佈。只要是與 ViT-B16-SWAG 進行配對，Jaccard Similarity 都會大幅度下降至 0.61 甚至 0.59 左右。這說明經由海量弱監督數據預訓練後，模型的決策邊界與失敗模式已經與常規 ImageNet-1K 訓練的模型產生了本質上的脫鉤。

# 3.5: Style-Specific 脆弱點
問：當 CLIP 以極高的信心度（Confidence）將某張 ImageNet-R 圖片歸類為特定風格（例如 painting 信心度 $>\! 0.9$）時，CNN 模型（ResNeXt/MobileNet）在該圖片上的錯誤率是否會顯著高於 CLIP 信心度較低的圖片？
> 假設性回答：標準訓練的 ViT-b-16 (V1) 展現出的 Wrong Confidence 情況會最嚴重；而 ViT-b-16 (SWAG) 的信心度分佈會明顯得到 Calibration（校準）。
推測原因：
標準的 ViT 在有限的 ImageNet-1K 數據上訓練時，容易對某些特定的 Patch 組合產生過擬合（Overfitting）或異常的高激活（Overactivation）。當 Domain Shift 帶來的風格圖案剛好觸發了這些高激活特徵時，標準 ViT 會毫無根據地給出極高的置信度。而 ViT-b-16 (SWAG) 在預訓練階段看過了數十億張、包含各種雜訊與奇特上下文的影像，其決策邊界（Decision Boundary）被高度平滑化，因此在面對未知的風格偏移時，即使猜錯，其輸出機率也會相對平均（Entropy 較高），表現出更理性的「不知道就說不知道」，顯著降低 Wrong Confidence。

驗證:
: h5_error_by_confidence_plot.png
: h5_wrong_conf_comparison_plot.png

- **相關性非常弱**：根據 `audit_h5_results.json`，CLIP 的風格信心度（Style Confidence）與模型的錯誤率（Error Rate）之間的 Pearson $r$ 僅在 **0.08 ~ 0.12** 之間。
- **過度自信普遍存在**：實驗顯示，模型在 Domain Shift 下的錯誤並非集中在風格特徵強烈的圖片上。無論 CLIP 的風格信心度高低，模型都展現出相似的錯誤率與高度的 "Wrong Confidence"（錯誤預測但置信度高），這說明模型對未知分佈的誤判是系統性的，而非由單一風格強度驅動。AI 的該項假說被完全推翻（Refuted）。

## 4. 總結與反思
- AI 生成的 5 個假說幾乎全被實驗推翻。
- 大型預訓練 (SWAG) 的影響力大於單純更換架構 (ViT vs CNN)。
- 模型在極端領域偏移下存在嚴重的過度自信 (Overconfidence)。
- 實驗驗證 (Empirical Audit) 是 AI 時代不可或缺的一環。

<!-- 
========================= ResNeXt-101 Style Analysis ==============
  Style              | Total  | Error  | Acc    | Drop    | Conf  
-------------------------------------------------------------------
  graffiti.......... | 323    | 288    |  10.8% | 0.6847 |  52.8%
  tattoo............ | 467    | 398    |  14.8% | 0.6453 |  51.8%
  plush object...... | 266    | 222    |  16.5% | 0.6277 |  53.6%
  pattern........... | 112    | 92     |  17.9% | 0.6145 |  61.2%
  embroidery........ | 475    | 373    |  21.5% | 0.5784 |  60.4%
  ========================= MobileNet-V3 Style Analysis ===========
  Style              | Total  | Error  | Acc    | Drop    | Conf  
-------------------------------------------------------------------
  tattoo............ | 467    | 426    |   8.8% | 0.6526 |  43.4%
  graffiti.......... | 323    | 289    |  10.5% | 0.6351 |  46.1%
  cartoon........... | 296    | 259    |  12.5% | 0.6154 |  55.1%
  plush object...... | 266    | 229    |  13.9% | 0.6013 |  50.7%
  toy............... | 53     | 45     |  15.1% | 0.5895 |  46.7%
  ========================= ViT-B16-V1 Style Analysis =============
  Style              | Total  | Error  | Acc    | Drop    | Conf  
-------------------------------------------------------------------
  tattoo............ | 467    | 411    |  12.0% | 0.6908 |  44.4%
  graffiti.......... | 323    | 274    |  15.2% | 0.6590 |  39.9%
  pattern........... | 112    | 92     |  17.9% | 0.6321 |  46.4%
  plush object...... | 266    | 216    |  18.8% | 0.6227 |  45.9%
  embroidery........ | 475    | 381    |  19.8% | 0.6128 |  51.8%
  ========================= ViT-B16-SWAG Style Analysis ===========
  Style              | Total  | Error  | Acc    | Drop    | Conf  
-------------------------------------------------------------------
  graffiti.......... | 323    | 206    |  36.2% | 0.4908 |  49.4%
  video game........ | 41     | 26     |  36.6% | 0.4871 |  55.3%
  origami........... | 119    | 75     |  37.0% | 0.4833 |  57.1%
  art............... | 45     | 27     |  40.0% | 0.4530 |  59.5%
  plush object...... | 266    | 159    |  40.2% | 0.4507 |  55.9%
 -->