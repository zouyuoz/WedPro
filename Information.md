# MILS作業: AI-Claim Audit for Robust Vision
## 用實驗檢查 AI對模型穩健性的解釋

**授課教師:** Chih-Chung Hsu(許志仲) Institute of Intelligent Systems, NYCU  
**MILS Assignment**

---

### 一、這次作業要練習什麼
這次作業不是要你們追求最高準確率，也不是要把 AI 產生的文字整理成報告。我要你們練習的是：**把 AI 的說法轉成可以被檢驗的技術 claim，然後用既有資料集、模型輸出與圖表去支持、推翻或修正它。**

你們可以用 AI 來理解概念、產生假說、協助寫程式與整理結果；但最後的結論必須由你們自己的 evidence 決定。AI 在本作業中是學習工具與待審核對象，不是答案產生器。

---

### 二、作業任務
請使用指定的 robustness/distribution-shift benchmark，分析至少兩個 pretrained vision models 在資料分布改變時的表現，並審核 AI 對這些 failure patterns 的預測與解釋是否正確。

* **基本要求:** 選一個指定資料集、至少兩個模型或模型設定、至少 5 個 AI claims，並定義自己的 robustness measure。
* **核心問題:** AI 能不能正確預測與解釋模型在 corruption、style shift、OOD-like samples 或 hard examples 下的失敗？
* **範圍限制:** 不需要訓練大型模型，不需要建立新資料庫，不需要跑完整大型 benchmark。
* **最重要的規則:** 每一個主要結論都必須連到 table、figure、results.csv 或 failure cases。

---

### 三、可使用資料集

| Track | Dataset/Benchmark | 適合分析的問題 |
| :--- | :--- | :--- |
| Basic | CIFAR-10-C | noise、blur、weather、digital corruption 對分類模型的影響。最推薦。 |
| Basic | CIFAR-100-C | 比 CIFAR-10-C 更細類別，可分析類別數增加後的 robustness。 |
| Advanced | ImageNet-C | 大型自然影像 corruption robustness；請使用合理子集即可。 |
| Advanced | ImageNet-R | rendition / style shift，例如 cartoon、painting、sketch。 |
| Advanced | ImageNet-A | 自然影像中的 hard/adversarial-like examples。 |

* **取樣原則:** 可以只跑子集，但要清楚說明抽樣方式。同一組比較必須使用相同測試條件。
* **標籤原則:** 請使用官方標籤；不要用 AI 產生 ground truth，也不需要人工重新標註。

---

### 四、模型與可做的輕量方法
* **至少兩個模型或設定:** 可使用 ResNet、VGG、DenseNet、ViT、ConvNeXt、CLIP / OpenCLIP zero-shot、RobustBench model zoo，或先前作業已完成的模型。
* **不可做的事:** 不可把重新訓練大型模型當成主要工作量。這份作業的重點是 evaluation、audit、analysis。
* **可做的 no-training method:** test-time augmentation、simple preprocessing、confidence rejection、model ensemble、prompt ensemble、model agreement check。即使沒有改善也可以，只要分析清楚。

---

### 五、robustness measure 可以自己定義
你們可以自行定義 robustness，但必須寫清楚公式或計算方式。請至少選一個主要指標與一個輔助指標。

| Measure | 簡單定義範例 |
| :--- | :--- |
| Accuracy Drop | Acc(clean) - Acc(shifted) |
| Relative Drop | [Acc(clean) - Acc(shifted)] / Acc(clean) |
| Average Robust Accuracy | 所有 corruption/shift conditions 的平均 accuracy |
| Worst-condition Accuracy | 所有 conditions 中最低的 accuracy |
| Wrong Confidence | 模型答錯時的平均 confidence |
| Failure Overlap | 兩個模型在同一批 samples 同時失敗的比例 |
| Rejection Robustness | 加入 confidence rejection 後的 coverage/accuracy trade-off |

---

### 六、AI-Claim Audit
請先讓 AI 對你們的資料集、模型與 robustness measure 提出至少 5 個可檢驗 claims。接著把 AI 的自然語言說法改寫成可測假說，最後用實驗結果審核每個 claim。

| Claim ID | AI 原始說法 | 可檢驗假說 | Evidence | Audit 結論 |
| :--- | :--- | :--- | :--- | :--- |
| C1 | Gaussian noise 通常比 brightness 更傷 CNN。 | 相同 severity 下，Gaussian noise 的 accuracy drop 大於 brightness。 | Table 1/Fig. 2 | Supported / Partially supported / Refuted / Unclear |
| C2 | Robust model 對 noise 會比較穩。 | Robust model 在 noise corruptions 的 relative drop 小於 standard model。 | Table 2 | 依你們結果填寫 |

---

### 七、For Example

| Example | 設定方式 |
| :--- | :--- |
| CIFAR-10-C Corruption Audit | **Conditions:** Gaussian noise、motion blur、fog、JPEG compression severity 1/3/5。<br>**Models:** standard ResNet vs. robust model。<br>**Measure:** relative accuracy drop + worst-condition accuracy。<br>**問題:** AI 預測 noise 比 JPEG 更難，實驗是否支持？ |
| ImageNet-R/ImageNet-A Semantic Shift Audit | **Dataset:** ImageNet-R 或 ImageNet-A 子集。<br>**Models:** ResNet-50、ViT、CLIP zero-shot。<br>**Measure:** shifted accuracy + wrong confidence + failure overlap。<br>**問題:** AI 預測 CLIP 對 style shift 較穩，實驗是否支持？ |

---

### 八、繳交內容
* **短報告:** 4-6 頁即可，可中文或英文。內容包含 problem setting、AI claims、實驗設定、結果、claim audit、failure analysis、學習反思。
* **Claim Audit Table:** 至少 5 個 claims，可放在報告內或另外附 CSV。
* **程式與結果:** Colab/notebook/GitHub 擇一，並附 results.csv 或等價的結果表。
* **AI collaboration log:** 不用貼完整聊天紀錄；簡短說明 AI 哪裡幫上忙、哪裡可能誤導你們、你們如何驗證。
* **Failure cases:** 至少 5 個代表性失敗案例，包含 image、true label、prediction、confidence，以及你們的解釋。

---

### 九、評分方式
評分重點放在學習過程與證據品質，不是比誰的模型最高分。只要定義清楚、實驗可重現、結論有 evidence，就能拿到主要分數。

| 項目 | 比例 | 評分重點 |
| :--- | :--- | :--- |
| Problem setting & robustness definition | 20% | 問題切得清楚；資料集、模型、robustness measure 有明確定義。 |
| AI claim audit | 25% | AI 的說法有被改寫成可檢驗假說，且每個 claim 都有審核結果。 |
| Experiments & evidence | 25% | 實驗能重現；圖表、results.csv、failure cases 能支撐結論。 |
| Failure analysis & reflection | 20% | 能說明模型錯在哪裡，以及 AI 哪裡幫助或誤導自己的理解。 |
| Report clarity | 10% | 報告清楚、格式乾淨、讀者能快速看懂你們做了什麼。 |

**補充:** no-training method 沒有讓 performance 變好也沒關係。只要能清楚說明哪些 cases 變好、哪些變差、可能原因是什麼，仍然是好的分析。

---
---

# MILS Assignment: AI-Claim Audit for Robust Vision
## Testing AI Explanations of Model Robustness with Evidence

**Instructor:** Chih-Chung Hsu (許志仲)  Institute of Intelligent Systems, NYCU  
**MILS Assignment**

---

### 1. Purpose
This assignment is not about achieving the highest accuracy, and it is not about submitting an AI-written report. The goal is to turn AI-generated explanations into testable technical claims, and then use experimental evidence to support, reject, or revise those claims.

You may use AI to understand concepts, generate hypotheses, organize code, and interpret preliminary results. However, the final conclusions must come from your own evidence. In this assignment, AI is a learning tool and a claim generator to be audited, not the answer.

---

### 2. Task
Use one assigned robustness or distribution-shift benchmark to analyze at least two pretrained vision models under data shifts, and audit whether AI can correctly predict and explain the failure patterns.

* **Basic requirement:** Choose one assigned dataset, compare at least two models or settings, generate at least 5 AI claims, and define your own robustness measure.
* **Main question:** Can AI correctly predict and explain model failures under corruptions, style shift, OOD-like samples, or hard examples?
* **Scope:** No large model training, no new dataset construction, and no need to run a complete large benchmark.
* **Evidence rule:** Every major conclusion must be linked to a table, figure, results.csv, or failure cases.

---

### 3. Allowed Datasets

| Track | Dataset/Benchmark | Suitable Questions |
| :--- | :--- | :--- |
| Basic | CIFAR-10-C | Noise, blur, weather, and digital corruption robustness. Recommended. |
| Basic | CIFAR-100-C | A more fine-grained version for analyzing robustness with more classes. |
| Advanced | ImageNet-C | Large-scale natural image corruption robustness. Use a reasonable subset. |
| Advanced | ImageNet-R | Rendition and style shifts such as cartoon, painting, and sketch. |
| Advanced | ImageNet-A | Natural hard examples and adversarial-like examples. |

* **Sampling:** You may use subsets, but clearly describe the sampling rule. Use the same test conditions for all compared models.
* **Labels:** Use official labels. Do not use AI-generated ground truth labels.

---

### 4. Models and Lightweight Methods
* **Models:** ResNet, VGG, DenseNet, ViT, ConvNeXt, CLIP/OpenCLIP zero-shot, RobustBench model zoo models, or a model you already built in previous assignments.
* **Do not:** Do not make large-scale model training the main workload. Focus on evaluation, audit, and analysis.
* **Allowed no-training methods:** test-time augmentation, simple preprocessing, confidence rejection, model ensemble, prompt ensemble, or model agreement checking. It is acceptable if the method does not improve performance, as long as you analyze it well.

---

### 5. Define Robustness
You may define robustness in your own way, but the formula or calculation must be explicit. Use at least one primary measure and one supporting measure.

| Measure | Simple Definition Example |
| :--- | :--- |
| Accuracy Drop | Acc(clean) - Acc(shifted) |
| Relative Drop | [Acc(clean) - Acc(shifted)] / Acc(clean) |
| Average Robust Accuracy | Average accuracy over all corruption or shift conditions |
| Worst-condition Accuracy | The lowest accuracy among all evaluated conditions |
| Wrong Confidence | Average confidence on wrong predictions |
| Failure Overlap | The proportion of samples where two models fail together |
| Rejection Robustness | Coverage / accuracy trade-off after confidence rejection |

---

### 6. AI-Claim Audit
Ask AI to generate at least 5 testable claims about your chosen dataset, models, and robustness measure. Rewrite each natural-language claim into a measurable hypothesis, and then audit it with your results.

| Claim ID | AI Claim | Testable Hypothesis | Evidence | Audit Decision |
| :--- | :--- | :--- | :--- | :--- |
| C1 | Gaussian noise usually hurts CNNs more than brightness. | At the same severity, Gaussian noise causes a larger accuracy drop than brightness. | Table 1/Fig. 2 | Supported / Partially supported / Refuted / Unclear |
| C2 | Robust models should be more stable under noise. | The robust model has a smaller relative drop than the standard model under noise corruptions. | Table 2 | Based on your results |

---

### 7. For Example

| Example | Setup |
| :--- | :--- |
| CIFAR-10-C Corruption Audit | **Conditions:** Gaussian noise, motion blur, fog, JPEG compression, severity 1/3/5.<br>**Models:** standard ResNet vs. robust model.<br>**Measures:** relative accuracy drop and worst-condition accuracy.<br>**Question:** AI predicts that noise is harder than JPEG. Does the evidence support it? |
| ImageNet-R/ImageNet-A Semantic Shift Audit | **Dataset:** ImageNet-R or ImageNet-A subset.<br>**Models:** ResNet-50, ViT, CLIP zero-shot.<br>**Measures:** shifted accuracy, wrong confidence, and failure overlap.<br>**Question:** AI predicts that CLIP is more stable under style shift. Does the evidence support it? |

---

### 8. Deliverables
* **Short report:** 4-6 pages, in Chinese or English. Include problem setting, AI claims, experimental setup, results, claim audit, failure analysis, and learning reflection.
* **Claim Audit Table:** At least 5 claims, either inside the report or as a CSV file.
* **Code and results:** Submit a Colab notebook, notebook, or GitHub link, plus results.csv or an equivalent result table.
* **AI collaboration log:** Briefly explain how you used AI, where it helped, where it may have misled you, and how you verified it.
* **Failure cases:** At least 5 representative failure cases with image, true label, prediction, confidence, and your explanation.

---

### 9. Grading
The grading focuses on the learning process and evidence quality, not on achieving the highest score. Clear definitions, reproducible experiments, and evidence-based conclusions are the main criteria.

| Item | Weight | What I Look For |
| :--- | :--- | :--- |
| Problem setting & robustness definition | 20% | The question is clear; dataset, models, and robustness measure are explicitly defined. |
| AI claim audit | 25% | AI statements are rewritten as testable hypotheses and audited with evidence. |
| Experiments & evidence | 25% | Experiments are reproducible; figures, tables, results.csv, and cases support conclusions. |
| Failure analysis & reflection | 20% | You explain what the models got wrong and how AI helped or misled your understanding. |
| Report clarity | 10% | The report is clear, clean, and easy to follow. |

**Final reflection question:** What did AI help you learn? Where did AI mislead you? How did you find out?