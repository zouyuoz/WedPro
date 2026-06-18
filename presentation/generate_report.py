import json
import pandas as pd
from weasyprint import HTML, CSS
import os
import base64

def get_image_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def generate_report():
    # Paths relative to presentation/
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_csv_path = os.path.join(root_dir, 'results.csv')
    style_analysis_path = os.path.join(root_dir, 'results/style_error_analysis.csv')
    audit_h3_path = os.path.join(root_dir, 'results/audit_h3_results.csv')
    audit_h4_path = os.path.join(root_dir, 'results/audit_h4_results.json')
    audit_h5_path = os.path.join(root_dir, 'results/audit_h5_results.json')
    output_pdf_path = os.path.join(root_dir, 'presentation/report.pdf')

    # Image Paths
    img_style_acc = get_image_base64(os.path.join(root_dir, 'results/style_accuracy_plot.png'))
    img_h5_error = get_image_base64(os.path.join(root_dir, 'results/h5_error_by_confidence_plot.png'))
    img_h5_wrong = get_image_base64(os.path.join(root_dir, 'results/h5_wrong_conf_comparison_plot.png'))
    img_image1 = get_image_base64(os.path.join(root_dir, 'presentation/image1.png'))
    img_image2 = get_image_base64(os.path.join(root_dir, 'presentation/image2.png'))

    # Load Data
    try:
        df_results = pd.read_csv(results_csv_path)
        df_style = pd.read_csv(style_analysis_path)
        df_h3 = pd.read_csv(audit_h3_path)
        with open(audit_h4_path, 'r') as f:
            h4_data = json.load(f)
        with open(audit_h5_path, 'r') as f:
            h5_data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Section 2 Table
    summary_rows = ""
    for _, row in df_results.iterrows():
        summary_rows += f"<tr><td>{row['ID']}</td><td>{row['Claim']}</td><td>{row['Evidence']}</td><td>{row['Decision']}</td></tr>"

    # Section 3.1 Table (H1)
    h1_table = """
    <tr><td>ResNeXt-101</td><td>79.31%</td><td>27.43%</td><td>0.5189</td></tr>
    <tr><td>MobileNet-V3</td><td>74.04%</td><td>21.30%</td><td>0.5274</td></tr>
    <tr><td>ViT-B16-V1</td><td>81.07%</td><td>28.07%</td><td>0.5299</td></tr>
    <tr><td>ViT-B16-SWAG</td><td>85.30%</td><td>50.30%</td><td>0.3500</td></tr>
    """
    
    # Section 3.2 Top 5 Error Styles
    style_tables_html = ""
    models = ['ResNeXt-101', 'MobileNet-V3', 'ViT-B16-V1', 'ViT-B16-SWAG']
    model_dfs = {}
    for model in models:
        model_df = df_style[df_style['Model'] == model].sort_values(by='Style_Accuracy', ascending=True).head(5)
        model_dfs[model] = model_df

    def format_style_table(model_name, df):
        rows = ""
        for _, row in df.iterrows():
            rows += f"<tr><td>{row['Style']}</td><td>{row['Total_Count']}</td><td>{row['Error_Count']}</td><td>{row['Style_Accuracy']:.1%}</td><td>{row['Accuracy_Drop']:.4f}</td><td>{row['Avg_Confidence']:.1%}</td></tr>"
        
        return f"""
        <div class="style-table-container">
            <h4>{model_name} Style Analysis</h4>
            <table>
                <thead><tr><th>Style</th><th>Total</th><th>Error</th><th>Acc</th><th>Drop</th><th>Conf</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """
    
    style_tables_html = "<div class='grid-container'>"
    style_tables_html += format_style_table('ResNeXt-101', model_dfs['ResNeXt-101'])
    style_tables_html += format_style_table('MobileNet-V3', model_dfs['MobileNet-V3'])
    style_tables_html += "</div><div class='grid-container'>"
    style_tables_html += format_style_table('ViT-B16-V1', model_dfs['ViT-B16-V1'])
    style_tables_html += format_style_table('ViT-B16-SWAG', model_dfs['ViT-B16-SWAG'])
    style_tables_html += "</div>"


    # Section 3.3 Tables (H3)
    h3_ensemble_rows = ""
    for _, row in df_h3.iterrows():
        h3_ensemble_rows += f"<tr><td>{row['Strategy']}</td><td>{row['Coverage']:.2%}</td><td>{row['Accuracy']:.2%}</td><td>{row['Gain_vs_Best_Baseline']:+.2%}</td></tr>"

    # Section 3.4 Table (H4)
    h4_rows = ""
    global_overlap = h4_data.get('Global_Overlap', {})
    for pair, metrics in global_overlap.items():
        jaccard = metrics.get('Jaccard_Similarity', 0)
        match_rate = metrics.get('Wrong_Prediction_Match_Rate', 0)
        h4_rows += f"<tr><td>{pair}</td><td>{jaccard:.4f}</td><td>{match_rate:.2%}</td></tr>"

    # HTML Template
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: "Noto Sans TC", sans-serif; line-height: 1.4; color: #333; max-width: 900px; margin: 0 auto; padding: 15px; font-size: 12px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 5px; font-size: 22px; text-align: center; }}
            h2 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 3px; margin-top: 20px; font-size: 18px; }}
            h3, h4 {{ color: #34495e; margin-top: 15px; font-size: 15px; }}
            h4 {{ font-size: 13px; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 10px; page-break-inside: avoid; }}
            th, td {{ border: 1px solid #ddd; padding: 5px; text-align: left; }}
            th {{ background-color: #f8f9fa; font-weight: bold; color: #2c3e50; }}
            tr:nth-child(even) {{ background-color: #fcfcfc; }}
            blockquote {{ background: #fdfdfd; border-left: 5px solid #3498db; margin: 10px 0; padding: 8px 15px; color: #555; font-style: italic; }}
            code {{ background: #f4f4f4; border: 1px solid #ddd; border-left: 3px solid #e67e22; color: #d35400; font-family: monospace; font-size: 11px; padding: 10px; display: block; white-space: pre-wrap; margin: 10px 0; }}
            .summary-table th:nth-child(1) {{ width: 5%; }}
            .summary-table th:nth-child(2) {{ width: 30%; }}
            .summary-table th:nth-child(3) {{ width: 50%; }}
            .summary-table th:nth-child(4) {{ width: 15%; }}
            .img-container {{ text-align: center; margin: 15px 0; page-break-inside: avoid; }}
            .img-container img {{ max-width: 85%; height: auto; border: 1px solid #eee; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }}
            .page-break {{ page-break-before: always; }}
            .definition-box {{ background-color: #f1f8ff; border: 1px solid #c8e1ff; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .grid-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .style-table-container {{ page-break-inside: avoid; }}
        </style>
    </head>
    <body>
        <h1>AI-Claim Audit for Robust Vision</h1>
        <p style="text-align: center; font-weight: bold;">MILS Assignment Report | Testing AI Explanations with Evidence</p>

        <h2>1. 實驗設定 (Problem Setting)</h2>
        <div class="definition-box">
            <strong>核心目標：</strong> 將 AI 的自然語言說法改寫成可測假說，並利用 ImageNet-R 資料集、ResNeXt、MobileNet 與 ViT 模型，透過實驗數據 (Accuracy Drop, Failure Overlap 等) 審核 AI 的預測是否正確。
        </div>
        
        <h3>1.1 穩健性分析條件</h3>
        <ul>
            <li><strong>Clean 資料集:</strong> ImageNet-1K (基準)</li>
            <li><strong>Shifted 資料集:</strong> ImageNet-R (包含 15 種風格變體，如 cartoon, painting, origami 等)</li>
            <li><strong>受測模型:</strong> 
                <ul>
                    <li>ResNeXt-101 (Heavy CNN)</li>
                    <li>MobileNet-V3 (Lightweight CNN)</li>
                    <li>ViT-B16-V1 (Standard Vision Transformer)</li>
                    <li>ViT-B16-SWAG (Transformer with billions of weakly supervised images)</li>
                </ul>
            </li>
        </ul>

        <h3>1.2 資料集風格展示</h3>
        <p>ImageNet-R 包含多樣化的 Rendition 風格，對於依賴紋理的傳統 CNN 具有極大挑戰性。</p>
        <div class="img-container">
            <img src="data:image/png;base64,{img_image1}" alt="ImageNet-R Styles">
            <p>圖 1: ImageNet-R 各類風格展示</p>
        </div>

        <h3>1.3 AI 協作與提示詞工程 (AI Collaboration)</h3>
        <p>本實驗使用 Gemini (網頁版與 CLI) 進行假說生成與程式碼輔助開發。以下為生成假說時所使用的核心 Prompt：</p>
        <code>
# AI-Claim Audit for Robust Vision (MILS 作業)
本專案旨在執行 AI-Claim Audit，透過實驗證據審核 AI 對模型穩健性的預測。
我們針對 ResNeXt-101 與 MobileNet-V3 兩個模型，在 ImageNet-R 資料集上進行風格偏移分析。

## 核心任務
- 資料集: 使用 ImageNet-V2 作為基準，ImageNet-R 作為偏移資料。
- 主要指標: Accuracy Drop, Wrong Confidence, Failure Overlap。
- 風格: ImageNet-R 包含 art, cartoon, graffiti, embroidery, graphics, origami... 等 15 種風格。
        </code>
        <p><em>一開始是先用 plan mode 將所有要求完成，但效果不好，就從輸出結果去"誘達" AI，產生第二版的 AI claims，再去驗證，得到本版本。</em></p>
        <div class="img-container">
            <img src="data:image/png;base64,{img_image2}" alt="AI Collaboration Screenshot">
            <p>圖 2: AI 假說審核與提示詞溝通過程</p>
        </div>

        <h3>1.4 評估指標定義 (Robustness Measures)</h3>
        <table>
            <thead>
                <tr><th>Measure</th><th>定義 / 公式</th></tr>
            </thead>
            <tbody>
                <tr><td>Accuracy Drop</td><td>Acc(clean) - Acc(shifted)</td></tr>
                <tr><td>Relative Drop</td><td>[Acc(clean) - Acc(shifted)] / Acc(clean)</td></tr>
                <tr><td>Wrong Confidence</td><td>模型答錯時的平均 confidence</td></tr>
                <tr><td>Failure Overlap</td><td>兩個模型在同一批 samples 同時失敗的比例 (Jaccard Similarity)</td></tr>
            </tbody>
        </table>

        <div class="page-break"></div>

        <h2>2. AI 假說審核總覽</h2>
        <p>我們使用 Gemini 對穩健性提出 5 個 Claim，並將其轉化為可驗證的假說：</p>
        <table class="summary-table">
            <thead>
                <tr><th>ID</th><th>假說內容 (Claim)</th><th>證據 (Evidence)</th><th>審核結果</th></tr>
            </thead>
            <tbody>
                {summary_rows}
            </tbody>
        </table>

        <h2>3. 各項 AI claim 深度分析</h2>

        <h3>3.1: 整體 Domain Shift 表現 (H1)</h3>
        <p>問：這四個模型，誰在進行 domain shift 之後表現會比較好，為甚麼？</p>
        <blockquote>
            AI 預測：ViT-b-16 (SWAG) > ViT-b-16 (V1) ≥ ResNeXt-101 > MobileNet<br>
            ViT-b-16 (SWAG) 經數十億張弱監督影像預訓練，打破 ImageNet-1K 限制，對未知 Domain Shift 具備極強 Generalization 能力；標準 ViT 依賴 Self-Attention，天生 Shape Bias 較高，面對 ImageNet-R 時抗跌幅能力持平或優於重型 CNN (ResNeXt)。而 MobileNet 的 Capacity 過小，極度依賴局部像素紋理 (Texture Bias)，面對風格偏移最易崩潰。
        </blockquote>
        <table>
            <thead>
                <tr><th>Model</th><th>Clean Acc</th><th>Shifted Acc</th><th>Acc Drop</th></tr>
            </thead>
            <tbody>
                {h1_table}
            </tbody>
        </table>
        <ul>
            <li><strong>實驗發現：</strong> SWAG 預訓練確實大幅降低了跌幅。然而，標準訓練的 <strong>ViT-B16-V1 的 Accuracy Drop (0.5299) 反而高過兩個 CNN 模型</strong>。</li>
            <li><strong>結論：</strong> AI 在 Shifted Acc 的絕對數值的排名預測正確，但高估了標準 ViT 架構的穩健性（其跌幅最劇烈）。</li>
        </ul>

        <h3>3.2: Style-Specific 脆弱點 (H2)</h3>
        <p>問：哪種風格的acc drop 會最大，為甚麼？可以四個模型分別回答，也可以說所有模型都對一種風格有最大的 acc drop</p>
        <blockquote>
            CNN 陣營 (ResNeXt, MobileNet)：在 "line drawing" 跌幅最大。因濾除顏色與陰影，僅剩稀疏線條，導致 CNN 的 Activation Map 嚴重稀疏化，無法提取有效特徵。</br>
            ViT 陣營 (V1, SWAG)：在 "pattern" 或 "embroidery" 跌幅最大。密集重複圖案在 Patch Embedding 階段即引入錯誤編碼，嚴重干擾後續 Self-Attention 的全域關聯計算。</br>
        </blockquote>
        <p>驗證：實驗結果顯示，AI 的具體風格預測被完全推翻：</p>
        <ul>
            <li><strong>CNN 陣營：</strong> ResNeXt-101 與 MobileNet-V3 表現最差的風格皆為 <strong>"graffiti" (塗鴉)</strong> 與 <strong>"tattoo" (刺青)</strong>。</li>
            <li><strong>ViT 陣營：</strong> 脆弱點同樣集中在 <strong>"tattoo"</strong> 與 <strong>"graffiti"</strong>。</li>
        </ul>
        <div class="img-container">
            <img src="data:image/png;base64,{img_style_acc}" alt="Style Accuracy Plot">
            <p>圖 3: 不同模型在各風格下的正確率分布</p>
        </div>
        
        <h4>四種模型判斷錯誤率最高的五種風格</h4>
        {style_tables_html}

        

        <h3>3.3: 非訓練提升策略驗證 (H3)</h3>
        <p>問：對於資料進行甚麼非訓練的處理 (包含**但不限**test-time augmentation、simple preprocessing、confidence rejection、model ensemble、prompt ensemble、model agreement check 等等)，能夠對於domain shift後的表現有最大的提升</p>
        <blockquote>
            Cross-Architecture Logit Ensemble：平均 ResNeXt-101 與 ViT-b-16 (SWAG) 的 Logits。利用 CNN（局部紋理）與 ViT（全域結構）不同的 Inductive Bias 實現特徵互補，抹平單一架構盲點。</br>
            Test-Time Resolution Scaling：測試期提升 ViT 輸入解析度並雙線性插值。這能減弱高頻風格噪聲，在 Patch 內部保留物體相對結構，提升 OOD 精準度。</br>
        </blockquote>
        <p>驗證：</p>
        <h4>Model Ensemble 效果：</h4>
        <table>
            <thead>
                <tr><th>Strategy</th><th>Coverage</th><th>Acc on Accepted</th><th>Gain vs Best Baseline</th></tr>
            </thead>
            <tbody>
                {h3_ensemble_rows}
            </tbody>
        </table>
        <ul>
            <li><strong>Ensemble 失敗：</strong> 單純 Logit 平均反而導致 <strong>-5.75%</strong> 的負收益。說明 CNN 與 ViT 的錯誤並未如預期般互補。</li>
            <li><strong>Agreement Rejection：</strong> 只有透過一致性拒絕機制，才能顯著提升準確率 (+14.23%)，但覆蓋率僅剩 37.85%。</li>
        </ul>

        <h4>Test-Time Augmentation (TTA) 效果：</h4>
        <table>
            <thead>
                <tr><th>Metric</th><th>Shifted Accuracy</th><th>TTA Shifted Accuracy</th><th>Δ Acc</th></tr>
            </thead>
            <tbody>
                <tr><td>ResNeXt-101-TTA</td><td>0.27425</td><td>0.28900</td><td>+0.01475</td></tr>
                <tr><td>MobileNet-V3-TTA</td><td>0.21300</td><td>0.21025</td><td>-0.00275</td></tr>
                <tr><td>ViT-B16-V1-TTA</td><td>0.28075</td><td>0.29175</td><td>+0.01100</td></tr>
                <tr><td>ViT-B16-SWAG-TTA</td><td>0.50300</td><td>0.51375</td><td>+0.01075</td></tr>
            </tbody>
        </table>
        <ul>
            <li>TTA 帶來的提升非常微小（約 1%），MobileNet 甚至出現微幅下降。</li>
        </ul>

        
        <h3>3.4: 失敗交集分析 (H4) (AI 自問自答)</h3>
        <p>問：在 Style Shift 下，這四個模型彼此之間的 Failure Overlap 表現如何？是相同架構的模型（兩個 ViT）錯在同一個地方的機率高，還是同等訓練量/表現接近的模型（ResNeXt-101 與 ViT-b-16 V1）錯在同一個地方的機率高？</p>
        <blockquote>
            同架構高重合：ViT-b-16 (V1) 與 SWAG 識別率雖異，但共享 Tokenization 與 Global Attention 底層數學邏輯，誤判視角一致，同時出錯時 Wrong Predictions 重合度極高。</br>
            異架構低重合：ResNeXt-101 與 ViT-b-16 (V1) 的 Failure Mechanism 相互獨立，錯誤點與錯誤類別高度發散。
        </blockquote>
        <p>驗證：</p>
        <table>
            <thead>
                <tr><th>Model Pair</th><th>Jaccard Similarity (錯誤交集)</th><th>Wrong Prediction Match Rate</th></tr>
            </thead>
            <tbody>
                {h4_rows}
            </tbody>
        </table>
        <ul>
            <li><strong>事實：</strong> 傳統 CNN (ResNeXt) 與標準 ViT (V1) 在同樣 1K 訓練下，Jaccard Similarity 高達 <strong>0.8361</strong>。</li>
            <li><strong>結論：</strong> 數據分佈才是決定模型「錯在哪裡」的主因，架構差異的影響次之。SWAG 因海量數據預訓練，其錯誤分佈與其他模型產生了本質上的脫鉤。</li>
        </ul>

        <h3>3.5: 風格信心度與過度自信 (H5) (AI 自問自答)</h3>
        <p>當模型面對無法正確分類的風格影像時，標準訓練的 ViT-b-16 (V1) 與海量預訓練的 ViT-b-16 (SWAG)，誰更容易給出極高信心度（例如 Confidence > 0.9）的錯誤預測？</p>
        <blockquote>
            標準 ViT 過度自信：ViT-b-16 (V1) 在有限數據上訓練，易對特定 Patch 產生 Overfitting 或 Overactivation。遇到特定風格時，會盲目給出高置信度的錯誤預測。</br>
            SWAG 有效校準：ViT-b-16 (SWAG) 看過數十億張影像，Decision Boundary 高度平滑。面對未知偏移時輸出機率分布更平均 (Entropy 較高)，顯著降低 Wrong Confidence。
        </blockquote>
        <p>驗證：</p>
        <div class="img-container">
            <img src="data:image/png;base64,{img_h5_wrong}" alt="H5 Error Plot">
            <p>圖 2: 模型的錯誤信心度隨 CLIP 風格信心度之變化 (幾乎為水平線)</p>
        </div>
        <ul>
            <li><strong>數據證據：</strong> Pearson 相關係數 $r$ 僅在 0.08 ~ 0.12，說明錯誤率與風格強度無關。</li>
            <li><strong>結論：</strong> 模型在 OOD 情況下的失敗是系統性的「過度自信」，不論風格強弱，模型皆展現出相似的錯誤傾向。</li>
        </ul>

        

        <h2>4. 總結與學習反思</h2>
        <ul>
            <li><strong>實驗審核的重要性：</strong> AI 雖然能提供看似合理的架構與數據分析（如 Shape Bias, Logit Ensemble），但本次實驗中 5 個假說幾乎全被推翻或修正。這證明了 <strong>Empirical Audit (實證審核)</strong> 是使用 AI 輔助開發時不可或缺的一環。</li>
            <li><strong>數據規模的決定力：</strong> ViT-SWAG 的強大表現與獨特的錯誤分佈，再次證實了大規模預訓練在穩健性上的統治地位，其影響力遠超架構變換。</li>
            <li><strong>AI 協作心得：</strong> AI 在撰寫評估腳本與解釋模型架構上極具效率，但在預測特定分佈偏移 (ImageNet-R) 的失敗模式時，仍存在過度簡化的偏見。</li>
        </ul>

        <div class="definition-box" style="margin-top: 30px;">
            <strong>致謝與工具：</strong> 本報告由 Gemini CLI 輔助開發，實驗數據經由 Python 腳本自動化驗證與生成。
        </div>
    </body>
    </html>
    """
    
    # Save PDF to presentation/ (local to the script)
    print("Generating PDF report in presentation/report.pdf...")
    HTML(string=html_template, base_url='.').write_pdf(output_pdf_path)
    print(f"Report successfully generated: {output_pdf_path}")

if __name__ == "__main__":
    generate_report()
