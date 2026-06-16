import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import io
import base64

def generate_presentation():
    os.makedirs('presentation', exist_ok=True)
    
    # Load Data
    metrics_path = 'results/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}
        
    df_h3 = pd.DataFrame()
    if os.path.exists('results/audit_h3_results.csv'):
        df_h3 = pd.read_csv('results/audit_h3_results.csv')
        
    h4_data = {}
    if os.path.exists('results/audit_h4_results.json'):
        with open('results/audit_h4_results.json', 'r') as f:
            h4_data = json.load(f)
            
    h5_data = {}
    if os.path.exists('results/audit_h5_results.json'):
        with open('results/audit_h5_results.json', 'r') as f:
            h5_data = json.load(f)
            
    df_claims = pd.DataFrame()
    if os.path.exists('results.csv'):
        df_claims = pd.read_csv('results.csv')

    # Quantitative Stats Calculation for all models
    def calculate_robustness_metrics(model_name, model_data):
        acc_clean = model_data.get('acc_clean', 0)
        acc_shift = model_data.get('acc_shift', 0)
        drop = acc_clean - acc_shift
        
        return {
            'name': model_name,
            'acc_clean': acc_clean,
            'acc_shift': acc_shift,
            'drop': drop
        }

    all_model_metrics = [calculate_robustness_metrics(name, data) for name, data in metrics.items() if 'TTA' not in name]
    
    # H5 Chart Generation
    h5_chart_b64 = ""
    if h5_data:
        plt.figure(figsize=(8, 4))
        for model_name, data in h5_data.items():
            bins = [d['style_conf_bin'] for d in data['Error_Rate_by_Bin']]
            error_rates = [d['error_rate'] for d in data['Error_Rate_by_Bin']]
            plt.plot(bins, error_rates, marker='o', label=model_name)
        
        plt.title('Error Rate by Style Confidence Bin', color='white')
        plt.xlabel('CLIP Style Confidence', color='white')
        plt.ylabel('Model Error Rate', color='white')
        plt.legend()
        plt.grid(True, color='gray')
        plt.tick_params(colors='white')
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', transparent=True)
        img_buf.seek(0)
        h5_chart_b64 = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()

    # Claims table rows
    claims_rows = ""
    if not df_claims.empty:
        for _, row in df_claims.iterrows():
            color = "green" if row['Decision'] == "Supported" else "red" if "Refuted" in row['Decision'] else "orange"
            claims_rows += f"""
            <tr style="font-size: 0.6em;">
                <td>{row['ID']}</td>
                <td style="text-align: left;">{row['Claim']}</td>
                <td style="text-align: left;">{row['Evidence']}</td>
                <td style="color: {color}; font-weight: bold;">{row['Decision']}</td>
            </tr>
            """

    # HTML Template using Reveal.js
    html_template = f"""
    <!doctype html>
    <html lang="zh-TW">
    <head>
        <meta charset="utf-8">
        <title>AI-Claim Audit Presentation</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.5.0/reset.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.5.0/reveal.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.5.0/theme/night.min.css" id="theme">
        <style>
            .reveal table th, .reveal table td {{ font-size: 0.8em; border-bottom: 1px solid #777; padding: 10px; }}
            .reveal h1, .reveal h2, .reveal h3 {{ text-transform: none; font-family: 'Microsoft JhengHei', sans-serif; }}
            .highlight {{ color: #42affa; }}
        </style>
    </head>
    <body>
        <div class="reveal">
            <div class="slides">
                <!-- Slide 1 -->
                <section>
                    <h2>AI-Claim Audit for Robust Vision</h2>
                    <h3>MILS Assignment Report</h3>
                    <p>Testing AI Explanations of Model Robustness with Evidence</p>
                </section>

                <!-- Slide 2 -->
                <section>
                    <h2>1. 實驗設定 (Problem Setting)</h2>
                    <p style="font-size: 0.8em;">比較不同模型架構在面對分佈偏移（Distribution Shift）時的穩健性。</p>
                    <ul>
                        <li><span class="highlight">Clean 資料集</span>: ImageNet-V2 / ImageNet-1K</li>
                        <li><span class="highlight">Shifted 資料集</span>: ImageNet-R (包含 16 種風格變體)</li>
                        <li><span class="highlight">受測模型</span>: ResNeXt-101, MobileNet-V3, ViT-B16-V1, ViT-SWAG</li>
                    </ul>
                </section>

                <!-- Slide 3 -->
                <section>
                    <h2>2. 基礎穩健性評估</h2>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Clean Acc</th>
                            <th>Shifted Acc</th>
                            <th>Acc Drop</th>
                        </tr>
                        {"".join([f"<tr><td>{m['name']}</td><td>{m['acc_clean']:.2%}</td><td>{m['acc_shift']:.2%}</td><td class='highlight'>{m['drop']:.4f}</td></tr>" for m in all_model_metrics])}
                    </table>
                    <p style="font-size: 0.7em;">發現：SWAG 預訓練大幅降低了 Accuracy Drop，而 MobileNet 表現最為脆弱。</p>
                </section>

                <!-- Slide 4: Overview -->
                <section>
                    <h2>3. AI 假說審核總覽</h2>
                    <table style="width: 100%;">
                        <tr>
                            <th style="width: 5%;">ID</th>
                            <th style="width: 45%;">假說內容 (Claim)</th>
                            <th style="width: 35%;">證據 (Evidence)</th>
                            <th style="width: 15%;">審核結果</th>
                        </tr>
                        {claims_rows}
                    </table>
                </section>

                <!-- Slide 5: H1 -->
                <section>
                    <h2>H1: 整體 Domain Shift 表現</h2>
                    <p style="font-size: 0.8em; color: #ccc;">Claim: ViT-b-16 (SWAG) > ViT-b-16 (V1) > ResNeXt-101 > MobileNet</p>
                    <ul style="font-size: 0.8em;">
                        <li>ViT-SWAG Drop: <span class="highlight">0.3500</span></li>
                        <li>ResNeXt-101 Drop: <span class="highlight">0.5189</span></li>
                        <li>ViT-V1 Drop: <span class="highlight">0.5299</span></li>
                        <li>MobileNet-V3 Drop: <span class="highlight">0.5344</span></li>
                    </ul>
                    <p style="font-size: 0.8em; margin-top: 20px;"><strong>審核結果: <span style="color: red;">Refuted (部分推翻)</span></strong><br>
                    雖然 SWAG 最穩，但高參數卷積模型 (ResNeXt) 的穩健性實際上優於標準的 ViT-V1。單純「Transformer 架構決定較高穩健性」的說法並不成立。</p>
                </section>

                <!-- Slide 6: H2 -->
                <section>
                    <h2>H2: Style-Specific 脆弱點</h2>
                    <p style="font-size: 0.8em; color: #ccc;">Claim: CNNs 最大的 Drop 在 line drawing，而 ViTs 最大的 Drop 在 pattern / embroidery。</p>
                    <table style="font-size: 0.8em; width: 80%; margin: 0 auto;">
                        <tr><th>Model</th><th>Worst-case Style (Max Drop)</th></tr>
                        <tr><td>ResNeXt-101 (CNN)</td><td>graffiti</td></tr>
                        <tr><td>MobileNet-V3 (CNN)</td><td>tattoo</td></tr>
                        <tr><td>ViT-B16-V1 (ViT)</td><td>tattoo</td></tr>
                        <tr><td>ViT-B16-SWAG (ViT)</td><td>graffiti</td></tr>
                    </table>
                    <p style="font-size: 0.8em; margin-top: 20px;"><strong>審核結果: <span style="color: red;">Refuted (推翻)</span></strong><br>
                    兩個陣營最脆弱的風格高度一致，並未在不同材質上產生分歧。</p>
                </section>

                <!-- Slide 7: H3 -->
                <section>
                    <h2>H3: 免訓練補救措施</h2>
                    <p style="font-size: 0.8em; color: #ccc;">Claim: Cross-Architecture Ensemble yields the highest boost.</p>
                    <table style="font-size: 0.8em;">
                        <tr><th>Strategy</th><th>Coverage</th><th>Acc on Accepted</th><th>Gain vs SWAG</th></tr>
                        {"".join([f"<tr><td>{row['Strategy']}</td><td>{row['Coverage']:.2%}</td><td>{row['Accuracy']:.2%}</td><td class='highlight'>{row['Gain_vs_Best_Baseline']:+.2%}</td></tr>" for _, row in df_h3.iterrows()])}
                    </table>
                    <p style="font-size: 0.8em; margin-top: 20px;"><strong>審核結果: <span style="color: red;">Refuted (推翻)</span></strong><br>
                    單純 Ensemble 反而變差。<strong>一致性拒絕 (Agreement Rejection)</strong> 才能顯著提升留存樣本準確度。</p>
                </section>

                <!-- Slide 8: H4 -->
                <section>
                    <h2>H4: 失敗重疊率分析</h2>
                    <p style="font-size: 0.8em; color: #ccc;">Claim: Architecture dictates Failure Overlap more than Data Scale.</p>
                    <ul style="font-size: 0.8em;">
                        <li><span class="highlight">跨架構 (ResNeXt vs ViT-V1)</span>: 錯誤重疊率高達 <strong>83.6%</strong></li>
                        <li><span class="highlight">同架構不同訓練 (ViT-V1 vs SWAG)</span>: 錯誤重疊率僅 <strong>63.6%</strong></li>
                    </ul>
                    <p style="font-size: 0.8em; margin-top: 20px;"><strong>審核結果: <span style="color: red;">Refuted (推翻)</span></strong><br>
                    訓練方式 (Data/Scale) 比 模型底層架構 (CNN/Transformer) 更能決定模型在 OOD 資料上的失敗模式。</p>
                </section>

                <!-- Slide 9: H5 -->
                <section>
                    <h2>H5: 風格強度與錯誤率</h2>
                    <p style="font-size: 0.8em; color: #ccc;">Claim: Higher CLIP style confidence correlates with higher CNN error rates.</p>
                    <img src="data:image/png;base64,{h5_chart_b64}" alt="Error Rate vs Style Confidence" style="max-height: 350px;">
                    <p style="font-size: 0.8em;"><strong>審核結果: <span style="color: red;">Refuted (推翻)</span></strong><br>
                    CLIP 風格信心度與模型錯誤率之間只有極弱的正相關 (Pearson r ~0.1)。</p>
                </section>

                <!-- Slide 10: Conclusion -->
                <section>
                    <h2>總結與反思</h2>
                    <ul style="font-size: 0.9em;">
                        <li>AI 生成的 5 個假說幾乎全被實驗推翻。</li>
                        <li>大型預訓練 (SWAG) 的影響力大於單純更換架構 (ViT vs CNN)。</li>
                        <li>模型在極端領域偏移下存在嚴重的<strong>過度自信 (Overconfidence)</strong>。</li>
                        <li class="highlight">實驗驗證 (Empirical Audit) 是 AI 時代不可或缺的一環。</li>
                    </ul>
                </section>
            </div>
        </div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.5.0/reveal.min.js"></script>
        <script>
            Reveal.initialize({{
                hash: true,
                transition: 'slide'
            }});
        </script>
    </body>
    </html>
    """
    
    with open('presentation/index.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("Presentation generated successfully at presentation/index.html")

if __name__ == "__main__":
    generate_presentation()
