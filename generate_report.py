import json
import pandas as pd
from weasyprint import HTML, CSS
import os
import matplotlib.pyplot as plt
import io
import base64

def generate_report():
    # Load Data
    metrics_path = 'results/metrics.json'
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found.")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    df_h3 = pd.read_csv('results/audit_h3_results.csv')
    with open('results/audit_h4_results.json', 'r') as f:
        h4_data = json.load(f)
    with open('results/audit_h5_results.json', 'r') as f:
        h5_data = json.load(f)

    # Quantitative Stats Calculation for all models
    def calculate_robustness_metrics(model_name, model_data):
        acc_clean = model_data['acc_clean']
        acc_shift = model_data['acc_shift']
        
        drop = acc_clean - acc_shift
        rel_drop = drop / acc_clean if acc_clean > 0 else 0
        
        styles = {}
        for r in model_data.get('results_shift', []):
            s = r.get('style', 'unknown')
            if s not in styles: styles[s] = {'correct': 0, 'total': 0}
            styles[s]['total'] += 1
            if r['is_correct']: styles[s]['correct'] += 1
        
        style_accs = {s: (v['correct']/v['total']) for s, v in styles.items() if v['total'] > 0}
        avg_robust = sum(style_accs.values()) / len(style_accs) if style_accs else 0
        worst_acc = min(style_accs.values()) if style_accs else 0
        worst_style = min(style_accs, key=style_accs.get) if style_accs else "unknown"
        
        wc = [r['confidence'] for r in model_data.get('results_shift', []) if not r['is_correct']]
        wrong_conf = sum(wc) / len(wc) if wc else 0
        
        sorted_results = sorted(model_data.get('results_shift', []), key=lambda x: x['confidence'], reverse=True)
        keep_count = int(len(sorted_results) * 0.8)
        top_results = sorted_results[:keep_count]
        rej_robust = sum([1 for r in top_results if r['is_correct']]) / keep_count if keep_count > 0 else 0
        
        return {
            'name': model_name,
            'acc_clean': acc_clean,
            'acc_shift': acc_shift,
            'drop': drop,
            'rel_drop': rel_drop,
            'avg_robust': avg_robust,
            'worst_acc': worst_acc,
            'worst_style': worst_style,
            'wrong_conf': wrong_conf,
            'rej_robust': rej_robust
        }

    all_model_metrics = [calculate_robustness_metrics(name, data) for name, data in metrics.items() if 'TTA' not in name]
    
    # H1 Data
    sorted_by_drop = sorted(all_model_metrics, key=lambda x: x['drop'])
    h1_evidence_text = ", ".join([f"{m['name']} ({m['drop']:.4f})" for m in sorted_by_drop])

    # H2 Data
    h2_evidence_text = "<ul>"
    for m in all_model_metrics:
        h2_evidence_text += f"<li>{m['name']}: {m['worst_style']}</li>"
    h2_evidence_text += "</ul>"

    # H3 Data processing
    h3_rows = ""
    for _, row in df_h3.iterrows():
        h3_rows += f"<tr><td>{row['Strategy']}</td><td>{row['Coverage']:.2%}</td><td>{row['Accuracy']:.2%}</td><td>{row['Gain_vs_Best_Baseline']:+.2%}</td></tr>"

    # H4 Data processing
    vit_pair_overlap = h4_data['Global_Overlap'].get('ViT-B16-V1 vs ViT-B16-SWAG', {}).get('Jaccard_Similarity', 0)
    cross_pair_overlap = h4_data['Global_Overlap'].get('ResNeXt-101 vs ViT-B16-V1', {}).get('Jaccard_Similarity', 0)
    
    # H5 Data processing & Chart Generation
    plt.figure(figsize=(10, 5))
    for model_name, data in h5_data.items():
        bins = [d['style_conf_bin'] for d in data['Error_Rate_by_Bin']]
        error_rates = [d['error_rate'] for d in data['Error_Rate_by_Bin']]
        plt.plot(bins, error_rates, marker='o', label=model_name)
    
    plt.title('Error Rate by Style Confidence Bin')
    plt.xlabel('CLIP Style Confidence')
    plt.ylabel('Model Error Rate')
    plt.legend()
    plt.grid(True)
    
    # Save chart to base64
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    h5_chart_b64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()

    # Create H5 Table
    h5_table_rows = ""
    for model_name, data in h5_data.items():
        corr_wc = data.get('Pearson_r_StyleConf_vs_WrongConf')
        corr_ep = data.get('Pearson_r_StyleConf_vs_Error_Prob')
        corr_wc_str = f"{corr_wc:.4f}" if corr_wc is not None else "N/A"
        corr_ep_str = f"{corr_ep:.4f}" if corr_ep is not None else "N/A"
        h5_table_rows += f"<tr><td>{model_name}</td><td>{corr_wc_str}</td><td>{corr_ep_str}</td></tr>"

    # HTML Template
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <title>AI-Claim Audit Report</title>
        <style>
            @page {{
                size: A4;
                margin: 15mm 10mm;
                @bottom-right {{
                    content: counter(page);
                    font-family: sans-serif;
                    font-size: 8pt;
                    color: #888888;
                }}
            }}
            body {{
                font-family: 'PingFang TC', 'Microsoft JhengHei', sans-serif;
                line-height: 1.4;
                color: #333;
                font-size: 9pt;
                max-width: 1000px;
                margin: auto;
            }}
            h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #2c3e50; padding-bottom: 8px; font-weight: bolder; font-size: 18pt; }}
            h2 {{ color: #2980b9; border-left: 5px solid #2980b9; padding-left: 10px; margin-top: 20px; font-weight: bolder; font-size: 14pt; page-break-after: avoid; }}
            h3 {{ color: #34495e; font-weight: bolder; font-size: 11pt; margin-top: 15px; page-break-after: avoid; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; table-layout: fixed; page-break-inside: avoid; }}
            th, td {{ border: 1px solid #ddd; padding: 6px; text-align: center; word-wrap: break-word; font-size: 8pt; }}
            th {{ background-color: #f2f2f2; color: #2c3e50; font-weight: bolder; }}
            .summary-box {{ background-color: #ecf0f1; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .audit-supported {{ color: #27ae60; font-weight: bold; }}
            .audit-refuted {{ color: #c0392b; font-weight: bold; }}
            .audit-unclear {{ color: #f39c12; font-weight: bold; }}
            .image-container {{ display: flex; flex-wrap: wrap; justify-content: flex-start; page-break-inside: avoid; }}
            .image-box {{ width: 24%; margin-bottom: 10px; text-align: center; border: 1px solid #eee; padding: 3px; }}
            .image-box img {{ max-width: 100%; height: auto; border-radius: 2px; }}
            .caption {{ font-size: 7pt; color: #666; margin-top: 2px; }}
        </style>
    </head>
    <body>
        <h1>AI-Claim Audit for Robust Vision</h1>
        <p style="text-align: center; font-style: italic;">MILS Assignment Report</p>

        <h2>1. 實驗設定 (Problem Setting)</h2>
        <div class="summary-box">
            <p>本次實驗旨在審核 AI 對於不同視覺模型架構在面對分佈偏移（Distribution Shift）時的穩健性預測與解釋是否正確。我們選用了 4 個預訓練模型，並在 ImageNet 類別子集上進行嚴格的量化測試與分析。</p>
            <ul>
                <li><strong>基準資料集 (Clean)</strong>: ImageNet-V2</li>
                <li><strong>偏移資料集 (Shifted)</strong>: ImageNet-R (包含 16 種如卡通、塗鴉等風格變體)</li>
                <li><strong>受測模型</strong>: ResNeXt-101, MobileNet-V3, ViT-B16-V1, ViT-B16-SWAG</li>
                <li><strong>預訓練權重</strong>: 使用官方 ImageNet-1K V1 及 SWAG 權重以對齊比較基礎。</li>
            </ul>
        </div>

        <h2>2. 實驗環境 (Experimental Environment)</h2>
        <table style="width: 50%; margin-left: 0; text-align: left;">
            <tr><th style="text-align: left;">項目</th><th style="text-align: left;">內容</th></tr>
            <tr><td>硬體</td><td>NVIDIA GeForce RTX 5090 (32GB VRAM)</td></tr>
            <tr><td>軟體</td><td>PyTorch 2.11.0, CUDA 12.8</td></tr>
            <tr><td>執行模式</td><td>GPU Inference (CUDA)</td></tr>
        </table>

        <h2>3. 基礎穩健性指標評估 (Robustness Metrics)</h2>
        <table>
            <tr>
                <th style="width: 20%;">指標 (Metrics)</th>
                {"".join([f"<th>{m['name']}</th>" for m in all_model_metrics])}
            </tr>
            <tr>
                <td>Clean Accuracy</td>
                {"".join([f"<td>{m['acc_clean']:.2%}</td>" for m in all_model_metrics])}
            </tr>
            <tr>
                <td>Shifted Accuracy</td>
                {"".join([f"<td>{m['acc_shift']:.2%}</td>" for m in all_model_metrics])}
            </tr>
            <tr>
                <td><strong>Accuracy Drop</strong></td>
                {"".join([f"<td>{m['drop']:.4f}</td>" for m in all_model_metrics])}
            </tr>
            <tr>
                <td><strong>Relative Drop</strong></td>
                {"".join([f"<td>{m['rel_drop']:.2%}</td>" for m in all_model_metrics])}
            </tr>
            <tr>
                <td><strong>Avg Robust Acc</strong></td>
                {"".join([f"<td>{m['avg_robust']:.2%}</td>" for m in all_model_metrics])}
            </tr>
            <tr>
                <td><strong>Worst-case Acc</strong></td>
                {"".join([f"<td>{m['worst_acc']:.2%}</td>" for m in all_model_metrics])}
            </tr>
            <tr>
                <td><strong>Wrong Conf</strong></td>
                {"".join([f"<td>{m['wrong_conf']:.2%}</td>" for m in all_model_metrics])}
            </tr>
            <tr>
                <td><strong>Rej Robust (80%)</strong></td>
                {"".join([f"<td>{m['rej_robust']:.2%}</td>" for m in all_model_metrics])}
            </tr>
        </table>

        <div style="page-break-after: always;"></div>

        <h2>4. 假說審核與證據 (AI Claim Audit)</h2>

        <h3>H1: 整體 Domain Shift 表現</h3>
        <p><strong>Claim:</strong> <code>ViT-b-16 (SWAG)</code> > <code>ViT-b-16 (V1)</code> > <code>ResNeXt-101</code> > <code>MobileNet</code> (就穩健性而言)</p>
        <p><strong>Evidence:</strong> 
        依據 Accuracy Drop 指標（越小越穩健），四個模型的降幅排序為：{h1_evidence_text}。
        </p>
        <p><strong>Decision:</strong> <span class="audit-refuted">Refuted (部分推翻)</span> - 雖然 SWAG 的降幅最小，但高參數卷積模型 ResNeXt-101 的穩健性實際上優於標準的 ViT-V1，打破了單純「Transformer 架構必然決定較高穩健性」的說法。</p>

        <h3>H2: Style-Specific 脆弱點</h3>
        <p><strong>Claim:</strong> CNNs 最大的 Drop 在 <code>line drawing</code>，而 ViTs 最大的 Drop 在 <code>pattern</code> / <code>embroidery</code>。</p>
        <p><strong>Evidence:</strong>
        根據分析，四個模型表現最差 (Worst-case) 的風格分別為：
        {h2_evidence_text}
        </p>
        <p><strong>Decision:</strong> <span class="audit-refuted">Refuted (推翻)</span> - 不論是 CNN 陣營還是 ViT 陣營，其最脆弱的風格高度一致（皆為 <code>graffiti</code> 或 <code>tattoo</code>），並未如 AI 預期般在不同材質（如 <code>line drawing</code> 或 <code>pattern</code>）上產生分歧。</p>

        <h3>H3: 免重新訓練的補救措施模擬</h3>
        <p><strong>Claim:</strong> Cross-Architecture Logit Ensemble yields the highest boost.</p>
        <p><strong>Evidence:</strong></p>
        <table>
            <tr><th>Strategy</th><th>Coverage</th><th>Accuracy on Accepted</th><th>Gain vs Best Baseline (SWAG)</th></tr>
            {h3_rows}
        </table>
        <p><strong>Decision:</strong> <span class="audit-refuted">Refuted (推翻)</span> - 實驗證明單純的 Confidence-Weighted Ensemble 反而拉低了 SWAG 模型的原始表現 (-1.67%)。真正能大幅提升系統可靠度的是 <strong>Agreement Rejection (一致性拒絕)</strong>，它在保留部分高可信度資料的情況下，讓子集準確度顯著提升 (+17.07%)。</p>

        <h3>H4: 失敗重疊率 (Failure Overlap)</h3>
        <p><strong>Claim:</strong> Architecture dictates Failure Overlap more than Data Scale.</p>
        <p><strong>Evidence:</strong>
        我們計算了模型在預測錯誤時的 Jaccard Similarity：
        <ul>
            <li><strong>Cross Pair (不同架構, 相似訓練):</strong> ResNeXt-101 vs ViT-B16-V1 錯誤重疊率為 <strong>{cross_pair_overlap:.2%}</strong></li>
            <li><strong>ViT Pair (同架構, 不同訓練):</strong> ViT-B16-V1 vs ViT-B16-SWAG 錯誤重疊率僅為 <strong>{vit_pair_overlap:.2%}</strong></li>
        </ul>
        </p>
        <p><strong>Decision:</strong> <span class="audit-refuted">Refuted (推翻)</span> - 實驗完美證明了「訓練資料的規模與方式（如 SWAG）」比「模型底層的架構（CNN vs ViT）」更能決定模型在領域偏移下的失敗模式。相同架構不同訓練的 ViT Pair 重疊率，遠低於不同架構但訓練設定類似的 Cross Pair。</p>

        <div style="page-break-after: always;"></div>

        <h3>H5: CLIP 風格信心度與模型錯誤率之相關性</h3>
        <p><strong>Claim:</strong> Higher CLIP style confidence correlates with higher CNN error rates and severe Wrong Confidence.</p>
        <p><strong>Evidence:</strong></p>
        <div style="text-align: center;">
            <img src="data:image/png;base64,{h5_chart_b64}" style="max-width: 80%;">
        </div>
        <table>
            <tr><th>Model</th><th>Pearson r (Style Conf vs Wrong Conf)</th><th>Pearson r (Style Conf vs Error Rate)</th></tr>
            {h5_table_rows}
        </table>
        <p><strong>Decision:</strong> <span class="audit-refuted">Refuted (推翻)</span> - 統計結果顯示，CLIP 的風格信心度與模型的錯誤機率或誤判信心值之間，只有極度微弱的正相關（Pearson r 僅約 0.08~0.12）。這代表「這張圖很像某種強烈風格」並不是導致模型答錯的決定性因素。</p>

        <h2>5. 失敗案例分析 (Failure Analysis)</h2>
        <div class="image-container">
            {"".join([f'''
            <div class="image-box">
                <img src="failure_cases/{name}_shifted_fail_0.png" onerror="this.style.display='none'">
                <div class="caption">{name} Fail</div>
            </div>
            ''' for name in [m['name'] for m in all_model_metrics]])}
        </div>
        <p><strong>觀察：</strong> 從具體的誤判資料（如 `misclassify_analysis.csv`）中我們發現，模型常將 `line drawing` 誤判為 `nematode` 或 `hook`，將 `tattoo` 誤判為 `Band_Aid`。這顯示模型嚴重依賴局部形狀與皮膚背景等捷徑特徵（Shortcut features），缺乏對整體語義的理解能力。</p>

        <h2>6. 結語與反思 (Reflection)</h2>
        <p><strong>AI 對我的幫助：</strong> AI 協助我快速梳理了穩健性評估框架，並構思了具備深度的實驗假說（如 H3 的免訓練補救、H4 的架構與資料規模之爭）。它也大幅加速了腳本開發，包含 CLIP 的 Zero-shot 風格標註整合。</p>
        <p><strong>AI 的誤導之處：</strong> 本次實驗的 5 個 AI Claims 幾乎全數被實驗數據推翻。AI 過於想當然地認為「架構決定一切」或「風格越明顯錯誤越嚴重」。然而實驗結果殘酷地證明：大規模弱監督訓練（SWAG）的影響力遠大於網路架構的選擇；而在面對真實世界的 Domain Shift 時，單純的模型集成可能弊大於利。這深刻提醒了我們：<strong>在 AI 時代，實驗驗證 (Empirical Audit) 比以往任何時候都更加不可或缺。</strong></p>
    </body>
    </html>
    """
    
    with open('report.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("Generating PDF report...")
    HTML(string=html_template, base_url='.').write_pdf('report.pdf')
    print("Report generated: report.pdf")

if __name__ == "__main__":
    generate_report()
