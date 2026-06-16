import json
import pandas as pd
from weasyprint import HTML, CSS
import os

def generate_report():
    # Load Data
    metrics_path = 'results/metrics.json'
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found.")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    df_claims = pd.read_csv('results.csv')
    
    # Quantitative Stats Calculation for all models
    def calculate_robustness_metrics(model_name, model_data):
        acc_clean = model_data['acc_clean']
        acc_shift = model_data['acc_shift']
        
        drop = acc_clean - acc_shift
        rel_drop = drop / acc_clean if acc_clean > 0 else 0
        
        styles = {}
        for r in model_data['results_shift']:
            s = r.get('style', 'unknown')
            if s not in styles: styles[s] = {'correct': 0, 'total': 0}
            styles[s]['total'] += 1
            if r['is_correct']: styles[s]['correct'] += 1
        
        style_accs = [v['correct']/v['total'] for v in styles.values() if v['total'] > 0]
        avg_robust = sum(style_accs) / len(style_accs) if style_accs else 0
        worst_acc = min(style_accs) if style_accs else 0
        
        wc = [r['confidence'] for r in model_data['results_shift'] if not r['is_correct']]
        wrong_conf = sum(wc) / len(wc) if wc else 0
        
        sorted_results = sorted(model_data['results_shift'], key=lambda x: x['confidence'], reverse=True)
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
            'wrong_conf': wrong_conf,
            'rej_robust': rej_robust
        }

    all_model_metrics = [calculate_robustness_metrics(name, data) for name, data in metrics.items()]
    
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
            h2 {{ color: #2980b9; border-left: 5px solid #2980b9; padding-left: 10px; margin-top: 20px; font-weight: bolder; font-size: 14pt; }}
            h3 {{ color: #34495e; font-weight: bolder; font-size: 11pt; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; table-layout: fixed; }}
            th, td {{ border: 1px solid #ddd; padding: 6px; text-align: center; word-wrap: break-word; font-size: 8pt; }}
            th {{ background-color: #f2f2f2; color: #2c3e50; font-weight: bolder; }}
            .summary-box {{ background-color: #ecf0f1; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .audit-supported {{ color: #27ae60; font-weight: bold; }}
            .audit-refuted {{ color: #c0392b; font-weight: bold; }}
            .audit-unclear {{ color: #f39c12; font-weight: bold; }}
            .image-container {{ display: flex; flex-wrap: wrap; justify-content: flex-start; }}
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
            <p>本次實驗旨在比較不同模型架構（高性能大型架構、輕量化架構、以及不同訓練方式的 Transformer）在面對分佈偏移（Distribution Shift）時的穩健性指標。</p>
            <ul>
                <li><strong>基準資料集 (Clean)</strong>: ImageNet-V2</li>
                <li><strong>偏移資料集 (Shifted)</strong>: ImageNet-R (包含 16 種風格變體)</li>
                <li><strong>模型列表</strong>: {", ".join(metrics.keys())}</li>
            </ul>
        </div>

        <h2>2. 實驗環境 (Experimental Environment)</h2>
        <table style="width: 50%; margin-left: 0; text-align: left;">
            <tr><th style="text-align: left;">項目</th><th style="text-align: left;">內容</th></tr>
            <tr><td>硬體</td><td>NVIDIA GeForce RTX 5090 (32GB VRAM)</td></tr>
            <tr><td>軟體</td><td>PyTorch 2.11.0, CUDA 12.8</td></tr>
        </table>

        <h2>3. 穩健性指標評估 (Robustness Metrics)</h2>
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

        <h2>4. AI Claim Audit (假說審核)</h2>
        <p style="font-size: 8pt; margin-bottom: 5px;">* 審核基於 ResNeXt-101 與 MobileNet-V3 之對比。</p>
        <table style="table-layout: auto;">
            <tr>
                <th style="width: 50px;">ID</th>
                <th>假說內容 (Claim)</th>
                <th>證據 (Evidence)</th>
                <th style="width: 80px;">審核結果</th>
            </tr>
            {generate_table_rows(df_claims)}
        </table>

        <h2>5. 深入分析與觀察 (In-depth Analysis)</h2>
        <p>實驗結果顯示，<strong>ViT-B16-SWAG 在所有模型中展現了最強的穩健性</strong>，其 Accuracy Drop 最低且在極端偏移下的準確度最高。這證明了大規模弱監督預訓練（SWAG）能顯著提升模型對分佈偏移的抵抗力。</p>
        <p>相對地，MobileNet-V3 雖然在基礎準確度上不俗，但在面對風格偏移時表現最為脆弱，其 Worst-case Accuracy 指標顯示在某些風格下幾近失效。</p>

        <div style="page-break-after: always;"></div>

        <h2>6. 失敗案例分析 (Failure Analysis)</h2>
        <div class="image-container">
            {"".join([f'''
            <div class="image-box">
                <img src="failure_cases/{name}_shifted_fail_0.png" onerror="this.style.display='none'">
                <div class="caption">{name} Fail</div>
            </div>
            ''' for name in metrics.keys()])}
        </div>

        <h2>7. 結語 (Conclusion)</h2>
        <p>本實驗透過對比四種不同規模與訓練方式的模型，揭示了模型架構與預訓練策略對穩健性的決定性影響。SWAG 預訓練與大型架構（ResNeXt）在處理複雜風格偏移時具有明顯優勢。</p>
    </body>
    </html>
    """
    
    with open('report.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("Generating PDF report...")
    HTML(string=html_template, base_url='.').write_pdf('report.pdf')
    print("Report generated: report.pdf")

def generate_table_rows(df):
    rows = ""
    for _, row in df.iterrows():
        decision_class = "audit-" + row['Decision'].lower()
        rows += f"""
        <tr>
            <td>{row['ID']}</td>
            <td style="text-align: left;">{row['Claim']}</td>
            <td style="text-align: left;">{row['Evidence']}</td>
            <td class="{decision_class}">{row['Decision']}</td>
        </tr>
        """
    return rows

if __name__ == "__main__":
    generate_report()
