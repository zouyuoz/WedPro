import json
import pandas as pd
from weasyprint import HTML, CSS
import os

def generate_report():
    # Load Data
    with open('results/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    df_claims = pd.read_csv('results.csv')
    
    m1_name = "ResNeXt-101"
    m2_name = "MobileNet-V3"
    
    m1_data = metrics[m1_name]
    m2_data = metrics[m2_name]
    
    # Quantitative Stats Calculation
    def calculate_robustness_metrics(model_data):
        acc_clean = model_data['acc_clean']
        acc_shift = model_data['acc_shift']
        
        # Accuracy Drop
        drop = acc_clean - acc_shift
        
        # Relative Drop
        rel_drop = drop / acc_clean if acc_clean > 0 else 0
        
        # Average Robust Accuracy & Worst-case (over styles)
        styles = {}
        for r in model_data['results_shift']:
            s = r.get('style', 'unknown')
            if s not in styles: styles[s] = {'correct': 0, 'total': 0}
            styles[s]['total'] += 1
            if r['is_correct']: styles[s]['correct'] += 1
        
        style_accs = [v['correct']/v['total'] for v in styles.values() if v['total'] > 0]
        avg_robust = sum(style_accs) / len(style_accs) if style_accs else 0
        worst_acc = min(style_accs) if style_accs else 0
        
        # Wrong Confidence
        wc = [r['confidence'] for r in model_data['results_shift'] if not r['is_correct']]
        wrong_conf = sum(wc) / len(wc) if wc else 0
        
        # Rejection Robustness (80% coverage)
        sorted_results = sorted(model_data['results_shift'], key=lambda x: x['confidence'], reverse=True)
        keep_count = int(len(sorted_results) * 0.8)
        top_results = sorted_results[:keep_count]
        rej_robust = sum([1 for r in top_results if r['is_correct']]) / keep_count if keep_count > 0 else 0
        
        return {
            'acc_clean': acc_clean,
            'acc_shift': acc_shift,
            'drop': drop,
            'rel_drop': rel_drop,
            'avg_robust': avg_robust,
            'worst_acc': worst_acc,
            'wrong_conf': wrong_conf,
            'rej_robust': rej_robust
        }

    m1_m = calculate_robustness_metrics(m1_data)
    m2_m = calculate_robustness_metrics(m2_data)
    
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
                margin: 20mm 15mm;
                @bottom-right {{
                    content: counter(page);
                    font-family: sans-serif;
                    font-size: 9pt;
                    color: #888888;
                }}
            }}
            body {{
                font-family: 'PingFang TC', 'Microsoft JhengHei', sans-serif;
                line-height: 1.6;
                color: #333;
                font-size: 10pt;
                max-width: 800px;
                margin: auto;
            }}
            h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; font-weight: bolder; }}
            h2 {{ color: #2980b9; border-left: 5px solid #2980b9; padding-left: 10px; margin-top: 30px; font-weight: bolder; }}
            h3 {{ color: #34495e; font-weight: bolder; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #f2f2f2; color: #2c3e50; font-weight: bolder; }}
            .summary-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .audit-supported {{ color: #27ae60; font-weight: bold; }}
            .audit-refuted {{ color: #c0392b; font-weight: bold; }}
            .audit-unclear {{ color: #f39c12; font-weight: bold; }}
            .image-container {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
            .image-box {{ width: 45%; margin-bottom: 20px; text-align: center; border: 1px solid #eee; padding: 5px; }}
            .image-box img {{ max-width: 100%; height: auto; border-radius: 3px; }}
            .caption {{ font-size: 0.8em; color: #666; margin-top: 5px; }}
        </style>
    </head>
    <body>
        <h1>AI-Claim Audit for Robust Vision</h1>
        <p style="text-align: center; font-style: italic;">MILS Assignment Report</p>

        <h2>1. 實驗設定 (Problem Setting)</h2>
        <div class="summary-box">
            <p>本次實驗旨在比較高性能大型架構與輕量化行動架構在面對分佈偏移（Distribution Shift）時的穩健性指標。</p>
            <ul>
                <li><strong>基準資料集 (Clean)</strong>: ImageNet-V2</li>
                <li><strong>偏移資料集 (Shifted)</strong>: ImageNet-R (16 種風格變體)</li>
                <li><strong>模型</strong>: {m1_name} (Heavy) vs. {m2_name} (Lightweight)</li>
                <li><strong>權重</strong>: 皆使用 IMAGENET1K_V2 預訓練權重</li>
            </ul>
        </div>

        <h2>2. 實驗環境 (Experimental Environment)</h2>
        <table>
            <tr><th>項目</th><th>內容</th></tr>
            <tr><td>硬體</td><td>NVIDIA GeForce RTX 5090 (32GB VRAM)</td></tr>
            <tr><td>執行模式</td><td>GPU Inference (CUDA 12.8)</td></tr>
        </table>

        <h2>3. 穩健性指標評估 (Robustness Metrics)</h2>
        <table>
            <tr>
                <th>指標 (Metrics)</th>
                <th>{m1_name}</th>
                <th>{m2_name}</th>
            </tr>
            <tr>
                <td>Clean Accuracy</td>
                <td>{m1_m['acc_clean']:.2%}</td>
                <td>{m2_m['acc_clean']:.2%}</td>
            </tr>
            <tr>
                <td>Shifted Accuracy</td>
                <td>{m1_m['acc_shift']:.2%}</td>
                <td>{m2_m['acc_shift']:.2%}</td>
            </tr>
            <tr>
                <td><strong>Accuracy Drop</strong></td>
                <td>{m1_m['drop']:.4f}</td>
                <td>{m2_m['drop']:.4f}</td>
            </tr>
            <tr>
                <td><strong>Relative Drop</strong></td>
                <td>{m1_m['rel_drop']:.2%}</td>
                <td>{m2_m['rel_drop']:.2%}</td>
            </tr>
            <tr>
                <td><strong>Average Robust Accuracy</strong></td>
                <td>{m1_m['avg_robust']:.2%}</td>
                <td>{m2_m['avg_robust']:.2%}</td>
            </tr>
            <tr>
                <td><strong>Worst-condition Accuracy</strong></td>
                <td>{m1_m['worst_acc']:.2%}</td>
                <td>{m2_m['worst_acc']:.2%}</td>
            </tr>
            <tr>
                <td><strong>Wrong Confidence</strong></td>
                <td>{m1_m['wrong_conf']:.2%}</td>
                <td>{m2_m['wrong_conf']:.2%}</td>
            </tr>
            <tr>
                <td><strong>Rejection Robustness (80%)</strong></td>
                <td>{m1_m['rej_robust']:.2%}</td>
                <td>{m2_m['rej_robust']:.2%}</td>
            </tr>
        </table>

        <h2>4. AI Claim Audit (假說審核)</h2>
        <table>
            <tr>
                <th>ID</th>
                <th>假說內容 (Claim)</th>
                <th>證據 (Evidence)</th>
                <th>審核結果 (Decision)</th>
            </tr>
            {generate_table_rows(df_claims)}
        </table>

        <h2>5. 深入分析與觀察 (In-depth Analysis)</h2>
        <p>實驗顯示，儘管 {m1_name} 在基準準確度上大幅領先，但兩者在面對風格偏移時皆表現出顯著的性能退化。值得注意的是，{m2_name} 雖然規模較小，但在某些特定穩健性指標（如 Relative Drop）上展現出了不俗的韌性。</p>

        <div style="page-break-after: always;"></div>

        <h2>6. 失敗案例分析 (Failure Analysis)</h2>
        <div class="image-container">
            <div class="image-box">
                <img src="failure_cases/{m1_name}_shifted_fail_0.png">
                <div class="caption">{m1_name} Fail #1</div>
            </div>
            <div class="image-box">
                <img src="failure_cases/{m2_name}_shifted_fail_0.png">
                <div class="caption">{m2_name} Fail #1</div>
            </div>
        </div>

        <h2>7. 結語 (Conclusion)</h2>
        <p>本實驗透過比較 ResNeXt 與 MobileNet，揭示了模型規模與穩健性之間的複雜關係。多維度指標的審核使我們能更客觀地評估 AI 模型的實際部署風險。</p>
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
            <td style="font-size: 8pt;">{row['Claim']}</td>
            <td style="font-size: 8pt;">{row['Evidence']}</td>
            <td class="{decision_class}">{row['Decision']}</td>
        </tr>
        """
    return rows

if __name__ == "__main__":
    generate_report()
