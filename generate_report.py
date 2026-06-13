import json
import pandas as pd
from weasyprint import HTML, CSS
import os

def generate_report():
    # Load Data
    with open('results/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    df_claims = pd.read_csv('results.csv')
    
    vgg = metrics['VGG-19']
    vit = metrics['ViT']
    
    # Calculate some extra stats if needed
    vgg_drop = vgg['acc_clean'] - vgg['acc_shift']
    vit_drop = vit['acc_clean'] - vit['acc_shift']
    
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
				font-size: 9pt;
                max-width: 800px;
                margin: auto;
            }}
            h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; font-weight: bolder; }}
            h2 {{ color: #2980b9; border-left: 5px solid #2980b9; padding-left: 10px; margin-top: 30px; font-weight: bolder; }}
            h3 {{ color: #34495e; font-weight: bolder; }}
            table {{ width: 100%; border-collapse: collapse; margin: 9px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; color: #2c3e50; font-weight: bolder; }}
            .summary-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .audit-supported {{ color: #27ae60; font-weight: bold; }}
            .audit-refuted {{ color: #c0392b; font-weight: bold; }}
            .audit-unclear {{ color: #f39c12; font-weight: bold; }}
            .image-container {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
            .image-box {{ width: 22%; margin-bottom: 20px; text-align: center; border: 1px solid #eee; padding: 10px; }}
            .image-box img {{ max-width: 100%; height: auto; border-radius: 3px; }}
            .caption {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
        </style>
    </head>
    <body>
        <h1>AI-Claim Audit for Robust Vision</h1>
        <p style="text-align: center; font-style: italic;">MILS Assignment Report</p>

        <h2>1. 實驗設定 (Problem Setting)</h2>
        <div class="summary-box">
            <p>本次實驗旨在比較卷積神經網路 (CNN) 與視覺轉換器 (Transformer) 在面對分佈偏移（Distribution Shift）時的穩健性。我們選擇了經典的 <strong>VGG-19</strong> 與 <strong>ViT-B/16</strong> 模型，並在 ImageNet 類別子集上進行測試。</p>
            <ul>
                <li><strong>基準資料集 (Clean)</strong>: ImageNet-V2</li>
                <li><strong>偏移資料集 (Shifted)</strong>: ImageNet-R (包含卡通、素描、油畫等風格變體)</li>
                <li><strong>類別數量</strong>: 隨機選定 20 個共同類別 (如：金魚、鯊魚、公雞等)</li>
            </ul>
        </div>

        <h2>2. 實驗環境 (Experimental Environment)</h2>
        <table>
            <tr><th>項目</th><th>內容</th></tr>
            <tr><td>硬體</td><td>NVIDIA GeForce RTX 5090 (32GB VRAM)</td></tr>
            <tr><td>作業系統</td><td>Linux (Ubuntu 24.04 核心)</td></tr>
            <tr><td>軟體版本</td><td>Python 3.12.3, PyTorch 2.11.0, CUDA 12.8</td></tr>
            <tr><td>執行模式</td><td>GPU Inference (NVIDIA RTX 5090 / CUDA)</td></tr>
        </table>

        <h2>3. 評估結果 (Quantitative Results)</h2>
        <table>
            <tr>
                <th>指標</th>
                <th>VGG-19 (CNN)</th>
                <th>ViT-B/16 (Transformer)</th>
            </tr>
            <tr>
                <td>Clean Accuracy</td>
                <td>75.50%</td>
                <td>84.00%</td>
            </tr>
            <tr>
                <td>Shifted Accuracy</td>
                <td>17.00%</td>
                <td>28.50%</td>
            </tr>
            <tr>
                <td><strong>Accuracy Drop</strong></td>
                <td><strong>0.5850</strong></td>
                <td><strong>0.5550</strong></td>
            </tr>
            <tr>
                <td>Avg Wrong Confidence</td>
                <td>41.68% (All)</td>
                <td>46.56% (All)</td>
            </tr>
        </table>

        <h2>4. AI Claim Audit (假說審核)</h2>
        <p>下表展示了我們對 AI 提出之假說的驗證結果：</p>
        <table>
            <tr>
                <th>ID</th>
                <th>假說內容 (Claim)</th>
                <th>證據 (Evidence)</th>
                <th>審核結果 (Decision)</th>
            </tr>
            
        <tr>
            <td>C1</td>
            <td>ViT has a smaller average Accuracy Drop on ImageNet-R than VGG-19.</td>
            <td>VGG Drop: 0.5850, ViT Drop: 0.5550</td>
            <td class="audit-supported">Supported</td>
        </tr>
        
        <tr>
            <td>C2</td>
            <td>VGG-19's Accuracy Drop is largest on 'cartoon' and 'sketch' sub-categories.</td>
            <td>Max VGG Drop on plush object (0.7550). </td>
            <td class="audit-refuted">Refuted</td>
        </tr>
        
        <tr>
            <td>C3</td>
            <td>Wrong Confidence is higher for VGG-19 than ViT.</td>
            <td>VGG Wrong Conf: 0.3443, ViT Wrong Conf: 0.3625</td>
            <td class="audit-refuted">Refuted</td>
        </tr>
        
        <tr>
            <td>C4</td>
            <td>Both models have their highest Accuracy Drop on 'sculpture' renditions.</td>
            <td>VGG Max: plush object, ViT Max: plush object</td>
            <td class="audit-refuted">Refuted</td>
        </tr>
        
        <tr>
            <td>C5</td>
            <td>The failure overlap between ViT and VGG-19 is less than 50%.</td>
            <td>Overlap Ratio: 80.17%</td>
            <td class="audit-refuted">Refuted</td>
        </tr>
        
        </table>

        <h2>5. 深入分析與觀察 (In-depth Analysis)</h2>
        <h3>5.1 穩健性差異</h3>
        <p>實驗結果支持了 <strong>ViT 比 VGG-19 更具穩健性</strong> 的說法。ViT 的 Accuracy Drop 為 0.5550，優於 VGG-19 的 0.5850。這可能歸功於 Transformer 的全局注意力機制，使其能捕捉到比局部卷積更具辨識度的形狀特徵，而非僅依賴局部紋理。</p>

        <h3>5.2 過度自信問題 (Overconfidence)</h3>
        <p>令人意外的是，ViT 在答錯時的平均信心值比 VGG-19 更高（Refuted C3）。這顯示 ViT 雖然準確率較高，但一旦遇到不熟悉的風格偏移，其預測往往更具誤導性，這在安全關鍵的應用中是一個潛在風險。</p>

        <h3>5.3 錯誤模式的重疊 (Failure Overlap)</h3>
        <p>儘管兩者架構迥異，但在 ImageNet-R 上的失敗案例重疊率高達 <strong>80.17%</strong>。這說明模型在面對極端的風格變換時，往往在相同的困難樣本上卡關，顯示目前的視覺模型在處理特定「本質性」缺失上仍有共同的弱點。</p>

        <div style="page-break-after: always;"></div>

        <h2>6. 失敗案例分析 (Failure Analysis)</h2>
        <p>以下展示了模型在風格偏移資料集中的典型錯誤：</p>
        <div class="image-container">
            <div class="image-box">
                <img src="failure_cases/VGG-19_shifted_fail_0.png" alt="VGG Fail">
                <div class="caption">VGG-19 失敗案例 #1 (Shifted)</div>
            </div>
            <div class="image-box">
                <img src="failure_cases/ViT_shifted_fail_0.png" alt="ViT Fail">
                <div class="caption">ViT-B/16 失敗案例 #1 (Shifted)</div>
            </div>
            <div class="image-box">
                <img src="failure_cases/VGG-19_shifted_fail_1.png" alt="VGG Fail">
                <div class="caption">VGG-19 失敗案例 #2 (Shifted)</div>
            </div>
            <div class="image-box">
                <img src="failure_cases/ViT_shifted_fail_1.png" alt="ViT Fail">
                <div class="caption">ViT-B/16 失敗案例 #2 (Shifted)</div>
            </div>
        </div>
        <p><strong>觀察：</strong> 許多錯誤發生在「素描」或「高抽象度」的影像中。模型往往將線條特徵誤認為其他類別（例如將細長線條誤認為針或天線）。</p>

        <h2>7. 結語與反思 (Reflection)</h2>
        <p><strong>AI 對我的幫助：</strong> AI 協助我快速產生了可測試的假說，並提供了實作穩健性評估框架的腳本雛形。</p>
        <p><strong>AI 的誤導之處：</strong> AI 最初預測 VGG-19 會有較高的 Wrong Confidence，且認為兩模型失敗案例會大不相同。然而實驗證明 ViT 更加過度自信，且兩者失敗模式高度一致。這顯示了「實驗驗證」在 AI 時代的重要性——我們不能盲目相信 AI 的推論，必須以數據說話。</p>
    </body>
    </html>
    
    """
    
    # Save HTML temporarily
    with open('report.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    # Generate PDF
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
            <td>{row['Claim']}</td>
            <td>{row['Evidence']}</td>
            <td class="{decision_class}">{row['Decision']}</td>
        </tr>
        """
    return rows

if __name__ == "__main__":
    generate_report()
