import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    metrics_path = 'results/metrics.json'
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found.")
        return

    with open(metrics_path, 'r') as f:
        all_metrics = json.load(f)

    # 定義目標模型與順序
    target_models = ["ResNeXt-101", "MobileNet-V3", "ViT-B16-V1", "ViT-B16-SWAG"]
    
    # 檢查模型是否存在
    available_models = [m for m in target_models if m in all_metrics]
    if len(available_models) < 4:
        print(f"Warning: Only found {available_models}. Expected all four base models.")

    if not available_models:
        print("No models found in metrics.json.")
        return

    # 提取所有出現過的風格
    all_styles = set()
    for m in available_models:
        results = all_metrics[m].get('results_shift', [])
        for res in results:
            if 'style' in res:
                all_styles.add(res['style'])
    
    styles = sorted(list(all_styles))
    
    # 準備繪圖數據: {style: [acc_model1, acc_model2, ...]}
    plot_data = {style: [] for style in styles}
    
    for model_name in available_models:
        results = all_metrics[model_name].get('results_shift', [])
        if not results:
            for style in styles:
                plot_data[style].append(0)
            continue
            
        df = pd.DataFrame(results)
        
        # 計算該模型下各風格的 Accuracy
        style_acc = df.groupby('style')['is_correct'].mean().to_dict()
        
        for style in styles:
            # 如果某個風格在該模型中沒出現，填入 0
            plot_data[style].append(style_acc.get(style, 0))

    # --- 繪製折線圖 ---
    plt.figure(figsize=(12, 8))
    
    for style in styles:
        plt.plot(available_models, plot_data[style], marker='o', label=style, linewidth=2)

    plt.title('Style Accuracy Across Models', fontsize=16)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.8, 0) # Invert Y-axis: 0.0 at top, 1.0 at bottom
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Styles")
    plt.tight_layout()

    # 儲存結果
    os.makedirs('results', exist_ok=True)
    plot_output = 'results/style_accuracy_plot.png'
    plt.savefig(plot_output)
    print(f"Plot saved to {plot_output}")

    # 同時輸出數值 CSV 供檢查
    csv_data = []
    for i, model in enumerate(available_models):
        row = {'Model': model}
        for style in styles:
            row[style] = plot_data[style][i]
        csv_data.append(row)
    
    df_csv = pd.DataFrame(csv_data)
    csv_output = 'results/style_error_analysis.csv'
    df_csv.to_csv(csv_output, index=False)
    print(f"Numerical analysis saved to {csv_output}")

if __name__ == "__main__":
    main()
