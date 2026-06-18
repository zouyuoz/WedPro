import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    metrics_path = 'results/metrics.json'
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found.")
        return

    with open(metrics_path, 'r') as f:
        data = json.load(f)

    # 目標模型
    target_models = ["ResNeXt-101", "MobileNet-V3", "ViT-B16-V1", "ViT-B16-SWAG"]
    
    plt.figure(figsize=(10, 6))

    # 定義 Bins
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

    for model_name in target_models:
        if model_name not in data:
            continue
            
        results_shift = data[model_name].get('results_shift', [])
        if not results_shift:
            continue
            
        df = pd.DataFrame(results_shift)
        
        # 進行分箱 (Binning)
        df['style_conf_bin'] = pd.cut(df['style_confidence'], bins=bins, labels=labels, include_lowest=True)
        
        # 計算每個分箱的 Error Rate
        bin_stats = df.groupby('style_conf_bin', observed=False)['is_correct'].agg(['count', 'sum'])
        error_rates = 1.0 - (bin_stats['sum'] / bin_stats['count'])
        
        # 繪圖
        plt.plot(labels, error_rates, marker='o', label=model_name, linewidth=2)

    plt.title('H5: Error Rate by Style Confidence Bin', fontsize=14)
    plt.xlabel('CLIP Style Confidence Bin', fontsize=12)
    plt.ylabel('Error Rate (1 - Accuracy)', fontsize=12)
    plt.ylim(0.3, 0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Models")
    
    # 儲存圖片
    os.makedirs('results', exist_ok=True)
    output_path = 'results/h5_error_by_confidence_plot.png'
    plt.savefig(output_path)
    print(f"H5 visualization saved to {output_path}")

if __name__ == "__main__":
    main()
