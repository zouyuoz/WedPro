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

    # 定義 Bins (CLIP 的風格信心度)
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

    for model_name in target_models:
        if model_name not in data:
            continue
            
        results_shift = data[model_name].get('results_shift', [])
        if not results_shift:
            continue
            
        df = pd.DataFrame(results_shift)
        
        # 只篩選出「預測錯誤」的樣本
        wrong_df = df[df['is_correct'] == False].copy()
        
        if wrong_df.empty:
            continue

        # 根據 CLIP 的 style_confidence 進行分箱
        wrong_df['style_conf_bin'] = pd.cut(wrong_df['style_confidence'], bins=bins, labels=labels, include_lowest=True)
        
        # 計算每個分箱中，模型本身的平均信心度 (confidence)
        bin_stats = wrong_df.groupby('style_conf_bin', observed=False)['confidence'].mean()
        
        # 繪圖
        plt.plot(labels, bin_stats, marker='s', label=f"{model_name} (Errors Only)", linewidth=2)

    plt.title('H5: Avg Model Confidence on Errors vs. CLIP Style Confidence', fontsize=13)
    plt.xlabel('CLIP Style Confidence Bin', fontsize=11)
    plt.ylabel('Avg Model Wrong Confidence', fontsize=11)
    plt.ylim(0.3, 0.6)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Models")
    
    # 儲存圖片
    os.makedirs('results', exist_ok=True)
    output_path = 'results/h5_wrong_conf_comparison_plot.png'
    plt.savefig(output_path)
    print(f"H5 Wrong Confidence comparison saved to {output_path}")

if __name__ == "__main__":
    main()
