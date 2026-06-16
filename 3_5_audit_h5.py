import json
import os
import pandas as pd
import numpy as np

def main():
    metrics_path = 'results/metrics.json'
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found.")
        return

    with open(metrics_path, 'r') as f:
        data = json.load(f)

    print("=" * 60)
    print("Audit Script 5: CLIP Confidence Correlation (H5)")
    print("=" * 60)

    results_summary = {}

    for model_name, model_data in data.items():
        if 'TTA' in model_name:
            continue
            
        print(f"\nProcessing {model_name}...")
        
        results_shift = model_data.get('results_shift', [])
        if not results_shift:
            continue
            
        df = pd.DataFrame(results_shift)
        
        # 1. Bin ImageNet-R samples by CLIP `style_confidence`
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        df['style_conf_bin'] = pd.cut(df['style_confidence'], bins=bins, labels=labels, include_lowest=True)
        
        # 2. For each bin, calculate the Error Rate (1 - Accuracy)
        bin_stats = df.groupby('style_conf_bin', observed=False)['is_correct'].agg(['count', 'sum']).reset_index()
        bin_stats['error_rate'] = 1.0 - (bin_stats['sum'] / bin_stats['count'])
        bin_stats['error_rate'] = bin_stats['error_rate'].where(bin_stats['count'] > 0, 0)
        
        # Print bin stats
        print("  Error Rate by Style Confidence Bin:")
        for _, row in bin_stats.iterrows():
            print(f"    - Bin {row['style_conf_bin']}: {row['error_rate']:.2%} (N={row['count']})")
        
        # 3. Calculate correlation between CLIP style_confidence and models' own confidence (on misclassified)
        wrong_df = df[df['is_correct'] == False]
        if len(wrong_df) > 1:
            corr_wrong_conf = wrong_df['style_confidence'].corr(wrong_df['confidence'], method='pearson')
        else:
            corr_wrong_conf = 0.0
            
        print(f"  Correlation (Style Conf vs Model Wrong Conf): {corr_wrong_conf:.4f}")
        
        # 4. Calculate correlation between CLIP confidence and Model Error Probability
        df['is_error'] = ~df['is_correct']
        corr_error_rate_point_biserial = df['style_confidence'].corr(df['is_error'], method='pearson')
        
        print(f"  Correlation (Style Conf vs Error Probability): {corr_error_rate_point_biserial:.4f}")
        
        # Convert bin_stats to dict, replacing NaN with None
        bin_records = bin_stats[['style_conf_bin', 'error_rate', 'count']].replace({np.nan: None}).to_dict(orient='records')
        # Cast categories to string for JSON serialization
        for rec in bin_records:
            rec['style_conf_bin'] = str(rec['style_conf_bin'])

        results_summary[model_name] = {
            "Error_Rate_by_Bin": bin_records,
            "Pearson_r_StyleConf_vs_WrongConf": float(corr_wrong_conf) if not np.isnan(corr_wrong_conf) else None,
            "Pearson_r_StyleConf_vs_Error_Prob": float(corr_error_rate_point_biserial) if not np.isnan(corr_error_rate_point_biserial) else None
        }

    # Save to JSON
    os.makedirs('results', exist_ok=True)
    out_file = 'results/audit_h5_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=4)
        
    print(f"\nSaved H5 correlation results to {out_file}")

if __name__ == "__main__":
    main()
