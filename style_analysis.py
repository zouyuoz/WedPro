import json
import pandas as pd
import os

def analyze_styles():
    metrics_path = 'results/metrics.json'
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found.")
        return

    with open(metrics_path, 'r') as f:
        data = json.load(f)

    analysis_results = {}

    for model_name in ['ResNeXt-101', 'MobileNet-V3']:
        model_data = data[model_name]
        acc_clean = model_data['acc_clean']
        results_shift = model_data['results_shift']

        # Convert to DataFrame for easier grouping
        df = pd.DataFrame(results_shift)
        
        # 1. Count errors (is_correct == False) per style
        error_df = df[df['is_correct'] == False]
        error_counts = error_df['style'].value_counts().to_dict()

        # 2. Accuracy Drop per style
        # First calculate Accuracy per style: (Correct / Total)
        style_stats = df.groupby('style')['is_correct'].agg(['sum', 'count'])
        style_stats['accuracy'] = style_stats['sum'] / style_stats['count']
        
        # Accuracy Drop = Global Clean Acc - Style-specific Shifted Acc
        style_stats['accuracy_drop'] = acc_clean - style_stats['accuracy']
        
        analysis_results[model_name] = {
            'error_counts_per_style': error_counts,
            'style_metrics': style_stats[['count', 'accuracy', 'accuracy_drop']].to_dict(orient='index')
        }

    # Print Results in a unified table format
    for model_name, results in analysis_results.items():
        print(f"\n{'='*25} {model_name} Style Analysis {'='*25}")
        print(f"  {'Style':<18} | {'Total':<6} | {'Error':<6} | {'Acc':<6} | {'Acc Drop':<10}")
        print("-" * 65)
        
        # Sort by accuracy drop descending
        sorted_metrics = sorted(results['style_metrics'].items(), key=lambda x: x[1]['accuracy_drop'], reverse=True)
        for style, metrics in sorted_metrics:
            error_count = results['error_counts_per_style'].get(style, 0)
            print(f"  {style:.<18} | {metrics['count']:<6} | {error_count:<6} | {metrics['accuracy']:6.2%} | {metrics['accuracy_drop']:.4f}")

    # Optionally save to a CSV
    rows = []
    for model_name, results in analysis_results.items():
        for style, metrics in results['style_metrics'].items():
            rows.append({
                'Model': model_name,
                'Style': style,
                'Error_Count': results['error_counts_per_style'].get(style, 0),
                'Total_Count': metrics['count'],
                'Style_Accuracy': metrics['accuracy'],
                'Accuracy_Drop': metrics['accuracy_drop']
            })
    
    pd.DataFrame(rows).to_csv('results/style_error_analysis.csv', index=False)
    print("\nDetailed analysis saved to results/style_error_analysis.csv")

if __name__ == "__main__":
    analyze_styles()
