import json
import pandas as pd
import os

def analyze_all():
    metrics_path = 'results/metrics.json'
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found. Run evaluate.py first.")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # ---------------------------------------------------------
    # Part 1: Overall metrics and Claims
    # ---------------------------------------------------------
    # We'll use ResNeXt-101 and MobileNet-V3 for the 5 agreed claims to maintain consistency
    # unless they are missing.
    m1_name = "ResNeXt-101"
    m2_name = "MobileNet-V3"
    
    if m1_name in metrics and m2_name in metrics:
        resnext = metrics[m1_name]
        mobilenet = metrics[m2_name]

        # 1. Accuracy Drop
        resnext_drop = resnext['acc_clean'] - resnext['acc_shift']
        mobilenet_drop = mobilenet['acc_clean'] - mobilenet['acc_shift']

        def get_style_acc(results):
            styles = {}
            for r in results:
                s = r.get('style', 'unknown')
                if s not in styles: styles[s] = {'correct': 0, 'total': 0}
                styles[s]['total'] += 1
                if r['is_correct']: styles[s]['correct'] += 1
            return {s: (v['correct']/v['total']) for s, v in styles.items() if v['total'] > 0}

        resnext_styles = get_style_acc(resnext['results_shift'])
        mobilenet_styles = get_style_acc(mobilenet['results_shift'])
        
        # 5. Wrong Confidence
        resnext_wrong_conf = [r['confidence'] for r in resnext['results_shift'] if not r['is_correct']]
        mobilenet_wrong_conf = [r['confidence'] for r in mobilenet['results_shift'] if not r['is_correct']]
        avg_resnext_wrong_conf = sum(resnext_wrong_conf) / len(resnext_wrong_conf) if resnext_wrong_conf else 0
        avg_mobilenet_wrong_conf = sum(mobilenet_wrong_conf) / len(mobilenet_wrong_conf) if mobilenet_wrong_conf else 0

        # 6. Failure Overlap
        resnext_fails = [not r['is_correct'] for r in resnext['results_shift']]
        mobilenet_fails = [not r['is_correct'] for r in mobilenet['results_shift']]
        both_fail = sum([1 for f1, f2 in zip(resnext_fails, mobilenet_fails) if f1 and f2])
        either_fail = sum([1 for f1, f2 in zip(resnext_fails, mobilenet_fails) if f1 or f2])
        overlap_ratio = both_fail / either_fail if either_fail > 0 else 0

        # Style-Specific Analysis
        resnext_style_drops = {s: (resnext['acc_clean'] - acc) for s, acc in resnext_styles.items()}
        mobilenet_style_drops = {s: (mobilenet['acc_clean'] - acc) for s, acc in mobilenet_styles.items()}

        # Audit Decisions
        resnext_max_drop_style = max(resnext_style_drops, key=resnext_style_drops.get)
        c2_evidence = f"Max ResNeXt Drop on {resnext_max_drop_style} ({resnext_style_drops.get(resnext_max_drop_style, 0):.4f}). "
        c2_decision = "Supported" if resnext_max_drop_style in ['cartoons', 'sketches'] else "Refuted"

        mobilenet_max_drop_style = max(mobilenet_style_drops, key=mobilenet_style_drops.get)
        c4_evidence = f"ResNeXt Max: {resnext_max_drop_style}, MobileNet Max: {mobilenet_max_drop_style}"
        c4_decision = "Supported" if (resnext_max_drop_style == 'sculptures' and mobilenet_max_drop_style == 'sculptures') else "Refuted"

        claims = [
            {
                "ID": "C1",
                "Claim": f"{m1_name} has a smaller average Accuracy Drop on ImageNet-R than {m2_name}.",
                "Evidence": f"{m1_name} Drop: {resnext_drop:.4f}, {m2_name} Drop: {mobilenet_drop:.4f}",
                "Decision": "Supported" if resnext_drop < mobilenet_drop else "Refuted"
            },
            {
                "ID": "C2",
                "Claim": f"{m1_name}'s Accuracy Drop is largest on 'cartoon' and 'sketch' sub-categories.",
                "Evidence": c2_evidence,
                "Decision": c2_decision
            },
            {
                "ID": "C3",
                "Claim": f"Wrong Confidence is higher for {m2_name} than {m1_name}.",
                "Evidence": f"{m1_name} Wrong Conf: {avg_resnext_wrong_conf:.4f}, {m2_name} Wrong Conf: {avg_mobilenet_wrong_conf:.4f}",
                "Decision": "Supported" if avg_mobilenet_wrong_conf > avg_resnext_wrong_conf else "Refuted"
            },
            {
                "ID": "C4",
                "Claim": "Both models have their highest Accuracy Drop on 'sculpture' renditions.",
                "Evidence": c4_evidence,
                "Decision": c4_decision
            },
            {
                "ID": "C5",
                "Claim": f"The failure overlap between {m1_name} and {m2_name} is less than 50%.",
                "Evidence": f"Overlap Ratio: {overlap_ratio:.2%}",
                "Decision": "Supported" if overlap_ratio < 0.5 else "Refuted"
            }
        ]

        df_claims = pd.DataFrame(claims)
        df_claims.to_csv('results.csv', index=False)
        print("Claim Audit Table updated in results.csv")

    # ---------------------------------------------------------
    # Part 2: Detailed Style Analysis for ALL models
    # ---------------------------------------------------------
    summary_rows = []
    style_rows = []

    for model_name, model_data in metrics.items():
        acc_clean = model_data['acc_clean']
        results_shift = model_data['results_shift']

        df = pd.DataFrame(results_shift)
        
        # Overall summary for terminal
        acc_shift = model_data['acc_shift']
        acc_drop = acc_clean - acc_shift
        
        # Wrong Confidence
        wc = [r['confidence'] for r in results_shift if not r['is_correct']]
        avg_wc = sum(wc) / len(wc) if wc else 0
        
        style_stats = df.groupby('style').agg({
            'is_correct': ['sum', 'count'],
            'confidence': 'mean'
        })
        style_stats.columns = ['sum', 'count', 'avg_conf']
        style_stats['accuracy'] = style_stats['sum'] / style_stats['count']
        style_stats['accuracy_drop'] = acc_clean - style_stats['accuracy']
        
        print(f"\n{'='*25} {model_name} Style Analysis {'='*25}")
        print(f"  {'Style':<18} | {'Total':<6} | {'Error':<6} | {'Acc':<6} | {'Drop':<7} | {'Conf':<6}")
        print("-" * 75)
        
        sorted_metrics = sorted(style_stats.iterrows(), key=lambda x: x[1]['accuracy_drop'], reverse=True)
        max_drop_style = "unknown"
        if sorted_metrics:
            max_drop_style = sorted_metrics[0][0]

        for style, m in sorted_metrics:
            error_count = m['count'] - m['sum']
            print(f"  {style:.<18} | {int(m['count']):<6} | {int(error_count):<6} | {m['accuracy']:6.1%} | {m['accuracy_drop']:6.4f} | {m['avg_conf']:6.1%}")
            
            style_rows.append({
                'Model': model_name,
                'Style': style,
                'Error_Count': int(error_count),
                'Total_Count': int(m['count']),
                'Style_Accuracy': m['accuracy'],
                'Accuracy_Drop': m['accuracy_drop'],
                'Avg_Confidence': m['avg_conf']
            })

        summary_rows.append({
            "Metric": model_name,
            "Clean Accuracy": acc_clean,
            "Shifted Accuracy": acc_shift,
            "Accuracy Drop": acc_drop,
            "Avg Wrong Confidence": avg_wc,
            "Max Drop Style": max_drop_style
        })

    # Print Summary Table
    print("\n" + "="*20 + " Summary Results " + "="*20)
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    pd.DataFrame(style_rows).to_csv('results/style_error_analysis.csv', index=False)
    print("\nDetailed analysis saved to results/style_error_analysis.csv")

if __name__ == "__main__":
    analyze_all()
