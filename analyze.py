import json
import pandas as pd
import os

def analyze():
    if not os.path.exists('results/metrics.json'):
        print("Error: results/metrics.json not found. Run evaluate.py first.")
        return

    with open('results/metrics.json', 'r') as f:
        metrics = json.load(f)

    resnext = metrics['ResNeXt-101']
    mobilenet = metrics['MobileNet-V3']

    # 1. Accuracy Drop
    resnext_drop = resnext['acc_clean'] - resnext['acc_shift']
    mobilenet_drop = mobilenet['acc_clean'] - mobilenet['acc_shift']

    # 2. Relative Drop
    resnext_rel_drop = resnext_drop / resnext['acc_clean'] if resnext['acc_clean'] > 0 else 0
    mobilenet_rel_drop = mobilenet_drop / mobilenet['acc_clean'] if mobilenet['acc_clean'] > 0 else 0

    # 3. Average Robust Accuracy (over styles)
    # 4. Worst-condition Accuracy (min over styles)
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
    
    resnext_avg_robust = sum(resnext_styles.values()) / len(resnext_styles) if resnext_styles else 0
    mobilenet_avg_robust = sum(mobilenet_styles.values()) / len(mobilenet_styles) if mobilenet_styles else 0
    
    resnext_worst_acc = min(resnext_styles.values()) if resnext_styles else 0
    mobilenet_worst_acc = min(mobilenet_styles.values()) if mobilenet_styles else 0

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

    # 7. Rejection Robustness (Accuracy at 80% coverage - rejecting 20% least confident)
    def calc_rejection_acc(results, coverage=0.8):
        sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        keep_count = int(len(sorted_results) * coverage)
        top_results = sorted_results[:keep_count]
        correct = sum([1 for r in top_results if r['is_correct']])
        return correct / keep_count if keep_count > 0 else 0

    resnext_rej_robust = calc_rejection_acc(resnext['results_shift'])
    mobilenet_rej_robust = calc_rejection_acc(mobilenet['results_shift'])

    # Style-Specific Analysis (Fix for C2, C4)
    resnext_style_drops = {s: (resnext['acc_clean'] - acc) for s, acc in resnext_styles.items()}
    mobilenet_style_drops = {s: (mobilenet['acc_clean'] - acc) for s, acc in mobilenet_styles.items()}

    # Audit Decisions
    # C2: ResNeXt-101's Accuracy Drop is largest on 'cartoons' and 'sketches'
    resnext_max_drop_style = max(resnext_style_drops, key=resnext_style_drops.get)
    c2_evidence = f"Max ResNeXt Drop on {resnext_max_drop_style} ({resnext_style_drops.get(resnext_max_drop_style, 0):.4f}). "
    c2_decision = "Supported" if resnext_max_drop_style in ['cartoons', 'sketches'] else "Refuted"

    # C4: Both models have their highest Accuracy Drop on 'sculptures'
    mobilenet_max_drop_style = max(mobilenet_style_drops, key=mobilenet_style_drops.get)
    c4_evidence = f"ResNeXt Max: {resnext_max_drop_style}, MobileNet Max: {mobilenet_max_drop_style}"
    c4_decision = "Supported" if (resnext_max_drop_style == 'sculptures' and mobilenet_max_drop_style == 'sculptures') else "Refuted"

    claims = [
        {
            "ID": "C1",
            "Claim": "ResNeXt-101 has a smaller average Accuracy Drop on ImageNet-R than MobileNet-V3.",
            "Evidence": f"ResNeXt Drop: {resnext_drop:.4f}, MobileNet Drop: {mobilenet_drop:.4f}",
            "Decision": "Supported" if resnext_drop < mobilenet_drop else "Refuted"
        },
        {
            "ID": "C2",
            "Claim": "ResNeXt-101's Accuracy Drop is largest on 'cartoon' and 'sketch' sub-categories.",
            "Evidence": c2_evidence,
            "Decision": c2_decision
        },
        {
            "ID": "C3",
            "Claim": "Wrong Confidence is higher for MobileNet-V3 than ResNeXt-101.",
            "Evidence": f"ResNeXt Wrong Conf: {avg_resnext_wrong_conf:.4f}, MobileNet Wrong Conf: {avg_mobilenet_wrong_conf:.4f}",
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
            "Claim": "The failure overlap between ResNeXt-101 and MobileNet-V3 is less than 50%.",
            "Evidence": f"Overlap Ratio: {overlap_ratio:.2%}",
            "Decision": "Supported" if overlap_ratio < 0.5 else "Refuted"
        }
    ]

    # Save to CSV
    df_claims = pd.DataFrame(claims)
    df_claims.to_csv('results.csv', index=False)
    print("Claim Audit Table updated in results.csv")

    # Final Summary Table
    summary = {
        "Metric": ["Clean Accuracy", "Shifted Accuracy", "Accuracy Drop", "Avg Wrong Confidence", "Max Drop Style"],
        "ResNeXt-101": [resnext['acc_clean'], resnext['acc_shift'], resnext_drop, avg_resnext_wrong_conf, resnext_max_drop_style],
        "MobileNet-V3": [mobilenet['acc_clean'], mobilenet['acc_shift'], mobilenet_drop, avg_mobilenet_wrong_conf, mobilenet_max_drop_style]
    }
    df_summary = pd.DataFrame(summary)
    print("\n--- Summary Results ---")
    print(df_summary)

if __name__ == "__main__":
    analyze()
