import json
import pandas as pd
import os

def analyze():
    if not os.path.exists('results/metrics.json'):
        print("Error: results/metrics.json not found. Run evaluate.py first.")
        return

    with open('results/metrics.json', 'r') as f:
        metrics = json.load(f)

    vgg = metrics['VGG-19']
    vit = metrics['ViT']

    # 1. Accuracy Drop
    vgg_drop = vgg['acc_clean'] - vgg['acc_shift']
    vit_drop = vit['acc_clean'] - vit['acc_shift']

    # 2. Wrong Confidence
    vgg_wrong_conf = [r['confidence'] for r in vgg['results_shift'] if not r['is_correct']]
    vit_wrong_conf = [r['confidence'] for r in vit['results_shift'] if not r['is_correct']]
    avg_vgg_wrong_conf = sum(vgg_wrong_conf) / len(vgg_wrong_conf) if vgg_wrong_conf else 0
    avg_vit_wrong_conf = sum(vit_wrong_conf) / len(vit_wrong_conf) if vit_wrong_conf else 0

    # 3. Failure Overlap
    vgg_fails = [not r['is_correct'] for r in vgg['results_shift']]
    vit_fails = [not r['is_correct'] for r in vit['results_shift']]
    both_fail = sum([1 for f1, f2 in zip(vgg_fails, vit_fails) if f1 and f2])
    either_fail = sum([1 for f1, f2 in zip(vgg_fails, vit_fails) if f1 or f2])
    overlap_ratio = both_fail / either_fail if either_fail > 0 else 0

    # 4. Style-Specific Analysis (Fix for C2, C4)
    def get_style_acc(results):
        styles = {}
        for r in results:
            s = r.get('style', 'unknown')
            if s not in styles: styles[s] = {'correct': 0, 'total': 0}
            styles[s]['total'] += 1
            if r['is_correct']: styles[s]['correct'] += 1
        
        return {s: (v['correct']/v['total']) for s, v in styles.items()}

    vgg_styles = get_style_acc(vgg['results_shift'])
    vit_styles = get_style_acc(vit['results_shift'])
    
    # Heuristic for Clean Baseline per style (using global clean acc as proxy since clean has no styles)
    vgg_style_drops = {s: (vgg['acc_clean'] - acc) for s, acc in vgg_styles.items()}
    vit_style_drops = {s: (vit['acc_clean'] - acc) for s, acc in vit_styles.items()}

    # Audit Decisions
    # C2: VGG-19's Accuracy Drop is largest on 'cartoons' and 'sketches'
    vgg_max_drop_style = max(vgg_style_drops, key=vgg_style_drops.get)
    c2_evidence = f"Max VGG Drop on {vgg_max_drop_style} ({vgg_style_drops.get(vgg_max_drop_style, 0):.4f}). "
    c2_decision = "Supported" if vgg_max_drop_style in ['cartoons', 'sketches'] else "Refuted"

    # C4: Both models have their highest Accuracy Drop on 'sculptures'
    vit_max_drop_style = max(vit_style_drops, key=vit_style_drops.get)
    c4_evidence = f"VGG Max: {vgg_max_drop_style}, ViT Max: {vit_max_drop_style}"
    c4_decision = "Supported" if (vgg_max_drop_style == 'sculptures' and vit_max_drop_style == 'sculptures') else "Refuted"

    claims = [
        {
            "ID": "C1",
            "Claim": "ViT has a smaller average Accuracy Drop on ImageNet-R than VGG-19.",
            "Evidence": f"VGG Drop: {vgg_drop:.4f}, ViT Drop: {vit_drop:.4f}",
            "Decision": "Supported" if vit_drop < vgg_drop else "Refuted"
        },
        {
            "ID": "C2",
            "Claim": "VGG-19's Accuracy Drop is largest on 'cartoon' and 'sketch' sub-categories.",
            "Evidence": c2_evidence,
            "Decision": c2_decision
        },
        {
            "ID": "C3",
            "Claim": "Wrong Confidence is higher for VGG-19 than ViT.",
            "Evidence": f"VGG Wrong Conf: {avg_vgg_wrong_conf:.4f}, ViT Wrong Conf: {avg_vit_wrong_conf:.4f}",
            "Decision": "Supported" if avg_vgg_wrong_conf > avg_vit_wrong_conf else "Refuted"
        },
        {
            "ID": "C4",
            "Claim": "Both models have their highest Accuracy Drop on 'sculpture' renditions.",
            "Evidence": c4_evidence,
            "Decision": c4_decision
        },
        {
            "ID": "C5",
            "Claim": "The failure overlap between ViT and VGG-19 is less than 50%.",
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
        "VGG-19": [vgg['acc_clean'], vgg['acc_shift'], vgg_drop, avg_vgg_wrong_conf, vgg_max_drop_style],
        "ViT": [vit['acc_clean'], vit['acc_shift'], vit_drop, avg_vit_wrong_conf, vit_max_drop_style]
    }
    df_summary = pd.DataFrame(summary)
    print("\n--- Summary Results ---")
    print(df_summary)

if __name__ == "__main__":
    analyze()
