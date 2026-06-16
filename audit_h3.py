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

    # We need models that were requested for ensemble
    model1_name = "ResNeXt-101"
    model2_name = "ViT-B16-SWAG"

    if model1_name not in data or model2_name not in data:
        print(f"Error: Required models '{model1_name}' or '{model2_name}' not found in metrics.")
        return

    print("=" * 60)
    print("Non-Training Remediation Simulator")
    print("=" * 60)

    results_m1 = data[model1_name]['results_shift']
    results_m2 = data[model2_name]['results_shift']

    # Baseline Accuracies
    acc_m1 = data[model1_name]['acc_shift']
    acc_m2 = data[model2_name]['acc_shift']

    print(f"Baseline {model1_name} Accuracy: {acc_m1:.2%}")
    print(f"Baseline {model2_name} Accuracy: {acc_m2:.2%}")

    # Ensure alignment (they should be identical in order if streaming was sequential)
    num_samples = len(results_m1)
    if num_samples != len(results_m2):
        print("Warning: Sample counts do not match between models.")
        num_samples = min(len(results_m1), len(results_m2))

    # --- Strategy 1: Simulate Ensemble (using confidence as a proxy for logits since we didn't save full logits) ---
    # Note: A true logit ensemble requires the full raw logits array. Since our metrics.json 
    # only saves the top prediction and its confidence, we will simulate an "Agreement / Confidence-weighted" 
    # ensemble proxy: If they agree, use it. If they disagree, trust the one with higher confidence.
    
    ensemble_correct = 0
    ensemble_results = []
    
    for i in range(num_samples):
        r1 = results_m1[i]
        r2 = results_m2[i]
        target = r1['target']
        
        # Simulated Ensemble Logic
        if r1['prediction'] == r2['prediction']:
            final_pred = r1['prediction']
            final_conf = (r1['confidence'] + r2['confidence']) / 2.0
        else:
            if r1['confidence'] > r2['confidence']:
                final_pred = r1['prediction']
                final_conf = r1['confidence']
            else:
                final_pred = r2['prediction']
                final_conf = r2['confidence']
                
        is_correct = (final_pred == target)
        if is_correct:
            ensemble_correct += 1
            
        ensemble_results.append({
            'confidence': final_conf,
            'is_correct': is_correct,
            'agreed': (r1['prediction'] == r2['prediction'])
        })

    acc_ensemble = ensemble_correct / num_samples
    print(f"\n[Strategy 1] Simulated Confidence-Weighted Ensemble")
    print(f"  Accuracy: {acc_ensemble:.2%} (Gain: {acc_ensemble - max(acc_m1, acc_m2):+.2%})")

    # --- Strategy 2: Model Agreement Check (Rejection) ---
    # Reject samples where the two models disagree. Calculate accuracy on the remaining accepted subset.
    agree_correct = sum(1 for r in ensemble_results if r['agreed'] and r['is_correct'])
    agree_total = sum(1 for r in ensemble_results if r['agreed'])
    
    if agree_total > 0:
        acc_agreement = agree_correct / agree_total
        coverage = agree_total / num_samples
        print(f"\n[Strategy 2] Agreement Rejection (Trust only when M1 & M2 agree)")
        print(f"  Coverage: {coverage:.2%} (Rejected {1-coverage:.2%})")
        print(f"  Accuracy on Accepted: {acc_agreement:.2%} (Gain over SWAG: {acc_agreement - acc_m2:+.2%})")
    else:
        acc_agreement = 0
        coverage = 0

    # --- Strategy 3: Confidence Rejection (Single Best Model - SWAG) ---
    # Reject the bottom 20% of samples based on confidence
    rejection_threshold = 0.8
    sorted_m2 = sorted(results_m2, key=lambda x: x['confidence'], reverse=True)
    keep_count = int(len(sorted_m2) * rejection_threshold)
    top_results = sorted_m2[:keep_count]
    acc_conf_reject = sum(1 for r in top_results if r['is_correct']) / keep_count
    
    print(f"\n[Strategy 3] Confidence Rejection ({model2_name} Only)")
    print(f"  Coverage: {rejection_threshold:.2%} (Rejected {1-rejection_threshold:.2%})")
    print(f"  Accuracy on Accepted: {acc_conf_reject:.2%} (Gain over SWAG baseline: {acc_conf_reject - acc_m2:+.2%})")


    # Save results
    output_data = [
        {
            "Strategy": "Baseline (ResNeXt-101)",
            "Coverage": 1.0,
            "Accuracy": acc_m1,
            "Gain_vs_Best_Baseline": acc_m1 - acc_m2
        },
        {
            "Strategy": "Baseline (ViT-B16-SWAG)",
            "Coverage": 1.0,
            "Accuracy": acc_m2,
            "Gain_vs_Best_Baseline": 0.0
        },
        {
            "Strategy": "Ensemble (Confidence-Weighted)",
            "Coverage": 1.0,
            "Accuracy": acc_ensemble,
            "Gain_vs_Best_Baseline": acc_ensemble - acc_m2
        },
        {
            "Strategy": "Agreement Rejection",
            "Coverage": coverage,
            "Accuracy": acc_agreement,
            "Gain_vs_Best_Baseline": acc_agreement - acc_m2
        },
        {
            "Strategy": f"Confidence Rejection ({rejection_threshold:.0%})",
            "Coverage": rejection_threshold,
            "Accuracy": acc_conf_reject,
            "Gain_vs_Best_Baseline": acc_conf_reject - acc_m2
        }
    ]

    os.makedirs('results', exist_ok=True)
    df_out = pd.DataFrame(output_data)
    csv_path = 'results/audit_h3_results.csv'
    df_out.to_csv(csv_path, index=False)
    print(f"\nRemediation metrics saved to {csv_path}")

if __name__ == "__main__":
    main()
