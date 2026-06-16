import json
import os
from itertools import combinations

def calculate_jaccard(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def calculate_wpmr(overlap_indices, preds1, preds2):
    if not overlap_indices:
        return 0
    match_count = sum(1 for idx in overlap_indices if preds1[idx] == preds2[idx])
    return match_count / len(overlap_indices)

def main():
    metrics_path = 'results/metrics.json'
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found.")
        return

    with open(metrics_path, 'r') as f:
        data = json.load(f)

    models = list(data.keys())
    pairs = list(combinations(models, 2))

    output_results = {
        "Global_Overlap": {},
        "Style_Specific_Overlap": {}
    }

    print("=" * 60)
    print("Failure Overlap Metrics (Jaccard & WPMR)")
    print("=" * 60)

    # Pre-process data
    # We assume results_shift order is identical across models (which is true for streaming w/o shuffle)
    model_errors_global = {}
    model_preds_global = {}
    
    # Store errors per style
    # style -> model -> set of error indices
    style_errors = {}
    style_preds = {}

    for model in models:
        results = data[model]['results_shift']
        
        err_indices = set()
        preds = {}
        
        for i, r in enumerate(results):
            preds[i] = r['prediction']
            if not r['is_correct']:
                err_indices.add(i)
                
                style = r.get('style', 'unknown')
                if style not in style_errors:
                    style_errors[style] = {m: set() for m in models}
                    style_preds[style] = {m: {} for m in models}
                    
                style_errors[style][model].add(i)
                style_preds[style][model][i] = r['prediction']
                
        model_errors_global[model] = err_indices
        model_preds_global[model] = preds

    # Global Metrics
    print("\n--- Global Pairwise Overlap ---")
    for m1, m2 in pairs:
        set1 = model_errors_global[m1]
        set2 = model_errors_global[m2]
        
        jaccard = calculate_jaccard(set1, set2)
        overlap = set1.intersection(set2)
        wpmr = calculate_wpmr(overlap, model_preds_global[m1], model_preds_global[m2])
        
        pair_name = f"{m1} vs {m2}"
        output_results["Global_Overlap"][pair_name] = {
            "Jaccard_Similarity": jaccard,
            "Shared_Errors_Count": len(overlap),
            "Wrong_Prediction_Match_Rate": wpmr
        }
        
        print(f"[{pair_name}]")
        print(f"  Jaccard Similarity : {jaccard:.2%}")
        print(f"  Wrong Pred Match   : {wpmr:.2%} (Same wrong guess)")

    # Target specific pairs requested in Task.md
    print("\n--- Task.md Specific Hypotheses Check ---")
    vit_pair = "ViT-B16-V1 vs ViT-B16-SWAG"
    cross_pair = "ResNeXt-101 vs ViT-B16-V1"
    
    if vit_pair in output_results["Global_Overlap"]:
        j_vit = output_results["Global_Overlap"][vit_pair]["Jaccard_Similarity"]
        print(f"ViT Pair (Shared Arch, Diff Scale) Overlap: {j_vit:.2%}")
    if cross_pair in output_results["Global_Overlap"]:
        j_cross = output_results["Global_Overlap"][cross_pair]["Jaccard_Similarity"]
        print(f"Cross Pair (Diff Arch, Similar Scale) Overlap: {j_cross:.2%}")

    # Style Specific Metrics
    for style in style_errors.keys():
        output_results["Style_Specific_Overlap"][style] = {}
        for m1, m2 in pairs:
            set1 = style_errors[style][m1]
            set2 = style_errors[style][m2]
            jaccard = calculate_jaccard(set1, set2)
            overlap = set1.intersection(set2)
            wpmr = calculate_wpmr(overlap, style_preds[style][m1], style_preds[style][m2])
            
            output_results["Style_Specific_Overlap"][style][f"{m1} vs {m2}"] = {
                "Jaccard_Similarity": jaccard,
                "Shared_Errors_Count": len(overlap),
                "Wrong_Prediction_Match_Rate": wpmr
            }

    # Save to JSON
    os.makedirs('results', exist_ok=True)
    out_file = 'results/audit_h4_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output_results, f, indent=4)
        
    print(f"\nSaved detailed overlap matrix to {out_file}")

if __name__ == "__main__":
    main()
