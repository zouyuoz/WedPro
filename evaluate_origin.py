import os
import json
import torch
from datasets import load_dataset
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image
import io
from transformers import CLIPProcessor, CLIPModel

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_style_model(hf_token=None):
    print("Loading Style Classifier (CLIP)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", token=hf_token).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", token=hf_token)
    return model, processor

def get_style(model, processor, image, style_list):
    # Create prompts like "an art rendition", "a cartoons rendition", etc.
    prompts = [f"a {s.lower()} rendition" for s in style_list]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    conf, style_idx = torch.max(probs, dim=1)
    return style_list[style_idx.item()].lower(), conf.item()

def evaluate_model(model, dataset, preprocess, num_samples, desc, model_name, split_name):
    correct = 0
    total = 0
    confidences = []
    results = []

    os.makedirs('failure_cases', exist_ok=True)
    failure_count = -1

    pbar = tqdm(total=num_samples, desc=desc)

    try:
        # Iterate over the already filtered and limited dataset
        for item in dataset:
            img = item['image']
            if not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img))

            img_rgb = img.convert('RGB')
            target = item['label']
            wnid = item.get('wnid', 'unknown')
            style = item.get('style', 'unknown')
            style_conf = item.get('style_confidence', 0.0)

            # Preprocess
            input_tensor = preprocess(img_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                conf, pred = torch.max(probabilities, dim=0)

            is_correct = (pred.item() == target)
            if is_correct:
                correct += 1
            else:
                # img_path = f"failure_cases/{model_name[:3]}_{split_name}_{failure_count}.png"
                # img_rgb.save(img_path)
                failure_count += 1

            total += 1
            confidences.append(conf.item())

            results.append({
                'is_correct': is_correct,
                'confidence': conf.item(),
                'prediction': pred.item(),
                'target': target,
                'wnid': wnid,
                'style': style,
                'style_confidence': style_conf,
                'failure_count': (-1 if is_correct else failure_count)
            })

            pbar.update(1)
    finally:
        pbar.close()

    accuracy = correct / total if total > 0 else 0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    return accuracy, avg_conf, results

def get_imagenet_mappings():
    with open('metadata/imagenet_class_index.json', 'r') as f:
        class_index = json.load(f)
    wnid_to_idx = {v[0]: int(k) for k, v in class_index.items()}
    return wnid_to_idx

def load_models():
    print("Loading ResNeXt-101 (IMAGENET1K_V1)...")
    resnext_weights = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
    resnext = models.resnext101_32x8d(weights=resnext_weights).to(device)
    resnext.eval()

    print("Loading MobileNet-V3 Large (IMAGENET1K_V1)...")
    mobilenet_weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    mobilenet = models.mobilenet_v3_large(weights=mobilenet_weights).to(device)
    mobilenet.eval()

    print("Loading ViT-B/16 (IMAGENET1K_V1)...")
    vit_v1_weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    vit_v1 = models.vit_b_16(weights=vit_v1_weights).to(device)
    vit_v1.eval()

    print("Loading ViT-B/16 (IMAGENET1K_SWAG_E2E_V1)...")
    vit_swag_weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    vit_swag = models.vit_b_16(weights=vit_swag_weights).to(device)
    vit_swag.eval()

    model_list = [
        ("ResNeXt-101", resnext, resnext_weights.transforms(), 0.7931),
        ("MobileNet-V3", mobilenet, mobilenet_weights.transforms(), 0.7404),
        ("ViT-B16-V1", vit_v1, vit_v1_weights.transforms(), 0.8107),
        ("ViT-B16-SWAG", vit_swag, vit_swag_weights.transforms(), 0.8530)
    ]
    return model_list

def main():
    # Load selected classes and renditions
    with open('metadata/selected_classes.json', 'r') as f:
        selected_wnids = json.load(f)

    with open('metadata/renditions.json', 'r') as f:
        style_list = json.load(f)

    # Load HF Token if exists
    hf_token = None
    if os.path.exists('HF_TOKEN.env'):
        with open('HF_TOKEN.env', 'r') as f:
            line = f.read().strip()
            if '=' in line:
                hf_token = line.split('=')[1]
            else:
                hf_token = line
        print("Found HF_TOKEN: ", hf_token)

    wnid_to_idx = get_imagenet_mappings()
    selected_indices = set([wnid_to_idx[wnid] for wnid in selected_wnids])
    idx_to_wnid = {idx: wnid for wnid, idx in wnid_to_idx.items()}

    model_list = load_models()
    style_model, style_processor = load_style_model(hf_token=hf_token)

    # 1. Prepare Datasets (Clean evaluation removed as per user request)
    print("Preparing Shifted Dataset (ImageNet-R)...")
    ds_shifted = load_dataset('axiong/imagenet-r', split='test', streaming=True, token=hf_token)

    def filter_shifted(item):
        return item['wnid'] in selected_wnids

    def process_shifted(item):
        img = item['image']
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img))

        style, style_conf = get_style(style_model, style_processor, img.convert('RGB'), style_list)
        return {
            'image': item['image'],
            'label': wnid_to_idx[item['wnid']],
            'wnid': item['wnid'],
            'style': style,
            'style_confidence': style_conf
        }

    # Filter and process shifted
    ds_shifted_final_base = ds_shifted.filter(filter_shifted).map(process_shifted)

    # Samples per class
    SHIFTED_SAMPLES_PER_CLASS = 100

    NUM_SHIFTED = len(selected_indices) * SHIFTED_SAMPLES_PER_CLASS

    all_metrics = {}

    for model_name, model, preprocess, theoretical_clean_acc in model_list:
        print(f"\n--- Evaluating {model_name} ---")

        # We re-apply take() on the filtered/mapped stream for each model
        model_ds_shifted = ds_shifted_final_base.take(NUM_SHIFTED)

        acc_shift, conf_shift, res_shift = evaluate_model(model, model_ds_shifted, preprocess, NUM_SHIFTED, f"{model_name} Shifted", model_name, "shifted")

        all_metrics[model_name] = {
            'acc_clean': theoretical_clean_acc, # Using theoretical metadata accuracy
            'acc_shift': acc_shift,
            'conf_clean': 0.0, # Removed clean evaluation
            'conf_shift': conf_shift,
            'results_clean': [], # Removed clean evaluation
            'results_shift': res_shift
        }

        print(f"{model_name} Results:")
        print(f"* Clean Acc (Theoretical): {theoretical_clean_acc:.4f}")
        print(f"* Shift Acc: {acc_shift:.4f}")
        print(f"* Acc Drop:  {theoretical_clean_acc - acc_shift:.4f}")

    # Save metrics for analysis script
    os.makedirs('results', exist_ok=True)
    with open('results/metrics.json', 'w') as f:
        # We don't save the full results list to JSON as it might be large,
        # but for 2400 items it's fine.
        json.dump(all_metrics, f)

    print("\nEvaluation complete. Metrics saved to results/metrics.json")

    # Use os._exit to bypass background thread cleanup crashes at finalization
    os._exit(0)

if __name__ == "__main__":
    main()

# 你可以開始更新整個 report.py 了，關於每個問題(AI claim + 支持/反對)，都要有完整的