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
    pbar = tqdm(total=num_samples, desc=desc)

    try:
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

            total += 1
            confidences.append(conf.item())

            results.append({
                'is_correct': is_correct,
                'confidence': conf.item(),
                'prediction': pred.item(),
                'target': target,
                'wnid': wnid,
                'style': style,
                'style_confidence': style_conf
            })

            pbar.update(1)
    finally:
        pbar.close()

    accuracy = correct / total if total > 0 else 0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    return accuracy, avg_conf, results

def evaluate_model_tta(model, dataset, preprocess_list, num_samples, desc, model_name, split_name):
    correct = 0
    total = 0
    confidences = []
    results = []

    os.makedirs('failure_cases', exist_ok=True)
    pbar = tqdm(total=num_samples, desc=desc)

    try:
        for item in dataset:
            img = item['image']
            if not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img))
            img_rgb = img.convert('RGB')
            target = item['label']
            wnid = item.get('wnid', 'unknown')
            style = item.get('style', 'unknown')
            style_conf = item.get('style_confidence', 0.0)

            # TTA: Run multiple transforms and average probabilities
            probs_accum = None
            with torch.no_grad():
                for p in preprocess_list:
                    input_tensor = p(img_rgb).unsqueeze(0).to(device)
                    output = model(input_tensor)
                    probs = torch.nn.functional.softmax(output[0], dim=0)
                    if probs_accum is None:
                        probs_accum = probs
                    else:
                        probs_accum += probs

            avg_probs = probs_accum / len(preprocess_list)
            conf, pred = torch.max(avg_probs, dim=0)

            is_correct = (pred.item() == target)
            if is_correct:
                correct += 1

            total += 1
            confidences.append(conf.item())
            results.append({
                'is_correct': is_correct,
                'confidence': conf.item(),
                'prediction': pred.item(),
                'target': target,
                'wnid': wnid,
                'style': style,
                'style_confidence': style_conf
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
    print("Loading ResNeXt-101 (IMAGENET1K_V1) for TTA...")
    resnext_weights = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
    resnext = models.resnext101_32x8d(weights=resnext_weights).to(device)
    resnext.eval()

    print("Loading ViT-B/16 (IMAGENET1K_SWAG_E2E_V1) for TTA...")
    vit_swag_weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    vit_swag = models.vit_b_16(weights=vit_swag_weights).to(device)
    vit_swag.eval()

    model_list = [
        ("ResNeXt-101", resnext, resnext_weights.transforms(), 0.7931),
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

    # 1. Prepare Dataset (Shifted only)
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

    ds_shifted_final_base = ds_shifted.filter(filter_shifted).map(process_shifted)

    # Samples per class
    SHIFTED_SAMPLES_PER_CLASS = 100
    NUM_SHIFTED = len(selected_indices) * SHIFTED_SAMPLES_PER_CLASS

    # Load existing metrics to append to
    metrics_path = 'results/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            all_metrics = json.load(f)
        print(f"Loaded existing metrics from {metrics_path}. New TTA results will be appended.")
    else:
        all_metrics = {}
        print(f"Warning: {metrics_path} not found. Starting with empty metrics.")

    # TTA Evaluation
    for model_name, model, base_preprocess, theoretical_clean_acc in model_list:
        tta_name = f"{model_name}-TTA"
        print(f"\n--- Evaluating {tta_name} (Multi-view TTA) ---")
        
        # Detect required size from base_preprocess
        crop_size = 224
        if hasattr(base_preprocess, 'crop_size'):
            crop_size = base_preprocess.crop_size[0] if isinstance(base_preprocess.crop_size, (list, tuple)) else base_preprocess.crop_size
        
        scale_size = int(crop_size * 1.14)
        
        preprocess_list = [
            base_preprocess,
            transforms.Compose([transforms.Resize(scale_size), transforms.CenterCrop(crop_size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            transforms.Compose([transforms.Resize(crop_size), transforms.CenterCrop(crop_size), transforms.GaussianBlur(kernel_size=5, sigma=0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            transforms.Compose([transforms.Resize(crop_size), transforms.CenterCrop(crop_size), transforms.ColorJitter(brightness=0.1, contrast=0.1), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        ]
        
        model_ds_shifted = ds_shifted_final_base.take(NUM_SHIFTED)
        acc_shift, conf_shift, res_shift = evaluate_model_tta(model, model_ds_shifted, preprocess_list, NUM_SHIFTED, f"{tta_name} Shifted", tta_name, "shifted")

        all_metrics[tta_name] = {
            'acc_clean': theoretical_clean_acc,
            'acc_shift': acc_shift,
            'conf_clean': 0.0,
            'conf_shift': conf_shift,
            'results_clean': [],
            'results_shift': res_shift
        }

        print(f"{tta_name} Results:")
        print(f"* Clean Acc (Theoretical): {theoretical_clean_acc:.4f}")
        print(f"* Shift Acc (TTA): {acc_shift:.4f}")
        print(f"* Acc Drop:  {theoretical_clean_acc - acc_shift:.4f}")

    # Save consolidated metrics
    os.makedirs('results', exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f)

    print(f"\nEvaluation complete. Consolidated metrics saved to {metrics_path}")
    os._exit(0)

if __name__ == "__main__":
    main()

# source .venv/bin/activate && python evaluate.py && python style_analysis.py && python misclassification_analysis.py && python generate_report.py && rm report.html