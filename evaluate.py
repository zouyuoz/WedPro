import os
import json
import torch
import requests
from datasets import load_dataset
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image
import io
# ResNeXt101_32X8D_Weights, MobileNet_V3_Large_Weights
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
    style_idx = probs.argmax().item()
    return style_list[style_idx].lower()

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
                'failure_count': (-1 if not is_correct else failure_count)
            })

            pbar.update(1)
    finally:
        pbar.close()

    accuracy = correct / total if total > 0 else 0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    return accuracy, avg_conf, results

def get_imagenet_mappings():
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    response = requests.get(url)
    response.raise_for_status()
    class_index = response.json()
    wnid_to_idx = {v[0]: int(k) for k, v in class_index.items()}
    return wnid_to_idx

def load_models():
    print("Loading ResNeXt-101 (IMAGENET1K_V2)...")
    resnext = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2).to(device)
    resnext.eval()

    print("Loading MobileNet-V3 Large (IMAGENET1K_V2)...")
    mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2).to(device)
    mobilenet.eval()

    # Common transform for IMAGENET1K_V2 (Resize 232 then CenterCrop 224)
    preprocess = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return resnext, mobilenet, preprocess

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

    resnext101, mobilenet_v3, preprocess = load_models()
    style_model, style_processor = load_style_model(hf_token=hf_token)

    # 1. Prepare Datasets
    print("Preparing Clean Dataset (ImageNet-V2)...")
    ds_clean = load_dataset('vaishaal/ImageNetV2', split='train', streaming=True, token=hf_token)

    def filter_clean(item):
        try:
            idx = int(item['__key__'].split('/')[1])
            return idx in selected_indices
        except:
            return False

    def process_clean(item):
        idx = int(item['__key__'].split('/')[1])
        return {
            'image': item['jpeg'],
            'label': idx,
            'wnid': idx_to_wnid[idx],
            'style': 'clean'
        }

    # Filter and process clean
    ds_clean_final_base = ds_clean.filter(filter_clean).map(process_clean)

    print("Preparing Shifted Dataset (ImageNet-R)...")
    ds_shifted = load_dataset('axiong/imagenet-r', split='test', streaming=True, token=hf_token)

    def filter_shifted(item):
        return item['wnid'] in selected_wnids

    def process_shifted(item):
        img = item['image']
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img))

        style = get_style(style_model, style_processor, img.convert('RGB'), style_list)
        return {
            'image': item['image'],
            'label': wnid_to_idx[item['wnid']],
            'wnid': item['wnid'],
            'style': style
        }

    # Filter and process shifted
    ds_shifted_final_base = ds_shifted.filter(filter_shifted).map(process_shifted)

    # Samples per class
    CLEAN_SAMPLES_PER_CLASS = 30 # maximum
    SHIFTED_SAMPLES_PER_CLASS = 200 # all selected classes of ImageNet-R has 200+ samples

    NUM_CLEAN = len(selected_indices) * CLEAN_SAMPLES_PER_CLASS
    NUM_SHIFTED = len(selected_indices) * SHIFTED_SAMPLES_PER_CLASS

    all_metrics = {}

    for model_name, model in [("ResNeXt-101", resnext101), ("MobileNet-V3", mobilenet_v3)]:
        print(f"\n--- Evaluating {model_name} ---")

        # We re-apply take() on the filtered/mapped stream for each model
        model_ds_clean = ds_clean_final_base.take(NUM_CLEAN)
        model_ds_shifted = ds_shifted_final_base.take(NUM_SHIFTED)

        acc_clean, conf_clean, res_clean = evaluate_model(model, model_ds_clean, preprocess, NUM_CLEAN, f"{model_name} Clean", model_name, "clean")
        acc_shift, conf_shift, res_shift = evaluate_model(model, model_ds_shifted, preprocess, NUM_SHIFTED, f"{model_name} Shifted", model_name, "shifted")

        all_metrics[model_name] = {
            'acc_clean': acc_clean,
            'acc_shift': acc_shift,
            'conf_clean': conf_clean,
            'conf_shift': conf_shift,
            'results_clean': res_clean,
            'results_shift': res_shift
        }

        print(f"{model_name} Results:")
        print(f"* Clean Acc: {acc_clean:.4f}")
        print(f"* Shift Acc: {acc_shift:.4f}")
        print(f"* Acc Drop:  {acc_clean - acc_shift:.4f}")

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
