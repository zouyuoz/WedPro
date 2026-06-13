import os
import random
import json
import requests
from datasets import load_dataset

def setup_data():
    # Set seed for reproducibility
    random.seed(42)
    
    # Load HF Token if exists
    hf_token = None
    if os.path.exists('HF_TOKEN.env'):
        with open('HF_TOKEN.env', 'r') as f:
            line = f.read().strip()
            if '=' in line:
                hf_token = line.split('=')[1]
            else:
                hf_token = line
    
    # 1. Fetch ImageNet-1K WNIDs
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    print("Fetching ImageNet-1K class mappings...")
    class_index = requests.get(url).json()
    wnids_1k = set(v[0] for v in class_index.values())
    
    # 2. Load ImageNet-R to find common WNIDs
    print("Loading ImageNet-R from Hugging Face...")
    ds_r = load_dataset("axiong/imagenet-r", split="test", streaming=True, token=hf_token)
    
    r_wnids = set()
    print("Identifying common classes between ImageNet-R and ImageNet-1K...")
    for item in ds_r:
        r_wnids.add(item['wnid'])
        if len(r_wnids) >= 200: # ImageNet-R has 200 classes
            break
            
    selected_wnids = sorted(list(r_wnids.intersection(wnids_1k)))[:20]
    
    print(f"Selected 20 common classes: {selected_wnids}")
    
    os.makedirs('metadata', exist_ok=True)
    with open('metadata/selected_classes.json', 'w') as f:
        json.dump(selected_wnids, f)
        
    print("Metadata saved to metadata/selected_classes.json.")

if __name__ == "__main__":
    setup_data()
