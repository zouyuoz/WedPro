import os
import random
import json
from datasets import load_dataset

def setup_data():
    # Set seed for reproducibility
    random.seed(42)
    
    # 1. Load ImageNet-1K WNIDs from local metadata
    print("Loading ImageNet-1K class mappings from local metadata...")
    if not os.path.exists('metadata/imagenet_class_index.json'):
        print("Error: metadata/imagenet_class_index.json not found.")
        return
        
    with open('metadata/imagenet_class_index.json', 'r') as f:
        class_index = json.load(f)
    wnids_1k = set(v[0] for v in class_index.values())
    
    # 2. Read WNIDs from wala.csv in order
    print("Reading class order from wala.csv...")
    if not os.path.exists('wala.csv'):
        print("Error: wala.csv not found.")
        return
    
    ordered_wnids = []
    with open('wala.csv', 'r') as f:
        lines = f.readlines()[1:] # Skip header
        for line in lines:
            wnid = line.split(',')[0].strip()
            if wnid in wnids_1k:
                ordered_wnids.append(wnid)
    
    # Select the top 40 valid classes from the ordered list
    selected_wnids = ordered_wnids[:40]
    
    print(f"Selected {len(selected_wnids)} classes based on wala.csv order.")
    
    os.makedirs('metadata', exist_ok=True)
    with open('metadata/selected_classes.json', 'w') as f:
        json.dump(selected_wnids, f)
        
    print("Metadata saved to metadata/selected_classes.json.")

if __name__ == "__main__":
    setup_data()
