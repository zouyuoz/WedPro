import json
from collections import Counter
import os

def load_metrics(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_class_index(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_misclassifications(metrics_data, class_index):
    output_data = {}
    
    for model_name, model_data in metrics_data.items():
        output_data[model_name] = {}
        
        # 用於暫存該 model 每個 style 的所有 prediction 標籤
        style_predictions = {}
        
        results_shift = model_data.get("results_shift", [])
        for entry in results_shift:
            # 僅統計錯誤分類的資料
            if not entry.get("is_correct", True):
                style = entry.get("style", "unknown")
                pred_idx = str(entry.get("prediction"))
                
                if pred_idx in class_index:
                    human_label = class_index[pred_idx][1]
                else:
                    human_label = f"unknown_class_{pred_idx}"
                
                if style not in style_predictions:
                    style_predictions[style] = []
                style_predictions[style].append(human_label)
        
        # 計算每個 style 下各個標籤的次數，並轉為字典結構
        for style, preds in style_predictions.items():
            counter = Counter(preds)
            # counter.most_common() 會依據次數由大到小排序
            output_data[model_name][style] = dict(counter.most_common())
            
    return output_data

if __name__ == "__main__":
    metrics_file = "results/metrics.json"
    output_file = "results/misclassify_analysis_result.json"
    class_index_file = "metadata/imagenet_class_index.json"

    try:
        metrics = load_metrics(metrics_file)
        class_index = load_class_index(class_index_file)

        # 進行分析並取得結構化字典
        result_json = analyze_misclassifications(metrics, class_index)

        # 儲存為 JSON 檔案
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=4, ensure_ascii=False)

        print(f"分析完成！結果已儲存至 {output_file}")

    except FileNotFoundError as e:
        print(f"Error: 找不到檔案, 原因: {e}")
    except Exception as e:
        print(f"發生錯誤: {e}")