import json
import os
from pathlib import Path

humorous_neurons = [29951, 17035, 42246, 43522, 33925         
] # humorous, funny, hilarious, amusing
creativity_neurons = [9869, 21227, 17035, 63055]
#creative, inventive, imaginary, imaginative, innovative, original, 
violence_neurons = [9727, 21273, 58683, 3535, 10583, 36377, 30075]
# violent, brutal, savage, aggressive, hostile, cruel
toxicity_neurons = [21637, 43242,38814, 10395, ]
# toxic, abusive, malicious, 

neurons = toxicity_neurons

ground_truth_dir = Path("oasst1_toxicity")
ground_truth_name = "high_toxicity_0.790"
ground_truth_files = list(ground_truth_dir.glob(f"{ground_truth_name}*"))


# 获取oasst1_all_emb目录下所有文件
emb_dir = Path("oasst1_all_emb")
json_files = list(emb_dir.glob("*.json"))

# 存储包含humorous neurons的index
matching_indices = []
total_samples = 0
matched_samples = 0
global_index = 0  # 添加全局索引计数器

# 遍历所有文件
for json_file in json_files:
    # print(f"Processing {json_file.name}...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查每个样本的embeddings是否包含humorous neurons
    for i, sample in enumerate(data):
        total_samples += 1
        # 每个sample是一个包含neuron索引的数组
        if isinstance(sample, list):
            # 检查数组中的任何neuron是否在humorous_neurons中
            if any(neuron in neurons for neuron in sample):
                matching_indices.append(global_index)  # 使用全局索引
                matched_samples += 1
        global_index += 1  # 增加全局索引

print(f"Total samples processed: {total_samples}")
print(f"Matched samples: {matched_samples}")
print(f"Match rate: {matched_samples/total_samples*100:.2f}%")

# 保存结果到JSON文件
output_file = "humorous_neurons_matches.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(matching_indices, f, ensure_ascii=False, indent=2)

print(f"Found {len(matching_indices)} samples containing humorous neurons")
print(f"Results saved to {output_file}")

# 加载ground truth数据
ground_truth_indices = set()
for ground_truth_file in ground_truth_files:
    #print(f"Loading ground truth from {ground_truth_file.name}...")
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 数据是列表格式，每个元素是包含index字段的字典
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'index' in item:
                    ground_truth_indices.add(item['index'])
        # 如果数据是字典格式，需要根据实际结构调整
        elif isinstance(data, dict):
            # 根据实际数据结构调整
            if 'indices' in data:
                ground_truth_indices.update(data['indices'])
            else:
                # 如果是其他格式，打印前几个元素来了解结构
                print(f"Unexpected data format in {ground_truth_file.name}, first few items: {list(data.items())[:3]}")

print(f"Ground truth indices loaded: {len(ground_truth_indices)}")

# 将我们的预测结果转换为集合
predicted_indices = set(matching_indices)

# 计算评估指标
true_positives = len(ground_truth_indices.intersection(predicted_indices))
false_positives = len(predicted_indices - ground_truth_indices)
false_negatives = len(ground_truth_indices - predicted_indices)

# 计算准确率、召回率和F1值
precision = true_positives / len(predicted_indices) if len(predicted_indices) > 0 else 0
recall = true_positives / len(ground_truth_indices) if len(ground_truth_indices) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n=== Evaluation Results ===")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# 保存评估结果
evaluation_results = {
    "true_positives": true_positives,
    "false_positives": false_positives,
    "false_negatives": false_negatives,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score,
    "ground_truth_count": len(ground_truth_indices),
    "predicted_count": len(predicted_indices)
}

# 确保evaluation_results目录存在
evaluation_dir = Path("evaluation_results")
evaluation_dir.mkdir(exist_ok=True)

with open(evaluation_dir / f"{ground_truth_name}.json", 'w', encoding='utf-8') as f:
    json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

print(f"Evaluation results saved to evaluation_results/{ground_truth_name}.json")


