import json
import os
from pathlib import Path

# 定义不同类型的neurons
humorous_neurons = [29951, 17035, 42246, 43522, 33925]  # humorous, funny, hilarious, amusing
creativity_neurons = [9869, 21227, 17035, 63055]  # creative, inventive, imaginary, imaginative, innovative, original
violence_neurons = [9727, 21273, 58683, 3535, 10583, 36377, 30075]  # violent, brutal, savage, aggressive, hostile, cruel
toxicity_neurons = [21637, 43242, 38814, 10395]  # toxic, abusive, malicious

# 选择要使用的neurons类型
neuron_type = "toxicity"
neurons = toxicity_neurons


# 获取output目录下所有embeddings文件
output_dir = Path("output")
json_files = list(output_dir.glob("*_embs_chunk_*.json"))
json_files.sort()  # 确保文件按顺序处理

# 存储所有样本的得分和索引
sample_scores = []
total_samples = 0
global_index = 0

print(f"Processing {len(json_files)} files...")

# 遍历所有文件
for json_file in json_files:
    print(f"Processing {json_file.name}...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查每个样本的embeddings
    for i, sample in enumerate(data):
        total_samples += 1
        
        # 计算该样本在目标neurons上的总得分
        total_score = 0.0
        neuron_count = 0
        
        # 每个sample现在是一个字典，包含neuron_id: score
        if isinstance(sample, dict):
            for neuron_id, score in sample.items():
                neuron_id = int(neuron_id)  # 确保neuron_id是整数
                if neuron_id in neurons:
                    total_score += score
                    neuron_count += 1
        
        # 计算平均得分（避免样本间neuron数量不同导致的偏差）
        avg_score = total_score / len(neurons) if len(neurons) > 0 else 0.0
        
        # 存储样本信息
        sample_info = {
            'global_index': global_index,
            'file_index': i,
            'file_name': json_file.name,
            'total_score': total_score,
            'avg_score': avg_score,
            'neuron_count': neuron_count,
            'target_neurons': len(neurons)
        }
        
        sample_scores.append(sample_info)
        global_index += 1

print(f"Total samples processed: {total_samples}")

# 按平均得分排序（从高到低）
sample_scores.sort(key=lambda x: x['avg_score'], reverse=True)

# 分段保存排序结果，每8000个样本一个chunk
chunk_size = 8000
total_chunks = (len(sample_scores) + chunk_size - 1) // chunk_size

for chunk_idx in range(total_chunks):
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, len(sample_scores))
    chunk_data = sample_scores[start_idx:end_idx]
    
    chunk_file = f"oasst1_ranking/{neuron_type}/{neuron_type}_neurons_ranking_chunk_{chunk_idx:03d}.json"
    with open(chunk_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved chunk {chunk_idx + 1}/{total_chunks}: {len(chunk_data)} samples to {chunk_file}")


# 显示前10个样本的得分
print(f"\n=== Top 10 samples by {neuron_type} neuron scores ===")
for i, sample in enumerate(sample_scores[:10]):
    print(f"Rank {i+1}: Global Index {sample['global_index']}, "
          f"File: {sample['file_name']}, "
          f"Avg Score: {sample['avg_score']:.6f}, "
          f"Total Score: {sample['total_score']:.6f}, "
          f"Neurons Found: {sample['neuron_count']}/{sample['target_neurons']}")

# 显示后10个样本的得分
print(f"\n=== Bottom 10 samples by {neuron_type} neuron scores ===")
for i, sample in enumerate(sample_scores[-10:]):
    print(f"Rank {len(sample_scores)-9+i}: Global Index {sample['global_index']}, "
          f"File: {sample['file_name']}, "
          f"Avg Score: {sample['avg_score']:.6f}, "
          f"Total Score: {sample['total_score']:.6f}, "
          f"Neurons Found: {sample['neuron_count']}/{sample['target_neurons']}")

# 统计信息
scores_list = [sample['avg_score'] for sample in sample_scores]
import numpy as np

print(f"\n=== Statistics ===")
print(f"Mean score: {np.mean(scores_list):.6f}")
print(f"Median score: {np.median(scores_list):.6f}")
print(f"Std score: {np.std(scores_list):.6f}")
print(f"Min score: {np.min(scores_list):.6f}")
print(f"Max score: {np.max(scores_list):.6f}")

# 保存统计信息
stats = {
    "neuron_type": neuron_type,
    "target_neurons": neurons,
    "total_samples": total_samples,
    "mean_score": float(np.mean(scores_list)),
    "median_score": float(np.median(scores_list)),
    "std_score": float(np.std(scores_list)),
    "min_score": float(np.min(scores_list)),
    "max_score": float(np.max(scores_list))
}

stats_file = f"oasst1_ranking/{neuron_type}/{neuron_type}_neurons_stats.json"
with open(stats_file, 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"Statistics saved to {stats_file}")



