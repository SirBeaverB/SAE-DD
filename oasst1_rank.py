from datasets import load_dataset
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

# 加载数据集
ds = load_dataset("OpenAssistant/oasst1")
train_data = [example for example in ds["train"] if example['lang'] == 'en']

label_name = "humor"  # humor, violence, creativity, toxicity
label_index = train_data[0]['labels']['name'].index(label_name)

# 获取所有样本的标签值和索引
high_value_indices = []
for i, example in enumerate(train_data):
    if example['labels'] is not None:
        # 处理不同长度的标签值数组
        if len(example['labels']['value']) < 11:
            if label_name in example['labels']['name']:
                label_idx = example['labels']['name'].index(label_name)
                if label_idx < len(example['labels']['value']):
                    high_value_indices.append({
                        'global_index': i,
                        'label_value': example['labels']['value'][label_idx]
                    })
        else:
            high_value_indices.append({
                'global_index': i,
                'label_value': example['labels']['value'][label_index]
            })

# 按label_value排序，取前8000个
high_value_indices.sort(key=lambda x: x['label_value'], reverse=True)
top_8000_indices = [item['global_index'] for item in high_value_indices[:8000]]

# 构建文件夹路径
emb_value_dir = Path("oasst1_ranking") / label_name
if not emb_value_dir.exists():
    print(f"Directory {emb_value_dir} does not exist")
    exit()

# 获取第一个json文件
json_files = list(emb_value_dir.glob("*.json"))
if not json_files:
    print(f"No JSON files found in {emb_value_dir}")
    exit()

first_json_file = json_files[0]
print(f"Reading from: {first_json_file}")

# 读取文件内容
with open(first_json_file, 'r', encoding='utf-8') as f:
    emb_data = json.load(f)

# 提取前8000个句子的index
emb_indices = []
for item in emb_data:
    if isinstance(item, dict) and 'global_index' in item:
        emb_indices.append(item['global_index'])
    elif isinstance(item, (int, float)):
        # 如果数据直接是index
        emb_indices.append(int(item))

print(f"Extracted {len(emb_indices)} indices from embedding file")

def calculate_ranking_similarity(rank1, rank2):
    """
    计算两个排名列表之间的相似度（修正版）
    """
    # 创建排名映射字典
    rank1_dict = {idx: rank for rank, idx in enumerate(rank1)}
    rank2_dict = {idx: rank for rank, idx in enumerate(rank2)}
    
    # 只考虑两个排名列表的交集
    common_indices = set(rank1) & set(rank2)
    
    if len(common_indices) == 0:
        return {
            'spearman_correlation': 0.0,
            'spearman_p_value': 1.0
        }
    
    # 创建排名向量（只包含共同的索引）
    rank1_vector = []
    rank2_vector = []
    
    for idx in common_indices:
        rank1_vector.append(rank1_dict[idx])
        rank2_vector.append(rank2_dict[idx])
    
    # 计算Spearman相关系数
    try:
        spearman_corr, spearman_p = spearmanr(rank1_vector, rank2_vector)
        if np.isnan(spearman_corr):
            spearman_corr = 0.0
    except:
        spearman_corr = 0.0
    
    return {
        'spearman_correlation': float(spearman_corr),
        'spearman_p_value': float(spearman_p) if 'spearman_p' in locals() else 0.0,
        'common_indices_count': len(common_indices)
    }

def calculate_jaccard_similarity(set1, set2):
    """
    计算Jaccard相似度
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

# 将emb_indices改为倒序
# emb_indices = emb_indices[::-1]


# 比较两个index列表
current_indices_set = set(top_8000_indices)
emb_indices_set = set(emb_indices)

# 计算交集和差异
intersection = current_indices_set.intersection(emb_indices_set)
only_in_current = current_indices_set - emb_indices_set
only_in_emb = emb_indices_set - current_indices_set

print(f"\n=== Comparison Results ===")
print(f"Current top 8000 indices count: {len(current_indices_set)}")
print(f"Embedding file indices count: {len(emb_indices_set)}")
print(f"Intersection count: {len(intersection)}")
print(f"Only in current: {len(only_in_current)}")
print(f"Only in embedding file: {len(only_in_emb)}")
print(f"Overlap percentage: {len(intersection)/len(current_indices_set)*100:.2f}%")

# 计算Jaccard相似度
jaccard_sim = calculate_jaccard_similarity(current_indices_set, emb_indices_set)
print(f"Jaccard similarity: {jaccard_sim:.4f}")

# 计算排名相似度
ranking_sim = calculate_ranking_similarity(top_8000_indices, emb_indices)
print(f"Spearman correlation: {ranking_sim['spearman_correlation']:.4f}")

# 保存完整的比较结果
comparison_results = {
    "label_name": label_name,
    "current_indices_count": len(current_indices_set),
    "emb_indices_count": len(emb_indices_set),
    "intersection_count": len(intersection),
    "only_in_current_count": len(only_in_current),
    "only_in_emb_count": len(only_in_emb),
    "overlap_percentage": len(intersection)/len(current_indices_set)*100,
    "jaccard_similarity": jaccard_sim,
    "spearman_correlation": ranking_sim['spearman_correlation'],
    "spearman_p_value": ranking_sim['spearman_p_value'],
    "current_top_8000_indices": top_8000_indices,
    "emb_top_8000_indices": emb_indices,
    "intersection_indices": list(intersection),
    "only_in_current_indices": list(only_in_current),
    "only_in_emb_indices": list(only_in_emb)
}

# 确保输出目录存在
os.makedirs("oasst1_rank_results", exist_ok=True)

# 保存结果到文件
output_file = f"oasst1_rank_results/comparison_results_{label_name}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(comparison_results, f, ensure_ascii=False, indent=2)

print(f"Comparison results saved to {output_file}")

# 打印总结
print(f"\n=== Summary ===")
print(f"- 重叠率: {len(intersection)/len(current_indices_set)*100:.2f}%")
print(f"- Jaccard相似度: {jaccard_sim:.4f}")
print(f"- Spearman相关系数: {ranking_sim['spearman_correlation']:.4f}")
