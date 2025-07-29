import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import glob


#json_file = 'top_50%_sentences.json'
#json_file = 'top_70%_sentences.json'
#json_file = 'top_80%_sentences.json'
json_file = 'top_90%_sentences.json'  


def analyze_embedding_indices():
    """
    读取csqa_embs目录中的所有JSON文件，统计embedding index的出现次数
    """
    

    '''
    {
  "percentage": "top_50%",
  "sentence_count": 4870,
  "sentences": [
    {
      "sentence_id": 7685,
      "score": 3.989443299643208,
      "latent_indices": [
        "4319",

      ],
      "embeddings_values": {
        "4319": 0.39065220952033997,

      }
    },
    '''
    
    if not json_file:
        print("没有找到JSON文件")
        return
    
    print(f"找到 {len(json_file)} 个JSON文件")

    # 用于统计所有embedding index的计数器
    all_indices = []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = data.get("sentences", [])
    
    for sentence in sentences:
        latent_indices = sentence.get("latent_indices", [])
        all_indices.extend(latent_indices)
    
    if not all_indices:
        print("没有找到有效的embedding index数据")
        return
    
    # 统计每个index的出现次数
    index_counts = Counter(all_indices)
    
    # 打印统计结果
    print(f"\n总共找到 {len(all_indices)} 个embedding index")
    print(f"唯一index数量: {len(index_counts)}")
    print(f"\n最常见的10个index:")
    for index, count in index_counts.most_common(10):
        print(f"Index {index}: {count} 次")
    
    # 生成统计图
    create_statistics_plots(index_counts)

def create_statistics_plots(index_counts):
    """
    创建统计图表
    """

    
    # 创建图形
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 12))
    
    # 1. 前20个最常见index的柱状图
    top_20 = index_counts.most_common(20)
    indices, counts = zip(*top_20)
    
    ax1.bar(range(len(indices)), counts, color='skyblue', alpha=0.7)
    ax1.set_title('20 most common Embedding Index')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Occurrence')
    ax1.set_xticks(range(len(indices)))
    ax1.set_xticklabels(indices, rotation=45)
    ax1.set_ylim(0, 10000)
    
    # 2. 所有index的分布散点图
    all_indices = list(index_counts.keys())
    all_counts = list(index_counts.values())
    
    # 将index转换为数值，保持原始顺序
    numeric_indices = [int(idx) for idx in all_indices]
    
    ax2.scatter(numeric_indices, all_counts, alpha=0.6, s=20, color='skyblue')
    ax2.set_xlabel('Embedding Index')
    ax2.set_ylabel('Occurrence')
    ax2.set_title('All Embedding Index occurrence distribution')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 10000)
    
    
    plt.tight_layout()
    plt.savefig(f'statistic_results/{json_file.split(".")[0]}_embedding_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n统计图已保存为: statistic_results/{json_file.split('.'[0])}_embedding_statistics.png")
    
    # 保存详细统计结果到文件
    save_detailed_statistics(index_counts)

def save_detailed_statistics(index_counts):
    """
    保存详细的统计结果到文件
    """
    with open(f'statistic_results/{json_file.split(".")[0]}_embedding_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("Embedding Index statistics\n")
        f.write("=" * 50 + "\n\n")
        
        total_occurrence = sum(index_counts.values())
        unique_indices = len(index_counts)
        avg_occurrence = total_occurrence / unique_indices
        max_occurrence = max(index_counts.values())
        min_occurrence = min(index_counts.values())
        occurrences = list(index_counts.values())
        variance = np.var(occurrences)
        range_occurrence = max_occurrence - min_occurrence

        f.write(f"Total occurrence: {total_occurrence}\n")
        f.write(f"undead index number: {unique_indices}\n")
        f.write(f"Average occurrence: {avg_occurrence:.2f}\n")
        f.write(f"Maximum occurrence: {max_occurrence}\n")
        f.write(f"Minimum occurrence: {min_occurrence}\n")
        f.write(f"Occurrence variance: {variance:.2f}\n")
        f.write(f"Occurrence range: {range_occurrence}\n\n")
        
        f.write("All Index statistics (sorted by occurrence):\n")
        f.write("-" * 30 + "\n")
        for index, count in index_counts.most_common():
            f.write(f"Index {index}: {count} times\n")
    
    print(f"详细统计结果已保存为: statistic_results/{json_file.split('.'[0])}_embedding_statistics.txt")

if __name__ == "__main__":
    analyze_embedding_indices()