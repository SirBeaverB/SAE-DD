import json
import os
import glob
from collections import Counter, defaultdict
import numpy as np

def analyze_latent_activation():
    """
    读取csqa_embs目录中的所有JSON文件，统计latent单元激活频次
    """
    json_files = glob.glob('csqa_embs/csqa_embs_chunk_*.json')
    
    if not json_files:
        print("未找到JSON文件")
        return None, None
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 统计每个latent单元的激活频次
    latent_frequencies = Counter()
    sentence_embeddings = {}  # 存储每个句子的embeddings
    
    # 遍历每个JSON文件
    global_sentence_idx = 0  # 全局句子索引
    
    for file_idx, file_path in enumerate(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for sentence_idx, item in enumerate(data):
                    if isinstance(item, dict):
                        # 提取latent indices
                        latent_indices = list(item.keys())
                        # 统计频次
                        latent_frequencies.update(latent_indices)
                        
                        # 存储句子embeddings，使用全局连续的句子ID
                        sentence_id = global_sentence_idx
                        sentence_embeddings[sentence_id] = item
                        global_sentence_idx += 1
            
            print(f"处理文件: {file_path}, 包含 {len(data) if isinstance(data, list) else 0} 个句子")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    print(f"\n总共处理了 {len(sentence_embeddings)} 个句子")
    print(f"唯一latent单元数量: {len(latent_frequencies)}")
    
    return latent_frequencies, sentence_embeddings

def calculate_sentence_scores(latent_frequencies, sentence_embeddings):
    """
    计算每个句子的score = sum(1/(f+1))，其中f是每个latent单元的激活频次
    """
    sentence_scores = {}
    
    for sentence_id, embeddings in sentence_embeddings.items():
        score = 0
        for latent_idx in embeddings.keys():
            # 计算权重 w = 1/(f+1)
            frequency = latent_frequencies[latent_idx]
            weight = 1 / (frequency + 1)
            score += weight
        
        sentence_scores[sentence_id] = score
    
    return sentence_scores

def split_sentences_by_percentage(sentence_scores, sentence_embeddings, percentages=[50, 70, 80, 90]):
    """
    按score排序并切分不同百分比的句子
    """
    # 按score从大到小排序
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    
    total_sentences = len(sorted_sentences)
    results = {}
    
    for percentage in percentages:
        # 计算切分点
        split_point = int(total_sentences * percentage / 100)
        
        # 获取前percentage%的句子
        top_sentences = sorted_sentences[:split_point]
        
        # 收集这些句子的latent indices和embeddings
        sentence_latent_indices = []
        sentence_embs = []
        
        for sentence_id, score in top_sentences:
            embeddings = sentence_embeddings[sentence_id]
            sentence_latent_indices.append(list(embeddings.keys()))
            sentence_embs.append(embeddings)
        
        results[f"top_{percentage}%"] = {
            'sentence_ids': [s[0] for s in top_sentences],
            'scores': [s[1] for s in top_sentences],
            'latent_indices': sentence_latent_indices,
            'embeddings': sentence_embs,
            'count': len(top_sentences)
        }
        
        print(f"前{percentage}%: {len(top_sentences)} 个句子")
    
    return results

def save_results(results, latent_frequencies):
    """
    保存结果到文件
    """
    # 保存latent频次统计
    with open('csqa_naive_distill/latent_frequencies.json', 'w', encoding='utf-8') as f:
        json.dump(dict(latent_frequencies), f, indent=2)
    
    # 保存每个百分比的结果
    for percentage_name, data in results.items():
        filename = f"csqa_naive_distill/{percentage_name}_sentences.json"
        
        # 准备保存的数据
        save_data = {
            'percentage': percentage_name,
            'sentence_count': data['count'],
            'sentences': []
        }
        
        for i, (sentence_id, score, latent_indices, embeddings) in enumerate(zip(
            data['sentence_ids'], 
            data['scores'], 
            data['latent_indices'], 
            data['embeddings']
        )):
            sentence_data = {
                'sentence_id': sentence_id,
                'score': score,
                'latent_indices': latent_indices,
                'embeddings_values': embeddings
            }
            save_data['sentences'].append(sentence_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"已保存 {filename}")
    
    # 保存统计信息
    with open('csqa_naive_distill/analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("Latent单元激活分析总结\n")
        f.write("=" * 50 + "\n")
        f.write(f"总句子数: {sum(data['count'] for data in results.values())}\n")
        f.write(f"唯一latent单元数: {len(latent_frequencies)}\n\n")
        
        for percentage_name, data in results.items():
            f.write(f"{percentage_name}:\n")
            f.write(f"  句子数: {data['count']}\n")
            f.write(f"  平均score: {np.mean(data['scores']):.4f}\n")
            f.write(f"  最高score: {max(data['scores']):.4f}\n")
            f.write(f"  最低score: {min(data['scores']):.4f}\n\n")
    
    print("所有结果已保存")
    print(f"90%的文件大小为{os.path.getsize('csqa_naive_distill/top_90%_sentences.json')/1024/1024:.2f}MB")

def main():
    """
    主函数
    """
    print("开始分析latent单元激活频次...")
    
    # 1. 分析latent激活频次
    latent_frequencies, sentence_embeddings = analyze_latent_activation()
    
    if latent_frequencies is None:
        return
    
    # 2. 计算句子scores
    print("\n计算句子scores...")
    sentence_scores = calculate_sentence_scores(latent_frequencies, sentence_embeddings)
    
    # 3. 按百分比切分句子
    print("\n按score排序并切分句子...")
    results = split_sentences_by_percentage(sentence_scores, sentence_embeddings)
    
    # 4. 保存结果
    print("\n保存结果...")
    save_results(results, latent_frequencies)
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()