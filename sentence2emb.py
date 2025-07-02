from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sae.sae import *
from sae.config import *
from safetensors.torch import load_model
from pathlib import Path
from collections import Counter
from sklearn.cluster import SpectralClustering
import json
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency
# from finding_map import get_significant_features
import os

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

sae_name = "EleutherAI/sae-pythia-410m-65k"
# 410m-65k  or 160m-32k
sae = Sae.load_from_hub(sae_name, hookpoint="layers.11.mlp")

model_name = "EleutherAI/pythia-410m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
dataname = "oasst1"

'''with open("wino_sentences/merged_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)'''

# Load all files from oasst1_all directory
data = []
import glob
import os

# Get all json files in the oasst1_all directory
json_files = glob.glob("oasst1_all/oasst1_text_with_index_*.json")
json_files.sort()  # Sort to ensure consistent order

# Load all files
for file_path in json_files:
    with open(file_path, "r", encoding="utf-8") as f:
        file_data = json.load(f)
        sentences = [item['text'] for item in file_data]
        data.extend(sentences)

print(f"Loaded {len(data)} total samples from {len(json_files)} files")

sentences_chosen = data

# 检查可用的GPU数量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Available GPUs: {num_gpus}")

# 加载模型到GPU
try:
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
sae = sae.to(device)
    print(f"Models loaded successfully on {device}")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# 启用多GPU处理
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs with DataParallel")
    model = nn.DataParallel(model)
    sae = nn.DataParallel(sae)
    batch_size = 4 * num_gpus  # 减少批处理大小以避免内存问题
else:
    print(f"Using single GPU: {device}")
    batch_size = 1  # 减少批处理大小以确保稳定性

print(f"Batch size: {batch_size}")

# 添加内存管理优化
import gc
import torch.cuda

# 分段保存设置
chunk_size = 2000  # 每处理100个样本就保存一次
embs = []
processed_count = 0

with torch.inference_mode():
    from tqdm import tqdm
    
    # 将句子分批处理
    for i in tqdm(range(0, len(sentences_chosen), batch_size), desc="Processing sentence batches"):
        try:
        batch_sentences = sentences_chosen[i:i+batch_size]
        
        # 批量tokenize
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)  # 添加最大长度限制
        inputs = inputs.to(device)
        
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # 清理内存并继续
            torch.cuda.empty_cache()
            gc.collect()
            continue
        
        # 处理每个样本的隐藏状态
        batch_embs = []
        for j in range(hidden_states.size(0)):
            try:
            # 获取当前样本的隐藏状态
            sample_hidden = hidden_states[j:j+1]
            
            # 使用SAE编码
            if num_gpus > 1:
                latent_acts = sae.module.encode(sample_hidden)
                encoder_features = sae.module.encoder.out_features
            else:
                latent_acts = sae.encode(sample_hidden)
                encoder_features = sae.encoder.out_features
            
            latent_features_sum = torch.zeros(encoder_features).to(device)
            latent_features_sum[latent_acts.top_indices.flatten()] += latent_acts.top_acts.flatten()
            latent_features_sum /= sample_hidden.numel() / sample_hidden.shape[-1]
                
                # 获取top-k的indices和scores
                top_k_values, top_k_indices = latent_features_sum.topk(k=128)
                
                # 将indices和scores组合成字典格式
                neuron_data = {}
                for idx, score in zip(top_k_indices.tolist(), top_k_values.tolist()):
                    neuron_data[idx] = score
                
                batch_embs.append(neuron_data)
            except Exception as e:
                print(f"Error processing sample {j} in batch {i}: {e}")
                # 添加空的neuron_data作为占位符
                batch_embs.append({})
                continue
        
        # 添加到结果列表
        embs.extend(batch_embs)
        processed_count += len(batch_embs)
        
        # 每处理chunk_size个样本就保存一次
        if processed_count >= chunk_size:
            chunk_start = len(embs) - processed_count
            chunk = embs[chunk_start:]
            
            # 保存包含indices和scores的数据
            chunk_save_path = output_dir / f"{dataname}_embs_chunk_{len(embs)//chunk_size}.json"
            with open(chunk_save_path, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
            
            print(f"已保存第 {len(embs)//chunk_size} 个chunk，包含 {len(chunk)} 个样本")
            processed_count = 0
        
        # 清理GPU内存
        try:
            del inputs, outputs, hidden_states, batch_embs
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Warning: Error during memory cleanup: {e}")
            # 强制清理
        torch.cuda.empty_cache()
        gc.collect()

# 保存剩余的样本
if processed_count > 0:
    chunk_start = len(embs) - processed_count
    chunk = embs[chunk_start:]
    
    # 保存包含indices和scores的数据
    chunk_save_path = output_dir / f"{dataname}_embs_chunk_{len(embs)//chunk_size + 1}.json"
    with open(chunk_save_path, "w", encoding="utf-8") as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)
    
    print(f"已保存最后一个chunk，包含 {len(chunk)} 个样本")

print(f"总共处理了 {len(embs)} 个样本")
print(f"结果已保存到 {output_dir} 目录")

# 注释掉后面的计算部分
"""
def indices_to_onehot(indices, total_neurons=32000):
    if isinstance(indices, set):
        indices = list(indices)
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    onehot_matrix = F.one_hot(indices_tensor, num_classes=total_neurons)
    onehot_vector = torch.sum(onehot_matrix, dim=0)
    onehot_vector = (onehot_vector > 0).long()
    return onehot_vector

onehots = []
sentence_flat = sentences_chosen  # Changed from sum(sentences_chosen.values(), []) since sentences_chosen is now a list

for sentence, emb in zip(sentence_flat, embs):
    total_neurons = sae.module.encoder.out_features if num_gpus > 1 else sae.encoder.out_features
    onehot = indices_to_onehot(emb, total_neurons=total_neurons)
    onehots.append(onehot)

    
# Convert to numpy arrays for easier computation
first_half_np = torch.stack(onehots[:first_length]).numpy()
second_half_np = torch.stack(onehots[first_length:]).numpy()
    
first_half_means = np.mean(first_half_np, axis=0)
second_half_means = np.mean(second_half_np, axis=0)
    
# Define a threshold for significant mean difference
threshold = 0.1

# Calculate absolute difference between means
mean_diffs = np.abs(first_half_means - second_half_means)

# Initialize significant features list
significant_features = []

# Iterate over mean differences to find significant features
for feature_id, mean_diff in enumerate(mean_diffs):
    if mean_diff > threshold:
        cluster = 0 if first_half_means[feature_id] > second_half_means[feature_id] else 1
        significant_features.append((feature_id, cluster))

    # Find the feature with maximum difference
top_k = 10
# Get indices of top k features with largest differences
max_diff_indices = np.argsort(mean_diffs)[-top_k:]
# Get the corresponding values in descending order
max_diff_features = max_diff_indices[::-1]
max_diff_value = mean_diffs[max_diff_features]

print(f"Feature with maximum difference: {max_diff_features}")
print(f"Difference value: {max_diff_value}")
print(f"First half mean: {first_half_means[max_diff_features]}")
print(f"Second half mean: {second_half_means[max_diff_features]}")
"""
