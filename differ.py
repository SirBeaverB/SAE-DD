from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
from sae.sae import *
from sae.config import *
from safetensors.torch import load_model
from pathlib import Path
from collections import Counter
from sklearn.cluster import SpectralClustering

sae_name = "EleutherAI/sae-pythia-410m-65k"
# 410m-65k  or 160m-32k
saes = Sae.load_many("EleutherAI/sae-pythia-410m-65k")
sae = Sae.load_from_hub(sae_name, hookpoint="layers.11.mlp")




model_name = "EleutherAI/pythia-410m"
model_name = "EleutherAI/pythia-410m"

"""sae_name = "EleutherAI/sae-llama-3-8b-32x"
saes = Sae.load_many("EleutherAI/sae-llama-3-8b-32x")
sae = Sae.load_from_hub(sae_name, hookpoint="layers.31")

model_name = "meta-llama/Meta-Llama-3-8B"
"""

tokenizer = AutoTokenizer.from_pretrained(model_name)

import json

with open("wino_sentences/merged_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)


sentences_chosen = data
#half_length = 40
banking_length = 792           #2071
hotel_length = 792             #1009


model = AutoModelForCausalLM.from_pretrained(model_name)
embs = []
with torch.inference_mode(): # no gradient
    for pn, sentences in list(sentences_chosen.items())[2:]:
        from tqdm import tqdm
        for sentence in tqdm(sentences, desc=f"Processing {pn} sentences"):
            inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)

            # for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
                # latent_acts.append(sae.encode(hidden_state))
            hidden_states = outputs.hidden_states[-1] # get last layer
            latent_acts = sae.encode(hidden_states) # put into SAE
            latent_features_sum = torch.zeros(sae.encoder.out_features).to(sae.encoder.weight.device)
            latent_features_sum[latent_acts.top_indices.flatten()] += latent_acts.top_acts.flatten() # sum up the latent features
            latent_features_sum /= hidden_states.numel() / hidden_states.shape[-1] # average
            # embs.append(latent_features_sum.nonzero().flatten())
            # print(latent_features_sum.count_nonzero())
            # print(latent_features_sum.topk(k=32))
            embs.append(latent_features_sum.topk(k=32).indices) # get top k indices

embs = [set(i.tolist()) for i in embs]



def indices_to_onehot(indices, total_neurons=32000):
    """
    将激活神经元的索引列表转换为 one-hot 向量。
    参数:
        indices: list 或 set，包含激活神经元的索引（注意：索引需要是 0-indexed，如果是 1-indexed，请先减 1）
        total_neurons: 整数，总神经元数量
    返回:
        torch.Tensor，形状为 (total_neurons, ) 的 one-hot 向量
    """
    if isinstance(indices, set):
        indices = list(indices)
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    onehot_matrix = F.one_hot(indices_tensor, num_classes=total_neurons) # (num_indices, total_neurons)
    onehot_vector = torch.sum(onehot_matrix, dim=0)
    onehot_vector = (onehot_vector > 0).long() # normalize to 0 or 1
    return onehot_vector

onehots = []
sentence_flat = sum(sentences_chosen.values(), [])

for sentence, emb in zip(sentence_flat, embs):
    onehot = indices_to_onehot(emb, total_neurons=sae.encoder.out_features)
    onehots.append(onehot)

# 将所有 one-hot 向量堆叠成一个矩阵，形状为 (num_sentences, total_neurons)
onehots_tensor = torch.stack(onehots).float()

# 计算内积矩阵，形状为 (num_sentences, num_sentences)
dot_product = torch.matmul(onehots_tensor, onehots_tensor.t())

# 计算每个向量的 L2 范数，形状为 (num_sentences, 1)
norms = torch.norm(onehots_tensor, dim=1, keepdim=True)
# 构造范数矩阵，两两相乘，形状为 (num_sentences, num_sentences)
norm_matrix = torch.matmul(norms, norms.t())

epsilon = 1e-8
cosine_similarity_matrix = dot_product / (norm_matrix + epsilon)

print("Pairwise cosine similarity matrix:")
print(cosine_similarity_matrix)
# Save the cosine similarity matrix to a text file
output_path = Path("cosine_similarity_matrix.txt")
with output_path.open("w") as f:
    for row in cosine_similarity_matrix:
        f.write(" ".join(f"{value:.6f}" for value in row.tolist()) + "\n")

# 将 torch.Tensor 转换为 numpy 数组
similarity_matrix_np = cosine_similarity_matrix.numpy()

# 谱聚类，分成两类
spectral_model = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
labels = spectral_model.fit_predict(similarity_matrix_np)

print("cluster result：", labels)

# 假设真实标签
true_labels = np.array([0]*banking_length + [1]*hotel_length)

# 假设 spectral_model 经过谱聚类得到的预测标签
predicted_labels = labels  # 这里 labels 是你前面谱聚类得到的结果

from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

# 计算 ARI 和 NMI（这些指标不受标签顺序影响）
ari = adjusted_rand_score(true_labels, predicted_labels)
nmi = normalized_mutual_info_score(true_labels, predicted_labels)

print("Adjusted Rand Index (ARI):", ari)
print("Normalized Mutual Information (NMI):", nmi)

# 计算聚类准确率，需要先找到最佳匹配
def cluster_accuracy(y_true, y_pred):
    """
    计算聚类准确率，通过匈牙利算法找到最佳匹配
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    # 构造混淆矩阵
    cost_matrix = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        cost_matrix[y_pred[i], y_true[i]] += 1
    # 使用匈牙利算法找到最佳匹配（这里用的是最小化 cost, 因此取最大值减去 cost 矩阵）
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    total_correct = cost_matrix[row_ind, col_ind].sum()
    return total_correct / len(y_pred)

acc = cluster_accuracy(true_labels, predicted_labels)
print("Clustering Accuracy:", acc)

from scipy.stats import chi2_contingency
X_np = onehots_tensor.numpy()  # 每一行代表一个样本的 one-hot 特征

def feature_significance(X, labels):
    """
    对于每个特征（即每个神经元是否激活），构造2x2列联表：
        - 行：聚类标签（0 或 1）
        - 列：该特征激活（1）或未激活（0）
    通过卡方检验计算每个特征在不同聚类中的分布是否存在显著差异，返回 p 值数组。
    """
    n_features = X.shape[1]
    p_values = []
    for i in range(n_features):
        contingency_table = np.zeros((2, 2))
        for cluster in [0, 1]:
            # 找到属于该聚类的样本
            cluster_indices = (labels == cluster)
            # 统计该特征在该聚类下激活（1）和未激活（0）的数量
            count_1 = np.sum(X[cluster_indices, i])
            count_0 = np.sum(1 - X[cluster_indices, i])
            contingency_table[cluster, 0] = count_0
            contingency_table[cluster, 1] = count_1
        # 进行卡方检验
        contingency_table += 1e-8
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        p_values.append(p)
    return np.array(p_values)

# 计算每个特征的 p 值
p_values = feature_significance(X_np, labels)
#print("每个特征的 p 值统计：", p_values)

# 筛选出 p 值小于 0.05 的特征，认为这些特征在不同簇中分布有显著差异
significant_features = np.where(p_values < 0.05)[0]

# 可以对显著特征按 p 值从小到大排序，越小说明分布差异越显著
sorted_idx = np.argsort(p_values[significant_features])
top_features = significant_features[sorted_idx]

top_print = [f"{i}: {p_values[i]:.8f}" for i in top_features]

print("features with p < 0.05): ", top_print)
