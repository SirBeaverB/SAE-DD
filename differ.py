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
import json
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency

# 创建输出目录
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

sae_name = "EleutherAI/sae-pythia-410m-65k"
# 410m-65k  or 160m-32k
saes = Sae.load_many("EleutherAI/sae-pythia-410m-65k")
sae = Sae.load_from_hub(sae_name, hookpoint="layers.11.mlp")

model_name = "EleutherAI/pythia-410m"


"""sae_name = "EleutherAI/sae-llama-3-8b-32x"
saes = Sae.load_many("EleutherAI/sae-llama-3-8b-32x")
sae = Sae.load_from_hub(sae_name, hookpoint="layers.31")

model_name = "meta-llama/Meta-Llama-3-8B"
"""

tokenizer = AutoTokenizer.from_pretrained(model_name)
dataname = "nllb_news_health"

with open("nllb_domain_sentences.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sentences_chosen = data
#half_length = 40
banking_length = 2000           #2071 for banking, 792 for wino, 2000 for nllb
hotel_length = 2000             #1009 for hotel, 792 for wino, 2000 for nllb


model = AutoModelForCausalLM.from_pretrained(model_name)
embs = []
with torch.inference_mode():
    for pn, sentences in list(sentences_chosen.items())[1:]:
        from tqdm import tqdm
        for sentence in tqdm(sentences, desc=f"Processing {pn} sentences"):
            inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            latent_acts = sae.encode(hidden_states)
            latent_features_sum = torch.zeros(sae.encoder.out_features).to(sae.encoder.weight.device)
            latent_features_sum[latent_acts.top_indices.flatten()] += latent_acts.top_acts.flatten()
            latent_features_sum /= hidden_states.numel() / hidden_states.shape[-1]
            embs.append(latent_features_sum.topk(k=32).indices)

embs = [set(i.tolist()) for i in embs]

def indices_to_onehot(indices, total_neurons=32000):
    if isinstance(indices, set):
        indices = list(indices)
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    onehot_matrix = F.one_hot(indices_tensor, num_classes=total_neurons)
    onehot_vector = torch.sum(onehot_matrix, dim=0)
    onehot_vector = (onehot_vector > 0).long()
    return onehot_vector

onehots = []
sentence_flat = sum(sentences_chosen.values(), [])

for sentence, emb in zip(sentence_flat, embs):
    onehot = indices_to_onehot(emb, total_neurons=sae.encoder.out_features)
    onehots.append(onehot)

onehots_tensor = torch.stack(onehots).float()
dot_product = torch.matmul(onehots_tensor, onehots_tensor.t())
norms = torch.norm(onehots_tensor, dim=1, keepdim=True)
norm_matrix = torch.matmul(norms, norms.t())

epsilon = 1e-8
cosine_similarity_matrix = dot_product / (norm_matrix + epsilon)

# Save the cosine similarity matrix to a text file
output_path = Path("cosine_similarity_matrix.txt")
with output_path.open("w") as f:
    for row in cosine_similarity_matrix:
        f.write(" ".join(f"{value:.6f}" for value in row.tolist()) + "\n")

similarity_matrix_np = cosine_similarity_matrix.numpy()
spectral_model = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
labels = spectral_model.fit_predict(similarity_matrix_np)

true_labels = np.array([0]*banking_length + [1]*hotel_length)
predicted_labels = labels

ari = adjusted_rand_score(true_labels, predicted_labels)
nmi = normalized_mutual_info_score(true_labels, predicted_labels)

def cluster_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        cost_matrix[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    total_correct = cost_matrix[row_ind, col_ind].sum()
    return total_correct / len(y_pred)

acc = cluster_accuracy(true_labels, predicted_labels)

X_np = onehots_tensor.numpy()

def feature_significance(X, labels):
    n_features = X.shape[1]
    p_values = []
    feature_means = np.zeros((2, n_features))
    for i in range(n_features):
        contingency_table = np.zeros((2, 2))
        for cluster in [0, 1]:
            cluster_indices = (labels == cluster)
            count_1 = np.sum(X[cluster_indices, i])
            count_0 = np.sum(1 - X[cluster_indices, i])
            contingency_table[cluster, 0] = count_0
            contingency_table[cluster, 1] = count_1
            
            feature_means[cluster, i] = np.mean(X[cluster_indices, i])
        
        contingency_table += 1e-8
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        p_values.append(p)
    
    return np.array(p_values), feature_means

p_values, feature_means = feature_significance(X_np, labels)
significant_features = np.where(p_values < 0.01)[0]
sorted_idx = np.argsort(p_values[significant_features])
top_features = significant_features[sorted_idx]

# 保存所有结果到一个文件
with (output_dir / f"{dataname}_results.txt").open("w") as f:
    f.write("\n=== Cluster Labels ===\n")
    f.write(" ".join(map(str, labels)) + "\n")
    
    f.write("\n=== Evaluation Metrics ===\n")
    f.write(f"Adjusted Rand Index (ARI): {ari}\n")
    f.write(f"Normalized Mutual Information (NMI): {nmi}\n")
    f.write(f"Clustering Accuracy: {acc}\n")
    
    f.write("\n=== Significant Features for Cluster 0 ===\n")
    for i in top_features:
        if feature_means[0, i] > feature_means[1, i]:
            f.write(f"Feature {i}: p-value = {p_values[i]:.8f}, Mean in Cluster 0 = {feature_means[0, i]:.8f}, Mean in Cluster 1 = {feature_means[1, i]:.8f}\n")
    
    f.write("\n=== Significant Features for Cluster 1 ===\n")
    for i in top_features:
        if feature_means[1, i] > feature_means[0, i]:
            f.write(f"Feature {i}: p-value = {p_values[i]:.8f}, Mean in Cluster 0 = {feature_means[0, i]:.8f}, Mean in Cluster 1 = {feature_means[1, i]:.8f}\n")

print(f"Results saved to {output_dir / f'{dataname}_results.txt'}")