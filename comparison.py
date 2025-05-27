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
from finding_map import get_significant_features

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
dataname = "wino"

with open("wino_sentences/merged_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sentences_chosen = data
#half_length = 40
first_length = 792           #2071 for banking, 792 for wino, 2000 for nllb
second_length = 792             #1009 for hotel, 792 for wino, 2000 for nllb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
sae = sae.to(device)
embs = []
with torch.inference_mode():
    for pn, sentences in list(sentences_chosen.items())[::2]:
        from tqdm import tqdm
        for sentence in tqdm(sentences, desc=f"Processing {pn} sentences"):
            inputs = tokenizer(sentence, return_tensors="pt")
            inputs = inputs.to(device)
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
