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
import os

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

feature_of_interest = "financial"

def find_feature_in_json(directory, feature):
    feature_mapping = {}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if item['token'] == feature:
                        neurons = [embedding[0] for embedding in item['embeddings'][:5]]
                        feature_mapping[feature] = neurons
    return feature_mapping

json_directory = 'embeddings_with_tokens_AVL'
feature_mapping = find_feature_in_json(json_directory, feature_of_interest)

if feature_mapping:
    print(f"Found tokens for feature {feature_of_interest}:")
    print(feature_mapping[feature_of_interest])
else:
    print(f"No mapping found for feature {feature_of_interest}")


# Convert embs to one-hot vectors
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

# Calculate means for each feature mapping index across all sentences
feature_indices = feature_mapping[feature_of_interest]
first_half_means = np.mean(first_half_np[:, feature_indices], axis=0)
second_half_means = np.mean(second_half_np[:, feature_indices], axis=0)

# Print the means for each feature index
print("First half means for feature indices:", first_half_means)
print("Second half means for feature indices:", second_half_means)
