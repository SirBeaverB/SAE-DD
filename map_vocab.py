from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sae.sae import Sae
from sae.config import *
from safetensors.torch import load_model
from pathlib import Path
from collections import Counter
import json
from tqdm import tqdm

sae_name = "EleutherAI/sae-pythia-160m-32k"
# sae-pythia-410m-65k
saes = Sae.load_many("EleutherAI/sae-pythia-160m-32k")
sae = Sae.load_from_hub(sae_name, hookpoint="layers.11.mlp")


model_name = "EleutherAI/pythia-160m"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def read_tokens_to_list(txt_file_path: str) -> list:
    tokens = []
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Token: "):
                content = line[len("Token: "):]
                
                parts = content.split("-> ID:")
                if len(parts) == 2:
                    token_str = parts[0].strip()
                    tokens.append(token_str)
    return tokens


token_list = read_tokens_to_list("vocab_list_410m.txt")
print("token_list loaded.")

model = AutoModelForCausalLM.from_pretrained(model_name)
embs = []
with torch.inference_mode(): # no gradient
    for token in tqdm(token_list, desc="Processing tokens"):
        inputs = tokenizer(token, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # get last layer
        latent_acts = sae.encode(hidden_states)  # put into SAE
        latent_features_sum = torch.zeros(sae.encoder.out_features).to(sae.encoder.weight.device)
        latent_features_sum[latent_acts.top_indices.flatten()] += latent_acts.top_acts.flatten()  # sum up the latent features
        latent_features_sum /= hidden_states.numel() / hidden_states.shape[-1]  # average
        topk = latent_features_sum.topk(k=32)
        embs.append(list(zip(topk.indices.tolist(), topk.values.tolist())))  # save indices and scores as tuples
print("embeddings calculated.")

embs = [set(i) for i in embs]
embs_with_tokens = []
for idx, (token, emb_set) in enumerate(zip(token_list, embs)):  # Use the same slice as above
    embs_with_tokens.append({
        "index": idx,
        "token": token,
        "embeddings": sorted(list(emb_set), key=lambda x: x[1], reverse=True)
    })
print("embeddings with tokens created.")

chunk_size = 5000
for i in range(0, len(embs_with_tokens), chunk_size):
    chunk = embs_with_tokens[i:i + chunk_size]
    file_name = f"embeddings_with_tokens_160/embeddings_with_tokens_part_{i // chunk_size + 1}.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(chunk, f, ensure_ascii=False, indent=4)