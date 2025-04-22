from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sae.sae import Sae
from sae.config import *
from safetensors.torch import load_model
from pathlib import Path
from collections import Counter
import json
from tqdm import tqdm

sae_name = "EleutherAI/sae-pythia-410m-65k"
# sae-pythia-160m-32k or sae-pythia-410m-65k
saes = Sae.load_many(sae_name)
sae = Sae.load_from_hub(sae_name, hookpoint="layers.11.mlp")


model_name = "EleutherAI/pythia-410m"

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


list_name = "vocab_list_OLMo2-8B-SuperBPE-t180k"
token_list = read_tokens_to_list(f"{list_name}.txt")
print("token_list loaded.")

model = AutoModelForCausalLM.from_pretrained(model_name)
chunk_size = 2000
current_chunk = []
chunk_index = 1

with torch.inference_mode(): # no gradient
    for idx, token in enumerate(tqdm(token_list, desc="Processing tokens")):
        inputs = tokenizer(token, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # get last layer
        latent_acts = sae.encode(hidden_states)  # put into SAE
        latent_features_sum = torch.zeros(sae.encoder.out_features).to(sae.encoder.weight.device)
        latent_features_sum[latent_acts.top_indices.flatten()] += latent_acts.top_acts.flatten()  # sum up the latent features
        latent_features_sum /= hidden_states.numel() / hidden_states.shape[-1]  # average
        topk = latent_features_sum.topk(k=32)
        emb_set = set(zip(topk.indices.tolist(), topk.values.tolist()))
        
        # Create token entry with sorted embeddings
        token_entry = {
            "index": idx,
            "token": token,
            "embeddings": sorted(list(emb_set), key=lambda x: x[1], reverse=True)
        }
        current_chunk.append(token_entry)
        
        # Save chunk when it reaches the desired size
        if len(current_chunk) >= chunk_size or idx == len(token_list) - 1:
            file_name = f"embeddings_with_tokens_OLMo2/embeddings_with_tokens_part_{chunk_index}.json"
            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(current_chunk, f, ensure_ascii=False, indent=4)
            print(f"Saved chunk {chunk_index}")
            current_chunk = []
            chunk_index += 1