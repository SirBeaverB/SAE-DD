from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sae.sae import Sae
from sae.config import *
from safetensors.torch import load_model
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, recall_score
from collections import Counter

sae_name = "EleutherAI/sae-pythia-160m-32k"
# sae-pythia-410m-65k
saes = Sae.load_many("EleutherAI/sae-pythia-160m-32k")
sae = Sae.load_from_hub(sae_name, hookpoint="layers.11")

# sae = ConvSae(768, ConvSaeConfig(kernel_size=7))
# device="cpu"
# path = Path("sae-ckpts/gpt_neox.layers.11")

# load_model(
#     model=sae,
#     filename=str(path / "sae.safetensors"),
#     device=str(device),
#     # TODO: Maybe be more fine-grained about this in the future?
#     strict=True,
# )

model_name = "EleutherAI/pythia-410m"
# 410m

tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab = tokenizer.get_vocab()
vocab = list(vocab.items())
vocab = sorted(vocab, key=lambda x: x[1])

# Save the list to a file
with open("vocab_list_410m.txt", "w", encoding="utf-8") as f:
    for token, idx in vocab:
        f.write(f"Token: {token} -> ID: {idx}\n")