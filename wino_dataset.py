from datasets import load_dataset
import re
import json

type1_anti = load_dataset("uclanlp/wino_bias", "type1_anti")
type1_pro = load_dataset("uclanlp/wino_bias", "type1_pro")
type2_anti = load_dataset("uclanlp/wino_bias", "type2_anti")
type2_pro = load_dataset("uclanlp/wino_bias", "type2_pro")

"""
The *_pro subsets contain sentences that reinforce gender stereotypes (e.g. mechanics are male, nurses are female), 
whereas the *_anti datasets contain "anti-stereotypical" sentences (e.g. mechanics are female, nurses are male).

The type1 (WB-Knowledge) subsets contain sentences for which world knowledge is necessary to resolve the co-references, 
and type2 (WB-Syntax) subsets require only the syntactic information present in the sentence to resolve them.
"""

# get tokens
type1_anti_tokens = type1_anti["validation"]["tokens"] + type1_anti["test"]["tokens"]
type1_pro_tokens = type1_pro["validation"]["tokens"] + type1_pro["test"]["tokens"]
type2_anti_tokens = type2_anti["validation"]["tokens"] + type2_anti["test"]["tokens"]
type2_pro_tokens = type2_pro["validation"]["tokens"] + type2_pro["test"]["tokens"]


#get sentences
def get_sentences(tokenized_list):
    sentences = []
    for tokenized_sentence in tokenized_list:
        sentence = " ".join(tokenized_sentence)
        # remove space before punctuation
        sentence = re.sub(r' (?=\W)', '', sentence)
        sentences.append(sentence)
    return sentences

type1_anti_sentences = get_sentences(type1_anti_tokens)
type1_pro_sentences = get_sentences(type1_pro_tokens)
type2_anti_sentences = get_sentences(type2_anti_tokens)
type2_pro_sentences = get_sentences(type2_pro_tokens)

saving = [type1_anti_sentences, type1_pro_sentences, type2_anti_sentences, type2_pro_sentences]
name = ["type1_anti_sentences", "type1_pro_sentences", "type2_anti_sentences", "type2_pro_sentences"]
for dataset, n in zip(saving, name):
    file_name = f"wino_sentences/{n}.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)


"""from transformers import AutoModelForCausalLM, AutoTokenizer
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

model = AutoModelForCausalLM.from_pretrained(model_name)
embs = []
with torch.inference_mode(): # no gradient
    for sentence in type1_anti_sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
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
sentences_with_embeddings = []
for idx, (sentence, emb_set) in enumerate(zip(type1_anti_sentences, embs)):  # Use the same slice as above
    sentences_with_embeddings.append({
        "index": idx,
        "sentence": sentence,
        "embeddings": sorted(list(emb_set), key=lambda x: x[1], reverse=True)
    })
print("embeddings with sentences created.")

file_name = "type1_anti_sentences_with_embeddings.json"
with open(file_name, "w", encoding="utf-8") as f:
    json.dump(sentences_with_embeddings, f, ensure_ascii=False, indent=4)"""







