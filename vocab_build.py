"""from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("UW/OLMo2-8B-SuperBPE-t180k")
model = AutoModelForCausalLM.from_pretrained("UW/OLMo2-8B-SuperBPE-t180k")

tokenizer.convert_ids_to_tokens(tokenizer.encode("By the way, I am a fan of the Milky Way."))
# ['ByĠtheĠway', ',ĠIĠam', 'Ġa', 'Ġfan', 'ĠofĠthe', 'ĠMilkyĠWay', '.']

model_name = "UW/OLMo2-8B-SuperBPE-t180k"
model_name_txt = "OLMo2-8B-SuperBPE-t180k"
# model_name = "EleutherAI/pythia-410m"
# 410m and 160m seems to have the same vocab

tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab = tokenizer.get_vocab()
vocab = list(vocab.items())
vocab = sorted(vocab, key=lambda x: x[1])

# Save the list to a file
with open(f"vocab_list_{model_name_txt}.txt", "w", encoding="utf-8") as f:
    for token, idx in vocab:
        f.write(f"Token: {token} -> ID: {idx}\n")
print(f"Vocabulary list saved to vocab_list_{model_name_txt}.txt")"""

# Read AVL_adj_only.txt and save in the same format as the commented example
with open("AVL_adj_only.txt", "r", encoding="utf-8") as f:
    adj_tokens = f.read().splitlines()

# Save the list to a file in the same format
with open("vocab_list_AVL.txt", "w", encoding="utf-8") as f:
    for idx, token in enumerate(adj_tokens):
        f.write(f"Token: {token} -> ID: {idx}\n")
print(f"Vocabulary list saved to vocab_list_AVL_adj_only.txt")
