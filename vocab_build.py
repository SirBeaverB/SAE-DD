from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "EleutherAI/pythia-410m"
# 410m and 160m seems to have the same vocab

tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab = tokenizer.get_vocab()
vocab = list(vocab.items())
vocab = sorted(vocab, key=lambda x: x[1])

# Save the list to a file
with open("vocab_list_410m.txt", "w", encoding="utf-8") as f:
    for token, idx in vocab:
        f.write(f"Token: {token} -> ID: {idx}\n")
print("Vocabulary list saved to vocab_list_410m.txt")