from datasets import load_dataset

ds = load_dataset("breakend/nllb-multi-domain", "eng_Latn-ayr_Latn")

"""
DatasetDict({
    train: Dataset({

    valid: Dataset({
        features: ['id', 'domain', 'sentence_eng_Latn', 'sentence_ayr_Latn'],
        num_rows: 1309
    })
    test: Dataset({
        features: ['id', 'domain', 'sentence_eng_Latn', 'sentence_ayr_Latn'],
        num_rows: 1500
    })
})
"""
domain_sentences = {}
for row in ds['train']:
    domain = row['domain']
    sentence = row['sentence_eng_Latn']
    if domain not in domain_sentences:
        domain_sentences[domain] = []
    domain_sentences[domain].append(sentence)

# Save to JSON file
import json
with open('nllb_domain_sentences.json', 'w', encoding='utf-8') as f:
    json.dump(domain_sentences, f, ensure_ascii=False, indent=4)