import json
import os

def extract_indices(data):
    indices = [item.split(':')[0] for item in data]
    return indices

def read_json_files(directory):
    json_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                json_data.extend(data)
    return json_data

def map_indices_to_tokens(indices, json_data):
    index_to_tokens = {}
    for index in indices:
        index = int(index)  # Convert index to integer for comparison
        for item in json_data:
            if item['embedding_index'] == index:
                tokens = [token['token'] for token in item['tokens'][:10]]
                index_to_tokens[index] = tokens
                break
    return index_to_tokens

# Example data
data = [
    '37950: 0.00000000', '29951: 0.00000000', '49362: 0.00000000', '31787: 0.00000000', '63317: 0.00000000', '7606: 0.00000000', '3770: 0.00000000', '21637: 0.00000000', '17035: 0.00000000', '52157: 0.00000000', '55491: 0.00000000', '33925: 0.00000000', '27483: 0.00000000', '25674: 0.00000000', '4889: 0.00000000', '17225: 0.00000000', '60711: 0.00000000', '14981: 0.00000000', '11268: 0.00000000', '54856: 0.00000000', '27756: 0.00000000', '14655: 0.00000000', '440: 0.00000000', '37068: 0.00000000', '43991: 0.00000000', '46491: 0.00000000', '10141: 0.00000000', '24372: 0.00000000', '51509: 0.00000000', '55880: 0.00000000', '5087: 0.00000000', '30046: 0.00000000', '9564: 0.00000000', '17084: 0.00000000', '4311: 0.00000000', '26076: 0.00000000'
    
]

# Extract indices
indices = extract_indices(data)

# Read JSON files from the directory
json_directory = 'countermap_parts_AVL' 
json_data = read_json_files(json_directory)

# Map indices to tokens
index_to_tokens = map_indices_to_tokens(indices, json_data)

# Save results to JSON file
output_file = 'index_to_tokens_mapping_news_health.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(index_to_tokens, f, ensure_ascii=False, indent=2)

print(f"Results saved to {output_file}")