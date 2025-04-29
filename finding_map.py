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

def get_significant_features(file_path, threshold=0.1):
    significant_features = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'Feature' in line:
                # Split using both ', ' and ': '
                parts = line.replace(': ', ', ').split(', ')
                if len(parts) >= 4:  # Ensure there are enough parts to process
                    feature_id = parts[0].split(' ')[1]
                    mean_cluster_0 = float(parts[2].split('= ')[1])
                    mean_cluster_1 = float(parts[3].split('= ')[1])
                    mean_diff = abs(mean_cluster_0 - mean_cluster_1)
                    if mean_diff > threshold:
                        cluster = 0 if mean_cluster_0 > mean_cluster_1 else 1
                        significant_features.append((feature_id, cluster))
    return significant_features


# Read JSON files from the directory
json_directory = 'countermap_parts_AVL' 
json_data = read_json_files(json_directory)

# Example usage
file_name = "nllb_chat_health"
file_path = f'output/{file_name}_results.txt'
significant_features = get_significant_features(file_path)

# Extract indices from the significant features
indices = [int(feature_id) for feature_id, _ in significant_features]

# Map indices to tokens
index_to_tokens = map_indices_to_tokens(indices, json_data)

# Map significant features to tokens and add cluster information
significant_index_to_tokens = {}
for feature_id, cluster in significant_features:
    index = int(feature_id)  # Convert feature_id to integer
    if index in index_to_tokens:
        significant_index_to_tokens[index] = {
            'tokens': index_to_tokens[index],
            'cluster': cluster
        }

# Save significant features mapping to JSON file
significant_output_file = f'map_result/{file_name}_mapping.json'
with open(significant_output_file, 'w', encoding='utf-8') as f:
    json.dump(significant_index_to_tokens, f, ensure_ascii=False, indent=2)

print(f"Significant features mapping saved to {significant_output_file}")