import json
import os

def merge_json_files(directory, output_file):
    merged_data = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                merged_data[filename] = data
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(merged_data, outfile, ensure_ascii=False, indent=2)

# Directory containing JSON files
json_directory = 'wino_sentences'  # Update this path

# Output file for the merged JSON
output_file = 'wino_sentences/merged_data.json'

# Merge JSON files
merge_json_files(json_directory, output_file)

print(f"Merged JSON data saved to {output_file}")