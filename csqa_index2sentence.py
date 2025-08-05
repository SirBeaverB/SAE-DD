

import json
import os
from typing import Dict, List, Any
import re

def load_json_file(file_path: str) -> Any:
    """Load JSON file and return its content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        return None

def create_index_mapping(sentences_data: List[Dict]) -> Dict[int, Dict]:
    """Create a mapping from index to sentence data"""
    index_mapping = {}
    for sentence in sentences_data:
        if 'index' in sentence:
            index_mapping[sentence['index']] = sentence
    return index_mapping

def extract_sentences_by_ids(top_sentences_data: Dict, index_mapping: Dict[int, Dict]) -> List[Dict]:
    """Extract sentences based on sentence_ids from top sentences"""
    extracted_sentences = []
    
    if 'sentences' not in top_sentences_data:
        print("Error: 'sentences' key not found in top sentences data")
        return extracted_sentences
    
    for sentence_info in top_sentences_data['sentences']:
        sentence_id = sentence_info.get('sentence_id')
        if sentence_id is not None and sentence_id in index_mapping:
            # Create a new entry with both top sentence info and original sentence data
            extracted_sentence = {
                'sentence_id': sentence_id,
                'score': sentence_info.get('score'),
                'latent_indices': sentence_info.get('latent_indices', []),
                'original_sentence': index_mapping[sentence_id]
            }
            extracted_sentences.append(extracted_sentence)
        else:
            print(f"Warning: sentence_id {sentence_id} not found in csqa_sentences.json")
    
    return extracted_sentences

def parse_sentence_text(text: str) -> Dict[str, str]:
    """Parse the sentence text to extract question, options, and answer"""
    parts = text.split(' | ')
    
    result = {
        'question': '',
        'options': '',
        'answer': ''
    }
    
    for part in parts:
        if part.startswith('question:'):
            result['question'] = part.replace('question:', '').strip()
        elif part.startswith('options:'):
            result['options'] = part.replace('options:', '').strip()
        elif part.startswith('answer:'):
            result['answer'] = part.replace('answer:', '').strip()
    
    return result

def extract_pure_sentences(extracted_sentences: List[Dict]) -> List[Dict]:
    """Extract pure sentences with separated question, options, and answer"""
    pure_sentences = []
    
    for sentence_data in extracted_sentences:
        # original_text = sentence_data['original_sentence']['text']
        # parsed_parts = parse_sentence_text(original_text)
        
        pure_sentence = sentence_data['original_sentence']['text']
        
        
        pure_sentences.append(pure_sentence)
    
    return pure_sentences

def save_extracted_sentences(extracted_sentences: List[Dict], output_file: str):
    """Save extracted sentences to a new JSON file"""
    output_data = {
        'total_extracted': len(extracted_sentences),
        'sentences': extracted_sentences
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved {len(extracted_sentences)} sentences to {output_file}")
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")

def save_pure_sentences(pure_sentences: List[Dict], output_file: str):
    """Save pure sentences to a new JSON file"""
    output_data = {
        'total_extracted': len(pure_sentences),
        'sentences': pure_sentences
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved {len(pure_sentences)} pure sentences to {output_file}")
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")

def main():
    # File paths
    csqa_sentences_file = "csqa_sentences.json"
    percentage = 80
    top_sentences_file = f"csqa_naive_distill/sentences/top_{percentage}%_sentences.json"
    output_file = f"csqa_naive_distill/original_sentences/original_top_{percentage}%_sentences.json"
    pure_output_file = f"csqa_naive_distill/pure_sentences/pure_top_{percentage}%_sentences.json"
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(pure_output_file), exist_ok=True)
    
    print("Loading csqa_sentences.json...")
    csqa_data = load_json_file(csqa_sentences_file)
    if csqa_data is None:
        return
    
    print("Loading top_50%_sentences.json...")
    top_sentences_data = load_json_file(top_sentences_file)
    if top_sentences_data is None:
        return
    
    print("Creating index mapping...")
    index_mapping = create_index_mapping(csqa_data)
    print(f"Created mapping for {len(index_mapping)} sentences")
    
    print("Extracting sentences based on sentence_ids...")
    extracted_sentences = extract_sentences_by_ids(top_sentences_data, index_mapping)
    
    print(f"Extracted {len(extracted_sentences)} sentences")
    
    print("Saving extracted sentences...")
    save_extracted_sentences(extracted_sentences, output_file)
    
    print("Extracting pure sentences (separated question, options, answer)...")
    pure_sentences = extract_pure_sentences(extracted_sentences)
    
    print("Saving pure sentences...")
    save_pure_sentences(pure_sentences, pure_output_file)
    
    print("Done!")

if __name__ == "__main__":
    main()
