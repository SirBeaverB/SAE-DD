 #!/usr/bin/env python3
"""
Script to convert csqa_sentences.json to pure format
"""

import json
import os
from typing import List, Dict, Any

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

def convert_csqa_to_pure(input_file: str, output_file: str):
    """Convert csqa_sentences.json to pure format"""
    
    print(f"Loading data from {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pure_sentences = []
        total_count = len(data)
        
        print(f"Processing {total_count} sentences...")
        
        for i, sentence_data in enumerate(data):
            if i % 1000 == 0:
                print(f"Processed {i}/{total_count} sentences...")
            
            if 'text' in sentence_data:
                # Keep the original text format as string
                text = sentence_data['text']
                pure_sentences.append(text)
        
        # Create output data structure
        output_data = {
            'total_extracted': len(pure_sentences),
            'sentences': pure_sentences
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully converted and saved {len(pure_sentences)} sentences to {output_file}")
        
        # Print some statistics
        print(f"\nStatistics:")
        print(f"Total sentences: {len(pure_sentences)}")
        
        # Show some examples
        print(f"\nFirst 3 examples:")
        for i, sentence in enumerate(pure_sentences[:3]):
            print(f"Example {i+1}:")
            print(f"  {sentence}")
            print()
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}")
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    # File paths
    input_file = "csqa_sentences.json"
    output_file = "csqa_naive_distill/pure_sentences/pure_all_sentences.json"
    
    print("Converting csqa_sentences.json to pure format...")
    convert_csqa_to_pure(input_file, output_file)
    print("Done!")

if __name__ == "__main__":
    main() 