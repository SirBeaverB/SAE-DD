#!/usr/bin/env python3
"""
Convert CSQA sentences to pure format
"""

import json
import os

def convert_csqa_to_pure(input_file, output_file):
    """Convert CSQA sentences to pure format"""
    
    print(f"Reading CSQA sentences from {input_file}...")
    
    # Read the original CSQA sentences
    with open(input_file, 'r', encoding='utf-8') as f:
        csqa_data = json.load(f)
    
    print(f"Found {len(csqa_data)} CSQA sentences")
    
    # Convert to pure format
    pure_sentences = []
    
    for item in csqa_data:
        if 'text' in item:
            # Extract the sentence text
            sentence_text = item['text']
            pure_sentences.append(sentence_text)
        else:
            print(f"Warning: Item missing 'text' field: {item}")
    
    # Create the pure format structure
    pure_data = {
        "total_extracted": len(pure_sentences),
        "sentences": pure_sentences
    }
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Write the pure format file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pure_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully converted {len(pure_sentences)} sentences")
    print(f"Pure format saved to: {output_file}")
    
    # Show some examples
    print(f"\nFirst 3 sentences:")
    for i, sentence in enumerate(pure_sentences[:3]):
        print(f"{i+1}. {sentence}")
    
    return pure_data

def main():
    # Input and output file paths
    input_file = "csqa_sentences.json"
    output_file = "csqa_naive_distill/pure_sentences/pure_top_100%_sentences.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        print("Please make sure the file exists in the current directory.")
        return
    
    try:
        # Convert the data
        pure_data = convert_csqa_to_pure(input_file, output_file)
        
        print(f"\nConversion completed successfully!")
        print(f"Total sentences: {pure_data['total_extracted']}")
        print(f"Output file: {output_file}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
