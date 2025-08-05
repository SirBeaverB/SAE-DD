#!/usr/bin/env python3
"""
Test script for fine-tuned GPT-2 model on CSQA test sentences
"""

import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm

def load_test_data(file_path: str) -> List[Dict]:
    """Load test sentences from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_sentences = []
        for sentence_data in data:
            if 'text' in sentence_data:
                test_sentences.append(sentence_data)
        
        print(f"Loaded {len(test_sentences)} test sentences from {file_path}")
        return test_sentences
    except Exception as e:
        print(f"Error loading test data from {file_path}: {e}")
        return []

def parse_question_text(text: str) -> Tuple[str, str]:
    """Parse question text to extract question and options"""
    parts = text.split(' | ')
    
    question = ""
    options = ""
    
    for part in parts:
        if part.startswith('question:'):
            question = part.replace('question:', '').strip()
        elif part.startswith('options:'):
            options = part.replace('options:', '').strip()
    
    return question, options

def extract_answer_from_generated_text(generated_text: str) -> str:
    """Extract answer from generated text"""
    # Look for patterns like "answer: A", "answer: B", etc.
    answer_patterns = [
        r'answer:\s*([A-E])',
        r'Answer:\s*([A-E])',
        r'([A-E])\.\s*[^A-E]*$',  # Last option mentioned
        r'([A-E])\s*$'  # Just the letter at the end
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # If no clear pattern, look for the last A-E option mentioned
    all_options = re.findall(r'([A-E])', generated_text)
    if all_options:
        return all_options[-1].upper()
    
    return ""

def generate_answer(model, tokenizer, question: str, options: str, max_length: int = 100) -> str:
    """Generate answer for a given question"""
    # Create prompt
    prompt = f"question: {question} | options: {options} | answer:"
    
    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones_like(inputs)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        attention_mask = attention_mask.cuda()
        model = model.cuda()
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated part (after the prompt)
    prompt_length = len(tokenizer.encode(prompt))
    generated_part = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
    return generated_part.strip()

def test_model(model_path: str, test_file: str, output_file: str = None):
    """Test the fine-tuned model on test sentences"""
    
    print(f"Loading model from {model_path}...")
    try:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load test data
    test_sentences = load_test_data(test_file)
    if not test_sentences:
        return
    
    results = []
    correct_count = 0
    total_count = len(test_sentences)
    
    print(f"Testing model on {total_count} sentences...")
    
    for sentence_data in tqdm(test_sentences, desc="Testing", unit="test"):
        text = sentence_data['text']
        question, options = parse_question_text(text)
        
        # Generate answer
        generated_answer = generate_answer(model, tokenizer, question, options)
        extracted_answer = extract_answer_from_generated_text(generated_answer)
        
        # For test data, we don't have ground truth answers, so we can't calculate accuracy
        # But we can save the results for manual inspection
        result = {
            'index': sentence_data.get('index', len(results)),
            'question': question,
            'options': options,
            'generated_text': generated_answer,
            'extracted_answer': extracted_answer
        }
        results.append(result)
    
    # Save results
    if output_file:
        output_data = {
            'total_tests': total_count,
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Total tests: {total_count}")
    print(f"Generated answers: {len([r for r in results if r['extracted_answer']])}")
    
    # Show first few answers
    print(f"First 5 answers: {[r['extracted_answer'] for r in results[:5]]}")

def main():
    parser = argparse.ArgumentParser(description='Test fine-tuned GPT-2 model on CSQA sentences')
    parser.add_argument('--model_path', type=str, 
                       default='finetune/csqa50%',
                       help='Path to the fine-tuned model')
    parser.add_argument('--test_file', type=str, 
                       default='csqa_test_sentences.json',
                       help='Path to the test sentences JSON file')
    parser.add_argument('--output_file', type=str, 
                       default='test_results.json',
                       help='Path to save test results')
    
    args = parser.parse_args()
    
    print("Testing fine-tuned GPT-2 model...")
    test_model(args.model_path, args.test_file, args.output_file)
    print("Testing completed!")

if __name__ == "__main__":
    main()
