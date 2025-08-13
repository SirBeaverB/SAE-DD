#!/usr/bin/env python3
"""
Test script for fine-tuned model on CSQA test sentences with accuracy calculation
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

def parse_question_text(text: str) -> Tuple[str, str, str]:
    """Parse question text to extract question, options, and answer"""
    parts = text.split(' | ')
    
    question = ""
    options = ""
    answer = ""
    
    for part in parts:
        if part.startswith('question:'):
            question = part.replace('question:', '').strip()
        elif part.startswith('options:'):
            options = part.replace('options:', '').strip()
        elif part.startswith('answer:'):
            answer = part.replace('answer:', '').strip()
    
    return question, options, answer

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

def generate_answer(model, tokenizer, question: str, options: str, max_length: int = 200) -> str:
    """Generate answer for a given question"""
    # Create prompt
    system_prompt = "Answer the question based on the options provided. Only output the answer letter."
    prompt = f"{system_prompt}\nquestion: {question} | options: {options} | answer:"
    
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

def test_model_with_accuracy(model_path: str, test_file: str, output_file: str = None):
    """Test the fine-tuned model on test sentences and calculate accuracy"""
    
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
        question, options, ground_truth_answer = parse_question_text(text)
        
        # Generate answer
        generated_answer = generate_answer(model, tokenizer, question, options)
        extracted_answer = extract_answer_from_generated_text(generated_answer)
        
        # Check if answer is correct
        is_correct = extracted_answer == ground_truth_answer
        if is_correct:
            correct_count += 1
        
        result = {
            'index': sentence_data.get('index', len(results)),
            'question': question,
            'options': options,
            'ground_truth': ground_truth_answer,
            'generated_text': generated_answer,
            'extracted_answer': extracted_answer,
            'is_correct': is_correct
        }
        results.append(result)
    
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # Save results
    if output_file:
        output_data = {
            'total_tests': total_count,
            'correct_count': correct_count,
            'accuracy': accuracy,
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Total tests: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Generated answers: {len([r for r in results if r['extracted_answer']])}")

def main():
    parser = argparse.ArgumentParser(description='Test fine-tuned GPT-2 model on CSQA sentences with accuracy')
    parser.add_argument('--model_path', type=str, 
                       default='finetune/csqa50%',
                       help='Path to the fine-tuned model')
    parser.add_argument('--test_file', type=str, 
                       default='csqa_test_sentences.json',
                       help='Path to the test sentences JSON file')
    parser.add_argument('--output_file', type=str, 
                       default='csqa_50%_test_results.json',
                       help='Path to save test results')
    
    args = parser.parse_args()
    
    print("Testing fine-tuned GPT-2 model with accuracy calculation...")
    test_model_with_accuracy(args.model_path, args.test_file, args.output_file)
    print("Testing completed!")

if __name__ == "__main__":
    main() 