#!/usr/bin/env python3
"""
Test script for fine-tuned model on CSQA test sentences using paper's method
Uses [start], [sep], [end] tokens and calculates P([end] | [start] question [sep] answer)
Based on the paper's approach for adapting pre-trained LMs to question answering
"""

import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
import numpy as np

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
    
    # Extract only the option letter from the answer
    # Handle cases like "A. bank" -> "A" or just "A"
    if answer and '.' in answer:
        answer_letter = answer.split('.')[0].strip()
    else:
        answer_letter = answer.strip()
    
    return question, options, answer_letter

def parse_options(options_text: str) -> List[Tuple[str, str]]:
    """Parse options text to extract individual options"""
    # Split by common option separators
    option_patterns = [
        r'([A-E])\.\s*([^A-E]+?)(?=\s*[A-E]\.|$)',
        r'([A-E])\s*([^A-E]+?)(?=\s*[A-E]|$)',
        r'([A-E])\s*:\s*([^A-E]+?)(?=\s*[A-E]|$)'
    ]
    
    parsed_options = []
    for pattern in option_patterns:
        matches = re.findall(pattern, options_text, re.IGNORECASE)
        if matches:
            for match in matches:
                option_letter = match[0].upper()
                option_text = match[1].strip()
                if option_text:
                    parsed_options.append((option_letter, option_text))
            break
    
    # If no pattern matches, try simple splitting
    if not parsed_options:
        parts = re.split(r'\s*([A-E])[\.:]\s*', options_text)
        for i in range(1, len(parts), 2):
            if i < len(parts) - 1:
                option_letter = parts[i].upper()
                option_text = parts[i + 1].strip()
                if option_text:
                    parsed_options.append((option_letter, option_text))
    
    return parsed_options

def calculate_option_probability(model, tokenizer, prompt_ids, option_ids) -> float:
    """Calculate conditional log-probability P(option | prompt) using logits method"""
    # Get device from model
    device = next(model.parameters()).device
    
    # Move both tensors to the same device as model
    prompt_ids = prompt_ids.to(device)
    option_ids = option_ids.to(device)
    
    # Set model to eval mode for inference
    model.eval()
    
    with torch.no_grad():
        # Create input: prompt + option (without last token of option)
        input_ids = torch.cat([prompt_ids, option_ids[:, :-1]], dim=1)
        
        # Forward pass to get logits
        outputs = model(input_ids)
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        
        # Calculate log probabilities for the option tokens
        prompt_length = prompt_ids.shape[1]
        option_length = option_ids.shape[1] - 1  # Exclude last token
        
        # Get log probabilities for option tokens
        option_log_probs = []
        for i in range(option_length):
            if prompt_length + i < logits.shape[1]:
                # Get log probability for the next token
                token_log_probs = torch.nn.functional.log_softmax(logits[0, prompt_length + i, :], dim=-1)
                target_token = option_ids[0, i + 1]  # Next token in option
                token_log_prob = token_log_probs[target_token].item()
                option_log_probs.append(token_log_prob)
        
        # Return average log probability, or -inf if no valid tokens
        if option_log_probs:
            avg_log_prob = sum(option_log_probs) / len(option_log_probs)
            # Check for numerical issues
            if torch.isnan(torch.tensor(avg_log_prob)) or torch.isinf(torch.tensor(avg_log_prob)):
                return float('-inf')
            return avg_log_prob
        else:
            return float('-inf')

def calculate_option_probability_optimized(model, tokenizer, prompt_ids, option_text: str) -> float:
    """Calculate conditional log-probability P(option | prompt) using optimized approach with pre-encoded prompt"""
    # Get device from model
    device = next(model.parameters()).device
    
    # Encode option
    option_ids = tokenizer(option_text, return_tensors="pt")["input_ids"]
    
    # Move both tensors to the same device as model
    prompt_ids = prompt_ids.to(device)
    option_ids = option_ids.to(device)
    
    # Concatenate prompt + option (remove last token of option)
    input_ids = torch.cat([prompt_ids, option_ids[:, :-1]], dim=1)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Get log probabilities for option tokens
        # Option starts from the last N tokens of input_ids
        option_length = option_ids.shape[1]
        option_log_probs = log_probs[0, -option_length:, :].gather(
            1, option_ids[0, :, None]
        ).squeeze(-1)
        
        return option_log_probs.mean().item()  # Average log probability

def calculate_all_options_probability_parallel(model, tokenizer, prompt_ids, option_texts: List[str], option_letters: List[str]) -> Dict[str, float]:
    """Calculate probabilities for all options in parallel using stable logits method"""
    if not option_texts:
        return {}
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Encode all options at once
    all_option_ids = []
    max_option_length = 0
    
    for option_text in option_texts:
        option_ids = tokenizer(option_text, return_tensors="pt")["input_ids"]
        all_option_ids.append(option_ids)
        max_option_length = max(max_option_length, option_ids.shape[1])
    
    # Pad all options to the same length for batch processing
    batch_size = len(option_texts)
    padded_option_ids = torch.zeros(batch_size, max_option_length, dtype=torch.long)
    option_lengths = []
    
    for i, option_ids in enumerate(all_option_ids):
        length = option_ids.shape[1]
        option_lengths.append(length)
        padded_option_ids[i, :length] = option_ids[0]
    
    # Move all tensors to the same device as model
    prompt_ids = prompt_ids.to(device)
    padded_option_ids = padded_option_ids.to(device)
    
    # Create batch input: [prompt_ids, option_ids[:-1]] for each option
    batch_input_ids = []
    prompt_length = prompt_ids.shape[1]
    
    for i in range(batch_size):
        # Remove last token from option to avoid shift issues
        option_ids = padded_option_ids[i, :option_lengths[i]-1]
        input_ids = torch.cat([prompt_ids[0], option_ids], dim=0)
        batch_input_ids.append(input_ids)
    
    # Pad all inputs to the same length
    max_input_length = max(len(ids) for ids in batch_input_ids)
    batch_input_tensor = torch.zeros(batch_size, max_input_length, dtype=torch.long)
    
    for i, input_ids in enumerate(batch_input_ids):
        batch_input_tensor[i, :len(input_ids)] = input_ids
    
    # Move to the same device as model
    batch_input_tensor = batch_input_tensor.to(device)
    
    with torch.no_grad():
        # Process all options in one forward pass
        outputs = model(batch_input_tensor)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Calculate probabilities for each option
        option_probabilities = {}
        for i, option_letter in enumerate(option_letters):
            option_length = option_lengths[i] - 1  # Exclude last token
            option_ids = padded_option_ids[i, :option_lengths[i]]
            
            # Calculate log probabilities for option tokens
            option_log_probs = []
            for j in range(option_length):
                if prompt_length + j < logits.shape[1]:
                    # Get log probability for the next token
                    token_log_probs = torch.nn.functional.log_softmax(logits[i, prompt_length + j, :], dim=-1)
                    target_token = option_ids[j + 1]  # Next token in option
                    token_log_prob = token_log_probs[target_token].item()
                    option_log_probs.append(token_log_prob)
            
            # Calculate average log probability
            if option_log_probs:
                avg_log_prob = sum(option_log_probs) / len(option_log_probs)
                # Check for numerical issues
                if torch.isnan(torch.tensor(avg_log_prob)) or torch.isinf(torch.tensor(avg_log_prob)):
                    option_probabilities[option_letter] = float('-inf')
                else:
                    option_probabilities[option_letter] = avg_log_prob
            else:
                option_probabilities[option_letter] = float('-inf')
    
    return option_probabilities

def select_best_option_parallel(model, tokenizer, question: str, options: str) -> Tuple[str, Dict[str, float]]:
    """Select the best option using GPU parallel processing for all options at once"""

    prompt = f"""Here are some examples of question answering:

            Example 1:
            question: What is the capital of France? | options: A. London B. Paris C. Berlin D. Madrid E.New York | answer: B

            Example 2:
            question: Which planet is closest to the Sun? | options: A. Venus B. Earth C. Mercury D. Mars E.Jupiter | answer: C

            Example 3:
            question: What is 2 + 2? | options: A. 3 B. 4 C. 5 D. 6 | answer: B

            Now answer this question:
            question: {question} | options: {options} | answer: """
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Encode prompt once
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_ids = prompt_ids.to(device)
    
    # Parse individual options
    parsed_options = parse_options(options)
    
    if not parsed_options:
        return "", {}
    
    # Prepare all options for batch processing
    option_texts = []
    option_letters = []
    for option_letter, option_text in parsed_options:
        full_option = " " + option_letter + ". " + option_text
        option_texts.append(full_option)
        option_letters.append(option_letter)
    
    # Calculate probabilities for all options in parallel
    option_probabilities = calculate_all_options_probability_parallel(
        model, tokenizer, prompt_ids, option_texts, option_letters
    )
    
    # Validate probabilities
    option_probabilities = validate_probabilities(option_probabilities)
    
    # Find the option with maximum probability
    if option_probabilities:
        # Filter out -inf values
        valid_probs = {k: v for k, v in option_probabilities.items() if v != float('-inf')}
        if valid_probs:
            best_option = max(valid_probs.items(), key=lambda x: x[1])
            return best_option[0], option_probabilities
        else:
            # If all probabilities are -inf, return the first option
            return option_letters[0] if option_letters else "", option_probabilities
    
    return "", {}

def select_best_option(model, tokenizer, question: str, options: str) -> Tuple[str, Dict[str, float]]:
    """Select the best option using paper's method with [start], [sep], [end] tokens"""
    
    # Parse individual options
    parsed_options = parse_options(options)
    
    if not parsed_options:
        return "", {}
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Calculate probabilities for each option using paper's method
    option_probabilities = {}
    
    for option_letter, option_text in parsed_options:
        # Use paper's format: [start] question [sep] answer [end]
        input_text = f"[start] {question} [sep] {option_text} [end]"
        
        # Encode the input
        input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(device)
        
        # Calculate probability of [end] token given the context
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # [batch, seq_len, vocab_size]
            
            # Find the position of [end] token
            end_token_id = tokenizer.encode("[end]", add_special_tokens=False)[0]
            
            # Get log probability of [end] token at the last position
            last_logits = logits[0, -1, :]  # Last position logits
            log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)
            end_token_log_prob = log_probs[end_token_id].item()
            
            option_probabilities[option_letter] = end_token_log_prob
    
    # Validate probabilities
    option_probabilities = validate_probabilities(option_probabilities)
    
    # Find the option with maximum probability
    if option_probabilities:
        # Filter out -inf values
        valid_probs = {k: v for k, v in option_probabilities.items() if v != float('-inf')}
        if valid_probs:
            best_option = max(valid_probs.items(), key=lambda x: x[1])
            return best_option[0], option_probabilities
        else:
            # If all probabilities are -inf, return the first option
            return option_letters[0] if option_letters else "", option_probabilities
    
    return "", {}

def validate_probabilities(option_probabilities: Dict[str, float]) -> Dict[str, float]:
    """Validate and clean probability values to ensure numerical stability"""
    if not option_probabilities:
        return {}
    
    # Check for numerical issues and clean up
    cleaned_probabilities = {}
    valid_probs = []
    
    for option, prob in option_probabilities.items():
        # Check for NaN or infinite values
        if torch.isnan(torch.tensor(prob)) or torch.isinf(torch.tensor(prob)):
            cleaned_probabilities[option] = float('-inf')
        else:
            cleaned_probabilities[option] = prob
            if prob != float('-inf'):
                valid_probs.append(prob)
    
    # If all probabilities are -inf, return original
    if not valid_probs:
        return option_probabilities
    
    # Check for duplicate probabilities (indicates numerical issues)
    unique_probs = set(valid_probs)
    if len(unique_probs) < len(valid_probs):
        print(f"Warning: Found duplicate probabilities, this may indicate numerical issues")
        # Add small noise to break ties
        for option in cleaned_probabilities:
            if cleaned_probabilities[option] != float('-inf'):
                cleaned_probabilities[option] += torch.rand(1).item() * 1e-6
    
    return cleaned_probabilities

def normalize_answer(answer: str) -> str:
    """Normalize answer format to ensure consistent comparison"""
    if not answer:
        return ""
    
    # Remove any extra whitespace
    answer = answer.strip()
    
    # Extract only the option letter (A, B, C, D, E)
    if '.' in answer:
        # Handle format like "A. bank" -> "A"
        answer_letter = answer.split('.')[0].strip()
    else:
        # Handle format like "A" -> "A"
        answer_letter = answer.strip()
    
    # Ensure it's a valid option letter
    if answer_letter.upper() in ['A', 'B', 'C', 'D', 'E']:
        return answer_letter.upper()
    else:
        return answer_letter  # Return as is if not a standard option

def test_model_with_probability(model_path: str, test_file: str, output_file: str = None, use_parallel: bool = True):
    """Test the fine-tuned model or HF pre-trained model on test sentences using probability scoring"""
    
    print(f"Loading model from {model_path}...")
    try:
        # Check if this is a known HF model name
        if model_path in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            print(f"Loading pre-trained model from Hugging Face: {model_path}")
        else:
            print(f"Loading fine-tuned model from local path: {model_path}")
        
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        # Add special tokens for paper's method if they don't exist
        special_tokens = ["[start]", "[sep]", "[end]"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        # Resize model embeddings if new tokens were added
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
            print(f"Model embeddings resized to accommodate new tokens. New vocab size: {len(tokenizer)}")
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"Model moved to GPU: {torch.cuda.get_device_name()}")
        else:
            print("Using CPU for inference")
        
        print("Model loaded successfully!")
        print(f"Special tokens added: {special_tokens}")
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
    
    # Choose the appropriate function based on parallel flag
    if use_parallel:
        print(f"Testing model on {total_count} sentences using PARALLEL probability scoring...")
        select_function = select_best_option_parallel
    else:
        print(f"Testing model on {total_count} sentences using SEQUENTIAL probability scoring...")
        select_function = select_best_option
    
    for i, sentence_data in tqdm(enumerate(test_sentences), desc="Testing", unit="test"):
        text = sentence_data['text']
        question, options, ground_truth_answer = parse_question_text(text)
        
        # Select best option based on probability using chosen function
        selected_answer, option_probabilities = select_function(model, tokenizer, question, options)
        
        # Validate and clean probabilities
        option_probabilities = validate_probabilities(option_probabilities)
        
        # Normalize answers for consistent comparison
        normalized_selected = normalize_answer(selected_answer)
        normalized_ground_truth = normalize_answer(ground_truth_answer)
        
        # Check if answer is correct
        is_correct = normalized_selected == normalized_ground_truth
        
        # Debug: Print answer comparison
        """if i < 3:  # Only for first 3 tests
            print(f"Debug - Test {i+1}:")
            print(f"  Raw selected: '{selected_answer}' -> Normalized: '{normalized_selected}'")
            print(f"  Raw ground truth: '{ground_truth_answer}' -> Normalized: '{normalized_ground_truth}'")
            print(f"  Is correct: {is_correct}")"""
        
        if is_correct:
            correct_count += 1
        
        result = {
            'index': sentence_data.get('index', len(results)),
            'question': question,
            'options': options,
            'ground_truth': normalized_ground_truth,  # Store normalized version
            'selected_answer': normalized_selected,   # Store normalized version
            'option_probabilities': option_probabilities,
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
    print(f"\nTest Summary (Paper's Method):")
    print(f"Total tests: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Selected answers: {len([r for r in results if r['selected_answer']])}")
    
    # Print some example probability distributions
    print(f"\nExample probability distributions (P([end] | [start] question [sep] answer)):")
    for i, result in enumerate(results[:3]):  # Show first 3 examples
        if result['option_probabilities']:
            print(f"\nTest {i+1}:")
            print(f"Question: {result['question'][:50]}...")
            print(f"Ground truth: {result['ground_truth']}")
            print(f"Selected: {result['selected_answer']}")
            print(f"End token probabilities: {result['option_probabilities']}")
            
            # Show which option has highest probability
            if result['option_probabilities']:
                valid_probs = {k: v for k, v in result['option_probabilities'].items() if v != float('-inf')}
                if valid_probs:
                    best_option = max(valid_probs.items(), key=lambda x: x[1])
                    print(f"Best option: {best_option[0]} (log_prob: {best_option[1]:.4f})")
                    
                    # Show if selection matches ground truth
                    if result['selected_answer'] == result['ground_truth']:
                        print(f"✓ CORRECT: Selected answer matches ground truth")
                    else:
                        print(f"✗ WRONG: Selected {result['selected_answer']}, expected {result['ground_truth']}")
                        
                        # Show ground truth probability
                        gt_prob = result['option_probabilities'].get(result['ground_truth'], float('-inf'))
                        print(f"Ground truth '{result['ground_truth']}' probability: {gt_prob:.4f}")
                        
                        # Show probability ranking
                        sorted_probs = sorted(valid_probs.items(), key=lambda x: x[1], reverse=True)
                        print(f"Probability ranking: {[(k, f'{v:.4f}') for k, v in sorted_probs]}")
                else:
                    print(f"Warning: All probabilities are -inf (numerical issues)")

def main():
    parser = argparse.ArgumentParser(description='Test fine-tuned model or HF pre-trained model on CSQA sentences with probability scoring')
    parser.add_argument('--model_path', type=str, 
                       default='finetune/csqa100%',
                       help='Path to the fine-tuned model or HF model name (e.g., gpt2)')
    parser.add_argument('--test_file', type=str, 
                       default='csqa_test_sentences.json',
                       help='Path to the test sentences JSON file')
    parser.add_argument('--output_file', type=str, 
                       default=None,
                       help='Path to save test results (auto-generated if not specified)')
    parser.add_argument('--sequential', action='store_true',
                       help='Use sequential processing instead of parallel GPU processing')
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not specified
    if args.output_file is None:
        if args.model_path in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            args.output_file = f'csqa_test_results/csqa_{args.model_path}_hf_results.json'
        else:
            args.output_file = f'csqa_test_results/csqa_{args.model_path.split("/")[-1]}_results.json'
    
    use_parallel = not args.sequential
    print(f"Testing model with probability scoring...")
    print(f"Model: {args.model_path}")
    print(f"Processing mode: {'PARALLEL' if use_parallel else 'SEQUENTIAL'}")
    print(f"Output file: {args.output_file}")
    
    test_model_with_probability(args.model_path, args.test_file, args.output_file, use_parallel)
    print("Testing completed!")

if __name__ == "__main__":
    main()