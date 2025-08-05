#!/usr/bin/env python3
"""
GPT-2 Fine-tuning script for CSQA sentences
"""

import json
import os
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    GPT2Config,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np
from typing import List, Dict, Any, Tuple
import argparse
from tqdm import tqdm

def load_sentences_data(file_path: str) -> List[str]:
    """Load sentences from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sentences = []
        if 'sentences' in data:
            for sentence_info in data['sentences']:
                if isinstance(sentence_info, str):
                    # If sentences are stored as strings directly
                    sentences.append(sentence_info)
                elif isinstance(sentence_info, dict):
                    # If sentences are stored as objects with text field
                    if 'text' in sentence_info:
                        sentences.append(sentence_info['text'])
                    elif 'question' in sentence_info and 'options' in sentence_info and 'answer' in sentence_info:
                        # Reconstruct the sentence from separated parts
                        text = f"question: {sentence_info['question']} | options: {sentence_info['options']} | answer: {sentence_info['answer']}"
                        sentences.append(text)
        
        print(f"Loaded {len(sentences)} sentences from {file_path}")
        return sentences
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return []

def prepare_dataset(sentences: List[str], tokenizer, max_length: int = 512, validation_split: float = 0.1) -> Tuple[Dataset, Dataset]:
    """Prepare dataset for training with validation split"""
    
    def tokenize_function(examples):
        # Tokenize the sentences
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Set labels to input_ids for language modeling
        tokenized['labels'] = tokenized['input_ids'].clone()
        
        # Create attention mask
        tokenized['attention_mask'] = torch.ones_like(tokenized['input_ids'])
        
        return tokenized
    
    # Create dataset
    dataset_dict = {'text': sentences}
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split into train and validation
    dataset = dataset.train_test_split(test_size=validation_split, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    # Tokenize the datasets
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_tokenized = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    return train_tokenized, eval_tokenized

def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    save_steps: int = 500,
    eval_steps: int = 500,
    warmup_steps: int = 100,
    use_multi_gpu: bool = False,
    gradient_accumulation_steps: int = 4
):
    """Train the model"""
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 uses causal language modeling, not masked
    )
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Training arguments with multi-GPU support
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # Effective batch size = batch_size * num_gpus * gradient_accumulation_steps
        save_steps=save_steps,
        save_total_limit=2,  # Keep only the last 2 checkpoints
        prediction_loss_only=True,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=100,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=True,  # Enable for better GPU utilization
        remove_unused_columns=False,
        # Multi-GPU settings
        dataloader_num_workers=4 if use_multi_gpu else 0,  # Number of workers for data loading
        fp16=True,  # Use mixed precision training for better GPU utilization
        dataloader_drop_last=True,  # Drop incomplete batches
        # Disable wandb
        report_to=None,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Using separate validation dataset
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")

def check_gpu_status():
    """Check GPU availability and status"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA is available. Found {num_gpus} GPU(s):")
        
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Set default device
        torch.cuda.set_device(0)
        print(f"Using GPU 0 as default device")
        return True
    else:
        print("CUDA is not available. Training will use CPU (very slow!)")
        return False

def generate_sample_text(model, tokenizer, prompt: str = "question:", max_length: int = 100):
    """Generate sample text using the fine-tuned model"""
    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Create attention mask
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
            temperature=0.8,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 on CSQA sentences')
    parser.add_argument('--data_file', type=str, 
                       default='csqa_naive_distill/pure_sentences/pure_top_50%_sentences.json',
                       help='Path to the sentences JSON file')
    parser.add_argument('--output_dir', type=str, 
                       default='finetune/csqa50%',
                       help='Output directory for the fine-tuned model')
    parser.add_argument('--model_name', type=str, 
                       default='gpt2',
                       help='Base model name (gpt2, gpt2-medium, gpt2-large, gpt2-xl)')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--use_multi_gpu', action='store_true',
                       help='Enable multi-GPU training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Number of gradient accumulation steps')
    parser.add_argument('--validation_split', type=float, default=0.1,
                       help='Validation split ratio (0.1 = 10% for validation)')
    
    args = parser.parse_args()
    
    # Check GPU status
    gpu_available = check_gpu_status()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading sentences from {args.data_file}...")
    sentences = load_sentences_data(args.data_file)
    
    if not sentences:
        print("No sentences loaded. Exiting.")
        return
    
    print(f"Loading base model: {args.model_name}...")
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Preparing dataset...")
    train_dataset, eval_dataset = prepare_dataset(sentences, tokenizer, args.max_length, args.validation_split)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_multi_gpu=args.use_multi_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Load the fine-tuned model for testing
    print("Loading fine-tuned model for testing...")
    fine_tuned_model = GPT2LMHeadModel.from_pretrained(args.output_dir)
    fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(args.output_dir)
    
    # Generate some sample text
    print("\nGenerating sample text with fine-tuned model:")
    sample_text = generate_sample_text(fine_tuned_model, fine_tuned_tokenizer, "question:")
    print(f"Generated: {sample_text}")
    
    
    print(f"\nFine-tuning completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()

