#!/usr/bin/env python3
"""
GPT-2 Fine-tuning script for CSQA sentences
"""

import json
import os
import sys

# Set custom temporary directory before importing torch
import tempfile
tempfile.tempdir = os.path.join(os.getcwd(), 'tmp')
os.makedirs(tempfile.tempdir, exist_ok=True)
os.environ['TMPDIR'] = tempfile.tempdir

import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    GPT2Config,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from dataclasses import dataclass
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
        
        if not sentences:
            print(f"Warning: No sentences found in {file_path}")
            return []
            
        print(f"Loaded {len(sentences)} sentences from {file_path}")
        return sentences
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return []

def prepare_dataset(sentences: List[str], tokenizer, max_length: int = 512, validation_split: float = 0.1) -> Tuple[Dataset, Dataset]:
    """Prepare dataset for training with validation split using pre-tokenization"""
    
    if not sentences:
        raise ValueError("No sentences provided for dataset preparation")
    
    print("Pre-tokenizing all sentences to avoid repeated tokenization...")
    
    # Pre-tokenize all sentences at once with better error handling
    try:
        all_tokenized = tokenizer(
            sentences,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,  # Return as list of lists
            verbose=False,
            add_special_tokens=True,  # Ensure special tokens are added
            return_attention_mask=True  # Explicitly request attention mask
        )
        
        # Validate tokenization results
        if not all_tokenized['input_ids'] or len(all_tokenized['input_ids']) != len(sentences):
            raise ValueError("Tokenization failed or produced incorrect number of sequences")
        
        # Create dataset with pre-tokenized data
        dataset_dict = {
            'input_ids': all_tokenized['input_ids'],
            'attention_mask': all_tokenized['attention_mask']
        }
        
        # Split into train and validation
        dataset = Dataset.from_dict(dataset_dict).train_test_split(
            test_size=validation_split, 
            seed=42
        )
        
        train_dataset = dataset['train']
        eval_dataset = dataset['test']
        
        print(f"Pre-tokenization completed successfully!")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Validation samples: {len(eval_dataset)}")
        print(f"  - Max sequence length: {max_length}")
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        print(f"Error during pre-tokenization: {e}")
        print("Falling back to traditional tokenization method...")
        
        # Fallback method
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None,
                return_attention_mask=True
            )
        
        # Create dataset and split
        dataset = Dataset.from_dict({'text': sentences}).train_test_split(
            test_size=validation_split, 
            seed=42
        )
        
        # Tokenize datasets
        train_dataset = dataset['train'].map(
            tokenize_function, 
            batched=True, 
            remove_columns=['text']
        )
        
        eval_dataset = dataset['test'].map(
            tokenize_function, 
            batched=True, 
            remove_columns=['text']
        )
        
        return train_dataset, eval_dataset

@dataclass
class CausalCollator:
    """Simplified data collator for causal language modeling"""
    tokenizer: GPT2Tokenizer
    pad_to_multiple_of: int = None
    
    def __call__(self, features):
        # Pad the batch to the same length
        batch = self.tokenizer.pad(
            features, 
            padding=True, 
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        
        # Get basic tensors
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # Create labels for causal LM (shifted by 1)
        labels = input_ids.clone()
        
        # Set padding tokens to -100 (ignore in loss calculation)
        labels[attention_mask == 0] = -100
        
        # Return simplified batch without position_ids - let GPT-2 handle it internally
        return {
            "labels": labels,
            "attention_mask": attention_mask,
            "input_ids": input_ids
        }

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
    gradient_accumulation_steps: int = 4,
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.001,
    resume_from_checkpoint: str = None
):
    """Train the model with safe multi-GPU support and early stopping"""
    
    # Use simplified causal collator
    data_collator = CausalCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=None  # Disable padding to multiple for stability
    )
    
    # Safe multi-GPU configuration
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Enable multi-GPU if requested and available
    if num_gpus > 1 and use_multi_gpu:
        print(f"Enabling multi-GPU training with {num_gpus} GPUs")
        # Use DataParallel for safer multi-GPU training
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)
            print("Model wrapped with DataParallel for multi-GPU training")
    else:
        print("Using single GPU training for maximum stability")
        use_multi_gpu = False
    
    # Calculate effective batch size
    effective_batch_size = batch_size * (num_gpus if use_multi_gpu else 1) * gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch_size} (per_device: {batch_size}, GPUs: {num_gpus if use_multi_gpu else 1}, accumulation: {gradient_accumulation_steps})")
    
    # Training arguments with early stopping and better validation
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        save_total_limit=5,  # Keep more checkpoints for safety
        save_strategy="steps",  # Explicitly set save strategy
        save_safetensors=False,  # Use pytorch format for better compatibility
        prediction_loss_only=True,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=100,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        load_best_model_at_end=False,  # Disable to prevent state corruption
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,  # Keep disabled for stability
        remove_unused_columns=False,
        # Safe multi-GPU settings
        dataloader_num_workers=0,  # Disable workers for maximum stability
        fp16=False,  # Keep disabled for stability
        dataloader_drop_last=False,
        # Enhanced logging
        logging_dir=f"{output_dir}/logs",
        logging_first_step=True,
        logging_strategy="steps",
        # Basic stability settings
        max_grad_norm=1.0,  # Gradient clipping
        weight_decay=0.01,  # L2 regularization
        # Disable wandb
        report_to=None,
        # Checkpoint stability
        save_on_each_node=True,  # Save on each node for distributed training
        # Memory management
        dataloader_prefetch_factor=None,  # Disable prefetching
        group_by_length=False,  # Disable length grouping
    )
    
    # Initialize trainer with custom callback and early stopping
    log_file = os.path.join(output_dir, "training.log")
    loss_callback = LossLoggingCallback(
        patience=early_stopping_patience, 
        min_delta=early_stopping_threshold,
        log_file=log_file
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Using separate validation dataset
        callbacks=[loss_callback],
    )
    
    # Train the model
    print("\n" + "="*60)
    print(" STARTING TRAINING")
    print("="*60)
    print(f" Training Configuration:")
    print(f"   - Epochs: {num_epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"   - Multi-GPU: {'Enabled' if use_multi_gpu else 'Disabled'}")
    print(f"   - Effective batch size: {batch_size * (torch.cuda.device_count() if use_multi_gpu else 1) * gradient_accumulation_steps}")
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(eval_dataset)}")
    print(f"   - Evaluation every: {eval_steps} steps")
    print(f"   - Early stopping patience: {early_stopping_patience}")
    print(f"   - Early stopping threshold: {early_stopping_threshold}")
    print(f"   - Logging every: 100 steps")
    print("="*60)
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Training completed
    print("\n" + "="*60)
    print(" TRAINING COMPLETED")
    print("="*60)
    
    # Get final training metrics
    final_metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    if 'train_loss' in final_metrics:
        print(f" Final Training Loss: {final_metrics['train_loss']:.6f}")
    if 'eval_loss' in final_metrics:
        print(f"Final Validation Loss: {final_metrics['eval_loss']:.6f}")
    
    # Display training statistics
    if loss_callback.eval_loss_history:
        print(f"\nTraining Statistics:")
        print(f"  - Total epochs completed: {loss_callback.epoch}")
        print(f"  - Total steps: {loss_callback.step}")
        print(f"  - Best validation loss: {loss_callback.best_eval_loss:.6f}")
        print(f"  - Training stopped early: {'Yes' if loss_callback.patience_counter >= loss_callback.patience else 'No'}")
        
        if len(loss_callback.eval_loss_history) > 1:
            final_eval_loss = loss_callback.eval_loss_history[-1]
            best_eval_loss = loss_callback.best_eval_loss
            improvement = best_eval_loss - final_eval_loss
            print(f"  - Overall improvement: {improvement:.6f}")
    
    print(f"Model saved to {output_dir}")
    print(f"Training log saved to: {log_file}")
    print("="*60)
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

class LossLoggingCallback(TrainerCallback):
    """Custom callback to log detailed loss information during training with early stopping"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001, log_file: str = None):
        self.step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        self.last_loss = None
        self.loss_history = []
        self.eval_loss_history = []
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        
        # Setup logging to file
        self.log_file = log_file
        if self.log_file:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Write header to log file
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("Training Log\n")
                f.write("=" * 50 + "\n")
                f.write(f"Patience: {patience}, Min Delta: {min_delta}\n")
                f.write("=" * 50 + "\n\n")
    
    def log_to_file(self, message: str):
        """Write message to log file"""
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each step"""
        pass
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        print(f" Training started! Will log loss every 100 steps.")
        print(f"   Total steps per epoch: {len(state.train_dataloader) if hasattr(state, 'train_dataloader') else 'Unknown'}")
        print(f"   Logging strategy: Every 100 steps")
        print(f"   Early stopping patience: {self.patience} evaluations")
        print(f"   Early stopping min delta: {self.min_delta}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step"""
        self.step = state.global_step
        
        # Only log every 100 steps to reduce output
        if self.step % 100 == 0:
            # Try to get loss from different sources
            current_loss = None
            
            # Method 1: Try to get from log_history
            if hasattr(state, 'log_history') and state.log_history:
                latest_log = state.log_history[-1]
                if 'loss' in latest_log:
                    current_loss = latest_log['loss']
            
            # Method 2: Try to get from trainer's state
            if current_loss is None and hasattr(state, 'train_loss'):
                current_loss = state.train_loss
            
            # Log the loss
            if current_loss is not None:
                self.last_loss = current_loss
                self.loss_history.append(current_loss)
                print(f"Step {self.step:6d} | Loss: {current_loss:.6f}")
                self.log_to_file(f"Step {self.step:6d} | Loss: {current_loss:.6f}")
            else:
                print(f"Step {self.step:6d} | Training... (Loss not available)")
                self.log_to_file(f"Step {self.step:6d} | Training... (Loss not available)")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation with early stopping logic"""
        if metrics:
            eval_loss = metrics.get('eval_loss', float('inf'))
            self.eval_loss_history.append(eval_loss)
            
            # Check if this is a new best validation loss
            if eval_loss < self.best_eval_loss - self.min_delta:
                self.best_eval_loss = eval_loss
                self.patience_counter = 0
                print(f"New best validation loss: {eval_loss:.6f} (Step {self.step})")
                self.log_to_file(f"New best validation loss: {eval_loss:.6f} (Step {self.step})")
            else:
                self.patience_counter += 1
                print(f"Validation loss: {eval_loss:.6f} (Best: {self.best_eval_loss:.6f}, Patience: {self.patience_counter}/{self.patience})")
                self.log_to_file(f"Validation loss: {eval_loss:.6f} (Best: {self.best_eval_loss:.6f}, Patience: {self.patience_counter}/{self.patience})")
            
            # Check if we should stop early 
            if self.patience_counter >= self.patience:
                if hasattr(control, 'should_training_stop'):
                    control.should_training_stop = True
                print(f"Early stopping triggered! No improvement for {self.patience} evaluations.")
                print(f"Best validation loss: {self.best_eval_loss:.6f}")
                self.log_to_file(f"Early stopping triggered! No improvement for {self.patience} evaluations.")
                self.log_to_file(f"Best validation loss: {self.best_eval_loss:.6f}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        print(f"Training started! Early stopping patience: {self.patience}")
        self.log_to_file(f"Training started! Early stopping patience: {self.patience}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        print(f"Training completed! Final step: {self.step}")
        print(f"   Best validation loss: {self.best_eval_loss:.6f}")
        self.log_to_file(f"Training completed! Final step: {self.step}")
        self.log_to_file(f"Best validation loss: {self.best_eval_loss:.6f}")
    
    def on_evaluate_begin(self, args, state, control, **kwargs):
        """Called before evaluation begins"""
        print(f"\nStarting evaluation at step {self.step}")
        self.log_to_file(f"Starting evaluation at step {self.step}")
    
    def on_evaluate_end(self, args, state, control, **kwargs):
        """Called after evaluation is complete - ensure model is in train mode"""
        # Force model back to training mode after evaluation
        if hasattr(state, 'model'):
            try:
                state.model.train()
                print(f"Model restored to training mode")
                self.log_to_file(f"Model restored to training mode")
            except Exception as e:
                print(f"Failed to restore model to train mode: {e}")
                self.log_to_file(f"Failed to restore model to train mode: {e}")
        else:
            print(f"No model found in state")
            self.log_to_file(f"No model found in state")
        
        # Clean up memory after evaluation
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Failed to clear GPU memory: {e}")
                self.log_to_file(f"Failed to clear GPU memory: {e}")
        
        print(f"Evaluation completed, resuming training")
        self.log_to_file(f"Evaluation completed, resuming training")
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each step - monitor training recovery"""
        '''# Only log every 100 steps to reduce output
        if self.step > 0 and self.step % 100 == 0:
            print(f"Step {self.step} - Training mode: {state.model.training if hasattr(state, 'model') else 'Unknown'}")'''
        pass
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch"""
        self.epoch += 1
        print(f"\n Starting Epoch {self.epoch}/{args.num_train_epochs}")
        print(f"   Current step: {self.step} | Loss history length: {len(self.loss_history)}")
        if self.loss_history:
            print(f"   Last loss: {self.loss_history[-1]:.6f}")
        if self.eval_loss_history:
            print(f"   Last validation loss: {self.eval_loss_history[-1]:.6f}")
        
        self.log_to_file(f"\nStarting Epoch {self.epoch}/{args.num_train_epochs}")
        self.log_to_file(f"Current step: {self.step} | Loss history length: {len(self.loss_history)}")
        if self.loss_history:
            self.log_to_file(f"Last loss: {self.loss_history[-1]:.6f}")
        if self.eval_loss_history:
            self.log_to_file(f"Last validation loss: {self.eval_loss_history[-1]:.6f}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        if hasattr(state, 'log_history') and state.log_history:
            # Get average loss for this epoch
            epoch_logs = [log for log in state.log_history if 'loss' in log]
            if epoch_logs:
                avg_loss = sum(log['loss'] for log in epoch_logs) / len(epoch_logs)
                print(f" Epoch {self.epoch} completed | Average Loss: {avg_loss:.6f}")
                print(f"   Total steps: {self.step}")
                self.log_to_file(f"Epoch {self.epoch} completed | Average Loss: {avg_loss:.6f}")
                self.log_to_file(f"Total steps: {self.step}")
        


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def check_gpu_status():
    """Check GPU availability and status with enhanced device detection"""
    device = get_device()
    
    if device.type == "cuda":
        num_gpus = torch.cuda.device_count()
        print(f"CUDA is available. Found {num_gpus} GPU(s):")
        
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            gpu_capability = torch.cuda.get_device_capability(i)
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB, Compute Capability: {gpu_capability[0]}.{gpu_capability[1]})")
        
        # Set default device
        torch.cuda.set_device(0)
        print(f"Using GPU 0 as default device")
        return True
        
    elif device.type == "mps":
        print("MPS (Apple Silicon) is available")
        return True
        
    else:
        print("CUDA/MPS is not available. Training will use CPU (very slow!)")
        print("Consider using a GPU for better performance")
        return False

def generate_sample_text(model, tokenizer, prompt: str = "question:", max_length: int = 100):
    """Generate sample text using the fine-tuned model with device flexibility"""
    # Get device from model
    device = next(model.parameters()).device
    
    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Create attention mask
    attention_mask = torch.ones_like(inputs)
    
    # Move to same device as model
    inputs = inputs.to(device)
    attention_mask = attention_mask.to(device)
    
    # Set model to eval mode for generation
    model.eval()
    
    with torch.no_grad():
        # Use standard generation without autocast for better stability
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
            early_stopping=True,
            # Enhanced generation parameters
            no_repeat_ngram_size=3,  # Prevent repetition
            length_penalty=1.0,      # Balanced length
            num_beams=1             # Greedy decoding for speed
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 on CSQA sentences')
    parser.add_argument('--data_file', type=str, 
                       default='csqa_naive_distill/pure_sentences/pure_top_100%_sentences.json',
                       help='Path to the sentences JSON file')
    parser.add_argument('--output_dir', type=str, 
                       default='finetune/csqa100%',
                       help='Output directory for the fine-tuned model')
    parser.add_argument('--model_name', type=str, 
                       default='gpt2',
                       help='Base model name (gpt2, gpt2-medium, gpt2-large, gpt2-xl)')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs (recommended: 10-20 for fine-tuning)')
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
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='Number of evaluations to wait before early stopping')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.001,
                       help='Minimum change in validation loss to be considered improvement')
    parser.add_argument('--eval_steps', type=int, default=300,
                       help='Evaluate every N steps (higher = more stable)')
    parser.add_argument('--resume_training', action='store_true',
                       help='Resume training from the latest checkpoint if available')
    parser.add_argument('--force_restart', action='store_true',
                       help='Force restart training even if checkpoint exists')
    
    args = parser.parse_args()
    
    # Check GPU status and configure multi-GPU
    gpu_available = check_gpu_status()
    
    # Configure multi-GPU based on user preference
    if gpu_available and torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs")
        if args.use_multi_gpu:
            print("Multi-GPU training enabled for speed")
        else:
            print("Multi-GPU training disabled. Use --use_multi_gpu to enable for faster training")
    elif gpu_available:
        print("Single GPU detected")
        args.use_multi_gpu = False
    else:
        print("No GPU detected, using CPU (very slow!)")
        args.use_multi_gpu = False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading sentences from {args.data_file}...")
    sentences = load_sentences_data(args.data_file)
    
    if not sentences:
        print("No sentences loaded. Exiting.")
        return
    
    print(f"Loading base model: {args.model_name}...")
    # Load model and tokenizer
    try:
        model = GPT2LMHeadModel.from_pretrained(args.model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Error loading model {args.model_name}: {e}")
        print("Please check if the model name is correct and accessible.")
        return
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Enhanced model initialization for numerical stability
    print(" Applying numerical stability enhancements...")
    
    # Don't reinitialize pre-trained weights - this can cause internal state corruption
    # The model already has good initialization from pre-training
    print(" Keeping pre-trained weights for stability")
    
    # Disable gradient checkpointing for better stability
    if hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_disable()
            print("Gradient checkpointing disabled for stability")
        except Exception as e:
            print(f"Warning: Could not disable gradient checkpointing: {e}")
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Model moved to GPU: {torch.cuda.get_device_name()}")
        print("Using full precision for better numerical stability")
        
        # Configure multi-GPU if requested
        if args.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Setting up multi-GPU training with {torch.cuda.device_count()} GPUs")
            # Use DataParallel for safer multi-GPU training
            if not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)
                print("Model wrapped with DataParallel for multi-GPU training")
            
            # Set default device to first GPU
            torch.cuda.set_device(0)
            print(f"Primary GPU set to: {torch.cuda.get_device_name(0)}")
    
    print("Model initialization completed")
    
    print("Preparing dataset...")
    try:
        train_dataset, eval_dataset = prepare_dataset(sentences, tokenizer, args.max_length, args.validation_split)
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        print("Please check your data format and try again.")
        return
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    try:
        # Unwrap model from DataParallel if needed for trainer
        training_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        
        # Check if we can resume from a previous checkpoint
        resume_from_checkpoint = None
        if args.resume_training and not args.force_restart:
            # Look for the latest checkpoint by finding the highest step number
            checkpoint_dirs = []
            if os.path.exists(args.output_dir):
                for item in os.listdir(args.output_dir):
                    if item.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, item)):
                        try:
                            step_num = int(item.split("-")[1])
                            checkpoint_dir = os.path.join(args.output_dir, item)
                            
                            # Check if checkpoint is complete and valid
                            required_files = [
                                'pytorch_model.bin',
                                'config.json',
                                'trainer_state.json'
                            ]
                            
                            missing_files = []
                            for file in required_files:
                                if not os.path.exists(os.path.join(checkpoint_dir, file)):
                                    missing_files.append(file)
                            
                            if not missing_files:
                                # Checkpoint is complete
                                checkpoint_dirs.append((step_num, item))
                                print(f"✓ Checkpoint {item} is complete")
                            else:
                                print(f"✗ Checkpoint {item} is incomplete (missing: {missing_files})")
                        except ValueError:
                            continue
            
            if checkpoint_dirs:
                # Sort by step number and get the latest
                checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
                latest_checkpoint = checkpoint_dirs[0][1]
                checkpoint_dir = os.path.join(args.output_dir, latest_checkpoint)
                resume_from_checkpoint = checkpoint_dir
                print(f"Found {len(checkpoint_dirs)} valid checkpoint(s): {[f'checkpoint-{step}' for step, _ in checkpoint_dirs]}")
                print(f"Resuming training from latest checkpoint: {resume_from_checkpoint}")
            else:
                print("No valid checkpoints found, starting fresh training")
        elif args.force_restart:
            print("Force restart: ignoring existing checkpoints")
        
        train_model(
            model=training_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=100,  # Add default warmup steps
            use_multi_gpu=args.use_multi_gpu,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            early_stopping_patience=args.early_stopping_patience, 
            early_stopping_threshold=args.early_stopping_threshold,
            eval_steps=args.eval_steps,
            resume_from_checkpoint=resume_from_checkpoint
        )
    except Exception as e:
        print(f"Error during training: {e}")
        print("Training failed. Please check your configuration and try again.")
        return
    
    # Load the fine-tuned model for testing
    print("Loading fine-tuned model for testing...")
    try:
        fine_tuned_model = GPT2LMHeadModel.from_pretrained(args.output_dir)
        fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(args.output_dir)
        
        # Generate some sample text
        print("\nGenerating sample text with fine-tuned model:")
        sample_text = generate_sample_text(fine_tuned_model, fine_tuned_tokenizer, "question:")
        print(f"Generated: {sample_text}")
        
    except Exception as e:
        print(f"Warning: Could not test the fine-tuned model: {e}")
        print("Model training completed successfully, but testing failed.")
    
    print(f"\nFine-tuning completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()