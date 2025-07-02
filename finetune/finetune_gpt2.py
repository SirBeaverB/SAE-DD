import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import json
import os
from pathlib import Path
import glob

def load_high_humor_data():
    """
    加载oasst1_humor文件夹中所有high_humor_0.061打头的文件
    """
    humor_dir = Path("oasst1_humor")
    humor_files = glob.glob(str(humor_dir / "high_humor_0.150*"))
    
    all_texts = []
    for file_path in humor_files:
        print(f"Loading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 数据格式是包含text字段的字典列表
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'text' in item:
                        all_texts.append(item['text'])
    
    print(f"Loaded {len(all_texts)} high humor texts")
    return all_texts

def prepare_dataset(texts, tokenizer, max_length=512):
    """
    准备训练数据集
    """
    def tokenize_function(examples):
        # 对文本进行tokenize
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # 设置labels为input_ids（用于语言模型训练）
        tokenized['labels'] = tokenized['input_ids'].clone()
        
        return tokenized
    
    # 创建数据集
    dataset = Dataset.from_dict({'text': texts})
    
    # 应用tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def save_model_in_chunks(model, tokenizer, output_dir, max_file_size=50*1024*1024):
    """
    将模型保存为小于50MB的文件块，以便上传到GitHub
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 保存tokenizer
    tokenizer.save_pretrained(output_path)
    
    # 保存模型配置
    model.config.save_pretrained(output_path)
    
    # 保存模型权重（使用safetensors格式，更安全且支持分片）
    model.save_pretrained(
        output_path,
        max_shard_size="50MB",  # 每个分片最大50MB
        safe_serialization=True  # 使用safetensors格式
    )
    
    print(f"Model saved to {output_path}")
    print("Files created:")
    for file in output_path.glob("*"):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name}: {size_mb:.2f} MB")

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading high humor data...")
    texts = load_high_humor_data()
    
    if not texts:
        print("No texts found! Please check the data files.")
        return
    
    # 加载预训练的GPT-2模型和tokenizer
    print("Loading GPT-2 model and tokenizer...")
    model_name = "gpt2"  # 使用基础GPT-2模型
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 准备数据集
    print("Preparing dataset...")
    dataset = prepare_dataset(texts, tokenizer)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./gpt2_humor_finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=100,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        report_to=None,
        run_name="gpt2_humor_finetune_0.150",  # 设置一个不同的run_name
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("Starting fine-tuning...")
    trainer.train()
    
    # 保存模型
    print("Saving model...")
    save_model_in_chunks(model, tokenizer, "./gpt2_humor_finetuned")
    
    # 测试生成
    print("\nTesting generation...")
    model.eval()
    test_prompt = "This is funny because"
    
    inputs = tokenizer.encode(test_prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=100,
            num_return_sequences=3,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print("Generated samples:")
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"{i+1}. {generated_text}")

if __name__ == "__main__":
    main()