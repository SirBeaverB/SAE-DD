# GPT-2 Fine-tuning for CSQA Sentences

这个项目用于微调GPT-2模型，使其能够生成类似CSQA（Commonsense Question Answering）格式的问题和答案。

## 功能特点

- 支持从JSON文件加载句子数据
- 自动处理不同格式的句子数据（字符串或对象格式）
- 支持多种GPT-2模型大小（gpt2, gpt2-medium, gpt2-large, gpt2-xl）
- **支持多GPU训练**，自动检测GPU数量和内存
- 支持混合精度训练（FP16）以提高训练速度
- 自动保存检查点和最终模型
- 包含模型测试和文本生成功能

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python finetune_gpt2.py
```

### 自定义参数

```bash
python finetune_gpt2.py \
    --data_file "csqa_naive_distill/pure_sentences/pure_top_90%_sentences.json" \
    --output_dir "gpt2_csqa_model" \
    --model_name "gpt2-medium" \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --max_length 512
```

### 使用脚本运行

```bash
# 单GPU训练
chmod +x run_finetune.sh
./run_finetune.sh

# 多GPU训练
chmod +x run_multi_gpu_finetune.sh
./run_multi_gpu_finetune.sh
```

## 参数说明

- `--data_file`: 输入句子数据文件路径
- `--output_dir`: 微调后模型保存目录
- `--model_name`: 基础模型名称（gpt2, gpt2-medium, gpt2-large, gpt2-xl）
- `--num_epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--max_length`: 最大序列长度
- `--use_multi_gpu`: 启用多GPU训练
- `--gradient_accumulation_steps`: 梯度累积步数（用于模拟更大的批次大小）

## 输入数据格式

支持两种输入格式：

### 1. 字符串格式
```json
{
  "sentences": [
    "question: What is the capital of France? | options: A. Paris, B. London, C. Berlin | answer: A. Paris",
    "question: What color is the sky? | options: A. Blue, B. Red, C. Green | answer: A. Blue"
  ]
}
```

### 2. 对象格式
```json
{
  "sentences": [
    {
      "question": "What is the capital of France?",
      "options": "A. Paris, B. London, C. Berlin",
      "answer": "A. Paris"
    }
  ]
}
```

## 输出

微调完成后，模型将保存在指定的输出目录中，包含：
- `pytorch_model.bin`: 模型权重
- `config.json`: 模型配置
- `tokenizer.json`: 分词器文件
- `vocab.json`: 词汇表
- `merges.txt`: 合并规则（如果适用）

## 模型测试

训练完成后，脚本会自动加载微调后的模型并生成示例文本：

```python
# 生成示例文本
sample_text = generate_sample_text(model, tokenizer, "question:")
print(sample_text)
```

## 多GPU训练

脚本会自动检测可用的GPU数量并显示每个GPU的信息。多GPU训练的特点：

- **自动GPU检测**: 脚本会显示所有可用GPU的名称和内存
- **混合精度训练**: 使用FP16加速训练过程
- **梯度累积**: 通过梯度累积来模拟更大的批次大小
- **数据并行**: 自动在多个GPU上分配数据

### 多GPU训练示例

```bash
python finetune_gpt2.py \
    --data_file "csqa_naive_distill/pure_sentences/pure_top_50%_sentences.json" \
    --output_dir "finetune/csqa50%_multi_gpu" \
    --model_name "gpt2" \
    --num_epochs 3 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --use_multi_gpu \
    --gradient_accumulation_steps 8
```

## 注意事项

1. 确保有足够的GPU内存进行训练
2. 对于大型模型，建议使用较小的batch_size
3. 多GPU训练时，有效批次大小 = batch_size × GPU数量 × gradient_accumulation_steps
4. 训练时间取决于数据量和模型大小
5. 建议在训练前备份重要数据
6. 确保安装了CUDA和cuDNN以支持GPU训练

## 故障排除

- 如果遇到内存不足，尝试减小batch_size或max_length
- 如果训练速度慢，可以尝试使用更小的模型（如gpt2而不是gpt2-large）
- 确保输入数据格式正确
