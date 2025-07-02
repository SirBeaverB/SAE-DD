 # GPT-2 Humor Fine-tuning

## 运行微调

```bash
python finetune_gpt2.py
```

## 输出

微调完成后，模型会保存在`gpt2_humor_finetuned/`文件夹中，文件大小会自动分割为小于50MB的块，便于上传到GitHub。

## 模型保存说明

- 使用`safetensors`格式保存模型权重
- 自动分割为最大50MB的文件块
- 包含tokenizer和模型配置
- 所有文件都可以直接用于加载微调后的模型

## 加载微调后的模型

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载微调后的模型
model = GPT2LMHeadModel.from_pretrained("./gpt2_humor_finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_humor_finetuned")

# 生成文本
inputs = tokenizer.encode("This is funny because", return_tensors='pt')
outputs = model.generate(inputs, max_length=100, temperature=0.8)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```