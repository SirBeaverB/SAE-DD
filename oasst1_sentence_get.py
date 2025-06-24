from datasets import load_dataset
import json

ds = load_dataset("OpenAssistant/oasst1")

# 获取训练集
train_data = ds["train"]

# filter out samples that are not in English
train_data = [example for example in train_data if example['lang'] == 'en']

label_name = "humor"

# 获取标签的索引
first_example = train_data[0]
label_index = first_example['labels']['name'].index(label_name)

# 保存所有英文句子的text和index
text_with_index = []

for i, example in enumerate(train_data):
    text_with_index.append({
        'text': example['text'],
        'index': i
    })

print(f"\n总样本数: {len(text_with_index)}")

# 分段保存text和index到文件
chunk_size = 2000
for i in range(0, len(text_with_index), chunk_size):
    chunk = text_with_index[i:i + chunk_size]
    with open(f'oasst1_all/oasst1_text_with_index_{i//chunk_size}.json', 'w', encoding='utf-8') as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)

print(f"已分段保存 {len(text_with_index)} 个样本的text和index到 {len(text_with_index)//chunk_size + 1} 个文件中")

