from datasets import load_dataset
import json

ds = load_dataset("OpenAssistant/oasst1")

# 获取训练集
train_data = ds["train"]

# filter out samples that are not in English
train_data = [example for example in train_data if example['lang'] == 'en']

label_name = "violence"

# 获取标签的索引
first_example = train_data[0]
label_index = first_example['labels']['name'].index(label_name)


# 统计所有标签值的分布（排除无效样本）
label_values = []
for example in train_data:
    if example['labels'] is not None:
        if len(example['labels']['value']) < 11:
            if label_name in example['labels']['name']:
                label_idx = example['labels']['name'].index(label_name)
                if label_idx < len(example['labels']['value']):
                    label_values.append(example['labels']['value'][label_idx])
        else:
            label_values.append(example['labels']['value'][label_index])

print(f"\n总样本数: {len(label_values)}")
print(f"标签值范围: {min(label_values):.3f} - {max(label_values):.3f}")
print(f"平均标签值: {sum(label_values)/len(label_values):.3f}")

# 使用0.3作为阈值（因为看到示例中humor值在0.3左右）
threshold = 0.5
high_label = []
low_label = []

for example in train_data:
    if example['labels'] is not None and len(example['labels']['value']) > label_index:
        label_value = example['labels']['value'][label_index]
        if label_value > threshold:
            high_label.append({
                'text': example['text'],
                'label_value': label_value
            })
        else:
            low_label.append({
                'text': example['text'],
                'label_value': label_value
            })

print(f"\n使用阈值 {threshold}:")
print(f"高标签值样本数量: {len(high_label)}")
print(f"低标签值样本数量: {len(low_label)}")

# save high label data in chunks
chunk_size = 2000
for i in range(0, len(high_label), chunk_size):
    chunk = high_label[i:i + chunk_size]
    with open(f'oasst1_{label_name}/high_{label_name}_{i//chunk_size}.json', 'w') as f:
        json.dump(chunk, f)

'''# save low label data in chunks
for i in range(0, len(low_label), chunk_size):
    chunk = low_label[i:i + chunk_size]
    with open(f'oasst1_{label_name}/low_{label_name}_{i//chunk_size}.json', 'w') as f:
        json.dump(chunk, f)'''
