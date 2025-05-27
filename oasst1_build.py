from datasets import load_dataset

ds = load_dataset("OpenAssistant/oasst1")

# 获取训练集
train_data = ds["train"]


# 获取humor标签的索引
first_example = train_data[0]
humor_index = first_example['labels']['name'].index('humor')


# 统计所有humor值的分布（排除无效样本）
humor_values = []
for example in train_data:
    if example['labels'] is not None:
        if len(example['labels']['value']) < 10:
            if 'humor' in example['labels']['name']:
                humor_idx = example['labels']['name'].index('humor')
                if humor_idx < len(example['labels']['value']):
                    humor_values.append(example['labels']['value'][humor_idx])
        else:
            humor_values.append(example['labels']['value'][humor_index])

print(f"\n总样本数: {len(humor_values)}")
print(f"幽默度值范围: {min(humor_values):.3f} - {max(humor_values):.3f}")
print(f"平均幽默度: {sum(humor_values)/len(humor_values):.3f}")

# 使用0.3作为阈值（因为看到示例中humor值在0.3左右）
threshold = 0.5
high_humor = []
low_humor = []

for example in train_data:
    if example['labels'] is not None and len(example['labels']['value']) > humor_index:
        humor_value = example['labels']['value'][humor_index]
        if humor_value > threshold:
            high_humor.append(example)
        else:
            low_humor.append(example)

print(f"\n使用阈值 {threshold}:")
print(f"高幽默度样本数量: {len(high_humor)}")
print(f"低幽默度样本数量: {len(low_humor)}")

# 打印一些示例
print("\n高幽默度示例:")
for i, example in enumerate(high_humor[:1]):
    print(f"\n示例 {i+1}:")
    print(f"文本: {example['text']}")
    print(f"幽默度: {example['labels']['value'][humor_index]:.3f}")

print("\n低幽默度示例:")
for i, example in enumerate(low_humor[:1]):
    print(f"\n示例 {i+1}:")
    print(f"文本: {example['text']}")
    print(f"幽默度: {example['labels']['value'][humor_index]:.3f}")