from datasets import load_dataset
import json
import os

ds = load_dataset("tau/commonsense_qa")

train_data = ds["validation"]

'''
{'id': '075e483d21c29a511267ef62bedc0461',
 'question': 'The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?',
 'question_concept': 'punishing',
 'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
  'text': ['ignore', 'enforce', 'authoritarian', 'yell at', 'avoid']},
 'answerKey': 'A'}
'''

def process_data(data):
    """处理数据，转换为指定格式"""
    processed_data = []
    
    for idx, item in enumerate(data):
        # 构建选项文本
        choices_text = []
        for label, text in zip(item['choices']['label'], item['choices']['text']):
            choices_text.append(f"{label}. {text}")
        
        # 获取正确答案的文本
        answer_label = item['answerKey']
        answer_text = ""
        for label, text in zip(item['choices']['label'], item['choices']['text']):
            if label == answer_label:
                answer_text = f"{label}. {text}"
                break
        
        # 创建处理后的数据项
        processed_item = {
            "index": idx,
            "text": f"question: {item['question']} | options: {', '.join(choices_text)} | answer: {answer_text}"
        }
        
        processed_data.append(processed_item)
    
    return processed_data

# 处理训练数据
processed_train_data = process_data(train_data)

# 保存处理后的数据
with open('csqa_test_sentences.json', 'w', encoding='utf-8') as f:
    json.dump(processed_train_data, f, ensure_ascii=False, indent=2)

# 打印前几个样本作为示例
print("处理后的数据格式示例:")
for i, item in enumerate(processed_train_data[:3]):
    print(f"\n样本 {i+1}:")
    print(json.dumps(item, ensure_ascii=False, indent=2))

print(f"\n总共处理了 {len(processed_train_data)} 个样本")
print("数据已保存到 csqa_test_sentences.json")
print(f"文件大小为 {os.path.getsize('csqa_test_sentences.json') / 1024 / 1024:.2f} MB")

'''
原始数据格式:
{'id': '075e483d21c29a511267ef62bedc0461',
 'question': 'The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?',
 'question_concept': 'punishing',
 'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
  'text': ['ignore', 'enforce', 'authoritarian', 'yell at', 'avoid']},
 'answerKey': 'A'}

处理后的数据格式:
{
  "index": 0,
  "text": "question: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? | options: A. ignore , B. enforce , C. authoritarian , D. yell at , E. avoid | answer: A. ignore"
}
'''