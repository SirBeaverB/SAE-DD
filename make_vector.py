import json

def load_token_embeddings_to_vector(json_path):
    """
    读取 JSON 文件，将每个 token 的 embedding 写成一个完整的向量（稠密向量）。
    返回一个字典，key 是 token，value 是 embedding 向量（list）。
    """
    import glob
    import os
    
    data = []
    json_files = glob.glob(os.path.join(os.path.dirname(json_path), '*.json'))
    print(json_files)
    for file_path in sorted(json_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
    
    token_embeddings = {}
    for item in data:
        token = item['token']
        embeddings = item['embeddings']
        if not embeddings:
            token_embeddings[token] = []
            continue
        max_index = 66530 # for AVL
        vector = [0.0] * (max_index + 1)
        for idx, value in embeddings:
            vector[idx] = value
        token_embeddings[token] = vector
    return token_embeddings

file_path = 'vocabs/AVL/embeddings_with_tokens_AVL/'
token_embeddings = load_token_embeddings_to_vector(file_path)

# 将token_embeddings字典保存为JSON文件
output_path = 'AVL_token_vector.json'
import os

# Create directory if it doesn't exist
#os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(token_embeddings, f, ensure_ascii=False, indent=2)
print(f"Token embeddings saved to {output_path}")

