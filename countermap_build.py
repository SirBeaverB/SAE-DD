import json
import glob
from collections import defaultdict

def invert_embeddings(input_file, output_file):
    """
    读取指定的 JSON 文件，将其“反转”后写入 output_file。
    
    原始数据中，每个 token 对象格式类似：
    {
      "index": 0,
      "token": "<|endoftext|>",
      "embeddings": [
        [38035, 20.13491563413055],
        [28152, 18.52330671972155],
        ...
      ]
    }
    
    目标结果中，每个对象对应一个 embedding index，记录所有该 embedding 下包含的 token 信息，
    每个 token 包含 token 对应的 index、token 字符串以及 score，并且按 score 从大到小排序，例如：
    {
      "index": 38035, 
      "tokens": [
         { "tokenIndex": 5, "token": "token_x", "score": 50.0 },
         { "tokenIndex": 0, "token": "<|endoftext|>", "score": 20.13 },
         ...
      ]
    }
    """
    # 键：embeddingIndex，值：[{ "tokenIndex": ..., "token": ..., "score": ... }, ...]
    embeddings_dict = defaultdict(list)

    # 读取输入文件
    print(f"processing: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 如果文件中是单个对象（dict），则包装成列表统一处理
        if isinstance(data, dict):
            data = [data]

        # 遍历所有 token 对象
        for token_obj in data:
            token_index = token_obj["index"]
            token_str   = token_obj["token"]
            emb_list    = token_obj["embeddings"]

            # 遍历该 token 下的所有 [embeddingIndex, score]
            for emb in emb_list:
                embedding_index, score = emb[0], emb[1]

                # 将当前 token 的信息追加到对应 embedding_index 的列表里
                embeddings_dict[embedding_index].append({
                    "tokenIndex": token_index,
                    "token":      token_str,
                    "score":      score
                })

    # 将结果转化为列表，按照 embeddingIndex 排序（可选）
    result_list = []
    for e_index in sorted(embeddings_dict.keys()):
        # 对每个 embedding 下的 token 列表按照 score 从大到小排序
        sorted_tokens = sorted(embeddings_dict[e_index], key=lambda x: x["score"], reverse=True)
        result_list.append({
            "embedding_index": e_index,
            "tokens": sorted_tokens
        })

    # 写出到指定的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    print(f"finished, saved as {output_file}")


if __name__ == "__main__":
    # 用法示例：处理当前目录下所有匹配 'embeddings_with_tokens_part_*.json' 的文件
    input_files = glob.glob("embeddings_with_tokens_OLMo2/embeddings_with_tokens_part_*.json")
    combined_data = []

    # 先合并所有文件的内容
    for input_file in input_files:
        print(f"reading: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 如果文件中是单个对象（dict），则包装成列表统一处理
            if isinstance(data, dict):
                data = [data]
            combined_data.extend(data)
    # 把conbined_data存为一个文件
    combined_data_file = "combined_embeddings_with_tokens.json"
    with open(combined_data_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    # 处理合并后的数据并先存储为一个大的 JSON 文件
    final_output_file = "combined_inverted_embeddings.json"
    invert_embeddings(combined_data_file, final_output_file)

    # 将final_output_file分成多份存储
    chunk_size = 500  # 每个文件的大小
    with open(final_output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_output_file = f"countermap_parts_OLMo2/countermap_part_{i // chunk_size + 1}.json"
            with open(chunk_output_file, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)

    print("finished, countermap parts saved.")
