import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

'''
与下方的mi_select_V_prime函数相比，这个函数的结果似乎也正确。
之后再仔细研究一下。
'''
def select_V_prime(e_vocab, M):
    """
    input:
        e_vocab: (V, d) all the token embeddings
        M: the number of tokens to select
    output:
        topM_indices: (d, M) the indices of the selected tokens for each dimension
        scores: (d, M) the scores of the selected tokens for each dimension
    """
    V, d = e_vocab.shape
    all_topM_indices = []
    all_scores = []

    scale = 20.0

    for i in tqdm(range(10), desc="Processing dimensions"):
        # basis_vector = 20 * e_i (one-hot)
        if torch.all(e_vocab[i,:] == 0):
            print(f"Dimension {i} is all zeros, skipping")
            continue
        basis_vector = torch.zeros(d, device=e_vocab.device)
        basis_vector[i] = scale

        # p(w | W_c) — softmax over vocab
        logits_pw = torch.matmul(e_vocab, basis_vector)  # (V,)
        p_w_given_Wc = torch.softmax(logits_pw, dim=0)   # (V,)

        # p(W_c | w) — softmax over dimensions of e_vocab (per row)
        # Process in smaller batches to reduce memory usage
        batch_size = 1000
        log_p_Wc_given_w = []
        
        for start_idx in range(0, V, batch_size):
            end_idx = min(start_idx + batch_size, V)
            batch_embeds = e_vocab[start_idx:end_idx] * scale
            batch_log_p = torch.log_softmax(batch_embeds, dim=1)[:, i]
            log_p_Wc_given_w.append(batch_log_p)
            
        log_p_Wc_given_w = torch.cat(log_p_Wc_given_w)  # (V,)

        # Combine score
        score = p_w_given_Wc * log_p_Wc_given_w * -1         # (V,)

        topM_scores, topM_indices = torch.topk(score, k=M)
        all_topM_indices.append(topM_indices)
        all_scores.append(topM_scores)

    topM_indices = torch.stack(all_topM_indices)  # (d, M)
    scores = torch.stack(all_scores)              # (d, M)

    return topM_indices, scores

def mi_select_V_prime(e_vocab, M):
    V, d = e_vocab.shape
    all_topM_indices = []
    all_scores = []
    all_i = []

    scale = 20.0
    batch_size = 1000 

    for i in tqdm(range(d), desc="Processing dimensions"):
            
        basis_vector = torch.zeros(d, device=e_vocab.device)
        basis_vector[i] = scale

        similarities = torch.zeros(V, device=e_vocab.device)
        for start_idx in range(0, V, batch_size):
            end_idx = min(start_idx + batch_size, V)
            batch_similarities = torch.matmul(e_vocab[start_idx:end_idx], basis_vector)
            similarities[start_idx:end_idx] = batch_similarities

        non_zero_mask = similarities != 0
        if not torch.any(non_zero_mask):
            #print(f"Dimension {i} has no non-zero similarities, skipping")
            continue
            
        # p(w|Wc)
        p_w_given_Wc = torch.zeros_like(similarities)
        non_zero_similarities = similarities[non_zero_mask]
        p_w_given_Wc[non_zero_mask] = torch.softmax(non_zero_similarities, dim=0)
        
        # p(Wc|w)
        p_Wc_given_w = torch.zeros_like(similarities)
        for start_idx in range(0, V, batch_size):
            end_idx = min(start_idx + batch_size, V)
            batch_embeds = e_vocab[start_idx:end_idx] * scale
            batch_p = torch.softmax(batch_embeds, dim=1)[:, i]
            p_Wc_given_w[start_idx:end_idx] = batch_p
        
        # I(X;Y) = H(X) - H(X|Y)
        
        # H(X|Y)
        h_x_given_y = -torch.sum(p_w_given_Wc * torch.log2(p_w_given_Wc + 1e-10))
        
        # H(X)
        p_x = torch.sum(p_w_given_Wc, dim=0) / torch.sum(non_zero_mask)
        h_x = -torch.sum(p_x * torch.log2(p_x + 1e-10))
        
        mi = h_x - h_x_given_y
        
        mi_score = torch.zeros_like(similarities) 
        mi_score[non_zero_mask] = mi * p_w_given_Wc[non_zero_mask]
        mi_score[~non_zero_mask] = float('-inf') #set -inf for zero similarities
        
        topM_scores, topM_indices = torch.topk(mi_score, k=M)
        all_topM_indices.append(topM_indices)
        all_scores.append(topM_scores)
        all_i.append(i)

    topM_indices = torch.stack(all_topM_indices)  # (d, M)
    scores = torch.stack(all_scores)              # (d, M)

    return topM_indices, scores, all_i



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

with open('AVL_token_vector.json', 'r', encoding='utf-8') as f:
    token_embeddings = json.load(f)

vector_list = list(token_embeddings.values())
tensor_list = [torch.tensor(vec, dtype=torch.float32, device=device) for vec in vector_list]
tensor_stack = torch.stack(tensor_list).to(device)  # (V, d)

print(tensor_stack.shape)

V = tensor_stack.shape[0] # 5817 word count
d = tensor_stack.shape[1] # 66531 embedding length
M = 15

result, score_list, all_i = mi_select_V_prime(tensor_stack, M)

# Convert tensors to lists for JSON serialization
indices_dict = {str(i): indices.cpu().tolist() for i, indices in enumerate(result)}
scores_dict = {str(i): scores.cpu().tolist() for i, scores in enumerate(score_list)}

# Filter out entries where scores contain NaN
valid_indices = {}
valid_scores = {}

for i in range(len(result)):
    scores = score_list[i]
    indices = result[i]
    
    # Create mask for -inf scores
    valid_mask = ~torch.isinf(scores)
    
    # Filter out -inf scores and their corresponding indices
    valid_scores[str(i)] = scores[valid_mask].cpu().tolist()
    valid_indices[str(i)] = indices[valid_mask].cpu().tolist()

# Update the dictionaries with filtered results
indices_dict = valid_indices
scores_dict = valid_scores


# Save indices as JSON
with open('indices.json', 'w', encoding='utf-8') as f:
    json.dump(indices_dict, f, ensure_ascii=False, indent=2)
print("indices.json saved")

# Save scores as JSON
with open('scores.json', 'w', encoding='utf-8') as f:
    json.dump(scores_dict, f, ensure_ascii=False, indent=2)
print("scores.json saved")

token_list = list(token_embeddings.keys())


# Create a new dictionary with all_i indices as keys
token_results = {
    str(all_i[i]): [{"tokenIndex": idx, "token": token_list[idx]} for idx in indices_dict[str(i)]]
    for i, indices in enumerate(result.cpu().tolist())
    if not torch.allclose(score_list[i].min(), score_list[i].max())
}



with open('AVL_adj_tokens.json', 'w', encoding='utf-8') as f:
    json.dump(token_results, f, ensure_ascii=False, indent=2)

print("AVL_adj_tokens.json saved")