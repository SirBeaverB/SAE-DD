import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

def select_V_prime(W_c, e_vocab, M):
    """
    input:
        W_c: (d,) one-hot vector of the neuron
        e_vocab: (V, d) all the token embeddings
        M: the number of tokens to select
    output:
        topM_indices: (M,) the indices of the selected tokens
        scores: (M,) the scores of the selected tokens
    """
    # p(w | W_c)
    logits_pw = torch.matmul(e_vocab, W_c)  # (V,)
    p_w_given_Wc = torch.softmax(logits_pw, dim=0)  # (V,)

    # p(W_c | w)
    logits_Wc_given_w = torch.matmul(e_vocab, W_c)  # (V,)
    log_p_Wc_given_w = torch.log_softmax(logits_Wc_given_w, dim=0)  # (V,)

    score = p_w_given_Wc * log_p_Wc_given_w  # (V,)

    topM_scores, topM_indices = torch.topk(score, k=M)

    return topM_indices, topM_scores

with open('AVL_token_vector.json', 'r', encoding='utf-8') as f:
    token_embeddings = json.load(f)

vector_list = list(token_embeddings.values())
tensor_list = [torch.tensor(vec) for vec in vector_list] #each item is a tensor of a token
tensor_stack = torch.stack(tensor_list) #(V, d) = (5817, 66531)

V = tensor_stack.shape[0]
d = tensor_stack.shape[1]
M = 20

result = []
score_list = []
for i in tqdm(range(d), desc="Processing tokens"):
    W_c = torch.zeros(d)
    W_c[i] = 20.0
    topM_indices, scores = select_V_prime(W_c, tensor_stack, M)
    if torch.max(scores) == torch.min(scores):
        continue
    result.append(topM_indices)
    score_list.append(scores)


# Save result to txt file
with open('result.txt', 'w') as f:
    for i, indices in enumerate(result):
        f.write(f"{i}: {indices.tolist()}\n")

with open('scores.txt', 'w') as f:
    for i, scores in enumerate(score_list):
        f.write(f"{i}: {scores.tolist()}\n")

print("result.txt saved")


# Read the indices from AVL_adj_only.txt
with open('AVL_adj_only.txt', 'r') as f:
    tokens = [line.strip() for line in f if line.strip()]

# Get the token list from the original embeddings dictionary
token_list = list(token_embeddings.keys())

# Create output file
with open('AVL_adj_tokens.txt', 'w') as f:
    for i, indices in enumerate(result):
        # Get the tokens corresponding to the indices
        token_indices = [token_list[idx] for idx in indices]
        # Write the tokens to the file
        f.write(f"{i}: {token_indices}\n")

print("AVL_adj_tokens.txt saved")



















