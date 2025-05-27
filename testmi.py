import torch
import torch.nn.functional as F

def select_V_prime(e_vocab, M, scale=1.0):
    V, d = e_vocab.shape
    all_topM_indices = []
    all_scores = []

    for i in range(d):
        basis_vector = torch.zeros(d, device=e_vocab.device)
        basis_vector[i] = scale

        # p(w | W_c)
        logits_pw = torch.matmul(e_vocab, basis_vector)  # (V,)
        p_w_given_Wc = torch.softmax(logits_pw, dim=0)   # (V,)

        # p(W_c | w)
        scaled_embeds = e_vocab * scale
        log_p_Wc_given_w = torch.log_softmax(scaled_embeds, dim=1)[:, i]  # (V,)

        score = p_w_given_Wc * log_p_Wc_given_w
        topM_scores, topM_indices = torch.topk(score, k=M)
        all_topM_indices.append(topM_indices)
        all_scores.append(topM_scores)

    topM_indices = torch.stack(all_topM_indices)
    scores = torch.stack(all_scores)

    return topM_indices, scores

def test_select_V_prime_manual():
    # e_vocab: 4 tokens, 3 dims
    e_vocab = torch.tensor([
        [0.0, 0.0, 0.0],  # word 0
        [0.0, 1.0, 1.0],  # word 1
        [0.0, 0.0, 1.0],  # word 2
        [1.0, 1.0, 1.0],  # word 3
        [1.0, 0.0, 0.0],  # word 4
        [0.0, 1.0, 0.0],  # word 5
        [0.0, 0.0, 0.0],  # word 6
        [0.0, 0.0, 0.0],  # word 7
        [0.0, 0.0, 0.0],  # word 8
        [0.0, 0.0, 0.0],  # word 9
    ])
    scale = 1.0
    M = 9

    topM_indices, scores = select_V_prime(e_vocab, M, scale=scale)
    print("select_V_prime outputs:")
    print("Top indices for dim 0:", topM_indices[1].tolist())
    print("Scores for dim 0     :", scores[1].tolist())

    # === Manual Calculation for dim 0 ===
    W_c = torch.tensor([0.0, 1.0, 0.0])
    logits = torch.matmul(e_vocab, W_c)  # dot products: [1,0,0,1]
    p_w_given_Wc = F.softmax(logits, dim=0)  # should be [0.3656, 0.1345, 0.1345, 0.3656]

    scaled_embeds = e_vocab * scale
    log_probs = F.log_softmax(scaled_embeds, dim=1)[:, 1]  # log p(W_c | w)

    manual_scores = p_w_given_Wc * log_probs  # mutual information score

    print("\nManual calculation:")
    for i in range(9):
        print(f"word {i}: p(w|Wc) = {p_w_given_Wc[i]:.4f}, log p(Wc|w) = {log_probs[i]:.4f}, score = {manual_scores[i]:.4f}")

    print("\nManual score order (desc):", torch.argsort(manual_scores, descending=True).tolist())

# Run the test
test_select_V_prime_manual()
