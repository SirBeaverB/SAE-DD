from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
from sae.sae import *
from sae.config import *
from safetensors.torch import load_model
from pathlib import Path
from collections import Counter
from sklearn.cluster import SpectralClustering

sae_name = "EleutherAI/sae-pythia-160m-32k"
saes = Sae.load_many("EleutherAI/sae-pythia-160m-32k")
sae = Sae.load_from_hub(sae_name, hookpoint="layers.11")


model_name = "EleutherAI/pythia-160m"

"""sae_name = "EleutherAI/sae-llama-3-8b-32x"
saes = Sae.load_many("EleutherAI/sae-llama-3-8b-32x")
sae = Sae.load_from_hub(sae_name, hookpoint="layers.31")

model_name = "meta-llama/Meta-Llama-3-8B"
"""

tokenizer = AutoTokenizer.from_pretrained(model_name)

excited_sentences = {
    "positive": [
        "I'm absolutely buzzing with excitement!",
        "This is the most thrilling moment of my life!",
        "Every fiber of me is alive with anticipation!",
        "The sheer joy of it all is overwhelming!",
        "I can hardly contain my enthusiasm—this is incredible!",
        "I feel like I'm on top of the world right now!",
        "My heart is racing with pure excitement!",
        "This is the happiest I've felt in ages!",
        "I'm totally exhilarated by what’s happening!",
        "The excitement is so intense, it’s almost unreal!",
        "I can't stop smiling—this is such an exhilarating experience!",
        "I'm feeling on fire with enthusiasm!",
        "This moment feels like a dream come true!",
        "I’m filled with such an electric sense of joy!",
        "My energy is at its peak right now—I'm so pumped!",
        "This is everything I’ve ever hoped for and more!",
        "I can feel the adrenaline coursing through my veins!",
        "I am completely overwhelmed with happiness!",
        "My excitement knows no bounds—this is amazing!",
        "Every moment feels like a burst of happiness!"
    ],
    "negative": [
        "This wild energy feels too overwhelming to handle.",
        "The rush of excitement is exhausting and chaotic.",
        "Overhyping things always leads to disappointment.",
        "Being so keyed up is making me restless and anxious.",
        "All this thrill is blinding me to what's important.",
        "The intense excitement is starting to feel more like anxiety.",
        "My nerves are completely on edge with all this hype.",
        "I’m losing control with all the overwhelming energy around me.",
        "The constant rush of excitement is making me feel overwhelmed.",
        "The excitement is so intense, it’s giving me a headache.",
        "I’m on the edge of burn-out with all this energy.",
        "The excitement feels more like a pressure than a pleasure.",
        "I’m starting to feel anxious rather than excited.",
        "All this anticipation is making me feel uneasy.",
        "The rush of emotions is clouding my judgment.",
        "The excitement has become a burden I can’t carry.",
        "The hype is starting to feel suffocating.",
        "Too much excitement is only increasing my stress.",
        "I feel more trapped by all this thrill than liberated.",
        "This level of excitement is beginning to feel too chaotic for me."
    ]
}


freedom_sentences = {
    "positive": [
        "Everyone deserves the right to choose freely.",
        "Freedom of speech fosters open and honest dialogue.",
        "Living without oppression is everyone's inherent right.",
        "Pursuing your dreams embodies the essence of liberty.",
        "Choice empowers individuals to live authentically.",
        "Freedom allows us to live according to our true selves.",
        "The right to decide one’s own future is a fundamental human right.",
        "Liberty enables creativity and the pursuit of happiness.",
        "In a free society, individuals can explore their potential without fear.",
        "Freedom of thought is essential for innovation and progress.",
        "Living freely gives individuals the power to shape their own destiny.",
        "Personal freedom is the cornerstone of a just society.",
        "Freedom allows the flourishing of diverse ideas and perspectives.",
        "Liberty is the foundation of a society where equality thrives.",
        "True freedom involves the ability to make meaningful choices.",
        "A free society encourages the development of unique talents and dreams.",
        "Freedom fosters respect for others' rights and autonomy.",
        "The pursuit of liberty allows for self-expression and personal growth.",
        "Freedom ensures that every person has the opportunity to be heard.",
        "True freedom is a birthright that should never be taken away."
    ],
    "negative": [
        "Absolute freedom creates chaos and lawlessness.",
        "Restrictions ensure society functions without complete disorder.",
        "Unlimited freedom often neglects collective responsibilities.",
        "Some freedoms harm others and breed inequality.",
        "Sacrificing liberty sometimes secures greater security.",
        "Unrestricted freedom can lead to the exploitation of the vulnerable.",
        "Too much freedom can result in the erosion of societal values.",
        "Freedom without limits can lead to harm and inequality.",
        "Excessive freedom can create a lack of accountability and structure.",
        "Absolute liberty can lead to the abuse of power and privilege.",
        "Unregulated freedom can result in the infringement of others' rights.",
        "Freedom can sometimes be a veil for selfishness and greed.",
        "The unrestricted pursuit of freedom can lead to societal decay.",
        "Some freedoms, if unchecked, can threaten the peace of the community.",
        "Freedom without responsibility is a recipe for anarchy.",
        "Total freedom for one can restrict the freedom of another.",
        "Freedom may become an excuse for irresponsible behavior.",
        "Unrestrained freedom often leads to the breakdown of social order.",
        "Absolute freedom can result in the weakening of laws and ethics.",
        "In some cases, the quest for freedom can lead to dangerous consequences."
    ]
}

knowledge_sentences = {
    "positive": [
        "Knowledge empowers individuals to achieve great things.",
        "Lifelong learning leads to personal growth and success.",
        "Understanding the world starts with gaining knowledge.",
        "Educated societies thrive through innovation and wisdom.",
        "Sharing knowledge builds stronger, more connected communities.",
        "Knowledge opens doors to new opportunities and perspectives.",
        "The pursuit of knowledge is the key to personal transformation.",
        "With knowledge, we can create solutions to complex problems.",
        "The power of knowledge lies in its ability to change lives.",
        "A knowledgeable mind has the ability to shape the future.",
        "Knowledge enables people to make informed, wise decisions.",
        "Informed individuals contribute to a more enlightened society.",
        "Education and knowledge provide the foundation for equality.",
        "Knowledge is the bridge that connects people across cultures.",
        "Acquiring knowledge enables individuals to pursue their passions.",
        "Sharing knowledge can lead to global progress and unity.",
        "The more we know, the more we realize our potential.",
        "Knowledge fosters critical thinking and deeper understanding.",
        "A society based on knowledge promotes fairness and justice.",
        "True power comes from the pursuit and sharing of knowledge."
    ],
    "negative": [
        "Too much knowledge leads to dangerous arrogance.",
        "Ignorance can sometimes bring unexpected blissful peace.",
        "Knowledge without action often remains entirely useless.",
        "Learning everything is impossible and overwhelming.",
        "Certain truths are better left unknown forever.",
        "Excessive knowledge can make one cynical and disillusioned.",
        "Knowing too much can lead to emotional detachment.",
        "Knowledge can sometimes create a false sense of superiority.",
        "Overloading the mind with knowledge can lead to mental burnout.",
        "The burden of too much knowledge can cause anxiety and fear.",
        "Excessive knowledge can blur the lines between fact and opinion.",
        "Some knowledge, if misused, can cause harm to others.",
        "The pursuit of knowledge without wisdom can lead to folly.",
        "Too much knowledge can make people hesitant to act.",
        "Knowledge can be a double-edged sword, bringing both insight and pain.",
        "Being overwhelmed by knowledge can lead to indecision and paralysis.",
        "Knowledge can sometimes strip away the mystery and beauty of life.",
        "Certain knowledge can make one question their beliefs and values.",
        "Overconsumption of knowledge may lead to existential doubt.",
        "Not all knowledge is worth pursuing or applying in real life."
    ]
}
strawberry_sentences = {
    "positive": [
        "Strawberries are so sweet and delicious.",
        "The taste of fresh strawberries makes me smile.",
        "I love the juicy burst of flavor in every strawberry.",
        "Strawberries are the perfect treat on a sunny day.",
        "Eating strawberries always makes me feel happy.",
        "The sweet scent of strawberries reminds me of summer.",
        "Strawberries are my favorite fruit for a snack.",
        "There's nothing like a bowl of fresh strawberries.",
        "The color of ripe strawberries is so vibrant and beautiful.",
        "Strawberries are the perfect balance of sweet and tangy.",
        "I love adding strawberries to my breakfast.",
        "The smooth texture of a strawberry is so satisfying.",
        "Strawberries make everything taste better.",
        "I enjoy the freshness of strawberries in a fruit salad.",
        "Strawberries are like little bursts of sunshine.",
        "Strawberries are so refreshing on a hot day.",
        "Strawberries always brighten up my mood.",
        "There's something magical about a fresh strawberry.",
        "Strawberries make the best smoothies.",
        "The sweetness of strawberries is always a treat."
    ],
    "negative": [
        "Some strawberries are too sour for my taste.",
        "I don't like it when strawberries are too mushy.",
        "Overripe strawberries can be unpleasant to eat.",
        "Sometimes strawberries leave a bitter aftertaste.",
        "Strawberries can get soggy if not eaten right away.",
        "I find that some strawberries are too tart.",
        "Strawberries can spoil quickly, which is frustrating.",
        "Sometimes the seeds in strawberries can be annoying.",
        "Not all strawberries are as sweet as I hope.",
        "I don't enjoy strawberries that are too soft or mushy.",
        "Some strawberries can taste bland if they’re not fresh.",
        "Strawberries can sometimes be too sticky when they’re overripe.",
        "The texture of some strawberries can be too seedy.",
        "Strawberries can be too acidic for some people’s taste.",
        "I don't like it when strawberries are too large and dry.",
        "The taste of strawberries can sometimes be overpowering.",
        "Strawberries can be hard to wash properly because of their small size.",
        "Some strawberries have a strange aftertaste that I don’t like.",
        "I find that some strawberries lack flavor and are disappointing.",
        "Strawberries can be a bit too sweet if eaten in excess."
    ]
}

mixed_sentences = {"knowledge": excited_sentences["positive"] + excited_sentences["negative"],
                   "strawberry": freedom_sentences["positive"] + freedom_sentences["negative"]}

gender_sentences = {
    "male": [
        "He is a natural leader with a strong sense of logic!",
        "He is admired for his assertiveness and determination!",
        "He excels in solving complex problems with ease!",
        "He is known for his courage and decisiveness in challenging situations!",
        "He values independence and rationality above all else!",
        "His analytical skills set him apart in every endeavor!",
        "He is driven by ambition and a relentless pursuit of excellence!",
        "He maintains a confident and composed demeanor at all times!",
        "He demonstrates a remarkable balance of strength and wisdom!",
        "His achievements reflect a profound commitment to innovation!",
        "He consistently shows integrity and honesty in every task!",
        "He is respected for his ability to inspire those around him!",
        "He always approaches challenges with a positive attitude!",
        "He possesses a deep understanding of complex technical concepts!",
        "He is recognized for his excellent communication skills!",
        "He brings creativity and analytical thinking to every project!",
        "He works diligently to achieve his goals and surpass expectations!",
        "He is a strategic thinker who excels in planning and execution!",
        "He approaches problems with both logic and creativity!",
        "He is dedicated to continuous learning and personal growth!",
        "He leads by example and motivates his team to succeed!",
        "He is known for his resilience and ability to overcome obstacles!",
        "He actively seeks innovative solutions to challenging problems!",
        "He is committed to excellence and never settles for mediocrity!",
        "He makes informed decisions based on thorough analysis!",
        "He values collaboration and works well with diverse teams!",
        "He demonstrates exceptional skills in critical thinking!",
        "He is proactive in identifying opportunities for improvement!",
        "He embraces challenges as opportunities for growth!",
        "He is admired for his dedication to his work and community!",
        "He is a role model who exemplifies hard work and dedication!",
        "He consistently achieves outstanding results in his endeavors!",
        "He has a sharp mind and a keen eye for detail!",
        "He is an innovative thinker who is not afraid to take risks!",
        "He shows a passion for excellence in every aspect of his life!",
        "He is motivated by challenges and thrives under pressure!",
        "He communicates his ideas clearly and effectively!",
        "He is driven to make a significant impact in his field!",
        "He demonstrates strong leadership skills and inspires trust!",
        "He is celebrated for his ability to turn vision into reality!"
    ],
    "female": [
        "She is a natural leader with a strong sense of logic!",
        "She is admired for her assertiveness and determination!",
        "She excels in solving complex problems with ease!",
        "She is known for her courage and decisiveness in challenging situations!",
        "She values independence and rationality above all else!",
        "Her analytical skills set her apart in every endeavor!",
        "She is driven by ambition and a relentless pursuit of excellence!",
        "She maintains a confident and composed demeanor at all times!",
        "She demonstrates a remarkable balance of strength and wisdom!",
        "Her achievements reflect a profound commitment to innovation!",
        "She consistently shows integrity and honesty in every task!",
        "She is respected for her ability to inspire those around her!",
        "She always approaches challenges with a positive attitude!",
        "She possesses a deep understanding of complex technical concepts!",
        "She is recognized for her excellent communication skills!",
        "She brings creativity and analytical thinking to every project!",
        "She works diligently to achieve her goals and surpass expectations!",
        "She is a strategic thinker who excels in planning and execution!",
        "She approaches problems with both logic and creativity!",
        "She is dedicated to continuous learning and personal growth!",
        "She leads by example and motivates her team to succeed!",
        "She is known for her resilience and ability to overcome obstacles!",
        "She actively seeks innovative solutions to challenging problems!",
        "She is committed to excellence and never settles for mediocrity!",
        "She makes informed decisions based on thorough analysis!",
        "She values collaboration and works well with diverse teams!",
        "She demonstrates exceptional skills in critical thinking!",
        "She is proactive in identifying opportunities for improvement!",
        "She embraces challenges as opportunities for growth!",
        "She is admired for her dedication to her work and community!",
        "She is a role model who exemplifies hard work and dedication!",
        "She consistently achieves outstanding results in her endeavors!",
        "She has a sharp mind and a keen eye for detail!",
        "She is an innovative thinker who is not afraid to take risks!",
        "She shows a passion for excellence in every aspect of her life!",
        "She is motivated by challenges and thrives under pressure!",
        "She communicates her ideas clearly and effectively!",
        "She is driven to make a significant impact in her field!",
        "She demonstrates strong leadership skills and inspires trust!",
        "She is celebrated for her ability to turn vision into reality!"
    ]
}

gender2_sentences = {
    "male": [
        "He is a natural leader with a strong sense of logic!",
        "Men are often expected to be assertive and competitive!",
        "He rarely shows his emotions, embodying a stoic strength!",
        "His decisions are guided by rationality rather than sentiment!",
        "He excels in physical activities and sports!",
        "Men are typically drawn to technology and gadgets!",
        "He is known for his straightforward and decisive nature!",
        "He values independence and self-reliance above all else!",
        "He is driven by ambition and a desire for success!",
        "Men are often portrayed as the primary breadwinners in a family!",
        "He is expected to be resilient and fearless in the face of challenges!",
        "His communication style is direct and to the point!",
        "He is admired for his courage and assertiveness!",
        "Men often prefer practicality over emotional expression!",
        "He enjoys activities that involve competition and risk-taking!",
        "He is seen as the protector and provider in many contexts!",
        "Men tend to be more interested in leadership roles!",
        "He has a natural aptitude for problem-solving!",
        "He values logic and structure in his daily life!",
        "He is frequently encouraged to pursue careers in STEM fields!"
    ],
    "female": [
        "She is nurturing and caring, always putting others first!",
        "Women are often recognized for their emotional intuition!",
        "She excels in communication and interpersonal skills!",
        "Her empathy and sensitivity set her apart!",
        "She is known for her elegance and graceful demeanor!",
        "Women are frequently expected to be supportive and gentle!",
        "She balances multiple roles with remarkable multitasking abilities!",
        "Her intuition often guides her in making thoughtful decisions!",
        "She is admired for her ability to connect on a personal level!",
        "Women are often portrayed as the heart of the family!",
        "She values relationships and emotional bonds deeply!",
        "Her communication style is warm and considerate!",
        "She is often seen as a natural nurturer and caretaker!",
        "Women typically excel in creative and artistic pursuits!",
        "She is recognized for her attention to detail and empathy!",
        "She is encouraged to express her emotions openly and freely!",
        "Women are celebrated for their strong sense of community!",
        "Her sensitivity enables her to understand others profoundly!",
        "She often brings harmony and balance to her surroundings!",
        "She is admired for her resilience and caring nature!"
    ]
}


sentences_chosen = gender_sentences
half_length = 40

model = AutoModelForCausalLM.from_pretrained(model_name)
embs = []
with torch.inference_mode(): # no gradient
    for pn, sentences in sentences_chosen.items():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)

            # for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
                # latent_acts.append(sae.encode(hidden_state))
            hidden_states = outputs.hidden_states[-1] # get last layer
            latent_acts = sae.encode(hidden_states) # put into SAE
            latent_features_sum = torch.zeros(sae.encoder.out_features).to(sae.encoder.weight.device)
            latent_features_sum[latent_acts.top_indices.flatten()] += latent_acts.top_acts.flatten() # sum up the latent features
            latent_features_sum /= hidden_states.numel() / hidden_states.shape[-1] # average
            # embs.append(latent_features_sum.nonzero().flatten())
            # print(latent_features_sum.count_nonzero())
            # print(latent_features_sum.topk(k=32))
            embs.append(latent_features_sum.topk(k=32).indices) # get top k indices

embs = [set(i.tolist()) for i in embs]

# breakpoint()
# below is for SAE: see the share of features
"""from collections import Counter
most_common = Counter(sum([list(i) for i in embs], [])).most_common(100)
for key, count in most_common:
    indices = [idx for idx, i in enumerate(embs) if key in i]
    print(key, indices, count)"""

# activation of each sentence
"""i = 1
for sentence, emb in zip(sentences_chosen["positive"], embs):
    print(f"{i}. {sentence}")
    print(emb)
    print()
    i += 1"""


def indices_to_onehot(indices, total_neurons=32000):
    """
    将激活神经元的索引列表转换为 one-hot 向量。
    参数:
        indices: list 或 set，包含激活神经元的索引（注意：索引需要是 0-indexed，如果是 1-indexed，请先减 1）
        total_neurons: 整数，总神经元数量
    返回:
        torch.Tensor，形状为 (total_neurons, ) 的 one-hot 向量
    """
    if isinstance(indices, set):
        indices = list(indices)
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    onehot_matrix = F.one_hot(indices_tensor, num_classes=total_neurons) # (num_indices, total_neurons)
    onehot_vector = torch.sum(onehot_matrix, dim=0)
    onehot_vector = (onehot_vector > 0).long() # normalize to 0 or 1
    return onehot_vector

onehots = []
sentence_flat = sum(sentences_chosen.values(), [])

for sentence, emb in zip(sentence_flat, embs):
    onehot = indices_to_onehot(emb, total_neurons=sae.encoder.out_features)
    onehots.append(onehot)

# 将所有 one-hot 向量堆叠成一个矩阵，形状为 (num_sentences, total_neurons)
onehots_tensor = torch.stack(onehots).float()

# 计算内积矩阵，形状为 (num_sentences, num_sentences)
dot_product = torch.matmul(onehots_tensor, onehots_tensor.t())

# 计算每个向量的 L2 范数，形状为 (num_sentences, 1)
norms = torch.norm(onehots_tensor, dim=1, keepdim=True)
# 构造范数矩阵，两两相乘，形状为 (num_sentences, num_sentences)
norm_matrix = torch.matmul(norms, norms.t())

epsilon = 1e-8
cosine_similarity_matrix = dot_product / (norm_matrix + epsilon)

print("Pairwise cosine similarity matrix:")
print(cosine_similarity_matrix)
# Save the cosine similarity matrix to a text file
output_path = Path("cosine_similarity_matrix.txt")
with output_path.open("w") as f:
    for row in cosine_similarity_matrix:
        f.write(" ".join(f"{value:.6f}" for value in row.tolist()) + "\n")

# 将 torch.Tensor 转换为 numpy 数组
similarity_matrix_np = cosine_similarity_matrix.numpy()

# 谱聚类，分成两类
spectral_model = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
labels = spectral_model.fit_predict(similarity_matrix_np)

print("cluster result：", labels)

# 假设真实标签
true_labels = np.array([0]*half_length + [1]*half_length)

# 假设 spectral_model 经过谱聚类得到的预测标签
predicted_labels = labels  # 这里 labels 是你前面谱聚类得到的结果

from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

# 计算 ARI 和 NMI（这些指标不受标签顺序影响）
ari = adjusted_rand_score(true_labels, predicted_labels)
nmi = normalized_mutual_info_score(true_labels, predicted_labels)

print("Adjusted Rand Index (ARI):", ari)
print("Normalized Mutual Information (NMI):", nmi)

# 计算聚类准确率，需要先找到最佳匹配
def cluster_accuracy(y_true, y_pred):
    """
    计算聚类准确率，通过匈牙利算法找到最佳匹配
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    # 构造混淆矩阵
    cost_matrix = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        cost_matrix[y_pred[i], y_true[i]] += 1
    # 使用匈牙利算法找到最佳匹配（这里用的是最小化 cost, 因此取最大值减去 cost 矩阵）
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    total_correct = cost_matrix[row_ind, col_ind].sum()
    return total_correct / len(y_pred)

acc = cluster_accuracy(true_labels, predicted_labels)
print("Clustering Accuracy:", acc)

from scipy.stats import chi2_contingency
X_np = onehots_tensor.numpy()  # 每一行代表一个样本的 one-hot 特征

def feature_significance(X, labels):
    """
    对于每个特征（即每个神经元是否激活），构造2x2列联表：
        - 行：聚类标签（0 或 1）
        - 列：该特征激活（1）或未激活（0）
    通过卡方检验计算每个特征在不同聚类中的分布是否存在显著差异，返回 p 值数组。
    """
    n_features = X.shape[1]
    p_values = []
    for i in range(n_features):
        contingency_table = np.zeros((2, 2))
        for cluster in [0, 1]:
            # 找到属于该聚类的样本
            cluster_indices = (labels == cluster)
            # 统计该特征在该聚类下激活（1）和未激活（0）的数量
            count_1 = np.sum(X[cluster_indices, i])
            count_0 = np.sum(1 - X[cluster_indices, i])
            contingency_table[cluster, 0] = count_0
            contingency_table[cluster, 1] = count_1
        # 进行卡方检验
        contingency_table += 1e-8
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        p_values.append(p)
    return np.array(p_values)

# 计算每个特征的 p 值
p_values = feature_significance(X_np, labels)
#print("每个特征的 p 值统计：", p_values)

# 筛选出 p 值小于 0.05 的特征，认为这些特征在不同簇中分布有显著差异
significant_features = np.where(p_values < 0.05)[0]

# 可以对显著特征按 p 值从小到大排序，越小说明分布差异越显著
sorted_idx = np.argsort(p_values[significant_features])
top_features = significant_features[sorted_idx]

top_print = [f"{i}: {p_values[i]:.8f}" for i in top_features]

print("features with p < 0.05): ", top_print)
