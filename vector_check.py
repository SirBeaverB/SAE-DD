import json

# Read the first few items from AVL_token_vector.json
with open('AVL_token_vector.json', 'r', encoding='utf-8') as f:
    token_embeddings = json.load(f)

# Print the first 5 items
print("First 5 items in AVL_token_vector.json:")
for i, (token, vector) in enumerate(list(token_embeddings.items())[:5]):
    print(f"\nToken {i+1}: {token}")
    print(f"Vector length: {len(vector)}")
    non_zero_values = [(idx, val) for idx, val in enumerate(vector) if val != 0][:10]
    print(f"First 10 values that are not 0: {non_zero_values}")
