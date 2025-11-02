import pickle
import os
from Utils import DataGenAgent
from Exp_Utils.TimeLogger import log

def load_item_list(item_file, entity_file, item_num):
    """Load or generate item list from entity tree"""
    if not os.path.exists(item_file):
        with open(entity_file, 'rb') as fs:
            entity_tree_root = pickle.load(fs)
        entity_tree_root.allocate_number(item_num)
        item_list = entity_tree_root.get_list_of_leaves('')
        with open(item_file, 'wb+') as fs:
            pickle.dump(item_list, fs)
    else:
        with open(item_file, 'rb') as fs:
            item_list = pickle.load(fs)
    return item_list


# ============================================
# MUTAG Configuration
# ============================================
descs_mutag = {
    'data_name': 'gen_data_mutag',
    'scenario_desc': 'mutagenicity_prediction_for_aromatic_and_heteroaromatic_compounds',
    'human_role': 'chemist',
    'interaction_verb': 'study',
    'initial_entity': 'molecular_compounds',
    'dataset_type': 'mutag'
}

# ============================================
# ZINC Configuration
# ============================================
descs_zinc = {
    'data_name': 'gen_data_zinc',
    'scenario_desc': 'drug_discovery_and_molecular_property_prediction',
    'human_role': 'medicinal_chemist',
    'interaction_verb': 'evaluate',
    'initial_entity': 'drug-like_molecules',
    'dataset_type': 'zinc'
}

# ============================================
# SELECT YOUR DATASET
# ============================================
# Uncomment the configuration you want to use:

# For MUTAG:
descs = descs_mutag

# For ZINC (uncomment to use):
# descs = descs_zinc

# ============================================
# File paths
# ============================================
file_root = 'gen_results/datasets/{data_name}/'.format(data_name=descs['data_name'])
entity_file = 'gen_results/tree_wInstanceNum_{initial_entity}_{scenario}_{dataset}.pkl'.format(
    initial_entity=descs['initial_entity'],
    scenario=descs['scenario_desc'],
    dataset=descs['dataset_type'])
embed_file = file_root + 'embedding_dict.pkl'

# Create directory if it doesn't exist
os.makedirs(file_root, exist_ok=True)

# ============================================
# Validate entity file exists
# ============================================
if not os.path.exists(entity_file):
    print(f"ERROR: Entity tree file not found: {entity_file}")
    print("Please run instance_number_estimation_hierarchical.py first.")
    exit(1)

print(f"Generating embeddings for: {descs['dataset_type'].upper()}")
print(f"Data name: {descs['data_name']}")
print(f"Scenario: {descs['scenario_desc']}")
print(f"Entity file: {entity_file}")
print("="*80)

# ============================================
# Load entity tree and generate item list
# ============================================
with open(entity_file, 'rb') as fs:
    entity_tree_root = pickle.load(fs)

# Allocate 1 instance to get all categories (with branches)
entity_tree_root.allocate_number(1)
item_list = entity_tree_root.get_list_of_leaves('', with_branches=True)

# Remove instance numbers (e.g., " #0") from item names
item_list = list(map(
    lambda item_name: item_name if ' #' not in item_name else item_name[:item_name.index(' #')], 
    item_list))

print(f"\nTotal items to embed: {len(item_list)}")
print("\nSample items:")
for i, item in enumerate(item_list[:5]):
    print(f"  {i+1}. {item}")
if len(item_list) > 5:
    print(f"  ... and {len(item_list) - 5} more")
print()

# ============================================
# Generate embeddings
# ============================================
print("Generating embeddings using Gemini API...")
print("This may take a while depending on the number of items.")
print("-"*80)

agent = DataGenAgent()
embedding_dict = dict()

for i, item in enumerate(item_list):
    # Progress logging
    if i % 10 == 0:
        log('{idx} / {tot} ({pct:.1f}%)'.format(
            idx=i, 
            tot=len(item_list),
            pct=100*i/len(item_list)))
    
    # Generate embedding for molecular category
    # Add context to improve embedding quality
    if descs['dataset_type'] == 'mutag':
        embedding_text = f"Mutagenic compound category: {item}"
    elif descs['dataset_type'] == 'zinc':
        embedding_text = f"Drug-like molecule category: {item}"
    else:
        embedding_text = item
    
    embedding = agent.openai_embedding(embedding_text)
    embedding_dict[item] = embedding
    
    # Log sample embeddings
    if i < 3:
        print(f"\nItem: {item}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding (first 5 dims): {embedding[:5]}")

# ============================================
# Save embeddings
# ============================================
with open(embed_file, 'wb') as fs:
    pickle.dump(embedding_dict, fs)

print("\n" + "="*80)
print(f"Embedding generation complete!")
print(f"Total embeddings generated: {len(embedding_dict)}")
print(f"Embeddings saved to: {embed_file}")
print(f"Embedding dimension: {list(embedding_dict.values())[0].shape[0]}")

# ============================================
# Validate embeddings
# ============================================
print("\n" + "="*80)
print("Validating embeddings...")

# Check all embeddings have the same dimension
embedding_dims = set([emb.shape[0] for emb in embedding_dict.values()])
if len(embedding_dims) == 1:
    print(f"✓ All embeddings have consistent dimension: {list(embedding_dims)[0]}")
else:
    print(f"✗ WARNING: Inconsistent embedding dimensions found: {embedding_dims}")

# Check for any NaN or inf values
import numpy as np
has_invalid = False
for item, emb in embedding_dict.items():
    if np.isnan(emb).any() or np.isinf(emb).any():
        print(f"✗ WARNING: Invalid values found in embedding for: {item}")
        has_invalid = True

if not has_invalid:
    print("✓ No NaN or inf values found in embeddings")

# Show some basic statistics
all_embeddings = np.array(list(embedding_dict.values()))
print(f"\nEmbedding statistics:")
print(f"  Mean: {np.mean(all_embeddings):.6f}")
print(f"  Std:  {np.std(all_embeddings):.6f}")
print(f"  Min:  {np.min(all_embeddings):.6f}")
print(f"  Max:  {np.max(all_embeddings):.6f}")

# Calculate pairwise similarities for a few examples
if len(item_list) >= 3:
    print(f"\nSample pairwise cosine similarities:")
    from numpy.linalg import norm
    
    for i in range(min(3, len(item_list))):
        for j in range(i+1, min(3, len(item_list))):
            item_i = item_list[i]
            item_j = item_list[j]
            emb_i = embedding_dict[item_i]
            emb_j = embedding_dict[item_j]
            
            # Cosine similarity
            cos_sim = np.dot(emb_i, emb_j) / (norm(emb_i) * norm(emb_j))
            print(f"  {item_i[:50]}...")
            print(f"  vs {item_j[:50]}...")
            print(f"  Similarity: {cos_sim:.4f}\n")

print("="*80)
print("Embedding generation and validation complete!")
print(f"\nNext step: Run human_item_generation_gibbsSampling_embedEstimation.py")