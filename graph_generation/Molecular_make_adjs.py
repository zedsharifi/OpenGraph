import pickle
import os
import argparse
from scipy.sparse import coo_matrix
import numpy as np
import random
import networkx as nx

# ============================================
# MUTAG Configuration
# ============================================
descs_mutag = {
    'data_name': 'gen_data_mutag',
}

params_mutag = {
    'itmfusion': True,  # Fuse item instances
    'kcore': 0,  # k-core filtering (0 = no filtering)
    'sep': 1,  # Sample separation (1 = keep all)
    'min_base': 0,
    'max_base': 100,
}

# ============================================
# ZINC Configuration
# ============================================
descs_zinc = {
    'data_name': 'gen_data_zinc',
}

params_zinc = {
    'itmfusion': True,
    'kcore': 0,  # No k-core filtering for molecular graphs
    'sep': 1,  # Keep all samples
    'min_base': 0,
    'max_base': 100,
}

# ============================================
# SELECT YOUR DATASET
# ============================================
# Uncomment the configuration you want to use:

# For MUTAG:
descs = descs_mutag
params = params_mutag

# For ZINC (uncomment to use):
# descs = descs_zinc
# params = params_zinc

# ============================================
# Command line arguments
# ============================================
parser = argparse.ArgumentParser(description='Generate adjacency matrices for molecular graphs')
parser.add_argument('--gen_iter', default=0, type=int, help='generation iteration')
parser.add_argument('--dataset', default=None, type=str, help='dataset name (mutag or zinc)')
args = parser.parse_args()

# Override dataset if specified in command line
if args.dataset:
    if args.dataset.lower() == 'mutag':
        descs = descs_mutag
        params = params_mutag
    elif args.dataset.lower() == 'zinc':
        descs = descs_zinc
        params = params_zinc
    else:
        print(f"Unknown dataset: {args.dataset}")
        print("Please use 'mutag' or 'zinc'")
        exit(1)

file_root = 'gen_results/datasets/{data_name}/'.format(data_name=descs['data_name'])
fuse_file_path = file_root + 'res/interaction_fuse_iter-{iter}.pkl'.format(iter=args.gen_iter)

print(f"Processing dataset: {descs['data_name'].upper()}")
print(f"Generation iteration: {args.gen_iter}")
print("="*80)
print("Parameters:")
for key, value in params.items():
    print(f"  {key}: {value}")
print("="*80)

def get_all_bases():
    """Get all available generation bases"""
    bases = set()
    prefix = 'interaction_base-'
    suffix = '_iter-'
    
    if not os.path.exists(file_root):
        print(f"ERROR: Directory not found: {file_root}")
        print("Please run human_item_generation_gibbsSampling_embedEstimation.py first.")
        exit(1)
    
    for filename in os.listdir(file_root):
        if prefix in filename and suffix in filename:
            prefix_idx = len(prefix)
            suffix_idx = filename.index(suffix)
            base = int(filename[prefix_idx: suffix_idx])
            if base >= params['min_base'] and base <= params['max_base']:
                bases.add(base)
    bases = list(bases)
    bases.sort()
    return bases

def fuse_bases():
    """Fuse all generation bases into a single interaction list"""
    # Create res directory
    os.makedirs(file_root + 'res/', exist_ok=True)
    
    if os.path.exists(fuse_file_path):
        print('Fused interaction file exists! Loading existing file...')
        print('WARNING: This may be from a previous run!')
        with open(fuse_file_path, 'rb') as fs:
            interactions = pickle.load(fs)
        return interactions
    
    all_bases = get_all_bases()
    if len(all_bases) == 0:
        print("ERROR: No interaction files found!")
        print(f"Looking for files matching: {file_root}interaction_base-*_iter-*.pkl")
        exit(1)
    
    print(f"Found {len(all_bases)} generation bases: {all_bases}")
    
    interactions = []
    for gen_base in all_bases:
        file_path = None
        for iter in range(args.gen_iter, -1, -1):
            file_path = file_root + 'interaction_base-{gen_base}_iter-{gen_iter}.pkl'.format(
                gen_base=gen_base, gen_iter=iter)
            if os.path.exists(file_path):
                break
        
        if file_path is None or not os.path.exists(file_path):
            print(f"WARNING: No file found for base {gen_base}")
            continue
        
        with open(file_path, 'rb') as fs:
            tem_cur_base_interactions = pickle.load(fs)
            cur_base_interactions = []
            for i in range(len(tem_cur_base_interactions)):
                cur_base_interactions.append(tem_cur_base_interactions[i])
        
        print(f"Loaded {len(cur_base_interactions)} graphs from base {gen_base}")
        interactions += cur_base_interactions
    
    # Sample separation (if needed)
    new_interactions = []
    for i in range(len(interactions)):
        if i % params['sep'] == 0:
            new_interactions.append(interactions[i])
    interactions = new_interactions
    
    print(f"\nTotal molecular graphs after fusion: {len(interactions)}")
    
    with open(fuse_file_path, 'wb+') as fs:
        pickle.dump(interactions, fs)
    
    return interactions

def make_id_map(interactions, itm_criteria=None):
    """Create ID mappings for molecules and categories"""
    u_num = len(interactions)  # Number of molecules
    i_set = set()  # Set of unique categories
    i_cnt = dict()  # Category frequency
    
    for interaction in interactions:
        for item in interaction:
            num_idx = item.index(' #')
            tem_item = item if not params['itmfusion'] else item[:num_idx]
            i_set.add(tem_item)
            if tem_item not in i_cnt:
                i_cnt[tem_item] = 0
            i_cnt[tem_item] += 1
    
    i_list = list(i_set)
    
    # Apply category filtering criteria if provided
    if itm_criteria is not None:
        tem_i_list = list()
        for item in i_list:
            if itm_criteria(i_cnt[item]):
                tem_i_list.append(item)
        print('Filtering categories: {new_num} / {old_num}'.format(
            new_num=len(tem_i_list), old_num=len(i_list)))
        i_list = tem_i_list
    
    i_num = len(i_list)
    i_mapp = dict()
    for i, item in enumerate(i_list):
        i_mapp[item] = i
    
    # Create edge list (molecule-category pairs)
    rows = []  # Molecule IDs
    cols = []  # Category IDs
    for uid, interaction in enumerate(interactions):
        for item in interaction:
            num_idx = item.index(' #')
            tem_item = item if not params['itmfusion'] else item[:num_idx]
            if tem_item not in i_mapp:
                continue
            iid = i_mapp[tem_item]
            rows.append(uid)
            cols.append(iid)
    
    return rows, cols, i_mapp, u_num, i_num

def id_map(nodes):
    """Create ID mapping for nodes"""
    uniq_nodes = list(set(nodes))
    dic = dict()
    for i, node in enumerate(uniq_nodes):
        dic[node] = i
    return dic

def k_core(rows, cols, i_mapp, i_num, k):
    """Apply k-core decomposition to filter the graph"""
    if k == 0:
        return rows, cols, i_mapp, len(set(rows)), i_num
    
    print(f"\nApplying {k}-core decomposition...")
    
    # Create bipartite graph (molecules + categories)
    edge_list = list(map(lambda idx: (rows[idx] + i_num, cols[idx]), range(len(rows))))
    G = nx.Graph(edge_list)
    
    # Apply k-core
    core_graph = nx.k_core(G, k=k)
    edge_list = list(core_graph.edges())
    
    rows = [None] * len(edge_list)
    cols = [None] * len(edge_list)
    
    for i, edge in enumerate(edge_list):
        if edge[0] < i_num:
            rows[i] = edge[1] - i_num
            cols[i] = edge[0]
        else:
            rows[i] = edge[0] - i_num
            cols[i] = edge[1]
    
    # Remap IDs to be contiguous
    row_map = id_map(rows)
    col_map = id_map(cols)
    new_rows = list(map(lambda x: row_map[x], rows))
    new_cols = list(map(lambda x: col_map[x], cols))
    
    # Update category mapping
    new_i_mapp = dict()
    for key in i_mapp:
        tem_item = i_mapp[key]
        if tem_item not in col_map:
            continue
        new_i_mapp[key] = col_map[tem_item]
    
    print(f"After {k}-core: {len(row_map)} molecules, {len(col_map)} categories, {len(new_rows)} edges")
    
    return new_rows, new_cols, new_i_mapp, len(row_map), len(col_map)

def make_mat(rows, cols, st, ed, u_num, i_num, perm, decrease=False):
    """Create sparse matrix from edge list"""
    rows = np.array(rows)[perm]
    cols = np.array(cols)[perm]
    rows = rows[st: ed]
    cols = cols[st: ed]
    
    if decrease:
        # Reduce test set size
        rows = rows[:len(rows)//3]
        cols = cols[:len(cols)//3]
    
    vals = np.ones_like(rows)
    return coo_matrix((vals, (rows, cols)), shape=[u_num, i_num])

def data_split(rows, cols, u_num, i_num):
    """Split data into train/val/test sets"""
    leng = len(rows)
    perm = np.random.permutation(leng)
    
    # 70% train, 5% val, 25% test (reduced to ~8% after decrease)
    trn_split = int(leng * 0.7)
    val_split = int(leng * 0.75)
    
    trn_mat = make_mat(rows, cols, 0, trn_split, u_num, i_num, perm)
    val_mat = make_mat(rows, cols, trn_split, val_split, u_num, i_num, perm)
    tst_mat = make_mat(rows, cols, val_split, leng, u_num, i_num, perm, decrease=True)
    
    return trn_mat, val_mat, tst_mat

# ============================================
# Main processing
# ============================================
print("\nStep 1: Fusing interaction files...")
interactions = fuse_bases()

print("\nStep 2: Creating ID mappings...")
rows, cols, i_mapp, u_num, i_num = make_id_map(interactions)

print(f"\nBefore filtering:")
print(f"  Molecules: {u_num}")
print(f"  Categories: {i_num}")
print(f"  Edges: {len(rows)}")
print(f"  Avg edges per molecule: {len(rows)/u_num:.2f}")

if params['kcore'] != 0:
    print("\nStep 3: Applying k-core decomposition...")
    rows, cols, i_mapp, u_num, i_num = k_core(rows, cols, i_mapp, i_num, params['kcore'])
else:
    print("\nStep 3: Skipping k-core decomposition (kcore=0)")

print(f"\nFinal graph statistics:")
print(f"  Molecules: {u_num}")
print(f"  Categories: {i_num}")
print(f"  Edges: {len(rows)}")
print(f"  Avg edges per molecule: {len(rows)/u_num:.2f}")
print(f"  Density: {len(rows)/(u_num*i_num):.6f}")

# ============================================
# Save category mapping
# ============================================
print("\nStep 4: Saving category mapping...")
res_dir = file_root + 'res/'
os.makedirs(res_dir, exist_ok=True)

imap_file = res_dir + 'iter-{gen_iter}_imap.pkl'.format(gen_iter=args.gen_iter)
with open(imap_file, 'wb+') as fs:
    pickle.dump(i_mapp, fs)
print(f"  Saved to: {imap_file}")

# ============================================
# Split and save datasets
# ============================================
print("\nStep 5: Splitting into train/val/test...")
trn_mat, val_mat, tst_mat = data_split(rows, cols, u_num, i_num)

print(f"  Train: {trn_mat.nnz} edges ({trn_mat.nnz/len(rows)*100:.1f}%)")
print(f"  Val:   {val_mat.nnz} edges ({val_mat.nnz/len(rows)*100:.1f}%)")
print(f"  Test:  {tst_mat.nnz} edges ({tst_mat.nnz/len(rows)*100:.1f}%)")

train_file = res_dir + 'iter-{gen_iter}_train.pkl'.format(gen_iter=args.gen_iter)
with open(train_file, 'wb+') as fs:
    pickle.dump(trn_mat, fs)
print(f"\n  Train saved to: {train_file}")

valid_file = res_dir + 'iter-{gen_iter}_valid.pkl'.format(gen_iter=args.gen_iter)
with open(valid_file, 'wb+') as fs:
    pickle.dump(val_mat, fs)
print(f"  Valid saved to: {valid_file}")

test_file = res_dir + 'iter-{gen_iter}_test.pkl'.format(gen_iter=args.gen_iter)
with open(test_file, 'wb+') as fs:
    pickle.dump(tst_mat, fs)
print(f"  Test saved to:  {test_file}")

# ============================================
# Summary
# ============================================
print("\n" + "="*80)
print("Dataset generation complete!")
print("="*80)
print(f"Dataset: {descs['data_name']}")
print(f"Generation iteration: {args.gen_iter}")
print(f"\nFiles created in: {res_dir}")
print(f"  - iter-{args.gen_iter}_imap.pkl (category mapping)")
print(f"  - iter-{args.gen_iter}_train.pkl (training set)")
print(f"  - iter-{args.gen_iter}_valid.pkl (validation set)")
print(f"  - iter-{args.gen_iter}_test.pkl (test set)")
print(f"  - interaction_fuse_iter-{args.gen_iter}.pkl (fused interactions)")

print(f"\nNext steps:")
print(f"  1. Copy the generated files to the datasets/{descs['data_name']}/ directory")
print(f"  2. Update data_handler.py to support the new dataset")
print(f"  3. Run training/testing with OpenGraph")

print("\nUsage example:")
print(f"  python make_adjs.py --gen_iter 0 --dataset {descs['data_name'].replace('gen_data_', '')}")