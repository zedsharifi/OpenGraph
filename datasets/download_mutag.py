import torch
from torch_geometric.datasets import TUDataset
import numpy as np
import pickle
import os
import scipy.sparse as sp

def download_and_format_mutag():
    print("Downloading MUTAG dataset...")
    # Download dataset
    dataset = TUDataset(root='./datasets/temp_mutag', name='MUTAG')
    
    # New directory for formatted data
    output_dir = './datasets/mutag/'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loaded {len(dataset)} graphs.")
    
    # --- --- --- --- --- --- --- --- ---
    # Workaround: Merge 188 graphs into one giant, disconnected graph
    # --- --- --- --- --- --- --- --- ---
    
    all_adj_matrices = []
    all_feature_matrices = []
    all_labels = []
    
    node_offset = 0
    
    for graph in dataset:
        # 1. Get Adjacency Matrix
        num_nodes = graph.num_nodes
        # Convert edge_index (COO) to a scipy sparse matrix
        adj = sp.coo_matrix((np.ones(graph.num_edges), (graph.edge_index[0], graph.edge_index[1])),
                            shape=(num_nodes, num_nodes))
        all_adj_matrices.append(adj)
        
        # 2. Get Features
        all_feature_matrices.append(graph.x.numpy())
        
        # 3. Get Labels (Workaround: Use graph label for every node)
        # 0 = non-mutagenic, 1 = mutagenic
        graph_label = graph.y.item()
        node_labels = np.full(num_nodes, graph_label, dtype=np.int32)
        all_labels.append(node_labels)

    print("Merging graphs into a single large component...")
    
    # --- --- --- --- --- --- --- --- ---
    # 1. Save Adjacency Matrix (adj_-1.pkl)
    # --- --- --- --- --- --- --- --- ---
    # Create a block-diagonal matrix: [adj1, 0, 0]
    #                                 [0, adj2, 0]
    #                                 [0, 0, adj3] ...
    merged_adj = sp.block_diag(all_adj_matrices)
    # The framework seems to use sparse.csr_matrix
    merged_adj_csr = merged_adj.tocsr()
    with open(os.path.join(output_dir, 'adj_-1.pkl'), 'wb') as f:
        pickle.dump(merged_adj_csr, f)
    print(f"Saved adj_-1.pkl (Shape: {merged_adj_csr.shape})")

    # --- --- --- --- --- --- --- --- ---
    # 2. Save Features (feats.pkl)
    # --- --- --- --- --- --- --- --- ---
    merged_feats = np.vstack(all_feature_matrices)
    # The framework expects a sparse matrix for feats, so convert
    merged_feats_csr = sp.csr_matrix(merged_feats)
    with open(os.path.join(output_dir, 'feats.pkl'), 'wb') as f:
        pickle.dump(merged_feats_csr, f)
    print(f"Saved feats.pkl (Shape: {merged_feats_csr.shape})")

    # --- --- --- --- --- --- --- --- ---
    # 3. Save Labels (label.pkl)
    # --- --- --- --- --- --- --- --- ---
    merged_labels = np.concatenate(all_labels)
    # The framework expects labels to be one-hot encoded
    num_classes = dataset.num_classes
    merged_labels_onehot = np.eye(num_classes)[merged_labels]
    with open(os.path.join(output_dir, 'label.pkl'), 'wb') as f:
        pickle.dump(merged_labels_onehot, f)
    print(f"Saved label.pkl (Shape: {merged_labels_onehot.shape})")

    # --- --- --- --- --- --- --- --- ---
    # 4. Save Masks (mask_-1.pkl)
    # --- --- --- --- --- --- --- --- ---
    # Create an 80/10/10 split based on the *nodes*
    num_total_nodes = merged_feats.shape[0]
    indices = np.arange(num_total_nodes)
    np.random.shuffle(indices)
    
    train_size = int(num_total_nodes * 0.8)
    val_size = int(num_total_nodes * 0.1)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]
    
    train_mask = np.zeros(num_total_nodes, dtype=bool)
    val_mask = np.zeros(num_total_nodes, dtype=bool)
    test_mask = np.zeros(num_total_nodes, dtype=bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    # --- FIX ---
    # Save as a dictionary of masks, as expected by data_handler.py
    # The traceback shows it expects keys: 'train', 'valid', 'test'
    masks = {
        'train': train_mask,
        'valid': val_mask,
        'test': test_mask
    }
    with open(os.path.join(output_dir, 'mask_-1.pkl'), 'wb') as f:
        pickle.dump(masks, f)
    print(f"Saved mask_-1.pkl (Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)})")
    
    print("\nFormat conversion complete. MUTAG dataset is ready at './datasets/mutag/'")

if __name__ == "__main__":
    # Ensure you have the dependencies:
    # pip install torch torch-geometric scipy numpy
    download_and_format_mutag()