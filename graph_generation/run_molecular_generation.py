"""
Complete pipeline for generating molecular graph datasets using Tree of Prompts.
Combines all steps into a single workflow.

Usage:
    python run_molecular_generation.py --config mutag
    python run_molecular_generation.py --config zinc --depth 4
"""

import os
import sys
import argparse
import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import random

# Import the molecular generation modules
from molecularCollecting_dfsIterator import MolecularDataGenAgent
from Molecular_Utils import MolecularEntityTreeConstructer, MolecularDataGenAgent as EmbedAgent

# Predefined configurations
CONFIGS = {
    'mutag': {
        'entity': 'mutagenic compounds',
        'scenario': 'Ames mutagenicity test',
        'depth': 3,
        'total_molecules': 200,
        'description': 'MUTAG-style mutagenicity prediction'
    },
    'zinc': {
        'entity': 'drug-like molecules',
        'scenario': 'pharmaceutical compound database',
        'depth': 4,
        'total_molecules': 500,
        'description': 'ZINC-style drug-like molecules'
    },
    'toxic': {
        'entity': 'toxic organic compounds',
        'scenario': 'environmental toxicity assessment',
        'depth': 3,
        'total_molecules': 300,
        'description': 'Environmental toxicity prediction'
    },
    'bioactive': {
        'entity': 'bioactive molecules',
        'scenario': 'biological activity screening',
        'depth': 4,
        'total_molecules': 400,
        'description': 'Bioactivity classification'
    }
}


class MolecularDatasetBuilder:
    """Build complete molecular graph dataset from entity tree"""
    
    def __init__(self, entity_file, total_molecules, output_dir):
        self.entity_file = entity_file
        self.total_molecules = total_molecules
        self.output_dir = output_dir
        
        # Load entity tree
        print(f"\n📂 Loading molecular category tree...")
        with open(entity_file, 'r') as f:
            self.entity_lines = [line.strip() for line in f if line.strip()]
        print(f"  ✓ Loaded {len(self.entity_lines)} molecular categories")
    
    def parse_molecule_from_entity(self, entity_line, mol_id):
        """Parse entity line and create molecule structure"""
        # Extract category hierarchy
        parts = entity_line.split(',')
        category_path = [p.strip() for p in parts]
        
        # Infer properties from category
        mutagenic = self.infer_mutagenicity(category_path)
        num_atoms = self.infer_atom_count(category_path)
        
        return {
            'id': mol_id,
            'category': entity_line,
            'category_path': category_path,
            'num_atoms': num_atoms,
            'label': mutagenic
        }
    
    def infer_mutagenicity(self, category_path):
        """Infer mutagenicity from category path"""
        category_str = ' '.join(category_path).lower()
        
        # High-risk indicators
        if any(x in category_str for x in ['nitro', 'aromatic amine', 'polycyclic aromatic',
                                             'epoxide', 'azo', 'quinone']):
            return 1
        
        # Medium-risk indicators
        elif any(x in category_str for x in ['aromatic', 'heterocycle', 'halogenated']):
            return random.choice([0, 1])  # 50/50 chance
        
        # Low-risk indicators
        elif any(x in category_str for x in ['aliphatic', 'alcohol', 'simple']):
            return 0 if random.random() > 0.3 else 1
        
        # Default: balanced
        return random.randint(0, 1)
    
    def infer_atom_count(self, category_path):
        """Infer reasonable atom count from category"""
        category_str = ' '.join(category_path).lower()
        
        # Large molecules
        if any(x in category_str for x in ['polycyclic', 'multi-ring', 'complex']):
            return random.randint(18, 25)
        
        # Medium molecules
        elif any(x in category_str for x in ['aromatic', 'benzene', 'heterocycle']):
            return random.randint(12, 18)
        
        # Small molecules
        elif any(x in category_str for x in ['simple', 'aliphatic', 'small']):
            return random.randint(8, 12)
        
        # Default: medium-sized
        return random.randint(10, 20)
    
    def generate_atom_features(self, molecule):
        """Generate atom features (one-hot encoded)"""
        num_atoms = molecule['num_atoms']
        
        # Assign atom types based on category
        category_str = ' '.join(molecule['category_path']).lower()
        atom_types = []
        
        for i in range(num_atoms):
            if 'nitrogen' in category_str or 'amine' in category_str:
                if i < num_atoms * 0.5:
                    atom_types.append(0)  # Carbon
                elif i < num_atoms * 0.7:
                    atom_types.append(1)  # Nitrogen
                else:
                    atom_types.append(7)  # Hydrogen
            elif 'oxygen' in category_str or 'hydroxyl' in category_str:
                if i < num_atoms * 0.5:
                    atom_types.append(0)  # Carbon
                elif i < num_atoms * 0.65:
                    atom_types.append(2)  # Oxygen
                else:
                    atom_types.append(7)  # Hydrogen
            else:
                # Default distribution
                if i < num_atoms * 0.6:
                    atom_types.append(0)  # Carbon
                elif i < num_atoms * 0.75:
                    atom_types.append(7)  # Hydrogen
                elif i < num_atoms * 0.85:
                    atom_types.append(1)  # Nitrogen
                else:
                    atom_types.append(2)  # Oxygen
        
        # One-hot encoding (8 atom types: C, N, O, S, F, Cl, Br, H)
        features = np.zeros((num_atoms, 8))
        for i, atom_type in enumerate(atom_types):
            features[i, atom_type] = 1
        
        return features
    
    def generate_edges(self, molecule):
        """Generate molecular bonds (edges)"""
        num_atoms = molecule['num_atoms']
        edges = []
        
        # Create backbone chain
        for i in range(num_atoms - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
        
        category_str = ' '.join(molecule['category_path']).lower()
        
        # Add rings for aromatic compounds
        if 'aromatic' in category_str or 'benzene' in category_str:
            if num_atoms >= 6:
                edges.append((0, 5))
                edges.append((5, 0))
                if num_atoms >= 10:  # Naphthalene-like
                    edges.append((5, 9))
                    edges.append((9, 5))
        
        # Add branches
        num_branches = min(3, num_atoms // 4)
        for _ in range(num_branches):
            i = random.randint(1, num_atoms - 2)
            j = random.randint(max(0, i - 3), min(num_atoms - 1, i + 3))
            if i != j and (i, j) not in edges:
                edges.append((i, j))
                edges.append((j, i))
        
        return list(set(edges))
    
    def build_dataset(self):
        """Build complete graph dataset"""
        print("\n" + "="*70)
        print("Building Molecular Graph Dataset")
        print("="*70)
        
        # Sample molecules from entity tree
        molecules_per_category = max(1, self.total_molecules // len(self.entity_lines))
        
        molecules = []
        mol_id = 0
        
        for cat_idx, entity_line in enumerate(self.entity_lines):
            if mol_id >= self.total_molecules:
                break
            
            # Generate multiple molecules per category
            for _ in range(molecules_per_category):
                if mol_id >= self.total_molecules:
                    break
                
                mol = self.parse_molecule_from_entity(entity_line, mol_id)
                molecules.append(mol)
                mol_id += 1
            
            if (cat_idx + 1) % 10 == 0:
                print(f"  Processed {cat_idx + 1}/{len(self.entity_lines)} categories...")
        
        print(f"\n✓ Generated {len(molecules)} molecules")
        
        # Build merged graph
        print("\n📊 Creating merged graph structure...")
        all_features = []
        all_labels = []
        all_edges_row = []
        all_edges_col = []
        node_offset = 0
        
        for idx, mol in enumerate(molecules):
            if idx % 50 == 0 and idx > 0:
                print(f"  Processing molecule {idx}/{len(molecules)}")
            
            features = self.generate_atom_features(mol)
            edges = self.generate_edges(mol)
            
            all_features.append(features)
            all_labels.append(np.full(mol['num_atoms'], mol['label'], dtype=np.int32))
            
            for i, j in edges:
                all_edges_row.append(node_offset + i)
                all_edges_col.append(node_offset + j)
            
            node_offset += mol['num_atoms']
        
        # Merge
        merged_features = np.vstack(all_features)
        merged_labels = np.concatenate(all_labels)
        
        num_nodes = merged_features.shape[0]
        adj_data = np.ones(len(all_edges_row), dtype=np.float32)
        merged_adj = coo_matrix(
            (adj_data, (all_edges_row, all_edges_col)),
            shape=(num_nodes, num_nodes)
        ).tocsr()
        
        print(f"\n✓ Merged graph statistics:")
        print(f"  Nodes: {num_nodes}")
        print(f"  Edges: {len(all_edges_row)}")
        print(f"  Label distribution: {np.bincount(merged_labels)}")
        
        # Create splits
        print("\n📊 Creating train/val/test splits...")
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        
        train_size = int(num_nodes * 0.8)
        val_size = int(num_nodes * 0.1)
        
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        # Save
        print(f"\n💾 Saving dataset to {self.output_dir}...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(os.path.join(self.output_dir, 'adj_-1.pkl'), 'wb') as f:
            pickle.dump(merged_adj, f)
        
        features_sparse = csr_matrix(merged_features)
        with open(os.path.join(self.output_dir, 'feats.pkl'), 'wb') as f:
            pickle.dump(features_sparse, f)
        
        labels_onehot = np.eye(2)[merged_labels]
        with open(os.path.join(self.output_dir, 'label.pkl'), 'wb') as f:
            pickle.dump(labels_onehot, f)
        
        masks = {'train': train_mask, 'valid': val_mask, 'test': test_mask}
        with open(os.path.join(self.output_dir, 'mask_-1.pkl'), 'wb') as f:
            pickle.dump(masks, f)
        
        print("\n" + "="*70)
        print("✓ Dataset Generation Complete!")
        print("="*70)
        print(f"Saved to: {self.output_dir}")
        print(f"  Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")


def main():
    parser = argparse.ArgumentParser(description='Generate molecular graph datasets')
    parser.add_argument('--config', type=str, default='mutag', choices=list(CONFIGS.keys()),
                       help='Predefined configuration')
    parser.add_argument('--depth', type=int, default=None,
                       help='Override tree depth')
    parser.add_argument('--num_molecules', type=int, default=None,
                       help='Override number of molecules')
    parser.add_argument('--skip_tree', action='store_true',
                       help='Skip tree generation (use existing tree file)')
    args = parser.parse_args()
    
    # Load configuration
    config = CONFIGS[args.config].copy()
    if args.depth:
        config['depth'] = args.depth
    if args.num_molecules:
        config['total_molecules'] = args.num_molecules
    
    print("\n" + "🧬"*35)
    print("Molecular Graph Dataset Generator")
    print("🧬"*35)
    print(f"\nConfiguration: {args.config}")
    print(f"Description: {config['description']}")
    print(f"Entity: {config['entity']}")
    print(f"Scenario: {config['scenario']}")
    print(f"Depth: {config['depth']}")
    print(f"Molecules: {config['total_molecules']}")
    print("="*70 + "\n")
    
    # File paths
    tree_file = f"gen_results/molecular_tree/{config['entity'].replace(' ', '_')}_{config['scenario'].replace(' ', '_')}.txt"
    output_dir = f"./datasets/{args.config}_tree_gen/"
    
    # Step 1: Generate entity tree
    if not args.skip_tree:
        print("\n" + "="*70)
        print("STEP 1: Generating Molecular Category Tree")
        print("="*70)
        
        agent = MolecularDataGenAgent(
            config['entity'],
            config['scenario'],
            config['depth']
        )
        
        nodes = agent.run()
        
        # Save tree
        os.makedirs(os.path.dirname(tree_file), exist_ok=True)
        with open(tree_file, 'w+') as fs:
            for node in nodes:
                fs.write(node + '\n')
        
        print(f"\n✓ Saved entity tree to: {tree_file}")
        print(f"  Total categories: {len(nodes)}")
        print(f"  Tokens used: {agent.token_num}")
    else:
        print(f"\n⏭ Skipping tree generation, using existing file: {tree_file}")
    
    # Step 2: Build dataset
    print("\n" + "="*70)
    print("STEP 2: Building Graph Dataset")
    print("="*70)
    
    builder = MolecularDatasetBuilder(
        tree_file,
        config['total_molecules'],
        output_dir
    )
    
    builder.build_dataset()
    
    # Final instructions
    print("\n" + "="*70)
    print("🎉 All Done!")
    print("="*70)
    print("\nTest your dataset with:")
    print("  cd node_classification/")
    print(f"  python main.py --load pretrn_gen1 --tstdata {args.config}_tree_gen")
    print("\nOr train from scratch:")
    print(f"  python main.py --tstdata {args.config}_tree_gen --save {args.config}_model")
    print()


if __name__ == "__main__":
    main()