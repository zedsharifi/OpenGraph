import random
from Utils import DataGenAgent
import pickle
import os
import Exp_Utils.TimeLogger as logger
from Exp_Utils.TimeLogger import log
from Exp_Utils.Emailer import SendMail
import numpy as np
from scipy.stats import norm
import copy

class MolecularGraphGeneration(DataGenAgent):
    def __init__(self, item_list, length_sampler, descs, hyperparams, text_embedding_dict=None):
        super(MolecularGraphGeneration, self).__init__()

        self.item_list = item_list
        self.length_sampler = length_sampler
        self.descs = descs
        self.hyperparams = hyperparams
        self.hyperparams['item_num'] = len(self.item_list)
        self.item_to_id = dict()
        for iid, item in enumerate(item_list):
            self.item_to_id[item] = iid
        self.reject_cnt = 0
        self.item_perm = np.random.permutation(len(self.item_list))
        self.score_history = []
        self.text_embedding_dict = dict() if text_embedding_dict is None else text_embedding_dict
    
    def binvec2list(self, bin_sample_vec):
        """Convert binary vector to list of items"""
        idxs = np.reshape(np.argwhere(bin_sample_vec != 0), [-1])
        return list(map(lambda x: self.item_list[x], idxs))
    
    def list_text(self, item_list, nums):
        """Format item list with frequencies"""
        item_num_list = list(zip(item_list, nums))
        item_num_list.sort(key=lambda x: x[1], reverse=True)
        ret = ''
        for i, pair in enumerate(item_num_list):
            item, num = pair[0], pair[1]
            ret += '{idx}. {item}. Frequency: {num}\n'.format(idx=i, item=item, num=num)
        return ret

    def summarize(self, item_list):
        """Summarize item list by aggregating similar categories"""
        def fuse(item_list, prefix):
            for i in range(len(item_list)):
                item = item_list[i]
                if item.startswith(prefix):
                    item_list[i] = prefix
            return item_list
        
        def count_and_shrink(item_list):
            dic = dict()
            for item in item_list:
                if item not in dic:
                    dic[item] = 0
                dic[item] += 1
            ret_item, ret_cnt = [], []
            for key, cnt in dic.items():
                ret_item.append(key)
                ret_cnt.append(cnt)
            return ret_item, ret_cnt
        
        def count_prefixes_of_different_depth(item_list, max_depth):
            ret_item_list = []
            prefix_dicts = [dict() for i in range(max_depth + 1)]
            for item in item_list:
                num_idx = item.index(' #')
                tem_item = item[:num_idx]
                ret_item_list.append(tem_item)
                entities = tem_item.split(', ')
                entities = list(map(lambda entity: entity.strip(), entities))
                for depth in range(max_depth + 1):
                    if depth + 1 >= len(entities):
                        break
                    tem_prefix = ', '.join(entities[:depth + 1])
                    if tem_prefix not in prefix_dicts[depth]:
                        prefix_dicts[depth][tem_prefix] = 0
                    prefix_dicts[depth][tem_prefix] += 1
            return prefix_dicts, ret_item_list
        
        max_depth = len(item_list[0].split(', '))
        prefix_dicts, ret_item_list = count_prefixes_of_different_depth(item_list, max_depth)
        if len(ret_item_list) < self.hyperparams['context_limit']:
            return ret_item_list, [1] * len(ret_item_list)
        
        # Greedy search to aggregate items
        flag = False
        for depth in range(max_depth, -1, -1):
            prefix_list = [(prefix, cnt) for prefix, cnt in prefix_dicts[depth].items()]
            prefix_list.sort(key=lambda x: x[1], reverse=True)
            for prefix, cnt in prefix_list:
                if cnt == 1:
                    break
                ret_item_list = fuse(ret_item_list, prefix)
                if depth != 0:
                    shrinked_prefix = ', '.join(prefix.split(', ')[:-1])
                    prefix_dicts[depth - 1][shrinked_prefix] -= cnt - 1
                if len(set(ret_item_list)) <= self.hyperparams['context_limit']:
                    flag=True
                    break
            if flag:
                return count_and_shrink(ret_item_list)
        return count_and_shrink(ret_item_list)
    
    def text_embedding(self, text):
        """Get or generate text embedding"""
        if text in self.text_embedding_dict:
            embeds = self.text_embedding_dict[text]
            return embeds
        print('Embedding not found!')
        print(text)
        exit()
    
    def similarity(self, fst_embed, scd_embed):
        """Compute similarity between two embeddings"""
        return np.sum(fst_embed * scd_embed)
    
    def dynamic_normalize(self, score):
        """Dynamically normalize scores based on history"""
        self.score_history.append(score)
        if len(self.score_history) > 5000:
            self.score_history = self.score_history[-5000:]
        if len(self.score_history) < 5:
            return max(min(1.0, score), 0.0)
        score_samples = np.array(self.score_history)
        mean = np.mean(score_samples)
        std = np.sqrt(np.mean((score_samples - mean) ** 2))
        minn = mean - 1.96 * std
        maxx = mean + 1.96 * std
        ret = (score - minn) / (maxx - minn)
        ret = max(min(ret, 1.0), 0.0)
        return ret

    def estimate_probability(self, bin_sample_vec, cur_dim, is_deleting=False):
        """Estimate probability for adding/removing an item (molecular category)"""
        item_list = self.binvec2list(bin_sample_vec)
        new_item = self.item_list[cur_dim]
        candidate_embedding = self.text_embedding(new_item[:new_item.index(' #')])

        # Average embedding similarity approach
        embed_list = list(map(lambda item: self.text_embedding(item[:item.index(' #')]), item_list))
        avg_embed = sum(embed_list) / len(embed_list)
        instance_score = np.sum(avg_embed * candidate_embedding)

        score = instance_score
        
        # Interaction number probability (molecular graph size preference)
        interaction_num = np.sum(bin_sample_vec != 0)
        interaction_prob = 1.0 / (1.0 + np.exp(
            (interaction_num - self.hyperparams['length_center']) / 
            (self.hyperparams['length_center'] // 2)))
        score = score * interaction_prob
        
        return score
    
    def update_sample(self, last_sample, cur_dim, should_include):
        """Update sample by including or excluding an item"""
        if last_sample[cur_dim] == 0.0 and should_include or last_sample[cur_dim] > 0.0 and not should_include:
            new_sample = copy.deepcopy(last_sample)
            new_sample[cur_dim] = 1.0 - last_sample[cur_dim]
            return new_sample, True
        else:
            return last_sample, False

    def Gibbs_Sampling(self):
        """Generate molecular graphs using Gibbs sampling"""
        samples = []
        idx = 0
        update_cnt = 0
        cur_community = 0
        
        for step in range(self.hyperparams['sample_num']):
            # Periodic restart
            if step % self.hyperparams['restart_step'] == 0:
                samples.append(self.random_sample())
                cur_community = (cur_community + 1) % self.hyperparams['community_num']
            
            last_sample = samples[-1]
            update_flag = False
            
            for small_step in range(self.hyperparams['gibbs_step']):
                cur_dim = self.item_perm[idx]
                nnz = np.sum(last_sample != 0)
                delete_dice = random.uniform(0, 1)
                
                # Deletion bias for large graphs
                if nnz > self.hyperparams['delete_nnz'] and delete_dice < 0.5:
                    cur_dim = np.random.choice(np.reshape(np.argwhere(last_sample > 0.0), [-1]))
                
                tem_delete_flag = False
                if last_sample[cur_dim] > 0.0:
                    tem_delete_flag = True
                    last_sample[cur_dim] = 0.0
                
                idx = (idx + 1) % len(self.item_list)
                self.failure = 0
                prob = self.estimate_probability(last_sample, cur_dim, tem_delete_flag)
                
                # Community modifier (for diversity)
                diff = abs(cur_community - cur_dim % self.hyperparams['community_num'])
                prob *= self.hyperparams['com_decay'] ** diff

                dice = random.uniform(0, 1) - self.hyperparams['dice_shift']
                last_sample, change_flag = self.update_sample(last_sample, cur_dim, dice < prob)
                
                if tem_delete_flag:
                    change_flag = not change_flag
                
                if change_flag:
                    if small_step == 0:
                        log('Sample Updated! Step {step}_{small_step}, update cnt {update_cnt}, num nodes {int_num}, sample num {samp_num}'.format(
                            step=step, small_step=small_step, update_cnt=update_cnt, 
                            int_num=np.sum(last_sample!=0.0), samp_num=len(samples)), oneline=True)
                    self.reject_cnt = 0
                    update_cnt += 1
                    update_flag = True
                else:
                    if small_step == 0:
                        log('Sample UNCHANGED! Step {step}_{small_step}, update cnt {update_cnt}, num nodes {int_num}, sample num {samp_num}'.format(
                            step=step, small_step=small_step, update_cnt=update_cnt, 
                            int_num=np.sum(last_sample!=0.0), samp_num=len(samples)), oneline=True)
                    self.reject_cnt += 1
                
                if self.reject_cnt > 50:
                    log('Consecutive rejection {rej_cnt} when sampling!'.format(rej_cnt=self.reject_cnt), save=True)
                    log('Last sample: {last_sample}'.format(last_sample=self.binvec2list(samples[-1])))
                    self.reject_cnt = 0
                    break
            
            if update_flag:
                if step % self.hyperparams['restart_step'] < self.hyperparams['gibbs_skip_step']:
                    samples[-1] = last_sample
                else:
                    samples.append(last_sample)
        
        return samples
    
    def random_sample(self):
        """Generate a random initial sample"""
        picked_idxs = random.sample(list(range(len(self.item_list))), self.hyperparams['seed_num'])
        last_interaction = np.zeros(len(self.item_list))
        last_interaction[picked_idxs] = 1.0
        return last_interaction
    
    def run(self):
        """Run the molecular graph generation process"""
        samples_binvec = self.Gibbs_Sampling()
        picked_items_list = []
        for vec in samples_binvec:
            picked_items = self.binvec2list(vec)
            picked_items_list.append(picked_items)
        return picked_items_list


def load_item_list(item_file, entity_file, item_num):
    """Load or generate item list"""
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

def get_gen_iter(file_root, interaction_file_prefix):
    """Get the next generation iteration number"""
    max_existing_iter = -1
    if not os.path.exists(file_root):
        return max_existing_iter
    for filename in os.listdir(file_root):
        cur_filename = file_root + filename
        if interaction_file_prefix in cur_filename:
            st_idx = len(interaction_file_prefix)
            ed_idx = cur_filename.index('_iter-0.pkl')
            cur_iter = int(cur_filename[st_idx: ed_idx])
            max_existing_iter = max(max_existing_iter, cur_iter)
    return max_existing_iter

def load_embedding_dict(embed_file):
    """Load pre-computed embeddings"""
    if not os.path.exists(embed_file):
        return None
    with open(embed_file, 'rb') as fs:
        ret = pickle.load(fs)
    return ret


if __name__ == '__main__':
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
    
    # MUTAG typically has small graphs with ~20 atoms/nodes per molecule
    hyperparams_mutag = {
        'seed_num': 3,  # Initial number of categories in each sample
        'item_num': 5000,  # Total molecular instances
        'sample_num': 500,  # Number of molecular graphs to generate
        'context_limit': 10,  # Max categories to consider in context
        'gibbs_step': 500,  # Gibbs sampling steps per iteration
        'gen_base': 0,
        'restart_step': 50,  # Restart frequency for diversity
        'gibbs_skip_step': 1,
        'delete_nnz': 2,  # Threshold for deletion bias
        'length_center': 20,  # Target graph size (nodes per molecule)
        'community_num': 5,  # Number of communities for diversity
        'itmfuse': True,  # Fuse item instances
        'com_decay': 0.95,  # Community decay factor
        'dice_shift': 0.1,  # Probability shift for acceptance
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
    
    # ZINC has larger, more complex drug-like molecules
    hyperparams_zinc = {
        'seed_num': 5,  # More categories for complex molecules
        'item_num': 50000,  # Total molecular instances
        'sample_num': 2000,  # More samples for diversity
        'context_limit': 15,  # Larger context for complex molecules
        'gibbs_step': 800,  # More steps for convergence
        'gen_base': 0,
        'restart_step': 80,
        'gibbs_skip_step': 1,
        'delete_nnz': 3,
        'length_center': 35,  # Larger target graph size for drug-like molecules
        'community_num': 7,  # More communities for diversity
        'itmfuse': True,
        'com_decay': 0.95,
        'dice_shift': 0.1,
    }

    # ============================================
    # SELECT YOUR DATASET
    # ============================================
    # Uncomment the configuration you want to use:

    # For MUTAG:
    descs = descs_mutag
    hyperparams = hyperparams_mutag

    # For ZINC (uncomment to use):
    # descs = descs_zinc
    # hyperparams = hyperparams_zinc

    # ============================================
    # File paths
    # ============================================
    file_root = 'gen_results/datasets/{data_name}/'.format(data_name=descs['data_name'])
    entity_file = 'gen_results/tree_wInstanceNum_{initial_entity}_{scenario}_{dataset}.pkl'.format(
        initial_entity=descs['initial_entity'],
        scenario=descs['scenario_desc'],
        dataset=descs['dataset_type'])
    item_file = file_root + 'item_list.pkl'
    embed_file = file_root + 'embedding_dict.pkl'

    # Create directory
    os.makedirs(file_root, exist_ok=True)

    # ============================================
    # Validate required files
    # ============================================
    if not os.path.exists(entity_file):
        print(f"ERROR: Entity tree file not found: {entity_file}")
        print("Please run instance_number_estimation_hierarchical.py first.")
        exit(1)
    
    if not os.path.exists(embed_file):
        print(f"ERROR: Embedding file not found: {embed_file}")
        print("Please run embedding_generation.py first.")
        exit(1)

    print(f"Generating molecular graphs for: {descs['dataset_type'].upper()}")
    print(f"Data name: {descs['data_name']}")
    print(f"Scenario: {descs['scenario_desc']}")
    print("="*80)
    print("Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    print("="*80)

    # ============================================
    # Load data
    # ============================================
    item_list = load_item_list(item_file, entity_file, hyperparams['item_num'])
    if hyperparams['itmfuse']:
        item_list = list(map(lambda x: x[:x.index(' #')] + ' #1', item_list))
    
    embedding_dict = load_embedding_dict(embed_file)
    if embedding_dict is None:
        print("ERROR: Failed to load embeddings!")
        exit(1)

    print(f"\nLoaded {len(item_list)} molecular categories")
    print(f"Loaded {len(embedding_dict)} embeddings")

    def length_sampler():
        """Sample graph length (not used in current implementation)"""
        min_len, max_len = hyperparams.get('min_len', 10), hyperparams.get('max_len', 50)
        return random.randint(min_len, max_len)

    # ============================================
    # Generate molecular graphs
    # ============================================
    print("\nStarting molecular graph generation via Gibbs sampling...")
    print("This may take a while depending on hyperparameters.")
    print("-"*80)

    generator = MolecularGraphGeneration(item_list, length_sampler, descs, hyperparams, embedding_dict)
    sampled_interactions = generator.run()

    # ============================================
    # Store results
    # ============================================
    interaction_file_prefix = file_root + 'interaction_base-'
    if 'gen_base' in hyperparams:
        gen_base = hyperparams['gen_base']
    next_interaction_file = interaction_file_prefix + str(gen_base) + '_iter-0.pkl'
    
    if os.path.exists(next_interaction_file):
        gen_base = get_gen_iter(file_root, interaction_file_prefix) + 1
        next_interaction_file = interaction_file_prefix + str(gen_base) + '_iter-0.pkl'
    
    with open(next_interaction_file, 'wb+') as fs:
        pickle.dump(sampled_interactions, fs)

    print("\n" + "="*80)
    print(f"Molecular graph generation complete!")
    print(f"Generated {len(sampled_interactions)} molecular graphs")
    print(f"Results saved to: {next_interaction_file}")
    
    # ============================================
    # Statistics
    # ============================================
    graph_sizes = [len(graph) for graph in sampled_interactions]
    print(f"\nGraph size statistics:")
    print(f"  Mean: {np.mean(graph_sizes):.2f} nodes")
    print(f"  Std:  {np.std(graph_sizes):.2f}")
    print(f"  Min:  {np.min(graph_sizes)} nodes")
    print(f"  Max:  {np.max(graph_sizes)} nodes")
    print(f"  Median: {np.median(graph_sizes):.2f} nodes")
    
    print(f"\nNext step: Run make_adjs.py to create train/val/test splits")