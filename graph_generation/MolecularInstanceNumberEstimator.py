import numpy as np
from Utils import DataGenAgent, EntityTreeConstructer, EntityTreeNode
import os
import time
import pickle
import Exp_Utils.TimeLogger as logger
from Exp_Utils.TimeLogger import log
from Exp_Utils.Emailer import SendMail

class MolecularInstanceNumberEstimator(DataGenAgent):
    def __init__(self, entity_tree_root, total_num, depth, initial_entity, scenario_desc, dataset_type='mutag'):
        super(MolecularInstanceNumberEstimator, self).__init__()

        self.entity_tree_root = entity_tree_root
        self.total_num = total_num
        self.initial_entity = initial_entity
        self.scenario_desc = scenario_desc
        self.depth = depth
        self.failure = 0
        self.dataset_type = dataset_type.lower()
    
    def _entity_list_to_text(self, entity_list):
        ret = ''
        for i, entity in enumerate(entity_list):
            ret += '{idx}. {entity}\n'.format(idx=i+1, entity=entity)
        return ret
    
    def interpret_one_answer(self, answer_text, subcategory):
        answer_lower = answer_text.lower()
        if subcategory.lower() not in answer_lower:
            log('ERROR: Entity name not found.', save=True)
            log('subcategory: {subcategory}'.format(subcategory=subcategory), save=True)
            log('answer_lower: {answer_lower}'.format(answer_lower=answer_lower), save=True)
            raise Exception('Entity name not found.')
        
        estimation_choices = [
            'average frequency', 
            '1.2 times more frequent', '1.2 times less frequent', 
            '1.5 times more frequent', '1.5 times less frequent', 
            '2 times more frequent', '2 times less frequent', 
            '4 times more frequent', '4 times less frequent', 
            '8 times more frequent', '8 times less frequent'
        ]
        estimation_scores = [1.0, 1.2, 1/1.2, 1.5, 1/1.5, 2.0, 1/2.0, 4.0, 1/4.0, 8.0, 1/8.0]
        
        for i, choice in enumerate(estimation_choices):
            if choice in answer_lower:
                return estimation_scores[i]
        raise Exception('Estimation not found.')

    def interpret(self, answers_text, subcategories):
        answer_list = answers_text.strip().split('\n')
        assert len(answer_list) == len(subcategories), 'Length does not match.'
        answers = []
        for i in range(len(answer_list)):
            answers.append(self.interpret_one_answer(answer_list[i], subcategories[i]))
        return answers
    
    def estimate_subcategories(self, subcategories, category):
        subcategories_text = self._entity_list_to_text(subcategories)
        
        # Dataset-specific context
        if self.dataset_type == 'mutag':
            context_info = '''In the context of {scenario_desc}, you are analyzing mutagenic compounds. 
            Consider that:
            - Aromatic compounds with electron-withdrawing groups are more common
            - Nitro-aromatic compounds are particularly important for mutagenicity
            - Polycyclic aromatic hydrocarbons appear frequently
            - Heterocyclic compounds with nitrogen are prevalent'''
        elif self.dataset_type == 'zinc':
            context_info = '''In the context of {scenario_desc}, you are analyzing drug-like molecules.
            Consider that:
            - Benzene and pyridine scaffolds are extremely common
            - Piperidine and piperazine rings appear frequently
            - Molecules with 2-5 rings are most typical
            - Hydrogen bond donors/acceptors are essential features'''
        else:
            context_info = 'In the context of {scenario_desc}'
        
        if category != self.initial_entity:
            text = context_info + ''', you are given a list of sub-categories below, which belong to the {category} category of {initial_entity}. 
            Using your knowledge of molecular chemistry and common patterns in molecular datasets, identify the frequency of these sub-categories 
            compared to the average frequency of all possible {category} {initial_entity}.'''.format(
                scenario_desc=self.scenario_desc, 
                initial_entity=self.initial_entity, 
                category=category)
        else:
            text = context_info + ''', you are given a list of sub-categories below, which belong to {initial_entity}. 
            Using your knowledge of molecular chemistry and common patterns in molecular datasets, identify the frequency of these sub-categories 
            compared to the average frequency of all possible {initial_entity}.'''.format(
                scenario_desc=self.scenario_desc, 
                initial_entity=self.initial_entity)
        
        text += '''Your answer should contain one line for each of the sub-categories, EXACTLY following the following format: 
        "[serial number]. [sub-category name same as in the input]; [your frequency estimation]; [one-sentence explanation for your estimation based on molecular chemistry]". 
        The frequency estimation should be one of the following choices: [average frequency, 1.2 times more/less frequent, 1.5 times more/less frequent, 
        2 times more/less frequent, 4 times more/less frequent, 8 times more/less frequent]. 
        No other words should be included in your response. The sub-categories list is as follows:\n\n''' + subcategories_text
        
        try:
            answers_text = self.openai(text)
            print('Answers text:')
            print(answers_text)
            return self.interpret(answers_text, subcategories)
        except Exception as e:
            self.failure += 1
            if self.failure < 5:
                log('Exception occurs when interpreting. Retry in 10 seconds.', save=True)
                log('Exception message: {exception}'.format(exception=e), save=True)
                log('Failure times: {failure}'.format(failure=self.failure), save=True)
                log('Prompt text:\n{prompt}'.format(prompt=text), save=True)
                log('Response text:\n{response}'.format(response=answers_text), save=True)
                time.sleep(10)
                return self.estimate_subcategories(subcategories, category)
            else:
                log('Exception occurs {failure} times when interpreting. CANNOT HANDLE.'.format(
                    failure=str(self.failure)), save=True)
                log('Exception message: {exception}'.format(exception=e), save=True)
                log('Prompt text:\n{prompt}'.format(prompt=text), save=True)
                log('Response text:\n{response}'.format(response=answers_text), save=True)
                log('Sending report email.', save=True)
                SendMail(logger.logmsg)
                logger.logmsg = ''
                return [1.0] * len(subcategories)
    
    def run(self):
        que = [self.entity_tree_root]
        while len(que) > 0:
            cur_entity = que[0]
            que = que[1:]
            if len(cur_entity.children) == 0:
                continue
            cur_children_entities = list(cur_entity.children.values())
            que = que + cur_children_entities
            cur_children_names = list(map(lambda x: x.entity_name, cur_children_entities))
            assert self.depth - cur_entity.depth > 0 and self.depth - cur_entity.depth < self.depth
            
            for _ in range(self.depth - cur_entity.depth + 1):
                self.failure = 0
                # Uncomment below to use LLM-based frequency estimation
                # answers = self.estimate_subcategories(cur_children_names, cur_entity.entity_name)
                # print(answers)
                # print('-----------------')
                # print()
                # for j, entity in enumerate(cur_children_entities):
                #     entity.frequency.append(answers[j])
                
                # Using uniform frequency for simplicity (can be changed)
                for j, entity in enumerate(cur_children_entities):
                    entity.frequency.append(1.0)
        
        self.entity_tree_root.allocate_number(self.total_num)
        
        output_file = 'gen_results/tree_wInstanceNum_{initial_entity}_{scenario}_{dataset}.pkl'.format(
            initial_entity=self.initial_entity.replace(' ', '_'),
            scenario=self.scenario_desc.replace(' ', '_'),
            dataset=self.dataset_type)
        
        with open(output_file, 'wb') as fs:
            pickle.dump(self.entity_tree_root, fs)
        
        print(f'\nTree with instance numbers saved to: {output_file}')


# ============================================
# MUTAG Configuration
# ============================================
# MUTAG dataset: 188 mutagenic aromatic and heteroaromatic nitro compounds
# Typical size: ~200 molecules with ~20 atoms per molecule on average
scenario_mutag = 'mutagenicity_prediction_for_aromatic_and_heteroaromatic_compounds'
initial_entity_mutag = 'molecular_compounds'
total_num_mutag = 5000  # Total number of molecular instances to generate
depth_mutag = 4  # Should match the depth used in itemCollecting_dfsIterator.py
dataset_type_mutag = 'mutag'

# ============================================
# ZINC Configuration
# ============================================
# ZINC dataset: Large database of commercially available drug-like compounds
# Typical subsets: 12K-250K molecules with diverse scaffolds
scenario_zinc = 'drug_discovery_and_molecular_property_prediction'
initial_entity_zinc = 'drug-like_molecules'
total_num_zinc = 50000  # Total number of molecular instances to generate
depth_zinc = 4  # Should match the depth used in itemCollecting_dfsIterator.py
dataset_type_zinc = 'zinc'

# ============================================
# SELECT YOUR DATASET
# ============================================
# Uncomment the configuration you want to use:

# For MUTAG:
dataset_type = dataset_type_mutag
scenario = scenario_mutag
initial_entity = initial_entity_mutag
total_num = total_num_mutag
depth = depth_mutag

# For ZINC (uncomment to use):
# dataset_type = dataset_type_zinc
# scenario = scenario_zinc
# initial_entity = initial_entity_zinc
# total_num = total_num_zinc
# depth = depth_zinc

# ============================================
# Load entities and run estimation
# ============================================
print(f"Running instance number estimation for: {dataset_type.upper()}")
print(f"Entity: {initial_entity}")
print(f"Scenario: {scenario}")
print(f"Total molecules to allocate: {total_num}")
print(f"Depth: {depth}")
print("="*60)

# Load entity file (generated from itemCollecting_dfsIterator.py)
file = os.path.join('gen_results/', '{entity}_{scenario}_{dataset}.txt'.format(
    entity=initial_entity, 
    scenario=scenario,
    dataset=dataset_type))

if not os.path.exists(file):
    print(f"ERROR: Entity file not found: {file}")
    print("Please run itemCollecting_dfsIterator.py first to generate the entity hierarchy.")
    exit(1)

entity_lines = []
with open(file, 'r') as fs:
    for line in fs:
        entity_lines.append(line)

print(f"Loaded {len(entity_lines)} entity lines from {file}")

# Construct entity tree
entity_tree_constructer = EntityTreeConstructer(entity_lines)
entity_tree_root = entity_tree_constructer.root

# Run estimation
estimator = MolecularInstanceNumberEstimator(
    entity_tree_root, 
    total_num=total_num, 
    depth=depth, 
    initial_entity=initial_entity, 
    scenario_desc=scenario,
    dataset_type=dataset_type)

estimator.run()

print(f"\nInstance number estimation complete!")
print(f"Total instances allocated: {total_num}")