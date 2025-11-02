import os
import google.generativeai as genai
import time
import json

from google.colab import userdata
# Load your Gemini key
API_KEY = userdata.get('API_key_1')
genai.configure(api_key=API_KEY)

class MolecularDataGenAgent:
    def __init__(self, initial_entity, scenario_desc, depth, dataset_type='mutag'):
        super(MolecularDataGenAgent, self).__init__()

        self.initial_entity = initial_entity
        self.scenario_desc = scenario_desc
        self.token_num = 0
        self.total_num = 0
        self.depth = depth
        self.dataset_type = dataset_type.lower()

        # Initialize Gemini model
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            safety_settings={
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            }
        )

    def openai(self, message):
        """
        Generates text using the Google Gemini API.
        NOTE: Kept original name 'openai' for compatibility with the script's logic.
        """
        try:
            completion = self.model.generate_content(message)
            response = completion.text
            
            time.sleep(1)
            
            token_count = self.model.count_tokens(message)
            self.token_num += token_count.total_tokens
            
            return response
        except Exception as e:
            print(f'Gemini Exception occurs: {e}. Retry in 10 seconds.')
            time.sleep(10)
            return self.openai(message)

    def check_if_concrete(self, entity_stack):
        entity_name = ', '.join(entity_stack)
        text = 'In the context of {scenario_desc}, is {entity_name} a concrete instance or category that can hardly be divided into sub-categories with prominent differences? Response should start with "True" or "False".'.format(
            scenario_desc=self.scenario_desc, entity_name=entity_name)
        answer = self.openai(text)
        if answer.startswith('True'):
            print('Concrete Check True')
            return True
        return False
    
    def category_enum(self, prefix, entity_name):
        if prefix == '':
            if self.dataset_type == 'mutag':
                text = '''List all distinct sub-categories of {entity_name} in the context of {scenario_desc}, ensuring a finer level of granularity. 
                The sub-categories should not overlap with each other. Focus on molecular properties relevant to mutagenicity such as:
                - Functional groups (nitro groups, aromatic amines, polycyclic aromatic hydrocarbons, etc.)
                - Molecular structure types (aromatic compounds, aliphatic compounds, heterocyclic compounds)
                - Chemical properties (electrophilic compounds, nucleophilic compounds, radical-forming compounds)
                Directly present the list EXACTLY following the form: "sub-category a, sub-category b, sub-category c, ..." without other words, format symbols, new lines, serial numbers.'''.format(
                    entity_name=entity_name, scenario_desc=self.scenario_desc)
            elif self.dataset_type == 'zinc':
                text = '''List all distinct sub-categories of {entity_name} in the context of {scenario_desc}, ensuring a finer level of granularity.
                The sub-categories should not overlap with each other. Focus on drug-like molecular properties such as:
                - Scaffold types (benzene-based, pyridine-based, piperidine-based, indole-based, etc.)
                - Pharmacophore patterns (hydrogen bond donors/acceptors, hydrophobic regions, charged groups)
                - Molecular frameworks (linear, branched, cyclic, polycyclic)
                - Chemical classes (alkaloids, peptides, terpenoids, steroids)
                Directly present the list EXACTLY following the form: "sub-category a, sub-category b, sub-category c, ..." without other words, format symbols, new lines, serial numbers.'''.format(
                    entity_name=entity_name, scenario_desc=self.scenario_desc)
            else:
                text = 'List all distinct sub-categories of {entity_name} in the context of {scenario_desc}, ensuring a finer level of granularity. The sub-categories should not overlap with each other. Directly present the list EXACTLY following the form: "sub-category a, sub-category b, sub-category c, ..." without other words, format symbols, new lines, serial numbers.'.format(
                    entity_name=entity_name, scenario_desc=self.scenario_desc)
        else:
            text = 'List all distinct sub-categories of {entity_name} within the {prefix} category in the context of {scenario_desc}, ensuring a finer level of granularity. The sub-categories should not overlap with each other. And a sub-category should be a smaller subset of {entity_name}. Directly present the list EXACTLY following the form: "sub-category a, sub-category b, sub-category c, ..." without other words, format symbols, new lines, serial numbers.'.format(
                entity_name=entity_name, prefix=prefix, scenario_desc=self.scenario_desc)
        
        answer = self.openai(text)
        return list(map(lambda x: x.strip().strip(',').strip('.'), answer.split(',')))

    def decompose_category(self, entity_stack, depth):
        entity_name = ', '.join(entity_stack)
        prefix = '' if depth == 1 else ', '.join(entity_stack[:-1])
        
        if depth >= self.depth:
            self.total_num += 1
            return [entity_name]
        
        print('\nCurrent entity: {entity_name}'.format(entity_name=entity_name))
        concrete_entities = []
        sub_entities = self.category_enum(prefix, entity_stack[-1])
        print('sub-categories of {entity_name} includes:'.format(entity_name=entity_name), sub_entities)
        
        for sub_entity in sub_entities:
            if sub_entity in entity_name:
                continue
            new_concrete_entities = self.decompose_category(entity_stack + [sub_entity], depth+1)
            concrete_entities += new_concrete_entities
        
        if depth <= 4:
            print('Depth {depth}, current num of nodes {num}, total num of nodes {total_num}, num of tokens {token}'.format(
                depth=depth, num=len(concrete_entities), total_num=self.total_num, token=self.token_num))
        
        if depth <= 2:
            print('Storing...')
            os.makedirs('gen_results/tem', exist_ok=True) 
            tem_file = 'gen_results/tem/{scenario}_depth{depth}_{cur_entity}'.format(
                scenario=self.scenario_desc.replace('/', '_'), 
                depth=str(depth), 
                cur_entity=entity_name.replace('/', '_'))
            with open(tem_file, 'w+') as fs:
                for node in concrete_entities:
                    fs.write(node + '\n')
        
        return concrete_entities
    
    def run(self):
        return self.decompose_category([self.initial_entity], 1)


# ============================================
# MUTAG Configuration (Mutagenicity Dataset)
# ============================================
# MUTAG consists of 188 mutagenic aromatic and heteroaromatic nitro compounds
# Goal: Predict whether a molecule is mutagenic or not
entity_mutag = 'molecular compounds'
scenario_mutag = 'mutagenicity prediction for aromatic and heteroaromatic compounds'
depth_mutag = 4  # Adjust depth as needed

# ============================================
# ZINC Configuration (Drug-like Molecules)
# ============================================
# ZINC contains drug-like molecules with diverse scaffolds
# Goal: Predict molecular properties for drug discovery
entity_zinc = 'drug-like molecules'
scenario_zinc = 'drug discovery and molecular property prediction'
depth_zinc = 4  # Adjust depth as needed

# ============================================
# SELECT YOUR DATASET
# ============================================
# Uncomment the configuration you want to use:

# For MUTAG:
dataset_type = 'mutag'
entity = entity_mutag
scenario = scenario_mutag
depth = depth_mutag

# For ZINC (uncomment to use):
# dataset_type = 'zinc'
# entity = entity_zinc
# scenario = scenario_zinc
# depth = depth_zinc

# ============================================
# Run the generation
# ============================================
print(f"Generating molecular graph dataset for: {dataset_type.upper()}")
print(f"Entity: {entity}")
print(f"Scenario: {scenario}")
print(f"Depth: {depth}")
print("="*60)

agent = MolecularDataGenAgent(entity, scenario, depth, dataset_type=dataset_type)
nodes = agent.run()

# Save results
os.makedirs('gen_results', exist_ok=True)
output_file = 'gen_results/{entity}_{scenario}_{dataset}.txt'.format(
    entity=entity.replace(' ', '_'),
    scenario=scenario.replace(' ', '_'),
    dataset=dataset_type)

with open(output_file, 'w+') as fs:
    for node in nodes:
        fs.write(node+'\n')

print(f"\nGeneration complete!")
print(f"Total nodes generated: {len(nodes)}")
print(f"Results saved to: {output_file}")