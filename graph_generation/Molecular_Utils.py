import time
import google.generativeai as genai
import json
import numpy as np
import Exp_Utils.TimeLogger as logger
from Exp_Utils.TimeLogger import log
from Exp_Utils.Emailer import SendMail
import time

from google.colab import userdata
# Configure Gemini API
API_KEY = userdata.get('API_key_1')
genai.configure(api_key=API_KEY)

class MolecularDataGenAgent:
    def __init__(self):
        super(MolecularDataGenAgent, self).__init__()
        self.token_num = 0
        
        # Initialize Gemini models - using faster model for molecular generation
        self.text_model = genai.GenerativeModel(
            'gemini-2.0-flash',
            safety_settings={
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            }
        )
        self.embedding_model = 'models/text-embedding-004'
    
    def openai_embedding(self, message):
        """
        Generates an embedding using the Google Gemini API.
        NOTE: Kept original name 'openai_embedding' for compatibility with other scripts.
        """
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=message,
                task_type="SEMANTIC_SIMILARITY"
            )
            embedding = result['embedding']
            return np.array(embedding)
        except Exception as e:
            print('Gemini request error (embedding): {err_msg}. Retry in 10 seconds.'.format(err_msg=e))
            time.sleep(10)
            return self.openai_embedding(message)

    def openai(self, message):
        """
        Generates text using the Google Gemini API.
        NOTE: Kept original name 'openai' for compatibility with other scripts.
        """
        try:
            completion = self.text_model.generate_content(message)
            response = completion.text
            
            token_count = self.text_model.count_tokens(message)
            self.token_num += token_count.total_tokens

            time.sleep(0.5)  # Reduced sleep for faster generation
            return response
        except Exception as e:
            print('Gemini request error (generation): {err_msg}. Retry in 10 seconds.'.format(err_msg=e))
            time.sleep(10)
            return self.openai(message)
    
    def handling_llm_exceptions(self, message, interpret_func, interpret_args, failure_tolerance):
        try:
            answers_text = self.openai(message)
            print('Answers text:')
            print(answers_text)
            print('----------\n')
            return 0, interpret_func(answers_text, *interpret_args)
        except Exception as e:
            self.failure += 1
            log('\n**********\nERROR\n')
            log('Exception occurs when interpreting. Exception message: {exception}'.format(exception=e), save=True)
            log('Failure times: {failure}'.format(failure=self.failure), save=True)
            log('Prompt text:\n{prompt}'.format(prompt=message), save=True)
            log('Response text:\n{response}'.format(response=answers_text), save=True)
            if self.failure < failure_tolerance:
                log('Retry in 10 seconds.', save=True)
                time.sleep(10)
                log('\n**********\n')
                return 1, None
            else:
                log('Reaching maximum failure tolerance. CANNOT HANDLE!'.format(failure=self.failure), save=True)
                log('Sending report email.', save=True)
                SendMail(logger.logmsg)
                logger.logmsg = ''
                log('\n**********\n')
                return 2, None


class MolecularEntityTreeNode:
    """
    Tree node for hierarchical molecular structure organization.
    Each node represents a molecular category (e.g., 'aromatic compounds', 'benzene derivatives')
    """
    def __init__(self, entity_name, depth, parent=None, mutagenic_ratio=0.5):
        self.entity_name = entity_name
        self.frequency = []
        self.quantity = -1
        self.children = dict()
        self.parent = parent
        self.depth = depth
        self.mutagenic_ratio = mutagenic_ratio  # Proportion of mutagenic molecules in this category
    
    def is_child(self, entity_name):
        return entity_name in self.children
    
    def to_child(self, entity_name):
        return self.children[entity_name]

    def add_child(self, entity_name, mutagenic_ratio=0.5):
        child = MolecularEntityTreeNode(entity_name, self.depth+1, self, mutagenic_ratio)
        self.children[entity_name] = child
    
    def iterate_children(self):
        for key, node in self.children.items():
            yield key, node
    
    def allocate_number(self, quantity):
        """
        Allocate number of molecules to each node in the tree.
        Adapted from OpenGraph's allocation strategy.
        """
        print('Allocating depth {depth} {entity_name}, quantity: {quantity}'.format(
            depth=self.depth, entity_name=self.entity_name, quantity=quantity))
        self.quantity = quantity
        
        if len(self.children) == 0:
            return
        
        child_list = list(self.children.values())
        child_freq = list(map(lambda x: x.frequency, child_list))
        child_freq = np.array(child_freq)  # N * T
        
        if child_freq.shape[1] == 0:
            # Equal distribution if no frequency info
            num_children = len(child_list)
            for child in child_list:
                child.allocate_number(quantity / num_children)
            return
        
        summ = np.sum(child_freq, axis=0, keepdims=True)  # 1 * T
        child_freq = child_freq / summ  # N * T
        child_num = np.mean(child_freq, axis=1) * self.quantity  # N
        
        for i, child in enumerate(child_list):
            child.allocate_number(child_num[i])
    
    def get_list_of_leaves(self, entity_name, with_branches=False):
        """
        Get all leaf molecular categories with instance numbers.
        Adapted to generate molecule instances with properties.
        """
        if len(self.children) == 0:
            # Leaf node: generate molecule instances
            num = max(1, int(self.quantity))
            entity_list = list()
            cur_entity_name = entity_name + ', ' + self.entity_name if entity_name else self.entity_name
            
            # Generate molecules with mutagenicity labels
            num_mutagenic = int(num * self.mutagenic_ratio)
            num_nonmutagenic = num - num_mutagenic
            
            for i in range(num_mutagenic):
                entity_list.append(self.entity_name + ' #MUT-{idx}'.format(idx=i))
            for i in range(num_nonmutagenic):
                entity_list.append(self.entity_name + ' #NON-{idx}'.format(idx=i))
            
            return entity_list
        
        entity_list = list()
        if with_branches:
            if self.depth <= 2:
                tem_entity_name = self.entity_name
            else:
                tem_entity_name = entity_name + ', ' + self.entity_name
            entity_list.append(tem_entity_name)
        
        for _, child in self.iterate_children():
            nxt_entity_name = self.entity_name if self.depth <= 2 else (entity_name + ', ' + self.entity_name)
            entities = child.get_list_of_leaves(nxt_entity_name, with_branches)
            entity_list = entity_list + entities
        
        return entity_list


class MolecularEntityTreeConstructer:
    """
    Constructs hierarchical tree from molecular entity lines.
    Adapted from OpenGraph's EntityTreeConstructer.
    """
    def __init__(self, entity_lines):
        super(MolecularEntityTreeConstructer, self).__init__()
        
        root_name = self.line_process(entity_lines[0])[0]
        self.root = MolecularEntityTreeNode(root_name, depth=1)
        self.root.frequency.append(1.0)
        self.construct_tree(entity_lines)
    
    def add_node(self, cur_node, descriptions, cur):
        parent_entity_name = descriptions[cur-1]
        if cur_node.entity_name != parent_entity_name:
            print(cur_node.entity_name, parent_entity_name)
            print(descriptions)
        assert cur_node.entity_name == parent_entity_name
        
        cur_entity_name = descriptions[cur]
        if not cur_node.is_child(cur_entity_name):
            # Infer mutagenicity ratio from category name
            mutagenic_ratio = self.infer_mutagenic_ratio(cur_entity_name)
            cur_node.add_child(cur_entity_name, mutagenic_ratio)
        
        if cur + 1 < len(descriptions):
            self.add_node(cur_node.to_child(cur_entity_name), descriptions, cur+1)
    
    def infer_mutagenic_ratio(self, entity_name):
        """
        Infer likelihood of mutagenicity based on molecular category name.
        High-risk groups (nitro, aromatic amines) → higher ratio
        Low-risk groups (aliphatic alcohols) → lower ratio
        """
        entity_lower = entity_name.lower()
        
        # High-risk functional groups
        if any(x in entity_lower for x in ['nitro', 'aromatic amine', 'polycyclic', 
                                             'epoxide', 'azo', 'azoxy']):
            return 0.8
        
        # Medium-risk
        elif any(x in entity_lower for x in ['aromatic', 'heterocycle', 'aldehyde', 
                                               'quinone', 'halogenated']):
            return 0.6
        
        # Low-risk
        elif any(x in entity_lower for x in ['aliphatic', 'alcohol', 'ether', 
                                               'ester', 'simple']):
            return 0.3
        
        # Default balanced
        return 0.5
    
    def line_process(self, entity_line, check=False):
        entity_line = entity_line.strip()
        descriptions = list(map(lambda x: x.strip(), entity_line.split(',')))
        
        if not check:
            return descriptions
        
        if descriptions[0] != self.root.entity_name:
            raise Exception('Cannot find root')
        if len(descriptions) <= 1:
            raise Exception('Fail to split')
        
        return descriptions
    
    def construct_tree(self, entity_lines):
        for entity_line in entity_lines:
            try:
                descriptions = self.line_process(entity_line, check=True)
            except Exception as e:
                print(str(e), ':', entity_line)
                continue
            self.add_node(self.root, descriptions, cur=1)