import time
# import openai # No longer needed
import google.generativeai as genai # Added for Gemini
import json
# import tiktoken # No longer needed
import numpy as np
import Exp_Utils.TimeLogger as logger
from Exp_Utils.TimeLogger import log
from Exp_Utils.Emailer import SendMail
import time

from google.colab import userdata
# Configure Gemini API
# Assumes you have your Gemini key saved in Colab Secrets as 'GEMINI_API_KEY'
API_KEY = userdata.get('API_key_1')
genai.configure(api_key=API_KEY)

class DataGenAgent:
    def __init__(self):
        super(DataGenAgent, self).__init__()
        self.token_num = 0
        # self.encoding = tiktoken.encoding_for_model('gpt-3.5-turbo') # No longer needed
        
        # Initialize Gemini models
        # Using 1.5 Pro to fulfill your "2.5 Pro" request
        self.text_model = genai.GenerativeModel('gemini-2.5-pro')
        self.embedding_model = 'models/text-embedding-004' # Google's latest embedding model
    
    def openai_embedding(self, message):
        """
        Generates an embedding using the Google Gemini API.
        NOTE: Kept original name 'openai_embedding' for compatibility with other scripts.
        """
        try:
            # Replaced OpenAI call with genai.embed_content
            result = genai.embed_content(
                model=self.embedding_model,
                content=message,
                task_type="SEMANTIC_SIMILARITY" # Best for the similarity checks in this repo
            )
            embedding = result['embedding']
            # time.sleep() # Original was commented out, kept as is
            return np.array(embedding)
        except Exception as e:
            # Updated error message
            print('Gemini request error (embedding): {err_msg}. Retry in 10 seconds.'.format(err_msg=e))
            time.sleep(10)
            return self.openai_embedding(message) # Retry

    def openai(self, message):
        """
        Generates text using the Google Gemini API.
        NOTE: Kept original name 'openai' for compatibility with other scripts.
        """
        try:
            # Replaced OpenAI call with Gemini's generate_content
            # Added safety_settings to prevent blocking on borderline prompts
            completion = self.text_model.generate_content(
                message,
                safety_settings={
                    'HATE_SPEECH': 'BLOCK_NONE',
                    'HARASSMENT': 'BLOCK_NONE',
                    'SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                    'DANGEROUS_CONTENT': 'BLOCK_NONE'
                }
            )
            response = completion.text
            
            # Replaced tiktoken with Gemini's token counter
            token_count = self.text_model.count_tokens(message)
            self.token_num += token_count.total_tokens

            time.sleep(1) # Kept original sleep
            return response
        except Exception as e:
            # Updated error message
            print('Gemini request error (generation): {err_msg}. Retry in 10 seconds.'.format(err_msg=e))
            time.sleep(10)
            return self.openai(message) # Retry
    
    def handling_llm_exceptions(self, message, interpret_func, interpret_args, failure_tolerance):
        # This function works as-is because we kept the 'self.openai' method name
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
            log('Failure times: {failure}'.format(failure=self.failure, save=True))
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

# --- The rest of the file is unchanged ---

class EntityTreeNode:
    def __init__(self, entity_name, depth, parent=None):
        self.entity_name = entity_name
        self.frequency = []
        self.quantity = -1
        self.children = dict()
        self.parent = parent
        self.depth = depth
    
    def is_child(self, entity_name):
        return entity_name in self.children
    
    def to_child(self, entity_name):
        return self.children[entity_name]

    def add_child(self, entity_name):
        child = EntityTreeNode(entity_name, self.depth+1, self)
        self.children[entity_name] = child
    
    def iterate_children(self):
        for key, node in self.children.items():
            yield key, node
    
    def allocate_number(self, quantity):
        print('Allocating depth {depth} {entity_name}, quantity: {quantity}'.format(depth=self.depth, entity_name=self.entity_name, quantity=quantity))
        self.quantity = quantity
        if len(self.children) == 0:
            return
        child_list = list(self.children.values())
        child_freq = list(map(lambda x: x.frequency, child_list))
        child_freq = np.array(child_freq) # N * T
        if child_freq.shape[1] == 0:
            raise Exception('No estimated frequency for children.')
        summ = np.sum(child_freq, axis=0, keepdims=True) # 1 * T
        child_freq = child_freq / summ # N * T
        child_num = np.mean(child_freq, axis=1) * self.quantity # N
        for i, child in enumerate(child_list):
            child.allocate_number(child_num[i])
    
    def get_list_of_leaves(self, entity_name, with_branches=False):
        if len(self.children) == 0:
            num = max(1, int(self.quantity))
            entity_list = list()
            cur_entity_name = entity_name + ', ' + self.entity_name
            for i in range(num):
                # entity_list.append(cur_entity_name + ' #{idx}'.format(idx=i))
                entity_list.append(self.entity_name + ' #{idx}'.format(idx=i))
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

class EntityTreeConstructer:
    def __init__(self, entity_lines):
        super(EntityTreeConstructer, self).__init__()
        
        root_name = self.line_process(entity_lines[0])[0]
        self.root = EntityTreeNode(root_name, depth=1)
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
            cur_node.add_child(cur_entity_name)
        if cur + 1 < len(descriptions):
            self.add_node(cur_node.to_child(cur_entity_name), descriptions, cur+1)
    
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