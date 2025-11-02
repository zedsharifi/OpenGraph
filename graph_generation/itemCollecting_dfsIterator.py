# from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI
import os
# import openai # No longer needed
import google.generativeai as genai # Added for Gemini
import time
# import tiktoken # No longer needed
import json

from google.colab import userdata
# Load your Gemini key
API_KEY = userdata.get('API_key_1')
genai.configure(api_key=API_KEY)

class DataGenAgent:
    def __init__(self, initial_entity, scenario_desc, depth):
        super(DataGenAgent, self).__init__()

        # self.openai = OpenAI(temperature=0) # Original line
        self.initial_entity = initial_entity
        self.scenario_desc = scenario_desc
        # self.encoding = tiktoken.encoding_for_model('gpt-3.5-turbo') # No longer needed
        self.token_num = 0
        self.total_num = 0
        self.depth = depth

        # Initialize Gemini model
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            # Added safety_settings to prevent blocking on borderline prompts
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
            # Replaced OpenAI call with Gemini's generate_content
            completion = self.model.generate_content(message)
            response = completion.text
            
            time.sleep(1) # Kept original sleep
            
            # Replaced tiktoken with Gemini's token counter
            token_count = self.model.count_tokens(message)
            self.token_num += token_count.total_tokens
            
            return response
        except Exception as e:
            # Updated error message
            print(f'Gemini Exception occurs: {e}. Retry in 10 seconds.')
            time.sleep(10)
            return self.openai(message) # Retry

    def check_if_concrete(self, entity_stack):
        entity_name = ', '.join(entity_stack)
        text = 'In the context of {scenario_desc}, is {entity_name} a concrete instance or category that can hardly be divided into sub-categories with prominent differences? Response should starts with "True" or "False".'.format(scenario_desc=self.scenario_desc, entity_name=entity_name)
        answer = self.openai(text)
        # print('answer to {entity_name}: {answer}'.format(entity_name=entity_name, answer=answer))
        if answer.startswith('True'):
            print('Concrete Check True')
            return True
        return False
    
    def category_enum(self, prefix, entity_name):
        if prefix == '':
            text = 'List all distinct sub-categories of {entity_name} in the context of {scenario_desc}, ensuring a finer level of granularity. The sub-categories should not overlap with each other. And a sub-category should be a smaller subset of {entity_name}. Directly present the list EXACTLY following the form: "sub-category a, sub-category b, sub-category c, ..." without other words, format symbols, new lines, serial numbers.'.format(entity_name=entity_name, prefix=prefix, scenario_desc=self.scenario_desc)
        else:
            text = 'List all distinct sub-categories of {entity_name} within the {prefix} category in the context of {scenario_desc}, ensuring a finer level of granularity. The sub-categories should not overlap with each other. And a sub-category should be a smaller subset of {entity_name}. Directly present the list EXACTLY following the form: "sub-category a, sub-category b, sub-category c, ..." without other words, format symbols, new lines, serial numbers.'.format(entity_name=entity_name, prefix=prefix, scenario_desc=self.scenario_desc)
            # text = 'List all distinct sub-categories of {entity_name} within the {prefix} category in the context of {scenario_desc}, ensuring a finer level of granularity. The sub-categories should not overlap with each other. Present the list exactly following the form: "sub-category a, sub-category b, sub-category c, ...". There should be no serial number, new lines or other format symbols. Separate each pair of sub-categories with a comma.'.format(entity_name=entity_name, prefix=prefix, scenario_desc=self.scenario_desc)
        answer = self.openai(text)
        return list(map(lambda x: x.strip().strip(',').strip('.'), answer.split(',')))

    def decompose_category(self, entity_stack, depth):
        entity_name = ', '.join(entity_stack)
        prefix = '' if depth == 1 else ', '.join(entity_stack[:-1])
        if depth >= self.depth:# or self.check_if_concrete(entity_stack) is True:
            # print('{entity_name} is considered a concrete instance.'.format(entity_name=entity_name))
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
            print('Depth {depth}, current num of nodes {num}, total num of nodes {total_num}, num of tokens {token}'.format(depth=depth, num=len(concrete_entities), total_num=self.total_num, token=self.token_num))
        if depth <= 2:
            print('Storing...')
            # Ensure the gen_results/tem directory exists
            os.makedirs('gen_results/tem', exist_ok=True) 
            tem_file = 'gen_results/tem/{scenario}_depth{depth}_{cur_entity}'.format(scenario=self.scenario_desc, depth=str(depth), cur_entity=entity_name.replace('/', '_')) # Added replace for safety
            with open(tem_file, 'w+') as fs:
                for node in concrete_entities:
                    fs.write(node + '\n')
        return concrete_entities
    
    def run(self):
        return self.decompose_category([self.initial_entity], 1)

entity = 'products'
scenario = 'e-commerce platform like Amazon'
depth = 3

# entity = 'movies'
# scenario = 'movie rating platform'

# entity = 'books'
# scenario = 'book rating platform'

# entity = 'business venues'
# scenario = 'venue rating platform like yelp'

# entity = 'movies'
# scenario = 'movie rating platform'
# depth = 5

# entity = 'deep learning papers'
# scenario = 'published paper list of top AI conferences'
# depth = 6

# entity = 'ideology'
# scenario = "people's political ideologies"
# depth = 4

# entity = 'jobs'
# scenario = "people's occupations and professions"
# depth = 5

agent = DataGenAgent(entity, scenario, depth)
nodes = agent.run()
# Ensure the gen_results directory exists
os.makedirs('gen_results', exist_ok=True)
with open('gen_results/{entity}_{scenario}.txt'.format(entity=entity, scenario=scenario), 'w+') as fs:
    for node in nodes:
        fs.write(node+'\n')