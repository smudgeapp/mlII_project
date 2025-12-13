from google import genai
from google.genai import types
import os
import faiss
import numpy as np
import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec
import json


"""
    This script holds the EmployAidAgent class.

    It connects to the API, loads the agent, sends, receives and processes user
    queries.

"""

class EmployAidAgent:
    #api_key = os.environ.get("GEMINI_API_KEY")

    model = "gemini-2.5-flash"
    models = {
        "Gemini 2.5 Flash": 'gemini-2.5-flash',
        "Gemini 2.5 Flash Lite": 'gemini-2.5-flash-lite',
        "Gemini 2.5 Pro": 'gemini-2.5-pro'
        }
    
    def __init__(self, api_key=None):
        access_key = api_key
        
        if api_key is None:
            access_key = os.environ.get("GEMINI_API_KEY")
            
        
        self.client = genai.Client(api_key=access_key)
        
        with open("system_instructions.json", mode='r') as f:
            self.instructions = json.load(f)
            
        self.launch_agent()
        self.total_tokens = 0

    """
        Helper function to instantiate chats session.

        Required to change model at runtime without changing the client.
    """

    def launch_agent(self, model_name=None):
        if model_name is not None:
            self.model = self.models[model_name]
            
        system_config = types.GenerateContentConfig(
            system_instruction=self.instructions['system_instruction'])
        self.agent = self.client.chats.create(model=self.model,
                                         config=system_config)

    """
        User entry point, each query is passed through this function that
        decides whether to do a function calling or just respond to the
        user query.

        This intro query sets the tone for the user and primes the LLM for
        further queries.
        
    """

    
    def intro_query(self, query):
        print('*** INTRO QUERY CALLED ***')
        config = types.GenerateContentConfig(response_mime_type='application/json',
                                             response_schema=self.instructions['intro'])
        response = self.agent.send_message(query, config=config)
        current_tokens = response.usage_metadata.total_token_count
        self.total_tokens += current_tokens
        response = json.loads(response.text)
        if response['firm'] == "" and response['job'] == "":
            return response['response'], current_tokens, self.total_tokens
        else:
            mid_query = response['query'] + " " + response['firm'] + " " + response['job']
            return self.call_agent(mid_query)

    """
        This is the primary method called at each query.

        This enables the LLM to decide whether to make a function call or
        simply respond to the query.
    """


    def call_agent(self, query):
        tools = types.Tool(function_declarations=self.instructions['functions'])
        tool_config = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO))
        config = types.GenerateContentConfig(tools=[tools], tool_config=tool_config)
        response = self.agent.send_message(query, config=config)
        part = response.candidates[0].content.parts[0]
        if part.function_call:
            func_call = part.function_call
            if func_call.name == 'company_query':
                return self.company_query(**func_call.args)
            elif func_call.name == 'job_role_query':
                return self.job_role_query(**func_call.args)
            elif func_call.name == 'find_company':
                return self.find_company(**func_call.args)
            elif func_call.name == 'find_job_role':
                return self.find_job_role(**func_call.args)
            else:
                return self.find_job_role_company(**func_call.args)
        else:
            text_response = part.text
            return self.send_response(text_response, context="A database search was not required for this query. Respond to user query directly. Use chat history for context.")


    """
        Connector method to link the LLM tool calls to user query responding
    """


    def send_response(self, query, context='Give a generic response and ask for more information'):
        msg = f'Respond to {query}.'
        instruct = [{'text': 'CRITICAL: retain prior instructions and persona.'},
                    {'text': f'CONTEXT_INFO: {context}'}
                    ]
        resp_config = types.GenerateContentConfig(system_instruction=instruct)
        response = self.agent.send_message(msg, config=resp_config)
        current_tokens = response.usage_metadata.total_token_count
        print(f'current tokens: {current_tokens}')
        self.total_tokens += current_tokens
        
        return response.text, current_tokens, self.total_tokens


    """
        Vector search function if the query is about a particular
        company.
    """
    
    def company_query(self, query, company):
        print("****CALLED: COMPANY QUERY****")
        
        data = pd.read_json('employaid.firm_pros_cons.json')

        company_data = data[data['_id'] == company]

        if len(company_data) <= 0:
            firm_model = Doc2Vec.load('firm_name')
            keys = firm_model.dv.key_to_index
            comp_vec = firm_model.infer_vector([company], epochs=1000)
              
            firm_embeds = np.array(data['title_embed'].tolist())
            keys = np.array(data.index.tolist(), dtype=np.int64)        
            index = faiss.IndexFlatL2(firm_embeds.shape[1])
            keyed_index = faiss.IndexIDMap(index)
            keyed_index.add_with_ids(firm_embeds, keys)

            comp_vec = comp_vec.reshape(1, -1)
            dist, idx = keyed_index.search(comp_vec, 1)
            
            company_data = data.iloc[idx[0][0]]
            

        context = " ".join(company_data['pros']) + " " + " ".join(company_data['cons'])

        return self.send_response(query, context)

    """
        Vector search function if the query is about a specific job role
    """


    def job_role_query(self, query, job_title):
        print("****CALLED: JOB TITLE QUERY****")
        data = pd.read_json('employaid.title_pros_cons.json')

        job_title_data = data[data['_id'] == job_title]

        if len(job_title_data) <= 0:
            job_model = Doc2Vec.load('job_title')
            keys = job_model.dv.key_to_index
            comp_vec = job_model.infer_vector([job_title], epochs=1000)
            
            title_embeds = np.array(data['title_embed'].tolist())
            keys = np.array(data.index.tolist(), dtype=np.int64)
            index = faiss.IndexFlatL2(title_embeds.shape[1])
            keyed_index = faiss.IndexIDMap(index)
            keyed_index.add_with_ids(title_embeds, keys)

            comp_vec = comp_vec.reshape(1, -1)
            dist, idx = keyed_index.search(comp_vec, 1)

            job_title_data = data.iloc[idx[0][0]]

        context = " ".join(job_title_data['pros']) + " " + " ".join(job_title_data['cons'])

        return self.send_response(query, context)



    """
        Vector search function if the query is about a specific work
        environment.
    """

    def find_company(self, query, preferences):
        print("****CALLED: FIND COMPANY****")

        data = pd.read_json('employaid.firm_pros_cons.json')
        
        summary = preferences['summary']
        pref_keys = [k for k in preferences.keys()]
        pref_keys.remove('summary')
        rating_input = [preferences[k] for k in pref_keys]
        rating_input = np.array(rating_input)
        procon_model = Doc2Vec.load('firm_pros_cons')
        summary_vec = procon_model.infer_vector([summary], epochs=1000)

        procon_vecs = np.array(data['procon_embed'].tolist())
        keys = np.array(data.index.tolist(), dtype=np.int64)

        procon_index = faiss.IndexFlatL2(procon_vecs.shape[1])
        procon_k_index = faiss.IndexIDMap(procon_index)
        procon_k_index.add_with_ids(procon_vecs, keys)

        summary_vec = summary_vec.reshape(1, -1)
        dist, idx = procon_k_index.search(summary_vec, 1)

        procon_entry = data.iloc[idx[0][0]]
        company = procon_entry['_id']

        context = f"Company={company}, Company review=" + " ".join(procon_entry['pros']) + " " + " ".join(procon_entry['cons'])
        

        rating_vecs = np.array(data['rating_embed'].tolist())

        rating_index = faiss.IndexFlatL2(rating_vecs.shape[1])
        rating_k_index = faiss.IndexIDMap(rating_index)
        rating_k_index.add_with_ids(rating_vecs, keys)

        rating_input = rating_input.reshape(1, -1)
        rating_dist, rating_idx = rating_k_index.search(rating_input, 1)
        rat_co = ''
        rat_context = ''
        
        if rating_idx[0][0] == idx[0][0]:
            pass
        else:
            rat_entry = data.iloc[rating_idx[0][0]]
            rat_co = rat_entry['_id']
            rat_context = f"Company={rat_co}, Company review=" + " ".join(rat_entry['pros']) + " " + " ".join(rat_entry['cons'])

        context += rat_context

        return self.send_response(query, context)

    """
        Vector search function if the query describes a particular type
        of work or job the user wants to pursue.
    """

    def find_job_role(self, query, preferences):
        print("****CALLED: FIND JOB TITLE****")

        data = pd.read_json('employaid.title_pros_cons.json')

        title_procon_model = Doc2Vec.load('title_pros_cons')
        pref_vector = title_procon_model.infer_vector([preferences], epochs=1000)

        procon_vecs = np.array(data['procon_embed'].tolist())
        keys = np.array(data.index.tolist(), dtype=np.int64)

        index = faiss.IndexFlatL2(procon_vecs.shape[1])
        k_index = faiss.IndexIDMap(index)
        k_index.add_with_ids(procon_vecs, keys)

        pref_vector = np.array(pref_vector).reshape(1, -1)
        dist, idx = k_index.search(pref_vector, 1)

        procon_entry = data.iloc[idx[0][0]]

        title = procon_entry['_id']
        
        context = f"Job Title={title}, Title review=" + " ".join(procon_entry['pros']) + " " + " ".join(procon_entry['cons'])

        return self.send_response(query, context)


    """
        Vector search if the query is about a specific job type in a specific
        work environment.
    """

    def find_job_role_company(self, query, preferences):
        print("****CALLED: FIND JOB ROLE AND COMPANY****")

        data = pd.read_json('employaid.firm_job.json')
        
        firm_job_model = Doc2Vec.load('firm_job_model')
        pref_vector = firm_job_model.infer_vector([preferences], epochs=1000)

        total_vecs = np.array(data['total_embed'].tolist())

        keys = np.array(data.index.tolist(), dtype=np.int64)

        index = faiss.IndexFlatL2(total_vecs.shape[1])
        k_index = faiss.IndexIDMap(index)
        k_index.add_with_ids(total_vecs, keys)

        pref_vector = np.array(pref_vector).reshape(1, -1)
        dist, idx = k_index.search(pref_vector, 3)

        make_id = []
        
        d_entries = data[idx[0]]
        context = "["

        for i, d in enumerate(d_entries):
            context += f'Company{i+1} = ' + d['_id']['firm'] + f' Job Title {i+1} = ' + d['_id']['job_title'] + " " + " ".join(d['pros']) + " " + " ".join(d['cons'])
            if i < len(d_entries) - 1:
                context += ","

        context += "]"

        return self.send_response(query, context)
        
