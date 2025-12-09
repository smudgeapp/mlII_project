from google import genai
from google.genai import types
import os
import faiss
import numpy as np
import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec
import json
from pymongo import MongoClient

class EmployAidAgent:
    #api_key = os.environ.get("GEMINI_API_KEY")

    model = "gemini-2.5-flash"
    
    def __init__(self, api_key=None):
        access_key = api_key
        
        if api_key is None:
            access_key = os.environ.get("GEMINI_API_KEY")
            
        
        client_gen = genai.Client(api_key=access_key)
        
        with open("system_instructions.json", mode='r') as f:
            self.instructions = json.load(f)
            
        self.client = genai.Client()
        system_config = types.GenerateContentConfig(
            system_instruction=self.instructions['system_instruction'])
        self.agent = self.client.chats.create(model=self.model,
                                         config=system_config)
        self.total_tokens = 0

    def intro_query(self, query):
        ##TODO make json schema
        config = types.GenerateContentConfig(response_mime_type='application/json',
                                             response_schema=self.instructions['intro'])
        response = self.agent.send_message(query, config=config)
        return response.text

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
            return self.send_response(text_response)



    def send_response(self, query, context='database entry was not found. Ask for more specific information'):
        full_context = f'Respond to {query} based on {context}.'
        print(full_context)
        response = self.agent.send_message(full_context)
        current_tokens = response.usage_metadata.total_token_count
        self.total_tokens += current_tokens
        
        return response.text, current_tokens, self.total_tokens
        
    
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
            

        context = " ".join(company_data['pros'].tolist()[0]) + " " + " ".join(company_data['cons'].tolist()[0])

        return self.send_response(query, context)

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

        context = " ".join(job_title_data['pros'].tolist()[0]) + " " + " ".join(job_title_data['cons'].tolist()[0])

        return self.send_response(query, context)


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
        keys = np.arange(data.index.tolist(), dtype=np.int64)

        procon_index = faiss.IndexFlatL2(procon_vecs.shape[1])
        procon_k_index = faiss.IndexIDMap(procon_index)
        procon_k_index.add_with_ids(procon_vecs, keys)

        summary_vec = summary_vec.reshape(1, -1)
        dist, idx = procon_k_index.search(summary_vec, 1)

        procon_entry = data.iloc[idx[0][0]]
        company = procon_entry['_id'].values[0]

        context = f"Company={company}, Company review=" + " ".join(procon_entry['pros'].tolist()[0]) + " " + " ".join(procon_entry['cons'].tolist()[0])
        

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
            rat_co = rat_entry['_id'].values[0]
            rat_context = f"Company={rat_co}, Company review=" + " ".join(rat_entry['pros'].tolist()[0]) + " " + " ".join(rat_entry['cons'].tolist()[0])

        context += rat_context

        return self.send_response(query, context)

    def find_job_role(self, query, preferences):
        print("****CALLED: FIND JOB TITLE****")

        data = pd.read_json('employaid.title_pros_cons.json')

        title_procon_model = Doc2Vec.load('title_pros_cons')
        pref_vector = title_procon_model.infer_vector([preferences], epochs=1000)

        procon_vecs = np.array(data['procon_embed'].tolist())
        keys = np.arange(data.index.tolist(), dtype=np.int64)

        index = faiss.IndexFlatL2(procon_vecs.shape[1])
        k_index = faiss.IndexIDMap(index)
        k_index.add_with_ids(procon_vecs, keys)

        pref_vector = np.array(pref_vector).reshape(1, -1)
        dist, idx = k_index.search(pref_vector, 1)

        procon_entry = data.iloc[idx[0][0]]

        title = procon_entry['_id'].values[0]
        
        context = f"Job Title={title}, Title review=" + " ".join(procon_entry['pros'].tolist()[0]) + " " + " ".join(procon_entry['cons'].tolist()[0])

        return self.send_response(query, context)
        

    def find_job_role_company(self, query, preferences):
        print("****CALLED: FIND JOB ROLE AND COMPANY****")

        data = pd.read_json('employaid.firm_job.json')
        
        firm_job_model = Doc2Vec.load('firm_job_model')
        pref_vector = firm_job_model.infer_vector([preferences], epochs=1000)

        total_vecs = np.array(data['total_embed'].tolist())

        keys = np.arange(data.index.tolist(), dtype=np.int64)

        index = faiss.IndexFlatL2(total_vecs.shape[1])
        k_index = faiss.IndexIDMap(index)
        k_index.add_with_ids(total_vecs, keys)

        pref_vector = np.array(pref_vector).reshape(1, -1)
        dist, idx = k_index.search(pref_vector, 3)

        make_id = []
        
        d_entries = data[idx[0]]
        context = "["

        for i, d in enumerate(d_entries):
            context += f'Company{i+1} = ' + d['_id']['firm'] + f' Job Title {i+1} = ' + d['_id']['job_title'] + " " + " ".join(d['pros'].tolist()[0]) + " " + " ".join(d['cons'].tolist()[0])
            if i < len(d_entries) - 1:
                context += ","

        context += "]"

        return self.send_response(query, context)
        
