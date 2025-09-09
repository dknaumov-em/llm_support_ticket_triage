import os
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import GenerationConfig

from LLMClasses import LLMmanager
from DualBERTRetriever import import_dual_BERT_encoder
from VectorDB import FAISSdb

# Devices

USE_GPU = True
DEVICE_ENCODER = \
    torch.device('mps' if torch.mps.is_available() and USE_GPU else ('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'))
DEVICE_LLM = \
    torch.device('mps' if torch.mps.is_available() and USE_GPU else ('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'))

# Data

DATA_FILENAME = 'llm_summarized_tickets_df.json'
DATA_DIRECTORY = Path.cwd().joinpath('Data')

# Encoder

ENCODER_NUM = 0
ENCODER_DICT = \
    [
        {'name' : 'SBERT-all-MiniLM-L12-v2',
         'repo_id' : 'sentence-transformers/all-MiniLM-L12-v2',
         'required_files' : ['config.json', 'config_sentence_transformers.json', 'data_config.json',
                             'sentence_bert_config.json', 'tokenizer_config.json', 'special_tokens_map.json',
                             'model.safetensors', 'modules.json', 'tokenizer.json'],
         'model_path' : ['SBERT', 'all-MiniLM-L12-v2']
        },
        
        {'name' : 'SBERT-all-mpnet-base-v2',
         'repo_id' : 'sentence-transformers/all-mpnet-base-v2',
         'required_files' : ['config.json', 'config_sentence_transformers.json', 'data_config.json',
                             'sentence_bert_config.json', 'tokenizer_config.json', 'special_tokens_map.json',
                             'model.safetensors', 'modules.json', 'tokenizer.json'],
         'model_path' : ['SBERT', 'all-mpnet-base-v2']
        }
    ]

ENCODER_PATH = Path.cwd().joinpath(*ENCODER_DICT[ENCODER_NUM]['model_path'])
ENCODER_Q_PATH = ENCODER_PATH.joinpath('q_encoder', f"Checkpoint_{ENCODER_DICT[ENCODER_NUM]['name']}")

# LLM

MODEL_DICT = \
    {'name' : 'Llama-3-8B-Instruct',
     'repo_id' : 'meta-llama/Meta-Llama-3-8B-Instruct',
     'required_files' : ['config.json', 'generation_config.json', 'model.safetensors',
                         'model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors',
                         'model-00003-of-00004.safetensors', 'model-00004-of-00004.safetensors',
                         'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'],
     'model_path' : ['LLaMa', '3.1-8B-Instruct']}

LLM_PATH = Path.cwd().joinpath(*MODEL_DICT['model_path'])
FINETUNED_MODEL_PATH = LLM_PATH.joinpath('SFT', f"Checkpoint_{MODEL_DICT['name']}")
SYSTEM_PROMPT_PATH = LLM_PATH.joinpath('system_prompt.txt')
CHAT_TEMPLATE_PATH = LLM_PATH.joinpath('generation_template.txt')

ROPE_SCALING = None
MAX_CONTEXT_LENGTH = 16000
ROPE_THETA = 1000000
MAX_GENERATION_LENGTH = 1000

# Vector DB

FAISS_PATH = Path.cwd().joinpath('FAISS')
Q_INDEX_FILENAME = 'ticket_query_embeddings.index'
A_INDEX_FILENAME = 'ticket_answer_embeddings.index'
INDEX_DB_ID_MAP_FILENAME = 'index_id_db_map.json'
EMBED_Q_DIM = 384
EMBED_A_DIM = 384

# 1. Retriever wrapper

class SupportTicketRetriever:
    
    def __init__(self, vector_db, encoder, tokenizer):
        
        self.vector_db = vector_db
        self.encoder = encoder
        self.tokenizer = tokenizer

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:

        input_ids = self.tokenizer(query, return_tensors = 'pt',
                                   add_special_tokens = False,
                                   truncation = True, max_length = 512)['input_ids']

        query_vec = self.encoder(input_ids).numpy()
        candidates, similarities = self.vector_db.rank(query_vec, k = top_k, query_sim_weight = 0.6)
        documents = self.vector_db.get_metadata(candidates[0]).to_dict(orient = 'records')
        
        return [(similarities[0][i], documents[i]['Query'] 
                    + '\n\n' 
                    + documents[i]['Summarized Answer']) for i in range(len(documents))]

# 2. LLM wrapper

class SupportLLM:
    
    def __init__(self, model, tokenizer, gen_config, chat_template = None, max_len = 15000, max_new_tokens = 1000):

        self.model = model

        self.tokenizer = tokenizer
        self.gen_config = gen_config
        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        
        self.max_len = max_len
        self.max_new_tokens = max_new_tokens

    def generate(self, conversation: list, retrieved_documents: list) -> str:

        prompt = self.tokenizer.apply_chat_template(
                      conversation = conversation,
                      documents = retrieved_documents,
                      add_generation_prompt = True) \
                   .strip()
        
        inputs = self.tokenizer(prompt, return_tensors = 'pt')
        outputs = self.model.generate(inputs['input_ids'],
                                      max_new_tokens = self.max_new_tokens,
                                      pad_token_id = tokenizer.pad_token_id,
                                      generation_config = self.gen_config,
                                      return_dict_in_generate = True)

        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens = True) \
                             .split('<|start_header_id|>assistant<|end_header_id|>')[-1].strip()
        
        return generated_text

# 3. App logic

class SupportTicketAssistantApp:
    
    def __init__(self, retriever: MyRetriever, llm: MyLLM, system_prompt: str):
        
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt
        self.add_system_prompt = 'Keep answering follow-up questions and respond to any clarifications as a helpful assistant.'
        self.conversation =  [{'role' : 'system', 'content' : self.system_prompt}]
        self.conversation_turns = []
        self.excluded_history = []
        self.retrieved_documents = None

        # Token count properties 
        
        self.chat_token_count = \
            len(self.llm.tokenizer.apply_chat_template(
                  conversation = [{'role' : 'system', 'content' : self.system_prompt},
                                  {'role' : 'system_add', 'content' : self.add_system_prompt}],
                  documents = None,
                  add_generation_prompt = False,
                  tokenize = True))
        self.static_token_count = self.chat_token_count
        self.turn_token_count = 0

    
    def run(self, query: str) -> str:

        # Retrieve documents at the begining 
    
        if not self.conversation_turns:
            self.retrieved_documents = self.retriever.retrieve(query)

            for i, (sim, doc) in enumerate(self.retrieved_documents):
                print('\n\nRetrieved documents:\n\n')
                print(f'Document: {i + 1} // {sim} similarity\n\n')
                print(doc)
                self.chat_token_count += len(self.llm.tokenizer(doc)['input_ids'])
                self.static_token_count += len(self.llm.tokenizer(doc)['input_ids'])
            
        # Append a query to a conversation history

        allowed_tokens = self.llm.max_len - self.llm.max_new_tokens - self.static_token_count - self.turn_token_count
        query_token_length = len(self.llm.tokenizer(query)['input_ids'])
        tokens_to_trim = 0 if (allowed_tokens >= query_token_length) else abs(allowed_tokens - query_token_length)

        if tokens_to_trim: 
            if tokens_to_trim > self.turn_token_count:
                raise ValueError('Query is too long.')
            else:
                trimmed_token_count = 0
                for turn in self.conversation_turns[1:]:
                    if turn['Included']:
                        turn['Included'] = False
                        trimmed_token_count += turn['TokenCount']
                        self.turn_token_count -= turn['TokenCount']

                        self.excluded_history.append(turn['QueryID'])
                        self.excluded_history.append(turn['AnswerID'])

                        if trimmed_token_count >= tokens_to_trim:
                            break

        query_id = len(self.conversation)
        query_chat = {'role' : 'assistant', 'content' : query}
        self.conversation.append(query_chat)

        # Generate an answer

        live_conversation = [conversation[i] for i in range(len(conversation)) if i not in self.excluded_history]
        
        answer = self.llm.generate(live_conversation)

        # Append an answer to a conversation history

        answer_id = len(self.conversation)
        answer_chat = {'role' : 'assistant', 'content' : answer}
        self.conversation.append(answer_chat)

        if not self.conversation_turns and self.add_system_prompt is not None:
            self.conversation.append({'role' : 'system_add', 'content' : self.add_system_prompt})

        # Update the turn list
        
        turn_token_length = \
            len(self.llm.tokenizer.apply_chat_template(
                          conversation = [query_chat, answer_chat],
                          documents = None,
                          add_generation_prompt = False,
                          tokenize = True))
        self.chat_token_count += turn_token_length

        if not self.conversation_turns:
            self.static_token_count += turn_token_length
        else:
            self.turn_token_count += turn_token_length

        self.conversation_turns.append(
            {'QueryID' : query_id,
             'AnswerID' : answer_id,
             'TokenCount' : turn_token_length,
             'Included' : True}
        )
        
        return answer

# 4.Programm execution

def main():

    # Import a Data

    tickets_df = pd.read_json(DATA_DIRECTORY.joinpath(DATA_FILENAME), 
                                         orient = 'index', typ = 'frame', dtype = str)
    tickets_df['STRUCTUREDSOLUTION'] = \
        tickets_df['STRUCTUREDSOLUTION'].str.split('\n\nTicket status history:\n\n').str[0] \
            + '\n\nActivities description:\n\n' \
            + tickets_df['SUMMARIZEDSOLUTION']
    
    tickets_df.drop(columns = ['PROBLEM', 'SOLUTION', 'SUMMARIZEDSOLUTION'], inplace = True)
    tickets_df.set_index('TICKETID', drop = True, inplace = True)
    tickets_df.rename(columns = {'STRUCTUREDPROBLEM' : 'Query', 'STRUCTUREDSOLUTION' : 'Summarized Answer'}, inplace = True)
    
    # Import an Encoder

    encoder_q, tokenizer_e = import_dual_BERT_encoder(ENCODER_PATH, ENCODER_Q_PATH, DEVICE_ENCODER)

    # Load the FAISS vector DB
    
    index_q, index_a, index_db_id_map = FAISSdb.read(
        FAISS_PATH,
        Q_INDEX_FILENAME,
        A_INDEX_FILENAME,
        INDEX_DB_ID_MAP_FILENAME,
        reinitialize_vector_index = False
        embed_q_dim = EMBED_Q_DIM, embed_a_dim = EMBED_A_DIM
    )
    
    faiss_db = FAISSdb(index_q, index_a, index_db_id_map, dataset = tickets_df)
    
    # Import generative LLM and supporting files
    
    llm_utility = \
        LLMmanager(LLM_PATH, DEVICE_LLM, finetuned_model_path = FINETUNED_MODEL_PATH,
                   peft = True, from_pretrained = False, load_8bit = False, use_cache = True, is_dummy = False,
                   rope_scaling = ROPE_SCALING, rope_theta = ROPE_THETA, max_context_length = MAX_CONTEXT_LENGTH)

    tokenizer_g, model, gen_config = llm_utility.load_model(load_finetuned = True, parallelize = True)

    with open(SYSTEM_PROMPT_PATH) as sp, open(CHAT_TEMPLATE_PATH) as gt:

        system_prompt = sp.read()
        generation_template = gt.read()

    # Define main application classes

    retriever = SupportTicketRetriever(faiss_db, encoder_q, tokenizer_e)

    llm = SupportLLM(model, tokenizer_g, gen_config, chat_template = generation_template, max_len = 15000, max_new_tokens = 1000)

    app = SupportTicketAssistantApp(retriever, llm, system_prompt)

    print('Support ticket LLM-assistant app has started. Write your enquiry or enter an empty string to end the chat.')
    
    while True:
        query = input('User: ')
        if query.lower() == '':
            break
        answer = app.run(query)
        print('Assistant:', answer)
        print("-" * 50)
        

if __name__ == "__main__":
    main()
