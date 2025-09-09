import os
import sys
import re
import random
import json
from pathlib import Path
import copy
import warnings
from collections import defaultdict
import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import GenerationConfig

from LLMClasses import AdapterLlamaMLP, LoRALinear, LLMmanager

start = datetime.datetime.now()

# Define model configuration and import parameters

PEFT = False
FROM_PRETRAINED = False
IS_DUMMY = False
DUMMY_PARAMETERS = {'hidden_size' : 2, 'intermediate_size' : 4, 'head_dim' : 8}
LOAD_8BIT = False
LOAD_FINETUNED = False
PARALLELIZE = True
USE_CACHE = True

USE_GPU = True
DEVICE = torch.device('mps' if torch.mps.is_available() and USE_GPU else ('cuda:0' if torch.cuda.is_available() and USE_GPU else 'cpu'))

print('Device: {0} ({1})'.format(DEVICE, torch.cuda.device_count()))

MODEL_DICT = \
    {'name' : 'Llama-3-8B-Instruct',
     'repo_id' : 'meta-llama/Meta-Llama-3-8B-Instruct',
     'required_files' : ['config.json', 'generation_config.json', 'model.safetensors',
                         'model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors',
                         'model-00003-of-00004.safetensors', 'model-00004-of-00004.safetensors',
                         'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'],
     'model_path' : ['LLaMa', '3.1-8B-Instruct']}

ROPE_SCALING = None
ROPE_THETA = 1000000
MAX_CONTEXT_LENGTH = 15000

BATCH_SIZE = 4
MAX_NEW_TOKENS = 1100
NUM_EXAMPLES = 0

# Model and data paths 

LLM_PATH = Path.cwd().joinpath(*MODEL_DICT['model_path'])
FINETUNED_MODEL_PATH = LLM_PATH.joinpath('SFT', f"Checkpoint_{MODEL_DICT['name']}")
DATA_FOLDER = Path.cwd().joinpath('Data')

TICKETS_DF_FILENAME = 'original_evaluation_df.json'
INSTRUCTION_FILENAME = 'instruction_prompt_judge.txt'
EXAMPLES_FILENAME = None
OUTPUT_FIlENAME = 'or_llm_judged_tickets.json'

###---------------Essential functions---------------###

# Define batch inference function

def batch_inference_llama(model, tokenizer, df, query_column, answer_column, gen_answer_column, instruction_prompt, gen_config,
                              examples = None, batch_size = 4, max_new_tokens = 1024, num_examples = 1):
    
    results = []
    max_context_len = gen_config.max_length

    orig_model = model.module if hasattr(model, 'module') else model
    
    # Iterate over batches
    
    for start in tqdm(range(0, len(df), batch_size), file = sys.stdout):

        conversation = []
        batch_texts_truncated = []
        
        # Build few-shot prompt

        random.shuffle(examples)
        
        if examples:
            for i in range(num_examples):
                conversation.append({'role' : 'user', 'content' : examples[i][0]})
                conversation.append({'role' : 'assistant', 'content' : examples[i][1]})
    
        # System role
        
        conversation.insert(0, {'role' : 'system', 'content' : instruction_prompt})
    
        # Tokenize static context
        
        static_context_tokens = tokenizer.apply_chat_template(
            conversation = conversation,
            add_generation_prompt = True,
            tokenize = True
        )
        static_token_count = len(static_context_tokens)
        
        batch_texts = ('### User Query:\n'
                           + df[query_column].iloc[start:start+batch_size] 
                           + '\n\n### Consultant Answer (Human):\n'
                           + df[answer_column].iloc[start:start+batch_size]
                           + '\n\n### Generated Answer (Model):\n'
                           + df[gen_answer_column].iloc[start:start+batch_size]
                      ).tolist()

        # Ensure a correct context length

        for text in batch_texts:

            text_tokens = tokenizer(text.strip(), return_tensors = 'pt')
            text_len = text_tokens.input_ids.shape[1]

            # Check if adding this text would exceed max context
    
            if static_token_count + text_len + max_new_tokens > max_context_len:

                allowed_len = max_context_len - static_token_count - max_new_tokens
                truncated_ids = text_tokens.input_ids[:, :allowed_len]
                truncated_text = tokenizer.decode(truncated_ids[0], skip_special_tokens = True)
                batch_texts_truncated.append(truncated_text)
                
            else:
                
                batch_texts_truncated.append(text)
        
        # Build prompts for batch
        
        prompts = [
            tokenizer.apply_chat_template(
                conversation = conversation + [{'role' : 'user', 'content': text.strip()}],
                add_generation_prompt = True,
                tokenize = False) \
            for text in batch_texts_truncated
        ]
        
        # Tokenize
        
        inputs = tokenizer(prompts, return_tensors = 'pt',
                               padding = True, padding_side = 'left',
                                   truncation = False).to(orig_model.device)
        
        # Generate outputs

        orig_model.eval()
        
        with torch.no_grad():
            outputs = orig_model.generate(
                **inputs,
                pad_token_id = tokenizer.pad_token_id,
                generation_config = gen_config,
                return_dict_in_generate = True,
                use_cache = True,
                max_new_tokens = max_new_tokens
            )
        
        decoded_outputs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens = False)
        
        # Extract only the part after assistant generation
        
        cleaned_outputs = []
        for full_text in decoded_outputs:
            if '<|start_header_id|>assistant<|end_header_id|>' in full_text:
                cleaned_outputs.append(full_text.split('<|start_header_id|>assistant<|end_header_id|>')[-1].strip())
            else:
                cleaned_outputs.append(full_text.strip())
        
        results.extend(cleaned_outputs)
    
    df = df.copy()
    df['SUMMARIZEDSOLUTION'] = results
    df['LLMJudgement'] = df['SUMMARIZEDSOLUTION'].str.extract(r'\*\*Final Verdict:\*\*\s*\**([A-Z]+_MATCH_?[A-Z]*)\**\b')[0]
    
    return df

###---------------Import required files---------------###

# Upload data

processed_tickets_df = pd.read_json(DATA_FOLDER.joinpath(TICKETS_DF_FILENAME),
                                    orient = 'index', typ = 'frame',
                                    dtype = str, precise_float = True)

# Upload auxiliary files 

with open(DATA_FOLDER.joinpath(INSTRUCTION_FILENAME)) as ip:
    instruction_prompt = ip.read()

if EXAMPLES_FILENAME is not None:
    with open(DATA_FOLDER.joinpath(EXAMPLES_FILENAME)) as e:
        examples = e.read()
    
    examples = [(example.split('|<[sep1]>|')[0].strip(), example.split('|<[sep1]>|')[1].strip())
                    for example in examples.split('|<[sep2]>|')]
else:
    examples = []

print('Data is successfully uploaded!')

# Upload a model

llm_utility = \
    LLMmanager(LLM_PATH, DEVICE, finetuned_model_path = FINETUNED_MODEL_PATH, use_cache = USE_CACHE,
               peft = PEFT, from_pretrained = FROM_PRETRAINED, load_8bit = LOAD_8BIT,
               is_dummy = IS_DUMMY, dummy_parameters = DUMMY_PARAMETERS,
               rope_scaling = ROPE_SCALING, rope_theta = ROPE_THETA, max_context_length = MAX_CONTEXT_LENGTH)

tokenizer, llm_model, gen_config = llm_utility.load_model(load_finetuned = LOAD_FINETUNED, parallelize = PARALLELIZE)

print('Model is successfully uploaded!')

###---------------Run actual jobs---------------###

# Apply data transformation

processed_tickets_df = \
    batch_inference_llama(llm_model, tokenizer, processed_tickets_df,
                          'input_text', 'target_text', 'prediction',
                          instruction_prompt, gen_config,
                          examples = examples, batch_size = BATCH_SIZE,
                          max_new_tokens = MAX_NEW_TOKENS, num_examples = NUM_EXAMPLES)

processed_tickets_df.to_json(DATA_FOLDER.joinpath(OUTPUT_FIlENAME),
                                 orient = 'index', double_precision = 15, index = True)


finish = datetime.datetime.now()

print(round((finish - start).total_seconds() / 60**2, 3))
