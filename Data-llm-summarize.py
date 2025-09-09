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

from safetensors.torch import load_file

start = datetime.datetime.now()

# Define model configuration and import parameters

FROM_PRETRAINED = False
IS_DUMMY = False
DUMMY_PARAMETERS = {'hidden_size' : 2, 'intermediate_size' : 4, 'head_dim' : 8}

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
MAX_CONTEXT_LENGTH = 15000
ROPE_THETA = 1000000

BATCH_SIZE = 4
MAX_NEW_TOKENS = 1100
NUM_EXAMPLES = 0

# Model and data paths 

llm_path = Path.cwd().joinpath(*MODEL_DICT['model_path'])
data_folder = Path.cwd().joinpath('Data')

###---------------Essential functions---------------###

# Define loading function

def load_model(model_path, device, load_8bit = False, from_pretrained = True, is_dummy = False,
                   dummy_parameters = {}, rope_scaling = None, max_context_length = None, rope_theta = None):

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only = True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    gen_config = GenerationConfig.from_pretrained(model_path)

    if type(max_context_length) == int:
        setattr(gen_config, 'max_length', max_context_length)

    if from_pretrained:
    
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map = 'auto' if device == torch.device('cuda:0') else None,
            torch_dtype = torch.float16 if device == torch.device('cuda:0') else torch.float32,
            load_in_8bit = load_8bit,
            local_files_only = True,
            use_safetensors = True,
            rope_scaling = rope_scaling
        ).to(device)

    else:

        # Pulling out the config

        config = AutoConfig.from_pretrained(model_path, local_files_only = True, use_safetensors = True)
        strict_loading = not is_dummy
        
        # Optional config editing for a dummy model
        
        if is_dummy:
            for name, value in dummy_parameters.items():
                setattr(config, name, value)

        if type(rope_scaling) == dict:
            setattr(config, 'rope_scaling', rope_scaling)

        if type(max_context_length) == int:
            setattr(config, 'max_position_embeddings', max_context_length)

        if rope_theta is not None:
            setattr(config, 'rope_theta', rope_theta)
        
        # Instantiate the model
        
        model = AutoModelForCausalLM.from_config(config).to(device)
        
        # Load parameters
        
        with open(model_path.joinpath('model.safetensors.index.json'), 'r') as file:
            index = json.load(file)
        
        safetensors_params_map = defaultdict(list)
        for param_name, safetensor_name in index['weight_map'].items():
            safetensors_params_map[safetensor_name].append(param_name)

        full_state_dict = {}
        
        for safetensor_name, param_names in safetensors_params_map.items():
            safetensor_path = model_path.joinpath(safetensor_name)
            shard_dict = load_file(safetensor_path)
            model_dict = model.state_dict()
        
            shard_dict = {
                key : value for key, value in shard_dict.items()
                if key in model_dict and model_dict[key].shape == value.shape
            }

            full_state_dict.update(shard_dict)
        
        with warnings.catch_warnings(action = 'ignore'):
            model.load_state_dict(full_state_dict, strict = strict_loading)

        # Parallelize if possible

        if torch.cuda.device_count() > 1:
            model = model.to(device)
            model = nn.DataParallel(model)

    return tokenizer, model, gen_config
    

# Define batch inference function

def batch_inference_llama(model, tokenizer, df, query_column, answer_column, instruction_prompt, gen_config,
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

        
        batch_texts = (df[query_column].iloc[start:start+batch_size] + '\n\n' +
                           df[answer_column].iloc[start:start+batch_size]).tolist()

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
    
    return df

###---------------Import required files---------------###

# Upload data

processed_tickets_df = pd.read_json(data_folder.joinpath('processed_tickets_df.json'),
                                    orient = 'index', typ = 'frame',
                                    dtype = str, precise_float = True)

# Upload auxiliary files 

with open(data_folder.joinpath('instruction_prompt_summarization.txt')) as ip, \
    open(data_folder.joinpath('examples.txt')) as e:
        
    instruction_prompt = ip.read()
    examples = e.read()

examples = [(example.split('|<[sep1]>|')[0].strip(), example.split('|<[sep1]>|')[1].strip())
                for example in examples.split('|<[sep2]>|')]

print('Data is successfully uploaded!')

# Upload a model

tokenizer, llm_model, gen_config = \
    load_model(llm_path, DEVICE,
               load_8bit = False, from_pretrained = FROM_PRETRAINED,
               is_dummy = IS_DUMMY, dummy_parameters = DUMMY_PARAMETERS,
               rope_scaling = ROPE_SCALING, max_context_length = MAX_CONTEXT_LENGTH, rope_theta = ROPE_THETA)

print('Model is successfully uploaded!')

###---------------Run actual jobs---------------###

# Apply data transformation

processed_tickets_df = \
    batch_inference_llama(llm_model, tokenizer, processed_tickets_df,
                          'STRUCTUREDPROBLEM', 'STRUCTUREDSOLUTION',
                          instruction_prompt, gen_config,
                          examples = examples, batch_size = BATCH_SIZE,
                          max_new_tokens = MAX_NEW_TOKENS, num_examples = NUM_EXAMPLES)

processed_tickets_df.to_json(data_folder.joinpath('llm_summarized_tickets_df.json'),
                                 orient = 'index', double_precision = 15, index = True)


finish = datetime.datetime.now()

print(round((finish - start).total_seconds() / 60**2, 3))

