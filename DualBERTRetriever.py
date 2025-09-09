import os
import json
from pathlib import Path
import copy
import warnings
from collections import defaultdict
from collections import OrderedDict
from collections import Counter
import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoModel
from transformers import AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer

from safetensors.torch import load_file

# Custom pooler logic

class BertAttentionPooler(nn.Module):
    
    def __init__(self, config, device):
        
        super().__init__()

        self.attention_mask = None

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.softmax = nn.Softmax(dim = -1)
        self.dimension_scaling = torch.sqrt(torch.tensor(config.hidden_size))
        
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

        # Initialize weights and move to device

        self.apply(lambda module: self.init_weights(module, config.initializer_range, device))

    def init_weights(self, module, initializer_range, device):
        
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean = 0.0, std = initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        module.to(device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        query_embeds = self.query(hidden_states)
        key_embeds = self.key(hidden_states)
        
        scaled_dot = (query_embeds * key_embeds).sum(dim = -1) / self.dimension_scaling

        if not isinstance(self.attention_mask, torch.Tensor) or self.attention_mask.size()[:2] != hidden_states.size()[:2]:
            self.attention_mask = torch.ones(hidden_states.size()[:2], device = hidden_states.device)
            
        mask = self.attention_mask == 0
        scaled_dot = scaled_dot.masked_fill(mask, float('-inf'))
        
        pool_weights = self.softmax(scaled_dot).unsqueeze(-1)

        pooled_output = (pool_weights * hidden_states).sum(dim = 1)
        
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)

        self.attention_mask = None
        
        return pooled_output

# Model import function

def import_dual_BERT_encoder(llm_path, encoder_q_path, DEVICE, encoder_a_path = None):
    
    config = AutoConfig.from_pretrained(llm_path, local_files_only = True)
    llm_model_q = AutoModel.from_config(config).to(DEVICE)
    
    tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only = True)

    llm_model_q.pooler = BertAttentionPooler(llm_model_q.config, DEVICE)
    state_dict_q = {}
    
    for file in os.listdir(encoder_q_path):
        if file.split('.')[-1] == 'safetensors':      
            state_dict = load_file(encoder_q_path.joinpath(file))
            state_dict_q.update(state_dict)
    with warnings.catch_warnings(action = 'ignore'):
        llm_model_q.load_state_dict(state_dict_q, strict = True)

    if encoder_a_path is None:
        return llm_model_q, tokenizer

    llm_model_a = copy.deepcopy(llm_model_q).to(DEVICE)
    state_dict_a = {}

    for file in os.listdir(encoder_a_path):
        if file.split('.')[-1] == 'safetensors':      
            state_dict = load_file(encoder_a_path.joinpath(file))
            state_dict_a.update(state_dict)
    with warnings.catch_warnings(action = 'ignore'):
        llm_model_a.load_state_dict(state_dict_a, strict = True)

    
    return llm_model_q, llm_model_a, tokenizer

# Create vector representations for all tickets (queries and answers separately)

def embed_dataset(dataloader, encoder_q, encoder_a):

    encoder_q.eval()
    encoder_a.eval()
    
    with torch.no_grad():

        embed_q_list = []
        embed_a_list = []
        query_list = []
        answer_list = []
        db_id_list = []
    
        for input_ids_q, input_ids_a, attention_mask_q, attention_mask_a, query, answer, _, db_id in dataloader:
            
            input_ids_q = input_ids_q.to(encoder_q.device)
            attention_mask_q = attention_mask_q.to(encoder_q.device)
    
            input_ids_a = input_ids_a.to(encoder_a.device)
            attention_mask_a = attention_mask_a.to(encoder_a.device)
    
            encoder_q.pooler.attention_mask = attention_mask_q
            encoder_a.pooler.attention_mask = attention_mask_a
    
            embed_q = encoder_q(input_ids_q, attention_mask = attention_mask_q).pooler_output
            embed_a = encoder_a(input_ids_a, attention_mask = attention_mask_a).pooler_output

            embed_q_list.extend(embed_q.tolist())
            embed_a_list.extend(embed_a.tolist())
            query_list.extend(query)
            answer_list.extend(answer)
            db_id_list.extend(db_id)

    return pd.DataFrame({
        'Query' : query_list,
        'Answer' : answer_list,
        'Query embedding' : embed_q_list,
        'Answer embedding' : embed_a_list
    }, index = db_id_list)


