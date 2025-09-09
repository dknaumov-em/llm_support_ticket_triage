import os
import json
from pathlib import Path
import copy
import warnings
from collections import defaultdict
from collections import OrderedDict
from collections import Counter
from tqdm import tqdm
import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import DistributedSampler

# Declare a special dataset class for CPT (Continued Pre-training) and SFT datasets

class FineTuneDataset(Dataset):
    
    def __init__(self, tokenizer, dataset, dataset_type, target_col = None, system_prompt = None, query_col = None, id_col = None,
                     context_dataset = None, query_con_col = None, answer_con_col = None, query_match_col = None, sim_match_col = None,
                         num_retrieved = None, includes_gold_prob = None, relevant_prob_dist = None,
                             irrelevant_prob_dist = None, train = True, max_context_length = 16000, max_generation_length = 1000):

        if dataset_type == 'SFT' and (query_col is None or context_dataset is None 
                                      or query_match_col is None or sim_match_col is None
                                      or query_con_col is None or answer_con_col is None):
            raise ValueError('SFT dataset must have valid query, context and embeddings')

        self.tokenizer = tokenizer
        self.train = train
        self.dataset_type = dataset_type
        self.max_context_length = max_context_length
        self.max_generation_length = max_generation_length

        # Context evaluation phrases for sampling

        self.relevant_comments = [
            'The retrieved cases are relevant and provide actionable insights applicable to this query.',
            'Relevant prior cases were found and directly informed the suggested actions.',
            'The context from retrieved tickets was useful and guided the resolution strategy.',
            'Prior cases align closely with the current issue and offer practical guidance.',
            'Useful precedents were identified, supporting the recommended troubleshooting steps.'
        ]

        self.semi_relevant_comments = [
            'The retrieved cases are partially relevant. Some suggestions may apply but require adaptation.',
            'Prior tickets contain useful hints but do not fully address the current situation.',
            'Context provides limited insights. Additional investigation may be necessary.',
            'The retrieved cases offer partial guidance, but not all information is directly applicable.',
            'Some elements from prior cases can inform the solution, but verification is needed.'
        ]

        self.non_relevant_comments = [
            'The retrieved cases do not appear to be relevant to this query.',
            'No prior cases provide actionable insights for the current issue.',
            'Context from retrieved tickets is not directly applicable. Alternative strategies should be considered.',
            'The prior cases seem irrelevant to the query. Guidance will be derived from general troubleshooting knowledge.',
            'Retrieved tickets do not contribute to resolving this query.'
        ]

        if self.dataset_type == 'CPT':
            self.samples = dataset

        if self.dataset_type == 'SFT':

            dataset = dataset.merge(context_dataset[[query_match_col, sim_match_col]],
                                        how = 'left', left_on = id_col, right_index = True)

            self.system_prompt = system_prompt
            self.system_prompt_length = self.tokenizer(self.system_prompt, add_special_tokens = False,
                                                           return_tensors = 'pt')['attention_mask'].size()[1]
            
            self.context_dataset = context_dataset
            self.context_len = len(context_dataset)
            self.min_context_len = 300
            self.query_con_col = query_con_col
            self.answer_con_col = answer_con_col
            self.num_retrieved = num_retrieved if num_retrieved is not None else 5
            self.includes_gold_prob = includes_gold_prob if includes_gold_prob is not None else 0.7
            self.relevant_prob_dist = relevant_prob_dist if relevant_prob_dist is not None else [1 / self.num_retrieved] * self.num_retrieved
            self.irrelevant_prob_dist = irrelevant_prob_dist if irrelevant_prob_dist is not None else \
                                            [1 / (self.num_retrieved + 1)] * (self.num_retrieved + 1)
            self.samples = list((query, context, target) for query, context, target in \
                                    zip(dataset[query_col].tolist(),
                                        zip(dataset[query_match_col].tolist(), dataset[sim_match_col].tolist()),
                                        dataset[target_col].tolist()))

    def __len__(self):
        
        return len(self.samples)

    def __getitem__(self, idx):

        # Get data item depending on CPT or SFT training

        if self.dataset_type == 'CPT':
            text = self.tokenizer.bos_token + self.samples[idx] + self.tokenizer.eos_token
            whitespace_sep_text = text.split()
            instruct_text = whitespace_sep_text[:int(len(whitespace_sep_text) / 2)]
            target_text = whitespace_sep_text[int(len(whitespace_sep_text) / 2):]

        if self.dataset_type == 'SFT':

            # Retrieve context documents and adjust context length based on global context length

            retrieved_context, comment = self.retrieve_context(idx)
            retrieved_context = [(rank, doc) for rank, doc in sorted(retrieved_context, key = lambda x: x[0], reverse = True)]
            answer = comment + '\n\n' + self.samples[idx][2].rstrip('<|eot_id|>')
            
            retrieved_context_tokenized = [
                (rank, self.tokenizer(doc, add_special_tokens = False, return_tensors = 'pt')['input_ids'].squeeze(0))
                    for rank, doc in retrieved_context
            ]

            retrieved_context_lengths = torch.tensor([ids.size(0) for rank, ids in retrieved_context_tokenized])

            query_length = self.tokenizer(self.samples[idx][0], add_special_tokens = False,
                                              return_tensors = 'pt')['attention_mask'].size()[1]
            answer_length = self.tokenizer(answer, add_special_tokens = False,
                                              return_tensors = 'pt')['attention_mask'].size()[1]

            excess_length = (self.max_context_length 
                                 - self.system_prompt_length 
                                 - query_length
                                 - answer_length
                                 - retrieved_context_lengths.sum().item())
            
            excess_length = -excess_length if excess_length < 0 else 0

            if excess_length > 0:
                
                allowed_context_lengths = torch.clamp(retrieved_context_lengths - 
                                                        torch.ceil((retrieved_context_lengths / retrieved_context_lengths.sum()) 
                                                                       * excess_length).to(torch.long), min = 300)
                
            else:
                
                allowed_context_lengths = retrieved_context_lengths

            # Apply chat templates and distingush between instructions and generation parts

            retrieved_documents = [
                {'content' : self.tokenizer.decode(ids[:allowed_context_lengths[i].item()], skip_special_tokens = True)}
                     for i, (rank, ids) in enumerate(retrieved_context_tokenized)
            ]
            
            text = self.tokenizer.apply_chat_template(
                conversation = [
                    {'role' : 'system', 'content' : self.system_prompt},
                    {'role' : 'user', 'content' : self.samples[idx][0]},
                    {'role' : 'assistant', 'content' : answer}
                ],
                documents = retrieved_documents,
                add_generation_prompt = False,
                tokenize = False
            ).strip()

            instruct_text = self.tokenizer.apply_chat_template(
                conversation = [
                    {'role' : 'system', 'content' : self.system_prompt},
                    {'role' : 'user', 'content' : self.samples[idx][0]},
                    {'role' : 'assistant', 'content' : ''}
                ],
                documents = retrieved_documents,
                add_generation_prompt = False,
                tokenize = False
            ).strip()

            target_text = text[len(instruct_text)-10:]

        # Tokenize text/template
            
        encoding = self.tokenizer(text, return_tensors = 'pt', add_special_tokens = False)
        input_ids = encoding['input_ids'].squeeze(0)

        # Get position to start loss calculation

        if self.dataset_type == 'CPT':

            labels = input_ids.clone()

        if self.dataset_type == 'SFT':
        
            mask_start = self.tokenizer(instruct_text, return_tensors = 'pt', add_special_tokens = False)['input_ids'].squeeze(0).size()[0]

            # Mask an input up to expected generated text
            
            labels = input_ids.clone()
            labels[:mask_start-1] = -100

        return input_ids, labels, instruct_text, target_text, self.train
        
    def retrieve_context(self, idx):

        includes_gold = np.random.rand() <= self.includes_gold_prob

        if includes_gold:

            random_num = self.num_retrieved - np.random.choice(list(range(self.num_retrieved)), p = self.relevant_prob_dist)
            retrieval_idx = [self.samples[idx][1][0][i] if i < random_num
                                 else self.context_dataset.index[np.random.randint(0, self.context_len)]
                                     for i in range(self.num_retrieved)]
            match_sim = [round(self.samples[idx][1][1][i], 3) if i < random_num 
                             else round(np.random.rand() * 0.05, 3)
                                 for i in range(self.num_retrieved)]
            
        else:

            random_num = self.num_retrieved - np.random.choice(list(range(self.num_retrieved + 1)), p = self.irrelevant_prob_dist)
            retrieval_idx = [self.samples[idx][1][0][1:][i] if i < random_num
                                 else self.context_dataset.index[np.random.randint(0, self.context_len)]
                                     for i in range(self.num_retrieved)]
            match_sim = [round(self.samples[idx][1][1][1:][i], 3) if i < random_num
                             else round(np.random.rand() * 0.05, 3)
                                 for i in range(self.num_retrieved)]
            
        content = list(zip(match_sim,
                           list(map(lambda i: self.context_dataset.loc[i][self.query_con_col] + '\n\n' \
                                        + self.context_dataset.loc[i][self.answer_con_col],
                                    retrieval_idx))))
        
        if includes_gold:
            comment = np.random.choice(self.relevant_comments)
        elif np.array(match_sim).max() >= 0.6:
            comment = np.random.choice(self.semi_relevant_comments)
        else:
            comment = np.random.choice(self.non_relevant_comments)
            
        return content, comment

    def batch_collate(self, batch):
        
        input_ids, labels, instruct_text, target_text, is_train = zip(*batch)
        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first = True, padding_value = self.tokenizer.pad_token_id)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first = True, padding_value = -100)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.long)
        
        return input_ids, attention_mask, labels, instruct_text, target_text, is_train

# Special evaluation instance

class EvaluationDataset(Dataset):
    
    def __init__(self, tokenizer, dataset, dataset_type, target_col = None, system_prompt = None, query_col = None, id_col = None,
                     context_dataset = None, query_con_col = None, answer_con_col = None, query_match_col = None, sim_match_col = None,
                         num_retrieved = None, includes_gold_prob = None, relevant_prob_dist = None,
                             irrelevant_prob_dist = None, train = True, max_context_length = 16000, max_generation_length = 1000):

        if dataset_type == 'SFT' and (query_col is None or context_dataset is None 
                                      or query_match_col is None or sim_match_col is None
                                      or query_con_col is None or answer_con_col is None):
            raise ValueError('SFT dataset must have valid query, context and embeddings')

        self.tokenizer = tokenizer
        self.train = train
        self.dataset_type = dataset_type
        self.max_context_length = max_context_length
        self.max_generation_length = max_generation_length

        # Context evaluation phrases for sampling

        self.relevant_comments = [
            'The retrieved cases are relevant and provide actionable insights applicable to this query.',
            'Relevant prior cases were found and directly informed the suggested actions.',
            'The context from retrieved tickets was useful and guided the resolution strategy.',
            'Prior cases align closely with the current issue and offer practical guidance.',
            'Useful precedents were identified, supporting the recommended troubleshooting steps.'
        ]

        self.semi_relevant_comments = [
            'The retrieved cases are partially relevant. Some suggestions may apply but require adaptation.',
            'Prior tickets contain useful hints but do not fully address the current situation.',
            'Context provides limited insights. Additional investigation may be necessary.',
            'The retrieved cases offer partial guidance, but not all information is directly applicable.',
            'Some elements from prior cases can inform the solution, but verification is needed.'
        ]

        self.non_relevant_comments = [
            'The retrieved cases do not appear to be relevant to this query.',
            'No prior cases provide actionable insights for the current issue.',
            'Context from retrieved tickets is not directly applicable. Alternative strategies should be considered.',
            'The prior cases seem irrelevant to the query. Guidance will be derived from general troubleshooting knowledge.',
            'Retrieved tickets do not contribute to resolving this query.'
        ]

        if self.dataset_type == 'CPT':
            self.samples = dataset

        if self.dataset_type == 'SFT':

            dataset = dataset.merge(context_dataset[[query_match_col, sim_match_col]],
                                        how = 'left', left_on = id_col, right_index = True)

            self.system_prompt = system_prompt
            self.system_prompt_length = self.tokenizer(self.system_prompt, add_special_tokens = False,
                                                           return_tensors = 'pt')['attention_mask'].size()[1]
            
            self.context_dataset = context_dataset
            self.context_len = len(context_dataset)
            self.min_context_len = 300
            self.query_con_col = query_con_col
            self.answer_con_col = answer_con_col
            self.num_retrieved = num_retrieved if num_retrieved is not None else 5
            self.includes_gold_prob = includes_gold_prob if includes_gold_prob is not None else 0.7
            self.relevant_prob_dist = relevant_prob_dist if relevant_prob_dist is not None else [1 / self.num_retrieved] * self.num_retrieved
            self.irrelevant_prob_dist = irrelevant_prob_dist if irrelevant_prob_dist is not None else \
                                            [1 / (self.num_retrieved + 1)] * (self.num_retrieved + 1)
            self.samples = list((query, context, target) for query, context, target in \
                                    zip(dataset[query_col].tolist(),
                                        zip(dataset[query_match_col].tolist(), dataset[sim_match_col].tolist()),
                                        dataset[target_col].tolist()))

    def __len__(self):
        
        return len(self.samples)

    def __getitem__(self, idx):

        # Get data item depending on CPT or SFT training

        if self.dataset_type == 'CPT':
            text = self.tokenizer.bos_token + self.samples[idx] + self.tokenizer.eos_token
            whitespace_sep_text = text.split()
            instruct_text = whitespace_sep_text[:int(len(whitespace_sep_text) / 2)]
            target_text = whitespace_sep_text[int(len(whitespace_sep_text) / 2):]

        if self.dataset_type == 'SFT':

            # Retrieve context documents and adjust context length based on global context length

            retrieved_context, comment, includes_gold, random_num  = self.retrieve_context(idx)
            retrieved_context = [(rank, doc) for rank, doc in sorted(retrieved_context, key = lambda x: x[0], reverse = True)]
            answer = comment + '\n\n' + self.samples[idx][2].rstrip('<|eot_id|>')
            
            retrieved_context_tokenized = [
                (rank, self.tokenizer(doc, add_special_tokens = False, return_tensors = 'pt')['input_ids'].squeeze(0))
                    for rank, doc in retrieved_context
            ]

            retrieved_context_lengths = torch.tensor([ids.size(0) for rank, ids in retrieved_context_tokenized])

            query_length = self.tokenizer(self.samples[idx][0], add_special_tokens = False,
                                              return_tensors = 'pt')['attention_mask'].size()[1]
            answer_length = self.tokenizer(answer, add_special_tokens = False,
                                              return_tensors = 'pt')['attention_mask'].size()[1]

            excess_length = (self.max_context_length 
                                 - self.system_prompt_length 
                                 - query_length
                                 - answer_length
                                 - retrieved_context_lengths.sum().item())
            
            excess_length = -excess_length if excess_length < 0 else 0

            if excess_length > 0:
                
                allowed_context_lengths = torch.clamp(retrieved_context_lengths - 
                                                        torch.ceil((retrieved_context_lengths / retrieved_context_lengths.sum()) 
                                                                       * excess_length).to(torch.long), min = 300)
                
            else:
                
                allowed_context_lengths = retrieved_context_lengths

            # Apply chat templates and distingush between instructions and generation parts

            retrieved_documents = [
                {'content' : self.tokenizer.decode(ids[:allowed_context_lengths[i].item()], skip_special_tokens = True)}
                     for i, (rank, ids) in enumerate(retrieved_context_tokenized)
            ]
            
            text = self.tokenizer.apply_chat_template(
                conversation = [
                    {'role' : 'system', 'content' : self.system_prompt},
                    {'role' : 'user', 'content' : self.samples[idx][0]},
                    {'role' : 'assistant', 'content' : answer}
                ],
                documents = retrieved_documents,
                add_generation_prompt = False,
                tokenize = False
            ).strip()

            instruct_text = self.tokenizer.apply_chat_template(
                conversation = [
                    {'role' : 'system', 'content' : self.system_prompt},
                    {'role' : 'user', 'content' : self.samples[idx][0]},
                    {'role' : 'assistant', 'content' : ''}
                ],
                documents = retrieved_documents,
                add_generation_prompt = False,
                tokenize = False
            ).strip()

            target_text = text[len(instruct_text)-10:]

        # Tokenize text/template
            
        encoding = self.tokenizer(text, return_tensors = 'pt', add_special_tokens = False)
        input_ids = encoding['input_ids'].squeeze(0)

        # Get position to start loss calculation

        if self.dataset_type == 'CPT':

            labels = input_ids.clone()

        if self.dataset_type == 'SFT':
        
            mask_start = self.tokenizer(instruct_text, return_tensors = 'pt', add_special_tokens = False)['input_ids'].squeeze(0).size()[0]

            # Mask an input up to expected generated text
            
            labels = input_ids.clone()
            labels[:mask_start-1] = -100

        return input_ids, labels, instruct_text, target_text, self.train, includes_gold, random_num
        
    def retrieve_context(self, idx):

        includes_gold = np.random.rand() <= self.includes_gold_prob

        if includes_gold:

            random_num = self.num_retrieved - np.random.choice(list(range(self.num_retrieved)), p = self.relevant_prob_dist)
            retrieval_idx = [self.samples[idx][1][0][i] if i < random_num
                                 else self.context_dataset.index[np.random.randint(0, self.context_len)]
                                     for i in range(self.num_retrieved)]
            match_sim = [round(self.samples[idx][1][1][i], 3) if i < random_num 
                             else round(np.random.rand() * 0.05, 3)
                                 for i in range(self.num_retrieved)]
            
        else:

            random_num = self.num_retrieved - np.random.choice(list(range(self.num_retrieved + 1)), p = self.irrelevant_prob_dist)
            retrieval_idx = [self.samples[idx][1][0][1:][i] if i < random_num
                                 else self.context_dataset.index[np.random.randint(0, self.context_len)]
                                     for i in range(self.num_retrieved)]
            match_sim = [round(self.samples[idx][1][1][1:][i], 3) if i < random_num
                             else round(np.random.rand() * 0.05, 3)
                                 for i in range(self.num_retrieved)]
            
        content = list(zip(match_sim,
                           list(map(lambda i: self.context_dataset.loc[i][self.query_con_col] + '\n\n' \
                                        + self.context_dataset.loc[i][self.answer_con_col],
                                    retrieval_idx))))
        
        if includes_gold:
            comment = np.random.choice(self.relevant_comments)
        elif np.array(match_sim).max() >= 0.6:
            comment = np.random.choice(self.semi_relevant_comments)
        else:
            comment = np.random.choice(self.non_relevant_comments)
            
        return content, comment, includes_gold, random_num

    def batch_collate(self, batch):
        
        input_ids, labels, instruct_text, target_text, is_train, includes_gold, random_num = zip(*batch)
        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first = True, padding_value = self.tokenizer.pad_token_id)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first = True, padding_value = -100)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.long)
        
        return input_ids, attention_mask, labels, instruct_text, target_text, is_train, includes_gold, random_num
