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

from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import GenerationConfig

import evaluate
import nltk
from bert_score import score as bert_score

from LLMClasses import AdapterLlamaMLP, LoRALinear, LLMmanager
from GeneratorDataClasses import EvaluationDataset

nltk.download('wordnet')
nltk.download('omw-1.4')

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
DEVICE = torch.device('mps' if torch.mps.is_available() and USE_GPU else ('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'))

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
MAX_GENERATION_LENGTH = 1000

# Data options

DATASET_TYPE = 'SFT'
NUM_RETRIEVED = 5
INCLUDES_GOLD_PROB = 0.6
RELEVANT_PROB_DIST = [0.5, 0.2, 0.1, 0.1, 0.1]
IRRELEVANT_PROB_DIST = [0.7, 0.1, 0.05, 0.05, 0.05, 0.05]

# Model import

LLM_PATH = Path.cwd().joinpath(*MODEL_DICT['model_path'])
FINETUNED_MODEL_PATH = LLM_PATH.joinpath('SFT', f"Checkpoint_{MODEL_DICT['name']}")

# Data parameters

BATCH_SIZE = 4

EVALUATION_STAT_FILENAME = 'original_evaluation_stats.json'
EVALUATION_DF_FILENAME = 'original_evaluation_df.json'

TRAIN_FILENAME = 'llm_processed_tickets_df_train.json'
VAL_FILENAME = 'llm_processed_tickets_df_val.json'

CONTEXT_FILENAME = 'embedded_context_df.json'
SUMMARY_FILENAME = 'llm_summarized_tickets_df.json'
DATA_DIRECTORY = Path.cwd().joinpath('Data')

context_embedding_cols = ['Query embedding', 'Answer embedding', 'context_sims']
contex_dtypes = {'Query' : str, 'Answer' : str,
                 'Query embedding' : list, 'Answer embedding' : list,
                 'context_ids' : list, 'context_sims' : list}

# Import context dataset

context_tickets_df = pd.read_json(DATA_DIRECTORY.joinpath(CONTEXT_FILENAME), 
                                  orient = 'index', typ = 'frame', 
                                  dtype = contex_dtypes, precise_float = True)

for col in context_embedding_cols:
    context_tickets_df[col] = context_tickets_df[col].apply(lambda x: np.array(x, dtype = np.float32))

# Import SFT dataset

processed_tickets_df_train = pd.read_json(DATA_DIRECTORY.joinpath(TRAIN_FILENAME), 
                                            orient = 'index', typ = 'frame', 
                                            dtype = str, precise_float = True)

processed_tickets_df_val = pd.read_json(DATA_DIRECTORY.joinpath(VAL_FILENAME), 
                                            orient = 'index', typ = 'frame', 
                                            dtype = str, precise_float = True)

# Import and integrate ticket history summary dataset

summarized_tickets_df = pd.read_json(DATA_DIRECTORY.joinpath(SUMMARY_FILENAME), 
                                     orient = 'index', typ = 'frame', 
                                     dtype = str, precise_float = True)

summarized_tickets_df.drop(columns = ['PROBLEM', 'SOLUTION', 'STRUCTUREDPROBLEM', 'STRUCTUREDSOLUTION'], inplace = True)
summarized_tickets_df.set_index('TICKETID', drop = True, inplace = True)

context_tickets_df = context_tickets_df.join(summarized_tickets_df)
context_tickets_df['Summarized Answer'] = \
    context_tickets_df['Answer'].str.split('\n\nTicket status history:\n\n').str[0] \
        + '\n\nActivities description:\n\n' \
        + context_tickets_df['SUMMARIZEDSOLUTION']

context_tickets_df.sort_index(inplace = True)

# Import and  prepare the LLM for fine-tuning

llm_utility = \
    LLMmanager(LLM_PATH, DEVICE, finetuned_model_path = FINETUNED_MODEL_PATH, use_cache = USE_CACHE,
               peft = PEFT, from_pretrained = FROM_PRETRAINED, load_8bit = LOAD_8BIT,
               is_dummy = IS_DUMMY, dummy_parameters = DUMMY_PARAMETERS,
               rope_scaling = ROPE_SCALING, rope_theta = ROPE_THETA, max_context_length = MAX_CONTEXT_LENGTH)

tokenizer, llm_model, gen_config = llm_utility.load_model(load_finetuned = LOAD_FINETUNED, parallelize = PARALLELIZE)

with open(LLM_PATH.joinpath('system_prompt.txt')) as sp, \
    open(LLM_PATH.joinpath('generation_template.txt')) as gt:
        
    system_prompt = sp.read()
    generation_template = gt.read()

tokenizer.chat_template = generation_template
tokenizer.padding_side = 'left'

# Instantiate a dataset

dataset_train = EvaluationDataset(tokenizer, processed_tickets_df_train, DATASET_TYPE,
                                  target_col = 'SOLUTION', system_prompt = system_prompt, query_col = 'PROBLEM',
                                  context_dataset = context_tickets_df, query_con_col = 'Query', id_col = 'TICKETID',
                                  answer_con_col = 'Summarized Answer', query_match_col = 'context_ids', sim_match_col = 'context_sims',
                                  num_retrieved = NUM_RETRIEVED, includes_gold_prob = INCLUDES_GOLD_PROB,
                                  relevant_prob_dist = RELEVANT_PROB_DIST, irrelevant_prob_dist = IRRELEVANT_PROB_DIST,
                                  max_context_length = MAX_CONTEXT_LENGTH, max_generation_length = MAX_GENERATION_LENGTH, train = True)


if VAL_FILENAME is not None:

    dataset_val = EvaluationDataset(tokenizer, processed_tickets_df_val, DATASET_TYPE,
                                    target_col = 'SOLUTION', system_prompt = system_prompt, query_col = 'PROBLEM',
                                    context_dataset = context_tickets_df, query_con_col = 'Query', id_col = 'TICKETID',
                                    answer_con_col = 'Summarized Answer', query_match_col = 'context_ids', sim_match_col = 'context_sims',
                                    num_retrieved = NUM_RETRIEVED, includes_gold_prob = INCLUDES_GOLD_PROB,
                                    relevant_prob_dist = RELEVANT_PROB_DIST, irrelevant_prob_dist = IRRELEVANT_PROB_DIST,
                                    max_context_length = MAX_CONTEXT_LENGTH, max_generation_length = MAX_GENERATION_LENGTH, train = False)

else:

    dataset_val = None
    
print('Data and the model are successfully downloaded! Starting the main process.')

# Post training evaluation functions

def distinct_n_corpus(corpus, n):

    all_ngrams = []
    total_ngrams = 0
    
    for sentence in corpus:
        tokens = sentence.split()
        total_ngrams += max(len(tokens) - n + 1, 0)
        all_ngrams.extend(list(zip(*[tokens[i:] for i in range(n)])))
        
    if total_ngrams == 0:
        return 0.0
        
    return len(set(all_ngrams)) / total_ngrams

def evaluate_text_generation(model, tokenizer, dataset_train, gen_config, dataset_val = None, batch_size = 4):

    setattr(gen_config, 'do_sample', False)
    orig_model = model.module if hasattr(model, 'module') else model
    
    ref_nlls = []
    preds, refs, is_train_list, input_list = [], [], [], []
    includes_gold_list, number_of_random_examples_list = [], []
    gen_tokens_length_list, prompt_tokens_length_list = [], []

    if dataset_val is None:
        dataloader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, collate_fn = dataset_train.batch_collate)
    else:
        dataloader = DataLoader(ConcatDataset([dataset_train, dataset_val]), batch_size = batch_size,
                                    shuffle = True, collate_fn = dataset_train.batch_collate)

    bleu_metric = evaluate.load('bleu')
    rouge_metric = evaluate.load('rouge')
    meteor_metric = evaluate.load('meteor')

    orig_model.eval()

    with torch.no_grad():
        
        for input_ids, attention_mask, labels, input_texts, target_texts, is_train, includes_gold, random_num in tqdm(dataloader):
                
            input_ids = input_ids.to(orig_model.device)
            labels = labels.to(orig_model.device)
    
            logits = orig_model(input_ids = input_ids, attention_mask = attention_mask).logits
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()

            vocab_size = shifted_logits.size(-1)
            
            losses_flat = nn.functional.cross_entropy(
                shifted_logits.view(-1, vocab_size),
                shifted_labels.view(-1),
                reduction = 'none',
                ignore_index = -100)

            token_losses = losses_flat.view(input_ids.size(0), -1)
            mask = (shifted_labels != -100).float()
            tokens_per_example = mask.sum(dim = 1).clamp(min = 1.0)

            nll_per_example = (token_losses * mask).sum(dim = 1) / tokens_per_example
            ref_nlls.extend(nll_per_example.cpu().tolist())

            # Generate predictions

            inputs = tokenizer(input_texts, return_tensors = 'pt', padding_side = 'left',
	    		      	 padding = True).to(orig_model.device)

            gen_tokens = orig_model.generate(inputs['input_ids'], attention_mask = inputs['attention_mask'],
                                            pad_token_id = tokenizer.pad_token_id, generation_config = gen_config,
                                               return_dict_in_generate = True, output_scores = False)

            prompt_lengths = inputs['attention_mask'].sum(dim = 1).long().cpu().tolist()

            sequences = gen_tokens.sequences 
            gen_tokens_list = []
            
            for i in range(sequences.size(0)):
                
                seq = sequences[i]
                p_len = prompt_lengths[i]
                gen = seq[inputs['input_ids'].size(1):]
                gen_list = gen.cpu().tolist()
                gen_len = 0
                
                for token in gen_list:
                    if token == tokenizer.eos_token_id or token == tokenizer.pad_token_id:
                        break
                    gen_len += 1

                prompt_tokens_length_list.append(p_len)
                gen_tokens_length_list.append(gen_len)
                    
                if gen_len == 0:
                    gen_tokens_list.append(torch.tensor([], dtype = torch.long))
                else:
                    gen_tokens_list.append(gen[:gen_len].cpu())

            pred_texts = [tokenizer.decode(gen.cpu().tolist(), skip_special_tokens = True) for gen in gen_tokens_list]

            input_list.extend(input_texts)
            preds.extend(pred_texts)
            refs.extend(target_texts)
            is_train_list.extend(is_train)
            includes_gold_list.extend(includes_gold)
            number_of_random_examples_list.extend(random_num)

    # BLEU, ROUGE, METEOR

    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    meteor_scores = []
    
    for pred, ref in zip(preds, refs):
        
        bleu = bleu_metric.compute(predictions = [pred], references=[[ref]])['bleu']
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rouge = rouge_metric.compute(predictions = [pred], references = [ref], use_aggregator = True)
            rouge1 = rouge.get('rouge1', 0.0)
            rouge2 = rouge.get('rouge2', 0.0) 
            rougeL = rouge.get('rougeL', 0.0)   
            
        meteor = meteor_metric.compute(predictions = [pred], references = [ref])['meteor']
        
        bleu_scores.append(bleu)
        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
        rougeL_scores.append(rougeL)
        meteor_scores.append(meteor)

    # BERTScore
    
    P, R, F1 = bert_score(preds, refs, lang = 'en', model_type = 'roberta-large',
                              rescale_with_baseline = True)

    bert_P = P.cpu().tolist()
    bert_R = R.cpu().tolist()
    bert_F1 = F1.cpu().tolist()

    # Diversity
    
    distinct1 = distinct_n_corpus([preds[i] for i in range(len(preds))], 1)
    distinct2 = distinct_n_corpus([preds[i] for i in range(len(preds))], 2)

    distinct1_gold = distinct_n_corpus([preds[i] for i in range(len(preds)) if includes_gold_list[i]], 1)
    distinct2_gold = distinct_n_corpus([preds[i] for i in range(len(preds)) if includes_gold_list[i]], 2)

    distinct1_related = distinct_n_corpus([preds[i] for i in range(len(preds)) if not includes_gold_list[i]], 1)
    distinct2_related = distinct_n_corpus([preds[i] for i in range(len(preds)) if not includes_gold_list[i]], 2)

    if dataset_val is not None:

        distinct1_train = distinct_n_corpus([preds[i] for i in range(len(preds)) if is_train_list[i]], 1)
        distinct2_train = distinct_n_corpus([preds[i] for i in range(len(preds)) if is_train_list[i]], 2)

        distinct1_val = distinct_n_corpus([preds[i] for i in range(len(preds)) if not is_train_list[i]], 1)
        distinct2_val = distinct_n_corpus([preds[i] for i in range(len(preds)) if not is_train_list[i]], 2)

    df = pd.DataFrame({
        'input_text': input_list,
        'prediction': preds,
        'target_text': refs,
        'is_train': is_train_list,
        'includes_gold': includes_gold_list,
        'n_random_examples': number_of_random_examples_list,
        'prompth_length' : prompt_tokens_length_list,
        'gen_length' : gen_tokens_length_list,
        'CEloss': ref_nlls,
        'Perplexity' : list(np.exp(np.clip(np.array(ref_nlls), -100, 100))),
        'BLEU': bleu_scores,
        'Rouge1': rouge1_scores,
        'Rouge2': rouge2_scores,
        'RougeL': rougeL_scores,
        'METEOR': meteor_scores,
        'bertscore_precision': bert_P,
        'bertscore_recall': bert_R,
        'bertscore_f1': bert_F1,
    })

    stat_list = ['CEloss', 'Perplexity', 'BLEU', 'Rouge1', 'Rouge2', 'RougeL', 'METEOR',
                 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']
    stat_dict = {}

    for col in stat_list:

        if dataset_val is not None:
            stat_dict[col] = {
                'Train' : df.loc[np.array(is_train_list), col].mean(),
                'Val' : df.loc[~np.array(is_train_list), col].mean(),
                'Gold' : df.loc[np.array(includes_gold_list), col].mean(),
                'Related' : df.loc[~np.array(includes_gold_list), col].mean(),
                'Overall' : df.loc[:, col].mean()
            }
        else:
            stat_dict[col] = {'Overall' : df.loc[:, col].mean(),}

    if dataset_val is not None:
        
        stat_dict['Distinct1'] = {
                    'Train' : distinct1_train,
                    'Val' : distinct1_val,
                    'Gold' : distinct1_gold,
                    'Related' : distinct1_related,
                    'Overall' : distinct1
        }

        stat_dict['Distinct2'] = {
            'Train' : distinct2_train,
            'Val' : distinct2_val,
            'Gold' : distinct2_gold,
            'Related' : distinct2_related,
            'Overall' : distinct2
            
        }
        
    else:
        
        stat_dict['Distinct1'] = {'Overall' : distinct1,
                                  'Gold' : distinct1_gold,
                                  'Related' : distinct1_related}
        stat_dict['Distinct2'] = {'Overall' : distinct2,
                                  'Gold' : distinct2_gold,
                                  'Related' : distinct2_related}
    
    
    return stat_dict, df

# Execute evaluation function and save results

evaluation_stats, evaluation_df = evaluate_text_generation(llm_model, tokenizer, dataset_train, gen_config,
                                                           dataset_val = dataset_val, batch_size = BATCH_SIZE)

with open(DATA_DIRECTORY.joinpath(EVALUATION_STAT_FILENAME), 'w', encoding = 'utf-8') as f:
    json.dump(evaluation_stats, f, ensure_ascii = False, indent = True)

evaluation_df.to_json(DATA_DIRECTORY.joinpath(EVALUATION_DF_FILENAME),
                                 orient = 'index', double_precision = 15, index = True)

finish = datetime.datetime.now()

print(round((finish - start).total_seconds() / 60**2, 3))
