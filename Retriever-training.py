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

from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoModel
from transformers import AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer

from huggingface_hub import snapshot_download
from huggingface_hub import login
from safetensors.torch import load_file

###---------------Data---------------###

# Data import

start = datetime.datetime.now()

filename = 'processed_tickets_df.json'
data_folder = Path.cwd().joinpath('Data')

processed_tickets_df = pd.read_json(data_folder.joinpath(filename), 
                                    orient = 'index', typ = 'frame', 
                                    dtype = str, precise_float = True)

processed_tickets_df.drop(columns = ['PROBLEM', 'SOLUTION'], inplace = True)
processed_tickets_df.rename(columns = {'STRUCTUREDPROBLEM' : 'PROBLEM', 'STRUCTUREDSOLUTION' : 'SOLUTION'}, inplace = True)

processed_tickets_df.head()

print('Data was succesfully downloaded!')

###---------------Model---------------###

# Define model configuration and import parameters

USE_GPU = True
DEVICE = torch.device('mps' if torch.mps.is_available() and USE_GPU else ('cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'))

MODEL_NUM = 0
MODEL_DICT = \
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

# Model import

llm_path = Path.cwd().joinpath(*MODEL_DICT[MODEL_NUM]['model_path'])
llm_path.mkdir(parents = True, exist_ok = True)

# Instantiate the model

llm_model = AutoModel.from_pretrained(llm_path, local_files_only = True, use_safetensors = True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only = True)

# Replace pooling logic

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

llm_model_q = llm_model
llm_model_a  = copy.deepcopy(llm_model)

llm_model_q.pooler = BertAttentionPooler(llm_model_q.config, DEVICE)
llm_model_a.pooler = BertAttentionPooler(llm_model_a.config, DEVICE)

print('Model was succesfully downloaded!')

###---------------Data Loader---------------###

# Declare a special dataset class for fine-tuning

class RetrievalDataset(Dataset):
    
    def __init__(self, tokenizer, dataset, query_col, answer_col, example_id_col, train = True):

        self.tokenizer = tokenizer
        self.train = train
        self.samples = list((db_id, query, answer) for db_id, query, answer in \
                                zip(dataset[example_id_col].tolist(), dataset[query_col].tolist(),
                                        dataset[answer_col].tolist()))

    def __len__(self):
        
        return len(self.samples)

    def __getitem__(self, idx):

        db_id, query, answer = self.samples[idx]

        # Tokenize text
            
        encoding_q = self.tokenizer(query, return_tensors = 'pt', add_special_tokens = False, truncation = True, max_length = 512)
        encoding_a = self.tokenizer(answer, return_tensors = 'pt', add_special_tokens = False, truncation = True, max_length = 512)
        
        input_ids_q = encoding_q['input_ids'].squeeze(0)
        input_ids_a = encoding_a['input_ids'].squeeze(0)

        return input_ids_q, input_ids_a, query, answer, self.train, db_id

    def batch_collate(self, batch):
        
        input_ids_q, input_ids_a, query, answer, is_train, db_id = zip(*batch)
        
        input_ids_q = nn.utils.rnn.pad_sequence(input_ids_q, batch_first = True, padding_value = self.tokenizer.pad_token_id)
        input_ids_a = nn.utils.rnn.pad_sequence(input_ids_a, batch_first = True, padding_value = self.tokenizer.pad_token_id)
        
        attention_mask_q = (input_ids_q != self.tokenizer.pad_token_id).to(torch.long)
        attention_mask_a = (input_ids_a != self.tokenizer.pad_token_id).to(torch.long)
        
        return input_ids_q, input_ids_a, attention_mask_q, attention_mask_a, query, answer, is_train, db_id

# Instantiate dataset and data loading

validation = False
validation_share = 0.1
batch_size = 32

if validation:

    validation_idx = np.random.choice(list(processed_tickets_df.index), int(len(processed_tickets_df.index) * validation_share))
    validation_mask = processed_tickets_df.index.isin(validation_idx)

    dataset_train = RetrievalDataset(tokenizer, processed_tickets_df[~validation_mask], 'PROBLEM', 'SOLUTION', 'TICKETID', train = True)
    dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, collate_fn = dataset_train.batch_collate)

    dataset_val = RetrievalDataset(tokenizer, processed_tickets_df[validation_mask], 'PROBLEM', 'SOLUTION', 'TICKETID', train = False)
    dataloader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle = True, collate_fn = dataset_val.batch_collate)
    
else:

    dataset_train = RetrievalDataset(tokenizer, processed_tickets_df, 'PROBLEM', 'SOLUTION', 'TICKETID', train = True)
    dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, collate_fn = dataset_train.batch_collate)

###---------------Fine-tuning Parameters and Functions---------------###

# Optimizer definition

# Learning rate hyperparameters

embed_base_lr_q = 1e-5
encoder_base_lr_q = 2e-5
head_base_lr_q = 5e-5

embed_base_lr_a = 2e-5
encoder_base_lr_a = 5e-5
head_base_lr_a = 1e-4

# Weight decay hyperparameters

weight_decay = 0.01
lr_layer_decay = 0.95

# Scheduling and regularization hyperparameters

epochs = 10
warmup = 0.05
num_training_steps = int(np.ceil((len(dataset_train) / batch_size) * epochs))
num_warmup_steps = int(np.ceil(num_training_steps * warmup))

# Temperature parameter

tau_sim = 0.07

# Construction of parameter-specific learning rates and weight decays

def prepare_training_params(model, embed_base_lr, encoder_base_lr, head_base_lr, weight_decay, lr_layer_decay):

    parameter_optim_list = []
    num_layers = len(model.encoder.layer)
    
    param_dict = {
        'embeddings' : [*model.embeddings.parameters()],
        'encoder' : [[*model.encoder.layer[i].parameters()] for i in range(num_layers)],
        'pooler' : [*model.pooler.parameters()]
    }
    
    for key, value in param_dict.items():
    
        if key == 'pooler':
            parameter_optim_list.append(
                {'params' : value,
                 'weight_decay': weight_decay,
                 'lr': head_base_lr
                })
        
        elif key == 'embeddings':
            parameter_optim_list.append(
                {'params' : value,
                 'weight_decay': weight_decay,
                 'lr': embed_base_lr
                })    
            
        elif key == 'encoder':
            for i in range(len(value)):
                parameter_optim_list.append(
                    {'params' : value[i],
                     'weight_decay': weight_decay,
                     'lr': encoder_base_lr * (lr_layer_decay ** (num_layers - (i + 1)))
                    })
                
    return parameter_optim_list

parameter_optim_list_q = prepare_training_params(llm_model_q, embed_base_lr_q, encoder_base_lr_q,
                                                     head_base_lr_q, weight_decay, lr_layer_decay)
parameter_optim_list_a = prepare_training_params(llm_model_a, embed_base_lr_a, encoder_base_lr_a,
                                                     head_base_lr_a, weight_decay, lr_layer_decay)

# Optimizer instantiation

encoder_optimizer_q = optim.AdamW(parameter_optim_list_q[0:-1])
pooler_optimizer_q = optim.AdamW([parameter_optim_list_q[-1]])

encoder_optimizer_a = optim.AdamW(parameter_optim_list_a[0:-1])
pooler_optimizer_a = optim.AdamW([parameter_optim_list_a[-1]])

# Learning rate scheduler instantiation

def get_schedule(optimizer, num_warmup_steps, num_training_steps, original,
                     num_cycles = 0.5, min_lambda_lr = 0.1, lr_scaling_power = 2):

    if original:

        def lr_lambda(current_step):

            current_val = current_step ** lr_scaling_power
            max_val = num_training_steps ** lr_scaling_power
            
            return max(0, current_val / max_val)

    else:

        def lr_lambda(current_step):
            
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            
            return max(min_lambda_lr , 0.5 * (1.0 + torch.cos(torch.tensor(num_cycles * torch.pi * 2.0 * progress))))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


scheduler_encoder_q = get_schedule(
    encoder_optimizer_q, 
    num_warmup_steps = num_warmup_steps, 
    num_training_steps = num_training_steps,
    original = True
)

scheduler_pooler_q = get_schedule(
    pooler_optimizer_q, 
    num_warmup_steps = num_warmup_steps, 
    num_training_steps = num_training_steps,
    original = False
)

scheduler_encoder_a = get_schedule(
    encoder_optimizer_a, 
    num_warmup_steps = num_warmup_steps, 
    num_training_steps = num_training_steps,
    original = True
)

scheduler_pooler_a = get_schedule(
    pooler_optimizer_a, 
    num_warmup_steps = num_warmup_steps, 
    num_training_steps = num_training_steps,
    original = False
)

# Define a loss function

def infoNCE_loss_fn(embed_q, embed_a, tau_sim):

    q_norm = nn.functional.normalize(embed_q, p = 2, dim = -1)
    a_norm = nn.functional.normalize(embed_a, p = 2, dim = -1)
    
    sim = torch.einsum('ae,be->ab', q_norm, a_norm) / tau_sim
    
    loss = -torch.diag(sim) + torch.logsumexp(sim, dim = 1)

    return loss.mean()

    
# Model checkpointing utility

trained_model_q_output_dir = llm_path.joinpath('q_encoder', f"Checkpoint_{MODEL_DICT[MODEL_NUM]['name']}")
trained_model_a_output_dir = llm_path.joinpath('a_encoder', f"Checkpoint_{MODEL_DICT[MODEL_NUM]['name']}")
txt_filename = 'fine_tuning_report.txt'

txt_template = \
"""
//Fine-tuning checkpoint//

Datetime: {0}
Last training step: {1}
Last epoch: {2}
Epoch train losses: {3}
Epoch validation losses: {4}
Checkpoint source: {5}

Last batch infoNCE losses: {6}

//Hyperparameters//

Query embedding base learning rate: {7}
Query encoder base learning rate: {8}
Query pooler base learning rate: {9}

Answer embedding base learning rate: {10}
Answer encoder base learning rate: {11}
Answer pooler base learning rate: {12}

Similarity loss temperature: {13}
Weight decay: {14}
Layer learning rate decay: {15}
LR Warmup: {16}
LR Sheduling: One cycle cosine annealing (Pooler) + Squared growth annealing (Embedding and Encoder)
Max number of epochs: {17}
Batch size: {18}
"""

def save_model_checkpoint(model_q: PreTrainedModel, model_a: PreTrainedModel,
                              tokenizer: PreTrainedTokenizer, output_dir_q: str, output_dir_a: str,
                                  txt_filename: str, txt_template: str, log_param_list : list):

    txt_log = txt_template.format(*log_param_list)
    
    os.makedirs(output_dir_q, exist_ok = True)
    os.makedirs(output_dir_a, exist_ok = True)
    
    model_q.save_pretrained(output_dir_q, safe_serialization = True)
    model_a.save_pretrained(output_dir_a, safe_serialization = True) 
    
    tokenizer.save_pretrained(output_dir_q)
    tokenizer.save_pretrained(output_dir_a)

    with open(output_dir_q.joinpath(txt_filename), 'w') as qf, \
             open(output_dir_a.joinpath(txt_filename), 'w') as af:
        qf.write(txt_log)
        af.write(txt_log)

###---------------Fine-tuning Loop---------------###

# Define the fine-tuning loop

# Hyper-parameters

validation_logging_factor = 1
gradient_accumulation = 1
checkpoint_frequency_steps = 100

tolerance = 0
patience = 2

best_score = np.inf
patience_epochs = 0
training_steps = 0

train_losses = []
val_losses = []

for epoch in range(epochs):

    # Training

    epoch_loss = []
    llm_model_q.train()
    llm_model_a.train()
    
    for input_ids_q, input_ids_a, attention_mask_q, attention_mask_a, _, _, _, _ in tqdm(dataloader_train):
        
        input_ids_q = input_ids_q.to(llm_model_q.device)
        attention_mask_q = attention_mask_q.to(llm_model_q.device)

        input_ids_a = input_ids_a.to(llm_model_a.device)
        attention_mask_a = attention_mask_a.to(llm_model_a.device)

        llm_model_q.pooler.attention_mask = attention_mask_q
        llm_model_a.pooler.attention_mask = attention_mask_a

        embed_q = llm_model_q(input_ids_q, attention_mask = attention_mask_q).pooler_output
        embed_a = llm_model_a(input_ids_a, attention_mask = attention_mask_a).pooler_output
        
        loss =  infoNCE_loss_fn(embed_q, embed_a, tau_sim) / gradient_accumulation
        loss_unnorm = loss.item() * gradient_accumulation

        loss.backward()

        training_steps += 1
        if training_steps % gradient_accumulation == 0:

            encoder_optimizer_q.step()
            pooler_optimizer_q.step()
            encoder_optimizer_a.step()
            pooler_optimizer_a.step()

            scheduler_encoder_q.step()
            scheduler_pooler_q.step()
            scheduler_encoder_a.step()
            scheduler_pooler_a.step()
            
            encoder_optimizer_q.zero_grad()
            pooler_optimizer_q.zero_grad()
            encoder_optimizer_a.zero_grad()
            pooler_optimizer_a.zero_grad()

        epoch_loss.append(loss_unnorm)

        if training_steps % checkpoint_frequency_steps == 0:

            checkpoint_source = 'Ongoing intermediate checkpointing'
            log_list = \
                [str(datetime.datetime.now()), training_steps, epoch + 1,
                 str(train_losses), str(val_losses), checkpoint_source, str(epoch_loss),
                 embed_base_lr_q, encoder_base_lr_q, head_base_lr_q, embed_base_lr_a, encoder_base_lr_a, head_base_lr_a,
                 tau_sim, weight_decay, lr_layer_decay, warmup, epochs, batch_size]
            save_model_checkpoint(llm_model_q, llm_model_a, tokenizer, trained_model_q_output_dir,
                                      trained_model_a_output_dir, txt_filename, txt_template, log_list)

    train_losses.append(torch.tensor(epoch_loss).mean().item())
    print(f'Epoch {epoch + 1} train loss: {torch.tensor(epoch_loss).mean().item():.4f}')

    # Validation

    if epoch % validation_logging_factor == 0 and validation::
    
        epoch_loss = []
        llm_model_q.eval()
        llm_model_a.eval()

        with torch.no_grad():
        
            for input_ids_q, input_ids_a, attention_mask_q, attention_mask_a, _, _, _, _ in dataloader_val:
                
                input_ids_q = input_ids_q.to(llm_model_q.device)
                attention_mask_q = attention_mask_q.to(llm_model_q.device)
        
                input_ids_a = input_ids_a.to(llm_model_a.device)
                attention_mask_a = attention_mask_a.to(llm_model_a.device)

                llm_model_q.pooler.attention_mask = attention_mask_q
                llm_model_a.pooler.attention_mask = attention_mask_a
        
                embed_q = llm_model_q(input_ids_q, attention_mask = attention_mask_q).pooler_output
                embed_a = llm_model_a(input_ids_a, attention_mask = attention_mask_a).pooler_output
                
                loss =  infoNCE_loss_fn(embed_q, embed_a, tau_sim)
        
                epoch_loss.append(loss.item())
        
            val_losses.append(torch.tensor(epoch_loss).mean().item())
            print(f'Epoch {epoch + 1} validation loss: {torch.tensor(epoch_loss).mean().item():.4f}')

    else:

        val_losses.append(np.nan)

    # Early stopping

    if validation:

        if not np.isnan(val_losses[-1]) and val_losses[-1] < best_score * (1 + tolerance):
            best_score = val_losses[-1]

            checkpoint_source = 'Post validation improvement checkpointing'
            log_list = \
                [str(datetime.datetime.now()), training_steps, epoch + 1,
                 str(train_losses), str(val_losses), checkpoint_source, str(epoch_loss),
                 embed_base_lr_q, encoder_base_lr_q, head_base_lr_q, embed_base_lr_a, encoder_base_lr_a, head_base_lr_a,
                 tau_sim, weight_decay, lr_layer_decay, warmup, epochs, batch_size]
            save_model_checkpoint(llm_model_q, llm_model_a, tokenizer, trained_model_q_output_dir,
                                      trained_model_a_output_dir, txt_filename, txt_template, log_list)
            patience_epochs = 0
            
        elif not np.isnan(val_losses[-1]) and val_losses[-1] >= best_score * (1 + tolerance):
            patience_epochs += 1
            if patience_epochs >= patience:
                print('Early stopping triggered!')
                break

    elif (epoch + 1) == epochs:

        checkpoint_source = 'Final training checkpoint'
        log_list = \
            [str(datetime.datetime.now()), training_steps, epoch + 1,
             str(train_losses), str(val_losses), checkpoint_source, str(epoch_loss),
             embed_base_lr_q, encoder_base_lr_q, head_base_lr_q, embed_base_lr_a, encoder_base_lr_a, head_base_lr_a,
             tau_sim, weight_decay, lr_layer_decay, warmup, epochs, batch_size]
        save_model_checkpoint(llm_model_q, llm_model_a, tokenizer, trained_model_q_output_dir,
                                  trained_model_a_output_dir, txt_filename, txt_template, log_list)

finish = datetime.datetime.now()

print(round((finish - start).total_seconds() / 60**2, 3))