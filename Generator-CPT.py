import os
import json
import gc
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

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import GenerationConfig

from huggingface_hub import snapshot_download
from huggingface_hub import login
from safetensors.torch import load_file

from LLMClasses import AdapterLlamaMLP, LoRALinear, LLMmanager
from GeneratorDataClasses import FineTuneDataset

###---------------Parameters---------------###

start = datetime.datetime.now()

# Define model configuration and import parameters

MULTI_NODE = False

PEFT = True
FROM_PRETRAINED = False
IS_DUMMY = False
DUMMY_PARAMETERS = {'hidden_size' : 2, 'intermediate_size' : 4, 'head_dim' : 8}
LOAD_8BIT = False
LOAD_FINETUNED = False
PARALLELIZE = False
USE_CACHE = False

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
MAX_CONTEXT_LENGTH = 16000
ROPE_THETA = 1000000
MAX_GENERATION_LENGTH = 1000

ADAPTER_DIM = 128
LORA_RANK = 16
LORA_ALPHA = 16

LORA_PARAMS = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
ORIGINAL_TRAINABLE_PARAMS = []
TRAINABLE_ADAPTER = True
TRAINABLE_LORA = False

# Data options

VALIDATION = False
DATASET_TYPE = 'CPT'

# Model import

LLM_PATH = Path.cwd().joinpath(*MODEL_DICT['model_path'])
LLM_PATH.mkdir(parents = True, exist_ok = True)
FINETUNED_MODEL_PATH = LLM_PATH.joinpath(DATASET_TYPE, f"Checkpoint_{MODEL_DICT['name']}")

# Learning rate hyperparameters

ATTN_BASE_LR = 5e-5 
FFN_BASE_LR = 5e-4
NORM_BASE_LR = 5e-5 * 0.1 
HEAD_BASE_LR = 5e-5 

FFN_GATE_LR = FFN_BASE_LR * 1.5
FFN_BIAS_LR = FFN_BASE_LR * 1

ATTN_LORA_A_LR = ATTN_BASE_LR
ATTN_LORA_B_LR = ATTN_BASE_LR * 0.5
ATTN_GATE_LR = ATTN_BASE_LR * 2
ATTN_BIAS_LR = ATTN_BASE_LR * 1

# Weight decay hyperparameters

WEIGHT_DECAY = 0.01
BIAS_DECAY = 0
NORM_DECAY = 0
LR_LAYER_DECAY = 0.95

# Batch size parameters

GLOBAL_BATCH_SIZE = 32
BATCH_SIZE = 1

# Scheduling and regularization hyperparameters

EPOCHS = 20
WARMUP = 0.05

LAMBDA_MAGNITUDE_DRIFT = 1e-4
MAGNITUDE_SHIFT_WARMUP = 0.05

# Loop hyper-parameters

VALIDATION_LOGGING_FACTOR = 1
CHECKPOINT_FREQUENCY_STEPS = 50

TOLERANCE = 0
PATIENCE = 2

GRADIENT_CHECKPOINTING = False

# Data files

CPT_FILENAME = 'sage_docs.json'
DATA_FOLDER = Path.cwd().joinpath('Data')

# Data preparation

with open(DATA_FOLDER.joinpath(CPT_FILENAME)) as f:
    sage_docs = json.load(f)
    
sage_docs = \
    [doc['info'] + '\n\n'
         + doc['metadata'].get('Keywords', '') + ('\n\n' if doc['metadata'].get('Keywords', '') else '')
         + doc['metadata'].get('Product', '') + ('\n\n' if doc['metadata'].get('Product', '') else '')
         + doc['title'] + '\n\n' 
         + doc['content'] 
             for doc in sage_docs]

# Import a tokenizer

with open(LLM_PATH.joinpath('generation_template.txt')) as gt:
    generation_template = gt.read()

tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, local_files_only = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.chat_template = generation_template

# Instantiate a dataset

if VALIDATION:

    validation_idx = np.random.choice(list(range(len(sage_docs))), int(len(sage_docs) * 0.1), replace = False)
    validation_mask = np.isin(list(range(len(sage_docs))), validation_idx)

    with open(DATA_FOLDER.joinpath('sage_docs_train.json'), 'w') as f:
        json.dump(np.array(sage_docs)[~validation_mask].tolist(), f, indent = 2)

    with open(DATA_FOLDER.joinpath('sage_docs_val.json'), 'w') as f:
        json.dump(np.array(sage_docs)[validation_mask].tolist(), f, indent = 2)

    dataset_train = FineTuneDataset(tokenizer, np.array(sage_docs)[~validation_mask].tolist(), DATASET_TYPE,
                                    max_context_length = MAX_CONTEXT_LENGTH, max_generation_length = MAX_GENERATION_LENGTH, train = True)

    dataset_val = FineTuneDataset(tokenizer, np.array(sage_docs)[validation_mask].tolist(), DATASET_TYPE,
                                  max_context_length = MAX_CONTEXT_LENGTH, max_generation_length = MAX_GENERATION_LENGTH, train = False)
    
else:

    dataset_train = FineTuneDataset(tokenizer, sage_docs, DATASET_TYPE,
                                    max_context_length = MAX_CONTEXT_LENGTH, max_generation_length = MAX_GENERATION_LENGTH, train = True)

print('Data was successfully imported - main process starts!')

###---------------Training functions---------------###

# Construction of parameter-specific learning rates and weight decays

def create_parameter_groups(model):

    parameter_optim_list = []
    num_layers = len(model.model.layers)
    
    for mname, module in model.named_modules():
    
        if len(mname.split('.')) >= 3 and 'layers' in mname.split('.'):
    
            layer_num = int(mname.split('.')[2]) + 1
            
            if isinstance(module, AdapterLlamaMLP):
                for pname, param in module.named_parameters():
                    if 'weight' in pname.split('.') and 'adapter' in pname.split('.'):
                        parameter_optim_list.append({'params' : param, 'weight_decay': WEIGHT_DECAY,
                                                         'lr': FFN_BASE_LR * (LR_LAYER_DECAY ** (num_layers - layer_num))})
                    elif 'weight' in pname.split('.') and 'adapter_gate' in pname.split('.'):
                        parameter_optim_list.append({'params' : param, 'weight_decay': WEIGHT_DECAY, 'lr': FFN_GATE_LR})
                    elif 'bias' in pname.split('.') and 'adapter' in pname.split('.'):
                        parameter_optim_list.append({'params' : param, 'weight_decay': BIAS_DECAY,
                                                         'lr': FFN_BIAS_LR * (LR_LAYER_DECAY ** (num_layers - layer_num))})
                    elif 'bias' in pname.split('.') and 'adapter_gate' in pname.split('.'):
                        parameter_optim_list.append({'params' : param, 'weight_decay': BIAS_DECAY, 'lr': FFN_BIAS_LR})
        
            if isinstance(module, LoRALinear):
                for pname, param in module.named_parameters():
                    if 'lora_weight_a' in pname.split('.'):
                        parameter_optim_list.append({'params' : param, 'weight_decay': WEIGHT_DECAY,
                                                         'lr': ATTN_LORA_A_LR * (LR_LAYER_DECAY ** (num_layers - layer_num))})
                    elif 'lora_weight_b' in pname.split('.'):
                        parameter_optim_list.append({'params' : param, 'weight_decay': WEIGHT_DECAY,
                                                         'lr': ATTN_LORA_B_LR * (LR_LAYER_DECAY ** (num_layers - layer_num))})
                    elif 'weight' in pname.split('.') and 'lora_gate' in pname.split('.'):
                        parameter_optim_list.append({'params' : param, 'weight_decay': WEIGHT_DECAY, 'lr': ATTN_GATE_LR})
                    elif 'bias' in pname.split('.') and 'lora_gate' in pname.split('.'):
                        parameter_optim_list.append({'params' : param, 'weight_decay': BIAS_DECAY, 'lr': ATTN_BIAS_LR})
                        
            if isinstance(module, model.model.norm.__class__):
                for pname, param in module.named_parameters():
                    parameter_optim_list.append({'params' : param, 'weight_decay': NORM_DECAY, 
                                                 'lr': NORM_BASE_LR * (LR_LAYER_DECAY ** (num_layers - layer_num))})
    
        if isinstance(module, model.model.norm.__class__) and 'layers' not in mname.split('.'):
            for pname, param in module.named_parameters():
                parameter_optim_list.append({'params' : param, 'weight_decay': NORM_DECAY, 'lr': NORM_BASE_LR})
                    
        if 'lm_head' in mname.split('.') and 'layers' not in mname.split('.'):
            for pname, param in module.named_parameters():
                parameter_optim_list.append({'params' : param, 'weight_decay': WEIGHT_DECAY, 'lr': HEAD_BASE_LR})
                
    return parameter_optim_list

# Learning rate scheduler instantiation

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles = 0.5, min_lambda_lr = 0.1):

    def lr_lambda(current_step):
        
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        return max(min_lambda_lr , 0.5 * (1.0 + torch.cos(torch.tensor(num_cycles * torch.pi * 2.0 * progress))).item())

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Define loss function

def llm_finetuning_loss_fn(logits, labels, lora_modules, lambda_magnitude_drift,
                               vocab_size, ignore_index = -100, label_smoothing = 0):
    
    cross_entropy = nn.CrossEntropyLoss(ignore_index = -100, reduction = 'mean', label_smoothing = label_smoothing) \
                        (logits.view(-1, vocab_size), labels.view(-1))
    magnitude_regularization = torch.stack([module.magnitude_drift_penalty() for module in lora_modules]).mean()
    
    return cross_entropy + lambda_magnitude_drift * magnitude_regularization

# Memory allocation tracking

def check_memory(device, size, label = '', reporting_threshold_1 = 0.2, reporting_threshold_2 = 0.7):

    available = torch.cuda.get_device_properties(device.index).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    max_alloc = torch.cuda.max_memory_allocated(device) / 1024 ** 3

    if (reserved / available) >= reporting_threshold_1:
        print(f"[{label}][rank{device}] allocated={allocated:.2f} GB, reserved={reserved:.2f} GB, peak={max_alloc:.2f} GB, tensor={size}")

    if (reserved / available) >= reporting_threshold_2:

        snapshot = torch.cuda.memory_snapshot()
        with open('snapshot.json', 'w') as sn, open('memory_summary.txt', 'w') as msu, open('memory_stats.txt', 'w') as mst:
            json.dump(snapshot, sn, indent = 2)
            msu.write(torch.cuda.memory_summary(device = device, abbreviated = False))
            json.dump(torch.cuda.memory_stats(device), mst, indent = 2)

        torch.cuda.empty_cache()
        gc.collect()

# Fine-Tuning Loop

def train(rank, world_size):

    # Initialize process group (if multi-GPU)

    global_rank = int(os.environ['SLURM_PROCID']) if MULTI_NODE else rank

    parallelized = False
    device = DEVICE

    if device == torch.device('cuda'):
        
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    
    if world_size > 1 and device == torch.device('cuda'):

        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        
        dist.init_process_group('nccl', rank = global_rank, world_size = world_size, device_id = device)
        parallelized = True

    # Initiate dynamically updated parameters

    best_score = np.inf
    patience_epochs = 0
    training_steps = 0
    steps = 0
    
    train_losses = []
    val_losses = []
    train_perplex = []
    val_perplex = []

    gradient_accumulation = \
        int(np.ceil(GLOBAL_BATCH_SIZE / (BATCH_SIZE * world_size)))

    num_training_steps = int(np.ceil((len(dataset_train) / GLOBAL_BATCH_SIZE) * EPOCHS))
    num_warmup_steps = int(np.ceil(num_training_steps * WARMUP))
        
    # Construct the model

    llm_utility = \
        LLMmanager(LLM_PATH, device, finetuned_model_path = FINETUNED_MODEL_PATH,
                   peft = PEFT, from_pretrained = FROM_PRETRAINED, load_8bit = LOAD_8BIT,
                   is_dummy = IS_DUMMY, dummy_parameters = DUMMY_PARAMETERS,
                   rope_scaling = ROPE_SCALING, rope_theta = ROPE_THETA, max_context_length = MAX_CONTEXT_LENGTH,
                   adapter_dim = ADAPTER_DIM, lora_rank = LORA_RANK, lora_alpha = LORA_ALPHA, lora_params = LORA_PARAMS,
                   original_trainable_params = ORIGINAL_TRAINABLE_PARAMS,
                   trainable_adapter = TRAINABLE_ADAPTER, trainable_lora = TRAINABLE_LORA,
                   use_cache = USE_CACHE)

    tokenizer, model, _ = llm_utility.load_model(load_finetuned = LOAD_FINETUNED, parallelize = PARALLELIZE)

    # Broadcast parameters from a source process
    
    if parallelized:
        
        dist.barrier()
        
        for param in model.parameters():
            dist.broadcast(param.data, src = 0)

        dist.barrier()

    # Set up dataloaders

    if parallelized:
        sampler_train = DistributedSampler(dataset_train, num_replicas = world_size, rank = global_rank, shuffle = True)
    else:
        sampler_train = None

    dataloader_train = DataLoader(dataset_train, batch_size = BATCH_SIZE, sampler = sampler_train, 
                                      shuffle = (sampler_train is None), collate_fn = dataset_train.batch_collate)

    if VALIDATION:

        if parallelized:
            sampler_val = DistributedSampler(dataset_val, num_replicas = world_size, rank = global_rank, shuffle = False)
        else:
            sampler_val = None
            
        dataloader_val = DataLoader(dataset_val, batch_size = BATCH_SIZE, sampler = sampler_val, 
                                        shuffle = (sampler_val is None), collate_fn = dataset_val.batch_collate)

    # Memory optimizations

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()

    # Wrap the model into DDP class and define optimizer and schedule

    if parallelized:
        model = DDP(model, device_ids = [rank], output_device = rank, gradient_as_bucket_view = True)
        model_module = model.module
    else:
        model_module = model

    optimizer = optim.AdamW(create_parameter_groups(model_module))

    scheduler = get_cosine_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps = num_warmup_steps, 
                    num_training_steps = num_training_steps
    )

    lora_modules = [module for name, module in model_module.named_modules() if isinstance(module, LoRALinear)]

    if global_rank == 0:
        print('The model was successfully imported!')
        #print('Trainable parameters:')
        #print([name for name, param in model_module.named_parameters() if param.requires_grad])

    # Training loop

    for epoch in range(EPOCHS):
    
        # Training

        if sampler_train is not None:
            sampler_train.set_epoch(epoch)
    
        epoch_train_loss = []
        model.train()

        accumulated_loss = torch.zeros(1, device = device) 
        
        for input_ids, attention_mask, labels, _, _, _ in tqdm(dataloader_train):
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
    
            logits = model(input_ids = input_ids, attention_mask = attention_mask).logits
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
    
            magnitude_schedule = min((training_steps + 1) / (MAGNITUDE_SHIFT_WARMUP * num_training_steps), 1)
            
            loss = llm_finetuning_loss_fn(shifted_logits, shifted_labels, lora_modules, magnitude_schedule * LAMBDA_MAGNITUDE_DRIFT,
                                             model_module.model.embed_tokens.num_embeddings, ignore_index = -100, label_smoothing = 0) \
                                                 / gradient_accumulation

            # Print memory usage statistics

            if device.type == 'cuda':
                check_memory(device, input_ids.size(), label = f'forward step {steps}')

            # Loss accumulation
            
            loss.backward(retain_graph = False)

            with torch.no_grad():
                accumulated_loss += loss.detach()

            # Optimization steps
    
            steps += 1
            if steps % gradient_accumulation == 0:
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none = True)

                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()

                with torch.no_grad():
                    if parallelized:
                        dist.all_reduce(accumulated_loss, op = dist.ReduceOp.SUM)
                        loss_avg = (accumulated_loss / dist.get_world_size()).item()
                    else:
                        loss_avg = accumulated_loss.item()
                
                epoch_train_loss.append(loss_avg)
                accumulated_loss = torch.zeros(1, device = device) 
                training_steps += 1

            # Intermediate checkpointing
    
            if training_steps % CHECKPOINT_FREQUENCY_STEPS == 0 and global_rank == 0:
    
                checkpoint_source = 'Ongoing intermediate checkpointing'
                log_list = \
                    [str(datetime.datetime.now()), dataset_train.dataset_type, training_steps, epoch + 1,
                     str(train_losses), str(val_losses), str(train_perplex), str(val_perplex), checkpoint_source, str(epoch_train_loss),
                     ATTN_BASE_LR, FFN_BASE_LR, NORM_BASE_LR, HEAD_BASE_LR, WEIGHT_DECAY, BIAS_DECAY, NORM_DECAY,
                     LR_LAYER_DECAY, WARMUP, EPOCHS, GLOBAL_BATCH_SIZE, LAMBDA_MAGNITUDE_DRIFT, MAGNITUDE_SHIFT_WARMUP]
                llm_utility.save_model_checkpoint(FINETUNED_MODEL_PATH, log_param_list = log_list, model = model_module, tokenizer = tokenizer)
    
        train_losses.append(torch.tensor(epoch_train_loss).mean().item())
        train_perplex.append(torch.exp(torch.tensor(epoch_train_loss).mean()).item())
        if global_rank == 0:
            print(f'Epoch {epoch + 1} train loss: {torch.tensor(epoch_train_loss).mean().item():.4f}')
    
        # Validation
    
        if (epoch + 1) % VALIDATION_LOGGING_FACTOR == 0 and VALIDATION:

            if sampler_val is not None:
                sampler_val.set_epoch(epoch)
                        
            epoch_val_loss = []
            model.eval()
    
            with torch.no_grad():
            
                for input_ids, attention_mask, labels, _, _, _ in dataloader_val:
                    
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
            
                    logits = model(input_ids = input_ids, attention_mask = attention_mask).logits
                    shifted_logits = logits[:, :-1, :].contiguous()
                    shifted_labels = labels[:, 1:].contiguous()
                    
                    loss = llm_finetuning_loss_fn(shifted_logits, shifted_labels, lora_modules, 0,
                                                    model_module.model.embed_tokens.num_embeddings,
                                                        ignore_index = -100, label_smoothing = 0.1)

                    loss_tensor = loss.detach()
                    if parallelized:
                        dist.all_reduce(loss_tensor, op = dist.ReduceOp.SUM)
                        loss_avg = (loss_tensor / dist.get_world_size()).item()
                    else:
                        loss_avg = loss_tensor.item()
                    epoch_val_loss.append(loss_avg)

                val_losses.append(torch.tensor(epoch_val_loss).mean().item())
                val_perplex.append(torch.exp(torch.tensor(epoch_val_loss).mean()).item())
                if global_rank == 0:
                    print(f'Epoch {epoch + 1} validation loss: {torch.tensor(epoch_val_loss).mean().item():.4f}')
    
        else:
            
            val_losses.append(np.nan)
            val_perplex.append(np.nan)
    
        # Early stopping
    
        if VALIDATION and global_rank == 0:

            # Post-validation checkpoint
    
            if not np.isnan(val_losses[-1]) and val_losses[-1] < best_score * (1 + TOLERANCE):
                best_score = val_losses[-1]
    
                checkpoint_source = 'Post validation improvement checkpointing'
                log_list = \
                    [str(datetime.datetime.now()), dataset_train.dataset_type, training_steps, epoch + 1,
                     str(train_losses), str(val_losses), str(train_perplex), str(val_perplex), checkpoint_source, str(epoch_train_loss),
                     ATTN_BASE_LR, FFN_BASE_LR, NORM_BASE_LR, HEAD_BASE_LR, WEIGHT_DECAY, BIAS_DECAY, NORM_DECAY,
                     LR_LAYER_DECAY, WARMUP, EPOCHS, GLOBAL_BATCH_SIZE, LAMBDA_MAGNITUDE_DRIFT, MAGNITUDE_SHIFT_WARMUP]
                llm_utility.save_model_checkpoint(FINETUNED_MODEL_PATH, log_param_list = log_list, model = model_module, tokenizer = tokenizer)
                patience_epochs = 0
                
            elif not np.isnan(val_losses[-1]) and val_losses[-1] >= best_score * (1 + TOLERANCE):
                patience_epochs += 1
                if patience_epochs >= PATIENCE:
                    print('Early stopping triggered!')
                    break

        # Final checkpoint
        
        elif (epoch + 1) == EPOCHS and global_rank == 0:
    
            checkpoint_source = 'Final training checkpoint'
            log_list = \
                [str(datetime.datetime.now()), dataset_train.dataset_type, training_steps, epoch + 1,
                 str(train_losses), str(val_losses), str(train_perplex), str(val_perplex), checkpoint_source, str(epoch_train_loss),
                 ATTN_BASE_LR, FFN_BASE_LR, NORM_BASE_LR, HEAD_BASE_LR, WEIGHT_DECAY, BIAS_DECAY, NORM_DECAY,
                 LR_LAYER_DECAY, WARMUP, EPOCHS, GLOBAL_BATCH_SIZE, LAMBDA_MAGNITUDE_DRIFT, MAGNITUDE_SHIFT_WARMUP]
            llm_utility.save_model_checkpoint(FINETUNED_MODEL_PATH, log_param_list = log_list, model = model_module, tokenizer = tokenizer)

    # Clean up processes
    
    if parallelized:
        dist.destroy_process_group()
        
def main():
    
    world_size = torch.cuda.device_count()

    if world_size > 1 and DEVICE == torch.device('cuda'):
        # Spawn processes, one per GPU
        mp.spawn(train, args = (world_size, ), nprocs = world_size, join = True)
        
    else:  
        # Single process training
        train(rank = 0, world_size = 1)

###---------------Run the actual job---------------###

if __name__ == '__main__':
    main()

finish = datetime.datetime.now()

print(round((finish - start).total_seconds() / 60**2, 3))
