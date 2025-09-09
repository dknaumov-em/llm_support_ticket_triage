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

from safetensors.torch import load_file

###---------------LLM Classes---------------###

# Define Adapter Class

class AdapterLlamaMLP(nn.Module):
    
    def __init__(self, original_module, adapter_dim):
        
        # Original MLP attributes
        
        super().__init__()
        self.config = original_module.config
        self.hidden_size = original_module.hidden_size
        self.intermediate_size = original_module.intermediate_size
        
        self.gate_proj = copy.deepcopy(original_module.gate_proj)
        self.up_proj = copy.deepcopy(original_module.up_proj)
        self.down_proj = copy.deepcopy(original_module.down_proj)
        self.act_fn = original_module.act_fn

        # Added adapter attributes

        self.adapter_dim = adapter_dim
        self.adapter = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.up_proj.in_features, self.adapter_dim, bias = True, 
                              device = self.up_proj.weight.device, dtype = self.up_proj.weight.dtype)),
            ('act_fn', nn.SiLU()),
            ('dropout', nn.Dropout(p = 0.1)),
            ('fc2', nn.Linear(self.adapter_dim, self.down_proj.out_features, bias = True,
                              device = self.down_proj.weight.device, dtype = self.up_proj.weight.dtype))
        ]))
        self.adapter_gate = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.hidden_size, 1, bias = True, device = self.up_proj.weight.device, dtype = self.up_proj.weight.dtype)),
            ('act_fn', nn.Sigmoid())
        ]))

        # Initialize new adapter weights

        for param in self.adapter_gate.parameters():
            nn.init.constant_(param, 0.0)

        for name, param in self.adapter.named_parameters():
            if 'fc2' in name.split('.'):
                nn.init.normal_(param, mean = 0.0, std = 0.003)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        adapt_proj = self.adapter(x) * self.adapter_gate(x)
        
        return down_proj + adapt_proj

    # Special function for initiating adapter injection

    @staticmethod
    def patch_mlp(model, adapter_dim):
        
        for i in range(len(model.model.layers)):
            model.model.layers[i].mlp = AdapterLlamaMLP(
                model.model.layers[i].mlp, adapter_dim = adapter_dim)
  

# Define LoRA adapter class

class LoRALinear(nn.Linear):
    
    def __init__(self, original_module: nn.Linear, rank: int = 16, alpha: float = 16):

        # Transfer original attributes
        
        super().__init__(original_module.in_features, original_module.out_features, bias = original_module.bias is not None,
                             device = original_module.weight.device, dtype = original_module.weight.dtype)

        self.weight = copy.deepcopy(original_module.weight)
        if original_module.bias is not None:
            self.bias = copy.deepcopy(original_module.bias)

        self.weight.requires_grad = original_module.weight.requires_grad
        if original_module.bias is not None:
            self.bias.requires_grad = original_module.bias.requires_grad

        self.original_weight_norm = self.weight.norm(p = 'fro')

        # Add LoRA custom attributes
        
        self.lora_rank = rank
        self.lora_alpha = alpha
        self.lora_scale = alpha / rank
        self.lora_weight_a = nn.Parameter(torch.zeros(self.out_features, self.lora_rank,
            device = self.weight.device, dtype = self.weight.dtype), requires_grad = True)
        self.lora_weight_b = nn.Parameter(torch.normal(mean = 0, std = 0.003, size = (self.lora_rank, self.in_features),
            device = self.weight.device, dtype = self.weight.dtype), requires_grad = True)
        self.lora_gate = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.in_features, 1, bias = True, device = self.weight.device, dtype = self.weight.dtype)),
            ('act_fn', nn.Sigmoid())
        ]))
 
        # Initialize gate 

        for param in self.lora_gate.parameters():
            nn.init.constant_(param, 0.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        return nn.functional.linear(input, self.weight + self.lora_gate(input.mean(dim = (0, 1))) \
                                    * self.lora_scale * torch.matmul(self.lora_weight_a, self.lora_weight_b), self.bias)

    def magnitude_drift_penalty(self) -> torch.Tensor:
        
        updated_weight = self.weight + torch.matmul(self.lora_weight_a, self.lora_weight_b)
        
        return (updated_weight.norm(p = 'fro') - self.original_weight_norm) ** 2

    @staticmethod
    def patch_attention(model, rank, alpha, params):
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(key in name.split('.')[-2:] for key in params):
                parent = dict(model.named_modules())[name.rsplit('.', 1)[0]]
                attr = name.split('.')[-1]
                setattr(parent, attr, LoRALinear(module, rank = rank, alpha = alpha))

# Define a single class for proper model loading

class LLMmanager:
    
    def __init__(self, model_path, device, peft = True, load_8bit = False, finetuned_model_path = None,
                     from_pretrained = False, is_dummy = False, dummy_parameters = {}, use_cache = None,
                         rope_scaling = None, rope_theta = None, max_context_length = None,
                             adapter_dim = None, lora_rank = None, lora_alpha = None, lora_params = None,
                                 original_trainable_params = None, trainable_adapter = None, trainable_lora = None):

        # Loading parameters 
        
        self.model_path = model_path
        self.finetuned_model_path = finetuned_model_path
        self.device = torch.device(device)
        self.load_8bit = load_8bit
        self.from_pretrained = from_pretrained
        self.peft = peft
        self.use_cache = use_cache

        # Dummy model control
        
        self.is_dummy = is_dummy
        self.dummy_parameters = dummy_parameters
        self.strict_loading = not self.is_dummy

        # Context length parameters

        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.max_context_length = max_context_length

        # Model files
        
        self.tokenizer = None
        self.model = None
        self.gen_config = None
        self.checkpoint_report_template = None

        # Internal configs

        self.config = None
        self.finetuning_config = {
            'adapter_dim' : 128 if adapter_dim is None else adapter_dim, 
            'lora_rank' : 16 if lora_rank is None else lora_rank,
            'lora_alpha' : 16 if lora_alpha is None else lora_alpha, 
            'lora_params' : ['q_proj', 'k_proj', 'v_proj', 'o_proj'] if lora_params is None else lora_params,
            'original_trainable_params' :  [] if original_trainable_params is None else original_trainable_params,
            'trainable_adapter' : True if trainable_adapter is None else trainable_adapter,
            'trainable_lora' : True if trainable_lora is None else trainable_lora
        }

        # Checkpoint parameters
        
        self.checkpoint_report_filename = 'finetuning_report.txt'
        self.finetuning_config_filename = 'finetuning_config.json'

    # Main loading function 
    
    def load_model(self, load_finetuned: bool = False, parallelize: bool = False) \
                        -> tuple[PreTrainedTokenizer, PreTrainedModel, GenerationConfig]:

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only = True)
    
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
        self.gen_config = GenerationConfig.from_pretrained(self.model_path)
    
        if type(self.max_context_length) == int:
            setattr(self.gen_config, 'max_length', self.max_context_length)

        if 'checkpoint_report_template.txt' in os.listdir(self.model_path):
            with open(self.model_path.joinpath('checkpoint_report_template.txt'), 'r') as file:
                self.checkpoint_report_template = file.read()
    
        if self.from_pretrained and not self.is_dummy:
        
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map = 'auto' if self.device == torch.device('cuda') else None,
                torch_dtype = torch.bfloat16 if self.device == torch.device('cuda') else torch.float32,
                load_in_8bit = self.load_8bit,
                local_files_only = True,
                use_safetensors = True,
                rope_scaling = self.rope_scaling
            ).to(self.device)

        else:

            # Pulling out the config
    
            self.config = AutoConfig.from_pretrained(self.model_path, local_files_only = True, use_safetensors = True)
            
            # Optional config editing for a dummy model
            
            if self.is_dummy:
                for name, value in self.dummy_parameters.items():
                    setattr(self.config, name, value)
    
            if type(self.rope_scaling) == dict:
                setattr(self.config, 'rope_scaling', self.rope_scaling)
    
            if type(self.max_context_length) == int:
                setattr(self.config, 'max_position_embeddings', self.max_context_length)
    
            if self.rope_theta is not None:
                setattr(self.config, 'rope_theta', self.rope_theta)

            if self.use_cache is not None:
                setattr(self.config, 'use_cache', self.use_cache)
            
            # Instantiate the model
            
            self.model = AutoModelForCausalLM.from_config(self.config).to(self.device)
            
            # Load parameters
    
            if (not load_finetuned) or (self.finetuned_model_path is None):
                self.load_safetensors()

        # Fine-tuning preparation

        if self.peft:
            
            if load_finetuned and self.finetuned_model_path is not None:
                with open(self.finetuned_model_path.joinpath('finetuning_config.json'), 'r') as file:
                    self.finetuning_config.update(json.load(file))
                    
            self.finetune_prepare(load_weights = load_finetuned)

        if parallelize and torch.cuda.device_count() > 1 and self.device == torch.device('cuda'):
            self.model = self.model.to(self.device)
            self.model = nn.DataParallel(self.model)

        return self.tokenizer, self.model, self.gen_config 
                
    # Safetensor loading function 
    
    def load_safetensors(self, model_path = None, model = None):

        if model_path is None:
            model_path = self.model_path

        if model is None:
            model = self.model

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
            model.load_state_dict(full_state_dict, strict = self.strict_loading)

    # Function for fine-tuning preparation
    
    def finetune_prepare(self, model = None, load_weights = False):

        if model is None:
            model = self.model
    
        # Freeze original parameters
    
        for param in model.parameters():
            param.requires_grad = False
    
        # Inject LoRA modules
    
        LoRALinear.patch_attention(model, self.finetuning_config['lora_rank'], self.finetuning_config['lora_alpha'],
                                       self.finetuning_config['lora_params'])
    
        # Inject Adapters
    
        AdapterLlamaMLP.patch_mlp(model, self.finetuning_config['adapter_dim'])
    
        if load_weights and self.finetuned_model_path is not None:
            self.load_safetensors(model_path = self.finetuned_model_path)
    
        # Unfreeze selected modules

        trainable_params = []
        trainable_params.extend(self.finetuning_config['original_trainable_params'])
        if self.finetuning_config['trainable_adapter']:
            trainable_params.extend(self.finetuning_config.get('adapter_param_names',
                [name for name, param in model.named_parameters() if 'adapter' in name]))
        if self.finetuning_config['trainable_lora']:
            trainable_params.extend(self.finetuning_config.get('lora_param_names',
                [name for name, param in model.named_parameters() if 'lora' in name]))
    
        for name, param in model.named_parameters():
            if any([pname in name for pname in trainable_params]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Utility for saving checkpoints

    def save_model_checkpoint(self, output_dir, log_param_list: list = None,
                                  model: PreTrainedModel = None, tokenizer: PreTrainedTokenizer = None):

        if model is None:
            model = self.model
            
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        # Collect model parameters
        
        adapter_dim = None
        lora_rank = None
        lora_alpha = None
        adapter_param_names = []
        lora_param_names = []
        
        is_adapter = False
        is_lora = False
        
        for mname, module in model.named_modules():
                
            if isinstance(module, AdapterLlamaMLP):
        
                if not is_adapter:
        
                    adapter_dim = module.adapter_dim
                    is_adapter = True
                
                for pname, param in module.named_parameters():
                    if any([(keyword in pname.split('.')) for keyword in ['adapter', 'adapter_gate']]):
                        adapter_param_names.append('.'.join([mname, pname]))
        
            if isinstance(module, LoRALinear):
        
                if not is_lora:
        
                    lora_rank = module.lora_rank
                    lora_alpha = module.lora_alpha
                    is_lora = True
                
                for pname, param in module.named_parameters():
                    if any([(keyword in pname.split('.')) for keyword in ['lora_weight_a', 'lora_weight_b', 'lora_gate']]):
                        lora_param_names.append('.'.join([mname, pname]))
    
        lora_params = list(set([name.split('.')[4] for name in lora_param_names]))
    
        finetuning_config = {
            'adapter_dim' : adapter_dim, 
            'lora_rank' : lora_rank,
            'lora_alpha' : lora_alpha, 
            'lora_params' : lora_params,
            'adapter_param_names' : adapter_param_names, 
            'lora_param_names' : lora_param_names
        }
            
        # Save files
        
        os.makedirs(output_dir, exist_ok = True)
        model.save_pretrained(output_dir, safe_serialization = True) 
        tokenizer.save_pretrained(output_dir)

        with open(output_dir.joinpath(self.finetuning_config_filename), 'w') as f:
            json.dump(finetuning_config, f, indent = 2)

        if log_param_list is not None and self.checkpoint_report_template is not None:
            txt_log = self.checkpoint_report_template.format(*log_param_list)
            with open(output_dir.joinpath(self.checkpoint_report_filename), 'w') as f:
                f.write(txt_log)
