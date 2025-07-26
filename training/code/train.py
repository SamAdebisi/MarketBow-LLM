""" 
Pretrain 7B transformers baased decoder model on code/text dataaset 
"""

import os 
import random
import sys 

import numpy as np 
import torch 
from typing import Optional 
from tqdm import tqdm 
from dataclasses import dataclass, field 

from torch.utils.data import IterableDataset 
from datasets import load_dataset 

from transformers import (
    AutoModelForCausalLM, 
    AutoConfig, 
    AutoTokenizer, 
    Trainer, 
    HfArgumentParser, 
    TrainingArguments, 
    set_seed, 
)

import fim 

os.environ["WANDB_PROJET"] = "MarketBow-LLM" 


# Define and parse arguments. 
@dataclass 
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from. 
    """
    
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    use_flash_attn: Optional[bool] = field(
        default=False, 
        metadata={"help": "Enables Flash attention for training."}, 
    )
    use_reentrant: Optional[bool] = field(
        default=False, 
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"}, 
    )
    
    
@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="smangrul/hug_stack",
        metadata={"help": "The preference dataset to use."},
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "Dataset field to use as input text."}
    )
    max_seq_length: Optional[int] = field(default=4096)
    test_size: Optional[float] = field(default=0.1)
    fim_rate: Optional[float] = field(default=0.5)
    fim_spm_rate: Optional[float] = field(default=0.5)
    splits: Optional[str] = field(
        default="train",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens