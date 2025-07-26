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