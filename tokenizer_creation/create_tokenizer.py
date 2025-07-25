# Copied from https://github.com/huggingface/transformers/tree/main/examples/research_projects/codeparrot

from datasets import load_dataset 
from tqdm import tqdm 
from dataclasses import dataclass, field 
from typing import Optional 
from transformers import AutoTokenizer, HfArgumentParser 
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode 


@dataclass 