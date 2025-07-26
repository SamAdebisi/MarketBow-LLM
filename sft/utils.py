from enum import Enum 
import os 
import torch 
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk 
from datasets.builder import DatasetGenerationError 
from tqdm import tqdm 
from peft import Loraconfig 
from peft.tuners.lora import LoraLayer 
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
)

DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"
    
    @classmethod 
    def list(cls):
        return [c.value for c in cls] 
    
    
class ChatmlSpecialTokens(str, Enum): 
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>" 
    
    @classmethod 
    def list(cls):
        return [c.value for c in cls]
    
    
def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    def preprocess(samples):
        batch = []
        for conversation in samples["messages"]:
            batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
        return {"content": batch}
    
    raw_datasets = DatasetDict()
    for split in data_args.splits.split(","):
        try: 
            # Try first if dataset on a Hub repo 
            dataset = load_dataset(data_args.dataset_name, split=split)
        except DatasetGenerationError:
            # If not, check local dataset 
            dataset = load_from_disk(os.path.join(data_args.dataset_name, split))
            
        if "train" in split:
            raw_datasets["train"] = dataset 
        elif "test" in split:
            raw_datasets["test"] = dataset 
        else: 
            raise ValueError(
                f"Split type {split} not recognized as one of test or train."
            )
            
    if apply_chat_template:
        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True, 
            remove_columns=raw_datasets["train"].column_names, 
        )
        
    train_data = raw_datasets["train"]
    valid_data = raw_datasets["test"]
    
