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


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            fim_rate (float): Rate (0.0 to 1.0) that sample will be permuted with FIM.
            fim_spm_rate (float): Rate (0.0 to 1.0) of FIM permuations that will use SPM.
            seed (int): Seed for random number generator.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed

        (
            self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,
        ) = fim.get_fim_token_ids(self.tokenizer)
        if not self.suffix_tok_id and self.fim_rate > 0:
            print("FIM is not supported by tokenizer, disabling FIM")
            self.fim_rate = 0

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        np_rng = np.random.RandomState(seed=self.seed)
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []

            for tokenized_input in tokenized_inputs:
                # optionally do FIM permutations
                if self.fim_rate > 0:
                    tokenized_input, np_rng = fim.permute(
                        tokenized_input,
                        np_rng,
                        self.suffix_tok_id,
                        self.prefix_tok_id,
                        self.middle_tok_id,
                        self.pad_tok_id,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )

                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


def create_datasets(tokenizer, args, seed):
    dataset = load_dataset(args.dataset_name, split=args.splits)
    dataset = dataset.train_test_split(
        test_size=args.test_size, seed=seed, shuffle=True
    )
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )
    chars_per_token = chars_token_ratio(train_data, tokenizer, args.dataset_text_field)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=args.dataset_text_field,
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        seed=seed,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=args.dataset_text_field,
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        seed=seed,
    )
    return train_dataset, valid_dataset



