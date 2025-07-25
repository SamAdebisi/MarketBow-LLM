# Copied from https://github.com/huggingface/transformers/tree/main/examples/research_projects/codeparrot

from datasets import load_dataset 
from tqdm import tqdm 
from dataclasses import dataclass, field 
from typing import Optional 
from transformers import AutoTokenizer, HfArgumentParser 
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode 


@dataclass 
class TokenizerTrainingArguments:
    """
    Configuration for tokenizer training. 
    """
    
    base_tokenizer: Optional[str] =field(
        default="bigcode/starcoder",
        metadata={"help": "Base tokenizer to build new tokenizer from."},
    )
    dataset_name: Optional[str] = field(
        default="smangrul/hug_stack",
        metadata={"help": "Dataset to train tokenizer on."}, 
    )
    text_column: Optional[str] = field(
        default="text", 
        metadata={"help": "Column containing text data to process."}
    )
    vocab_size: Optional[int] = field(
        default=50_000, metadata={"help": "Number of examples to train tokenizer on."}
    )
    n_examples: Optional[int] = field(
        default=6500,
        metadata={"help": "Number of examples to train the tokenizer on."},
    )
    tokenizer_name: Optional[str] = field(
        default="hugcoder", metadata={"help": "Name of new tokenizer."}
    )
    push_to_hub: Optional[bool] = field(
        default=True, metadata={"help": "Push saved tokenizer to the hub."}
    )


def main():
    # Iterator for Training
    def batch_iterator(batch_size=10):
        for _ in tqdm(range(0, args.n_examples, batch_size)):
            yield [next(iter_dataset)[args.text_column] for _ in range(batch_size)]

    # Configuration
    parser = HfArgumentParser(TokenizerTrainingArguments)
    args = parser.parse_args()

    # Base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    base_vocab = list(bytes_to_unicode().values())

    # Load dataset
    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    iter_dataset = iter(dataset)

    # Training and saving
    new_tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(), vocab_size=args.vocab_size, initial_alphabet=base_vocab
    )
    new_tokenizer.save_pretrained(args.tokenizer_name, push_to_hub=args.push_to_hub)


if __name__ == "__main__":
    main()
