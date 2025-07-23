import os 
import pandas as pd 
import gzip 
import json 
from datasets import Dataset 

DATAFOLDER = "hf_stack"
HF_DATASET_NAME = "hug_stack" 


def load_gzip_jsonl(file_path):
    data = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data 


def create_hf_dataset():
    df = None 
    for file in os.listdir(DATAFOLDER):
        data = load_gzip_jsonl(os.path.join(DATAFOLDER, file))
        if df is None:
            df = pd.DataFrame(data)
        else:
            df = pd.concat([df, pd.DataFrame(data)])