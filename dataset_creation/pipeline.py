from datatrove.executor.base import PipelineExecutor 
from datatrove.executor.local import LocalPipelineExecutor 
from datatrove.pipeline.dedup import MinhashDedupSignature 
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig, 
    MinhashDedupBuckets,
    MinhashDedupCluster, 
    MinhashDedupFilter, 
)
from datatrove.pipeline.tokens import TokensCounter 
from datatrove.pipeline.readers import JsonlReader 
from datatrove.pipeline.writers.jsonl import JsonlWriter 
from reader import PersonalCopilotDatasetReader 
from filter import BasicodeFilter 

MIRROR_DIRECTORY = "MarketBow-LLM" 
TOTAL_TASKS = 16 

# you can also change ngrams or the number of buckets and their size here 
minhash_config = MinhashConfig(
    use_64bit_hashes=True, 
) # better precision -> fewer false positives (collisions) 


def run_code_dataset_generation():
    # stage 0 reads the code data and does basic filtering 
    pipeline_0 = [
        PersonalCopilotDatasetReader(data_folder=MIRROR_DIRECTORY),
        BasicodeFilter(),
        JsonlWriter(output_folder="filtered_data"), 
    ]
    
    # stage 1 computes minhash signatures for each task (each task gets a set of files)
    pipeline_1 = [
        JsonlReader("filtered_data"),
        MinhashDedupSignature(
            output_folder="signatures",
            config=minhash_config, 
        ), 
    ]
    
    # stage 2 finds matches between signatures in each bucket 
    pipeline_2 = [
        MinhashDedupBuckets(
            input_folder="signatures", 
            output_folder="buckets",
            config=minhash_config, 
        )
    ]
    
    # stage 3 creates clusters of duplicates using the results from all buckets 
    pipeline_3 = [
        MinhashDedupCluster(
            input_folder="buckets",
            output_folder="remove_ids",
            config=minhash_config, 
        )
    ]