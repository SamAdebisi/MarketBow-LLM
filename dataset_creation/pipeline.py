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