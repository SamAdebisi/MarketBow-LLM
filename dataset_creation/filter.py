from datatrove.data import Document 
from datatrove.pipeline.filters.base_filter import BaseFilter 
from datatrove.pipeline.writers.disk_base import DiskWriter 


def get_basic_stats(text):
    line_lengths = [len(line) for line in text.split("\n")]
    max_line_length = max(line_lengths)
    mean_line_length = sum(line_lengths) / len(line_lengths)
    alphanum_count = sum(char.isalpha() or char.isdigit() for char in text)
    alphanum_ratio = alphanum_count / len(text)
    return max_line_length, mean_line_length, alphanum_ratio 


class BasicCodeFilter(BaseFilter):
    name = "🧑🏽‍💻 Code Filter"
    
    def __init__(
        self, 
        
    ):
        pass 