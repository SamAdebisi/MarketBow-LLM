from datatrove.data import Document 
from datatrove.pipeline.filters.base_filter import BaseFilter 
from datatrove.pipeline.writers.disk_base import DiskWriter 


def get_basic_stats(text):
    line_lengths = [len(line) for line in text.split("\n")]
    max_line_length = max(line_lengths)
    mean_line_length = sum(line_lengths) / len(line_lengths)