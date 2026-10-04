"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .pascal_voc import VOCSegmentation
datasets = {
    'pascal_voc': VOCSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
