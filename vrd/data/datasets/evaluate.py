from tqdm import tqdm
import logging
import cv2
import numpy as np
import json
import zipfile
import os

import torch
from torch.functional import F


def evaluate(dataset, predictions, save_json_file="", visualize_dir=""):
    """evaluate dataset using different methods based on dataset type.
    Args:
    Returns:
    """
    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("vrd.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))
    all_logits = predictions[0]
    for idx, logits in enumerate(all_logits):
        print(logits.shape)
        print(dataset.all_categories[idx].shape)
    
