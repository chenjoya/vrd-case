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
    top1, top5 = 0, 0
    count = 0
    for idx, logits in enumerate(all_logits):
        _, candidates = logits.topk(5, dim=1)
        categories = dataset.all_categories[idx]
        # calc accuracy
        for candidate, category in zip(candidates, categories):
            if category in candidate:
                top5 += 1
            if category == candidate[0]:
                top1 += 1
            count += 1
    top1 /= count
    top5 /= count
    logger.info("Overall count: {}, Top 1 accuracy: {:.4}, Top 5 accuracy: {:.4}".format(count, top1, top5))
    
        

    
