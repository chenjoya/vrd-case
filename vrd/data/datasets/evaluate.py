from tqdm import tqdm
import logging
import cv2
import numpy as np
import json
import zipfile
import os

import torch
from torch.functional import F

from .vr import VR

# VR bounding box format is [ymin, ymax, xmin, xmax]
def iou(a, b):
    ymin = max(a[0], b[0])
    ymax = min(a[1], b[1])
    xmin = max(a[2], b[2])
    xmax = min(a[3], b[3])
    inter = max(0, ymax - ymin + 1) * max(0, xmax - xmin + 1)
    a_area = (a[1] - a[0] + 1) * (a[3] - a[2] + 1)
    b_area = (b[1] - b[0] + 1) * (b[3] - b[2] + 1)
    return inter / (a_area + b_area - inter)
    
def split(predictions, lengths):
    _predictions = []
    for ps in predictions:
        segments = []
        begin = 0
        for i in lengths:
            end = begin + i
            segments.append(ps[begin:end])
            begin = end
        _predictions.append(segments)
    return _predictions

def inform(indicator, rates, Ks):
    infos = [f'{indicator}@{k}: {rate:.4f}' for rate, k in zip(rates, Ks)]
    return ', '.join(infos)

def eval_phrdet(rs, ubs, gt_rs, gt_ubs):
    phrdet = 0
    gt_detected = torch.zeros(len(gt_rs), dtype=torch.bool)

    for r, ub in zip(rs, ubs):
        iou_max, g_max = -1, -1
        r, ub = r.tolist(), ub.tolist()
        for g, (gt_r, gt_ub) in enumerate(zip(gt_rs, gt_ubs)):

            if r != list(gt_r) or gt_detected[g]:
                continue
            
            u = iou(ub, gt_ub)
            if u >= 0.5 and u > iou_max:
                iou_max, g_max = u, g
        
        if g_max >= 0:
            phrdet += 1
            gt_detected[g_max] = True
    
    return phrdet

def eval_reldet(rs, sbs, obs, gt_rs, gt_sbs, gt_obs):
    reldet = 0
    gt_detected = torch.zeros(len(gt_rs), dtype=torch.bool)

    for r, sb, ob in zip(rs, sbs, obs):
        iou_max, g_max = -1, -1
        r, sb, ob = r.tolist(), sb.tolist(), ob.tolist()
        for g, (gt_r, gt_sb, gt_ob) in enumerate(zip(gt_rs, gt_sbs, gt_obs)):
            
            if r != list(gt_r) or gt_detected[g]:
                continue
            
            u = min(iou(sb, gt_sb), iou(ob, gt_ob))
            if u >= 0.5 and u > iou_max:
                iou_max, g_max = u, g
        
        if g_max >= 0:
            reldet += 1
            gt_detected[g_max] = True
    
    return reldet

def evaluate_relationships(relationships, gt_relationships, typ=None):
    relations = relationships.relations
    sbboxes, ubboxes, obboxes = relationships.sbboxes, relationships.ubboxes, relationships.obboxes
    lens = relationships.lens
    idxs = relationships.idxs
    relations, sbboxes, ubboxes, obboxes = split(
        (relations, sbboxes, ubboxes, obboxes), 
        lens
    )

    count, phrdet, reldet = 0, 0, 0
    for gt_idx, gt in enumerate(gt_relationships):
        if gt.empty():
            continue
        
        gt_rs, gt_sbs, gt_ubs, gt_obs = gt.return_gt(typ)
        count += len(gt_rs)

        idx = (idxs == gt_idx).nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        idx = idx[0].item()
        rs, sbs, ubs, obs = relations[idx], sbboxes[idx], ubboxes[idx], obboxes[idx]

        phrdet += eval_phrdet(rs, ubs, gt_rs, gt_ubs)
        reldet += eval_reldet(rs, sbs, obs, gt_rs, gt_sbs, gt_obs)
        
    return 100*phrdet/count, 100*reldet/count, count

def evaluate(dataset, predictions, task, save_json_file="", visualize_dir=""):
    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("vrd.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))
    # eval("evaluate_" + task.lower())(dataset, predictions, logger)

    relationships_25, relationships_50, relationships_100 = predictions
    gt_relationships = dataset.relationships
    
    for t in range(2):
        phr_25, rel_25, count = evaluate_relationships(relationships_25, gt_relationships, t)
        phr_50, rel_50, count = evaluate_relationships(relationships_50, gt_relationships, t)
        phr_100, rel_100, count = evaluate_relationships(relationships_100, gt_relationships, t)
        logger.info(f"Type: {VR.TYPE_NAMES[t]}, Overall count: {count}, {inform('PhrDet Recall', (phr_25, phr_50, phr_100), Ks=(25, 50, 100))}, {inform('RelDet Recall', (rel_25, rel_50, rel_100), Ks=(25, 50, 100))}")

    phr_25, rel_25, count = evaluate_relationships(relationships_25, gt_relationships)
    phr_50, rel_50, count = evaluate_relationships(relationships_50, gt_relationships)
    phr_100, rel_100, count = evaluate_relationships(relationships_100, gt_relationships)
    logger.info(f"Type: All, Overall count: {count}, {inform('PhrDet Recall', (phr_25, phr_50, phr_100), Ks=(25, 50, 100))}, {inform('RelDet Recall', (rel_25, rel_50, rel_100), Ks=(25, 50, 100))}")


    
    
        

    
