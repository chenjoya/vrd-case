from tqdm import tqdm
import logging
import cv2
import numpy as np
import json
import zipfile
import os

import torch
import torch.distributed as dist
from torch.functional import F
from torchvision.ops import box_iou

from .vr import VR
from vrd.utils.comm import reduce_sum

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

def eval_phrdet(pbboxes, rcats, gt_pbboxes, gt_rcats):
    phrdet = 0
    gt_detected = torch.zeros(len(gt_pbboxes), dtype=torch.bool)

    iou_matrix = box_iou(pbboxes, gt_pbboxes)

    for i, rcat in enumerate(rcats):
        iou_max, g_max = -1, -1
        for g, gt_rcat in enumerate(gt_rcats):
            if gt_detected[g] or not torch.equal(rcat, gt_rcat):
                continue
            
            u = iou_matrix[i][g]
            if u >= 0.5 and u > iou_max:
                iou_max, g_max = u, g
        
        if g_max >= 0:
            phrdet += 1
            gt_detected[g_max] = True
    
    return phrdet

def eval_reldet(sbboxes, rcats, obboxes, gt_sbboxes, gt_rcats, gt_obboxes):
    reldet = 0
    gt_detected = torch.zeros(len(gt_rcats), dtype=torch.bool)

    siou_matrix = box_iou(sbboxes, gt_sbboxes)
    oiou_matrix = box_iou(obboxes, gt_obboxes)

    for i, rcat in enumerate(rcats):
        iou_max, g_max = -1, -1
        for g, gt_rcat in enumerate(gt_rcats):
            if gt_detected[g] or not torch.equal(rcat, gt_rcat):
                continue
            
            u = min(siou_matrix[i][g], oiou_matrix[i][g])
            if u >= 0.5 and u > iou_max:
                iou_max, g_max = u, g
        
        if g_max >= 0:
            reldet += 1
            gt_detected[g_max] = True
    
    return reldet

def evaluate_relationships(all_rs, all_gt_rs, typ=None):

    count, phrdet, reldet = 0, 0, 0
    rs_idxs = []
    
    for rs in all_rs:
        sbboxes, scats, pbboxes, pcats, obboxes, ocats = rs.return_gt()
        rcats = torch.stack([scats, pcats, ocats], dim=1)
        
        if rs.idx >= len(all_gt_rs) or rs.idx in rs_idxs:
            continue
        rs_idxs.append(rs.idx)
        
        gt_rs = all_gt_rs[rs.idx]

        if gt_rs.empty():
            continue

        gt_rs = gt_rs.tensor().to(sbboxes.device)

        gt_sbboxes, gt_scats, gt_pbboxes, gt_pcats, gt_obboxes, gt_ocats = gt_rs.return_gt(typ)
        gt_rcats = torch.stack([gt_scats, gt_pcats, gt_ocats], dim=1)

        phrdet += eval_phrdet(pbboxes, rcats, gt_pbboxes, gt_rcats)
        reldet += eval_reldet(sbboxes, rcats, obboxes, gt_sbboxes, gt_rcats, gt_obboxes)
        count += len(gt_pbboxes)
    
    phrdet = reduce_sum(phrdet)
    reldet = reduce_sum(reldet)
    count = reduce_sum(count)

    return 100*phrdet/count, 100*reldet/count, count

def evaluate(dataset, predictions, task, visualize_dir=""):
    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("vrd.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))

    phrs = [0, 0, 0]
    rels = [0, 0, 0]
    counts = [0, 0, 0]

    for i, all_rs in enumerate(predictions):
        phrs[i], rels[i], counts[i] = evaluate_relationships(all_rs, dataset.all_relationships)
        
    logger.info(
        f"Type: All, Overall count: {counts[0]}, {inform('PhrDet Recall', phrs, Ks=(25, 50, 100))}, {inform('RelDet Recall', rels, Ks=(25, 50, 100))}"
    )
    
    
        

    
