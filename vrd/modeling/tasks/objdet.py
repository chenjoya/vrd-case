import json

import torch
from torch import nn
from torch.functional import F
from torchvision import models
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from vrd.structures.relationships import Relationships
from ..utils import pairwise_pbboxes

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()

        detector = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = detector.roi_heads.box_predictor.cls_score.in_features
        num_obj_classes = cfg.MODEL.NUM_OBJ_CLASSES
        detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_obj_classes+1)
        detector.roi_heads.detections_per_img = cfg.MODEL.NUM_DETECTIONS

        self.detector = detector
        self.num_obj_classes = num_obj_classes
        
    def forward(self, batches):
        images, batch_gt_instances, batch_gt_relationships = batches
        
        if self.training:
            batch_targets = []

            for gt_instances in batch_gt_instances:
                bboxes = gt_instances.bboxes
                bcats = gt_instances.bcats
                
                # unique operation
                bboxes, inverses = bboxes.unique(sorted=False, return_inverse=True, dim=0)

                labels = torch.zeros(len(bboxes), dtype=torch.long,
                    device=bboxes.device)
                labels[inverses] = bcats + 1 # for det training

                batch_targets.append(
                    dict(boxes=bboxes, labels=labels)
                )
            
            losses = self.detector(images, batch_targets)
            return losses

        # produce eval 
        Ks = [25, 50, 100]
        batch_topK_relationships = [[] for _ in Ks]

        batch_detections = self.detector(images)
        batch_bboxes = [dets['boxes'].int() for dets in batch_detections]
        batch_bcats = [dets['labels'] - 1 for dets in batch_detections] # not we should -1
        batch_bscores = [dets['scores'] for dets in batch_detections]

        for image, bboxes, bcats, bscores, gt_relationships in \
            zip(images, batch_bboxes, batch_bcats, batch_bscores, batch_gt_relationships):

            if gt_relationships.empty():
                continue

            for i, K in enumerate(Ks):
                sbboxes, scats, pbboxes, obboxes, ocats = \
                    pairwise_pbboxes(bboxes, bscores, top=K, bcats=bcats)
                pcats = torch.zeros_like(scats, dtype=scats.dtype, device=scats.device)
                
                batch_topK_relationships[i].append(
                    Relationships(gt_relationships.idx,
                        sbboxes, 
                        scats,
                        pbboxes, 
                        pcats,
                        obboxes, 
                        ocats,
                    )
                )
        
        return batch_topK_relationships

def build_objdet(cfg):
    return Model(cfg)