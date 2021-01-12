import json

import torch
from torch import nn
from torch.functional import F
from torchvision import models
from torchvision import transforms as T

from vrd.structures.relationships import Relationships
from ..utils import pairwise_pbboxes, triplet_matmul, unique_pass

class Model(nn.Module):
    def __init__(self, cfg, objcls_net, predcls_net):
        super(Model, self).__init__()

        self.objcls_net = objcls_net
        self.predcls_net = predcls_net
        self.detector = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        self.detector.roi_heads.detections_per_img = cfg.MODEL.NUM_DETECTIONS
        self.num_obj_classes = cfg.MODEL.NUM_OBJ_CLASSES
        self.num_pred_classes = cfg.MODEL.NUM_PRED_CLASSES
        
        self.transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def forward(self, batches):
        images, _, batch_gt_relationships = batches
        
        assert not self.training
        
        batch_detections = self.detector(images)
        batch_bboxes = [dets['boxes'].int() for dets in batch_detections]
        batch_bscores = [dets['scores'] for dets in batch_detections]

        # produce eval 
        Ks = [25, 50, 100]
        batch_topK_relationships = [[] for _ in Ks]

        for image, bboxes, bscores, gt_relationships in \
            zip(images, batch_bboxes, batch_bscores, batch_gt_relationships):
            
            if gt_relationships.empty():
                continue
            
            sbboxes, pbboxes, obboxes = pairwise_pbboxes(bboxes, bscores, top=150)
            
            sprobs = unique_pass(self.objcls_net, self.transform, 
                image, sbboxes).sigmoid_()
            
            pprobs = unique_pass(self.predcls_net, self.transform, 
                image, pbboxes).sigmoid_()

            oprobs = unique_pass(self.objcls_net, self.transform, 
                image, obboxes).sigmoid_()
            
            spoprobs = triplet_matmul(sprobs, pprobs, oprobs)
            spomasks = torch.zeros_like(spoprobs, 
                device=spoprobs.device, dtype=torch.bool)
            spomasks_view = spomasks.view(-1)

            for i, K in enumerate(Ks):
                _, idxs = spoprobs.view(-1).topk(K)
                spomasks_view[idxs] = 1
                idxs, scats, pcats, ocats = spomasks.nonzero(as_tuple=True)

                batch_topK_relationships[i].append(
                    Relationships(gt_relationships.idx,
                        sbboxes[idxs], scats,
                        pbboxes[idxs], pcats,
                        obboxes[idxs], ocats,
                    )
                )

                spomasks.zero_()
        
        return batch_topK_relationships

def build_predet(cfg):
    from . import build_objcls
    from . import build_predcls
    objcls_net = build_objcls(cfg).objcls_net
    predcls_net = build_predcls(cfg).predcls_net
    return Model(cfg, objcls_net, predcls_net)