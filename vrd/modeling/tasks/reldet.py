import json

import torch
from torch import nn
from torch.functional import F
from torchvision import models
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from vrd.structures.relationships import Relationships
from ..utils import pairwise_pbboxes, match_and_sample, unique_pass, fake_loss

class Model(nn.Module):
    def __init__(self, cfg, detector):
        super(Model, self).__init__()

        predcls_net = getattr(models, "vgg16")(pretrained=True)
        in_features = predcls_net.classifier._modules['6'].in_features
        predcls_net.classifier._modules['6'] = nn.Linear(in_features, 
            cfg.MODEL.NUM_PRED_CLASSES)
        
        prior = torch.tensor(json.load(open("priors.json"))['predcls'])
        bias = -torch.log((1 - prior) / prior)
        predcls_net.classifier._modules['6'].bias.data = bias

        self.predcls_net = predcls_net
        self.detector = detector
        self.num_obj_classes = cfg.MODEL.NUM_OBJ_CLASSES
        self.num_pred_classes = cfg.MODEL.NUM_PRED_CLASSES
        self.neg2pos = cfg.MODEL.NEG2POS
        self.num_pred_samples = cfg.MODEL.NUM_PRED_SAMPLES
        self.transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def forward(self, batches):
        images, _, batch_gt_relationships = batches
        
        if self.detector.training:
            self.detector.eval()
        with torch.no_grad():
            batch_detections = self.detector(images)
        
        batch_bboxes = [dets['boxes'].int() for dets in batch_detections]
        batch_bcats = [dets['labels'] - 1 for dets in batch_detections] # not we should -1
        batch_bscores = [dets['scores'] for dets in batch_detections]

        batch_pimages, batch_plabels = [], []

        if self.training:
            for image, bboxes, bcats, bscores, gt_relationships in \
                zip(images, batch_bboxes, batch_bcats, batch_bscores, batch_gt_relationships):

                if gt_relationships.empty():
                    continue
                
                sbboxes, scats, pbboxes, obboxes, ocats, soscores = \
                    pairwise_pbboxes(bboxes, bscores, top=None, bcats=bcats)

                pbboxes, plabels = match_and_sample(
                    scats, pbboxes, ocats, gt_relationships, 
                    self.num_pred_samples, self.neg2pos, 
                    self.num_pred_classes
                )

                pimages = torch.stack(
                    [self.transform(image[:, ymin:ymax, xmin:xmax]) 
                    for xmin, ymin, xmax, ymax in pbboxes]
                )

                batch_pimages.append(pimages)
                batch_plabels.append(plabels)

            if len(batch_pimages) == 0:
                return dict(predcls_loss=fake_loss(self.predcls_net)) 

            batch_pimages = torch.cat(batch_pimages)
            batch_plabels = torch.cat(batch_plabels)
            
            batch_plogits = self.predcls_net(batch_pimages)

            predcls_loss = F.binary_cross_entropy_with_logits(
                batch_plogits, batch_plabels
            ) * self.num_pred_classes

            return dict(predcls_loss=predcls_loss)
        
        # produce eval 
        Ks = [25, 50, 100]
        batch_topK_relationships = [[] for _ in Ks]

        for image, bboxes, bcats, bscores, gt_relationships in \
            zip(images, batch_bboxes, batch_bcats, batch_bscores, batch_gt_relationships):

            if gt_relationships.empty():
                continue
            
            sbboxes, scats, pbboxes, obboxes, ocats, soscores = \
                pairwise_pbboxes(bboxes, bscores, top=None, bcats=bcats)

            pscores = unique_pass(self.predcls_net, self.transform, image, pbboxes).sigmoid_()
            pscores = soscores.view(-1, 1) * pscores 

            idxs, pcats = pscores.nonzero(as_tuple=True)
            pscores = pscores[idxs, pcats]
            
            ranks = pscores.sort(descending=True)[1]
            idxs = idxs[ranks]
            pcats = pcats[ranks]
            
            sbboxes = sbboxes[idxs]
            scats = scats[idxs]
            pbboxes = pbboxes[idxs]
            obboxes = obboxes[idxs]
            ocats = ocats[idxs]

            for i, K in enumerate(Ks):
                batch_topK_relationships[i].append(
                    Relationships(gt_relationships.idx,
                        sbboxes[:K], 
                        scats[:K],
                        pbboxes[:K], 
                        pcats[:K],
                        obboxes[:K], 
                        ocats[:K],
                    )
                )
        
        return batch_topK_relationships

def build_reldet(cfg):
    from . import build_objdet
    objdet = build_objdet(cfg).detector
    return Model(cfg, objdet)