import json

import torch
from torch import nn
from torch.functional import F
from torchvision import models
from torchvision import transforms as T

from vrd.structures.relationships import Relationships

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        
        num_obj_classes = cfg.MODEL.NUM_OBJ_CLASSES
        objcls_net = getattr(models, "vgg16")(pretrained=True)
        in_features = objcls_net.classifier._modules['6'].in_features
        objcls_net.classifier._modules['6'] = nn.Linear(in_features, num_obj_classes)

        prior = torch.tensor(json.load(open("priors.json"))['objcls'])
        bias = -torch.log((1 - prior) / prior)
        objcls_net.classifier._modules['6'].bias.data = bias

        self.objcls_net = objcls_net
        self.num_obj_classes = num_obj_classes
        self.transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def forward(self, batches):
        images, batch_gt_instances, batch_gt_relationships = batches

        if self.training:
            batch_bimages, batch_targets = [], []

            for image, gt_instances in zip(images, batch_gt_instances):
                bboxes = gt_instances.bboxes

                # unique operation
                bboxes, inverses = bboxes.unique(sorted=False, return_inverse=True, dim=0)

                bimages = torch.stack(
                    [self.transform(image[:, ymin:ymax, xmin:xmax]) 
                    for xmin, ymin, xmax, ymax in bboxes]
                )
                
                targets = torch.zeros(len(bboxes), self.num_obj_classes,
                    dtype=torch.float, device=bboxes.device)

                for inv, bcat in zip(inverses, gt_instances.bcats):
                    targets[inv, bcat] = 1
                
                batch_bimages.append(bimages)
                batch_targets.append(targets)
            
            batch_bimages = torch.cat(batch_bimages)
            batch_targets = torch.cat(batch_targets)

            batch_logits = self.objcls_net(batch_bimages)

            objcls_loss = F.binary_cross_entropy_with_logits(
                batch_logits, 
                batch_targets
            ) * self.num_obj_classes

            return dict(objcls_loss=objcls_loss)

        # produce eval 
        Ks = [25, 50, 100]
        batch_topK_relationships = [[] for _ in Ks]
        
        for image, gt_relationships in zip(images, batch_gt_relationships):
            if gt_relationships.empty():
                continue

            sbboxes = gt_relationships.sbboxes
            obboxes = gt_relationships.obboxes

            simages = torch.stack(
                [self.transform(image[:, ymin:ymax, xmin:xmax]) 
                for xmin, ymin, xmax, ymax in sbboxes]
            )
            oimages = torch.stack(
                [self.transform(image[:, ymin:ymax, xmin:xmax]) 
                for xmin, ymin, xmax, ymax in obboxes]
            )

            sprobs = self.objcls_net(simages).sigmoid_()
            oprobs = self.objcls_net(oimages).sigmoid_()
            
            soprobs = torch.matmul(
                sprobs.view(-1, self.num_obj_classes, 1),
                oprobs.view(-1, 1, self.num_obj_classes),
            )
            somasks = torch.zeros_like(soprobs, 
                device=soprobs.device, dtype=torch.bool)
            somasks_view = somasks.view(-1)

            for i, K in enumerate(Ks):
                _, idxs = soprobs.view(-1).topk(K)
                somasks_view[idxs] = 1
                idxs, scats, ocats = somasks.nonzero(as_tuple=True)

                batch_topK_relationships[i].append(
                    Relationships(gt_relationships.idx,
                        gt_relationships.sbboxes[idxs], 
                        scats,
                        gt_relationships.pbboxes[idxs], 
                        gt_relationships.pcats[idxs],
                        gt_relationships.obboxes[idxs], 
                        ocats,
                    )
                )

                somasks.zero_()
        
        return batch_topK_relationships

    
def build_objcls(cfg):
    return Model(cfg)