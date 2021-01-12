import json

import torch
from torch import nn
from torch.functional import F
from torchvision import models
from torchvision import transforms as T

from vrd.structures.relationships import Relationships
from ..utils import fake_loss, triplet_matmul

class Model(nn.Module):
    def __init__(self, cfg, objcls_net):
        super(Model, self).__init__()

        self.objcls_net = objcls_net

        num_obj_classes = cfg.MODEL.NUM_OBJ_CLASSES
        num_pred_classes = cfg.MODEL.NUM_PRED_CLASSES
        
        predcls_net = getattr(models, "vgg16")(pretrained=True)
        in_features = predcls_net.classifier._modules['6'].in_features
        predcls_net.classifier._modules['6'] = nn.Linear(in_features, num_pred_classes)

        prior = torch.tensor(json.load(open("priors.json"))['predcls'])
        bias = -torch.log((1 - prior) / prior)
        predcls_net.classifier._modules['6'].bias.data = bias

        self.predcls_net = predcls_net
        self.num_obj_classes = num_obj_classes
        self.num_pred_classes = num_pred_classes
        
        self.transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def forward(self, batches):
        images, _, batch_gt_relationships = batches
        
        if self.training:
            batch_pimages, batch_targets = [], []

            for image, gt_relationships in zip(images, batch_gt_relationships):
                if gt_relationships.empty():
                    continue

                pbboxes = gt_relationships.pbboxes

                # unique operation
                pbboxes, inverses = pbboxes.unique(sorted=False, 
                    return_inverse=True, dim=0)
                
                pimages = torch.stack(
                    [self.transform(image[:, ymin:ymax, xmin:xmax]) 
                    for xmin, ymin, xmax, ymax in pbboxes]
                )
                
                targets = torch.zeros(len(pbboxes), self.num_pred_classes,
                    dtype=torch.float, device=pbboxes.device)

                for i, inv in enumerate(inverses):
                    pcat = gt_relationships.pcats[i]
                    targets[inv][pcat] = 1
                
                batch_pimages.append(pimages)
                batch_targets.append(targets)
            
            if len(batch_pimages) == 0:
                # produce a fake loss
                return dict(predcls_loss=fake_loss(self.predcls_net))

            batch_pimages = torch.cat(batch_pimages)
            batch_targets = torch.cat(batch_targets)

            batch_logits = self.predcls_net(batch_pimages)

            predcls_loss = F.binary_cross_entropy_with_logits(
                batch_logits, 
                batch_targets
            ) * self.num_pred_classes
            return dict(predcls_loss=predcls_loss)

        # produce eval 
        Ks = [25, 50, 100]
        batch_topK_relationships = [[] for _ in Ks]
        
        for image, gt_relationships in zip(images, batch_gt_relationships):
            if gt_relationships.empty():
                continue

            sbboxes = gt_relationships.sbboxes
            pbboxes = gt_relationships.pbboxes
            obboxes = gt_relationships.obboxes

            simages = torch.stack(
                [self.transform(image[:, ymin:ymax, xmin:xmax]) 
                for xmin, ymin, xmax, ymax in sbboxes]
            )
            pimages = torch.stack(
                [self.transform(image[:, ymin:ymax, xmin:xmax]) 
                for xmin, ymin, xmax, ymax in pbboxes]
            )
            oimages = torch.stack(
                [self.transform(image[:, ymin:ymax, xmin:xmax]) 
                for xmin, ymin, xmax, ymax in obboxes]
            )

            sprobs = self.objcls_net(simages).sigmoid_()
            pprobs = self.predcls_net(pimages).sigmoid_()
            oprobs = self.objcls_net(oimages).sigmoid_()
            
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

def build_predcls(cfg):
    from . import build_objcls
    return Model(cfg, build_objcls(cfg).objcls_net)