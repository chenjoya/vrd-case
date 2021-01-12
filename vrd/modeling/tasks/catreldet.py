import json

import torch
from torch import nn
from torch.functional import F
from torchvision import models
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from vrd.structures.relationships import Relationships
from ..encoder import build_encoder
from ..utils import pairwise_pbboxes, match, unique_pass, fake_loss, split

class Model(nn.Module):
    def __init__(self, cfg, detector):
        super(Model, self).__init__()

        hidden_channels = cfg.MODEL.HIDDEN_CHANNELS
        self.encoder = build_encoder("resnet34", hidden_channels)

        self.bert_relations = torch.load("bert/relations_69.pth").cuda()

        self.vnet = nn.Sequential(
            nn.Linear(hidden_channels*3, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )

        priors = json.load(open("priors.json"))
        rfc = nn.Linear(hidden_channels, cfg.MODEL.NUM_PRED_CLASSES)
        prior = torch.tensor(priors['all_predcls'])
        rfc.bias.data = -torch.log((1 - prior) / prior)
        self.rnet = nn.Sequential(
            nn.Linear(hidden_channels*2, hidden_channels),
            nn.ReLU(inplace=True),
            rfc
        )

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

    def attention(self, rfeatures, relations):
        _relations = []
        for rfeature, relation in zip(rfeatures, relations):
            _relation = torch.cosine_similarity(
                rfeature.view(1, -1), relation
            ).softmax(dim=0).matmul(relation)
            _relations.append(_relation)
        relations = torch.stack(_relations)
        return torch.cat([rfeatures, relations], dim=1)

    def forward(self, batches):
        images, _, batch_gt_relationships = batches
        
        bert_relations = self.bert_relations

        if self.detector.training:
            self.detector.eval()
        with torch.no_grad():
            batch_detections = self.detector(images)
        
        batch_bboxes = [dets['boxes'].int() for dets in batch_detections]
        batch_bcats = [dets['labels'] - 1 for dets in batch_detections] # not we should -1
        batch_bscores = [dets['scores'] for dets in batch_detections]

        batch_scats, batch_pcats, batch_ocats = [], [], []
        batch_pimages, batch_pinvs  = [], []
        batch_soimages, batch_soinvs = [], []
        batch_sbboxes, batch_pbboxes, batch_obboxes, batch_soscores = [], [], [], []

        for image, bboxes, bcats, bscores, gt_relationships in \
            zip(images, batch_bboxes, batch_bcats, batch_bscores, batch_gt_relationships):

            if gt_relationships.empty():
                continue

            # generate subject & object bbox and their categories
            # generate union box 
            sbboxes, scats, pbboxes, obboxes, ocats, soscores = \
                pairwise_pbboxes(bboxes, bscores, top=None, bcats=bcats)
            
            if self.training:
                sbboxes, scats, pbboxes, pcats, obboxes, ocats, soscores = match(
                    sbboxes, scats, pbboxes, obboxes, ocats, soscores,
                    gt_relationships, 
                    self.num_pred_samples, self.neg2pos, 
                    self.num_pred_classes
                )
                batch_pcats.append(pcats)
            
            batch_sbboxes.append(sbboxes)
            batch_pbboxes.append(pbboxes)
            batch_obboxes.append(obboxes)
            batch_soscores.append(soscores)
            batch_scats.append(scats)
            batch_ocats.append(ocats)

            sobboxes = torch.cat([sbboxes, obboxes])
            sobboxes, soinvs = sobboxes.unique(
                sorted=False, return_inverse=True, dim=0
            )
            soimages = torch.stack(
                [self.transform(image[:, ymin:ymax, xmin:xmax]) 
                for xmin, ymin, xmax, ymax in sobboxes]
            )
            batch_soimages.append(soimages)
            batch_soinvs.append(soinvs)
            
            pbboxes, pinvs = pbboxes.unique(
                sorted=False, return_inverse=True, dim=0
            )
            pimages = torch.stack(
                [self.transform(image[:, ymin:ymax, xmin:xmax]) 
                for xmin, ymin, xmax, ymax in pbboxes]
            )
            batch_pimages.append(pimages)
            batch_pinvs.append(pinvs)
            
        if len(batch_pimages) == 0:
            return dict(predcls_loss=fake_loss(self.predcls_net)) 

        solens = [len(_) for _ in batch_soimages]
        plens = [len(_) for _ in batch_pimages]
        batch_soimages = torch.cat(batch_soimages)
        batch_pimages = torch.cat(batch_pimages)

        batch_sofeatures, batch_pfeatures = \
            self.encoder(batch_soimages, batch_pimages)
        batch_sofeatures = split(batch_sofeatures, solens)
        batch_pfeatures = split(batch_pfeatures, plens)

        batch_plogits = []
        for sofeatures, soinvs, scats, ocats, pfeatures, pinvs in \
            zip(batch_sofeatures, batch_soinvs, batch_scats, batch_ocats,
                batch_pfeatures, batch_pinvs):
            sofeatures = sofeatures[soinvs]
            sfeatures = sofeatures[:len(sofeatures)//2]
            ofeatures = sofeatures[len(sofeatures)//2:]
            pfeatures = pfeatures[pinvs]

            vfeatures = torch.cat([sfeatures, pfeatures, ofeatures], dim=1)
            vfeatures = self.vnet(vfeatures)
            rfeatures = self.attention(vfeatures, bert_relations[scats,:,ocats])

            plogits = self.rnet(rfeatures) 
            batch_plogits.append(plogits)
        
        if self.training:
            batch_plogits = torch.cat(batch_plogits)
            batch_plabels = F.one_hot(
                torch.cat(batch_pcats) + 1, 
                self.num_pred_classes + 1
            )[:, 1:].float()
            predcls_loss = F.binary_cross_entropy_with_logits(
                batch_plogits, batch_plabels
            ) * self.num_pred_classes
            return dict(predcls_loss=predcls_loss)
        
        # produce eval 
        Ks = [25, 50, 100]
        batch_topK_relationships = [[] for _ in Ks]

        for sbboxes, scats, pbboxes, plogits, obboxes, ocats, soscores, gt_relationships in \
            zip(batch_sbboxes, batch_scats, batch_pbboxes, batch_plogits, 
                batch_obboxes, batch_ocats, batch_soscores, batch_gt_relationships):

            if gt_relationships.empty():
                continue
            
            pscores = plogits.sigmoid_()
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

def build_catreldet(cfg):
    from . import build_objdet
    objdet = build_objdet(cfg).detector
    return Model(cfg, objdet)