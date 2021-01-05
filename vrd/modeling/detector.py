import json

from torchvision import models, transforms 
import torch
from torch import nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .roi_heads import BceROIHeads

class FasterRCNN(nn.Module):
    def __init__(self, num_classes, 
        fg_iou_thresh, bg_iou_thresh, score_thresh, nms_thresh, num_detections,
        bce_roi_heads=False):
        super(FasterRCNN, self).__init__()
        detector = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        if bce_roi_heads:
            in_features = detector.roi_heads.box_predictor.cls_score.in_features
            detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            # bias initialization
            prior = torch.tensor(json.load(open("priors.json"))['objcls'])
            bias = -torch.log((1 - prior) / prior)
            detector.roi_heads.box_predictor.cls_score.bias.data = bias

            detector.roi_heads = BceROIHeads(
                detector.roi_heads.box_roi_pool, 
                detector.roi_heads.box_head, 
                detector.roi_heads.box_predictor,
                fg_iou_thresh, bg_iou_thresh,
                score_thresh=score_thresh, 
                nms_thresh=nms_thresh, 
                detections_per_img=num_detections
            )

        self.detector = detector
        self.num_classes = num_classes

    def forward(self, images, bboxes=None, bcats=None): # list of images
        if self.training:
            # ymin,ymax,xmin,xmax -> xmin,ymin,xmax,ymax
            targets = [
                {'boxes': bboxes[:,(2,0,3,1)], 'labels': bcats.float()}
                    for bboxes, bcats in zip(bboxes, bcats)
            ]
            return self.detector(images, targets)
        else:
            detections = self.detector(images)
            bboxes, labels, scores = [], [], []
            for dets in detections:
                # # xmin,ymin,xmax,ymax -> ymin,ymax,xmin,xmax
                bs = dets['boxes'][:,(1,3,0,2)].int()
                ls, ss = dets['labels'], dets['scores']

                #ps = torch.zeros(len(bs), self.num_classes, device=bs.device)
                #idxs = torch.arange(len(bs), device=bs.device)
                #ps[idxs, ls] = ss

                # drop too small bboxes
                squares = (bs[:,1] - bs[:,0]) * (bs[:,3] - bs[:,2])
                mask = squares > 10
                bboxes.append(bs[mask])
                labels.append(ls[mask])
                scores.append(ss[mask])
            return bboxes, labels, scores

def build_detector(cfg, num_classes, bce_roi_heads=False):
    fg_iou_thresh = cfg.MODEL.DETECTOR.FG_IOU_THRESH
    bg_iou_thresh = cfg.MODEL.DETECTOR.BG_IOU_THRESH 
    score_thresh = cfg.MODEL.DETECTOR.SCORE_THRESH 
    nms_thresh = cfg.MODEL.DETECTOR.NMS_THRESH 
    num_detections = cfg.MODEL.DETECTOR.NUM_DETECTIONS 
    return FasterRCNN(num_classes, fg_iou_thresh, bg_iou_thresh, 
        score_thresh, nms_thresh, num_detections,
        bce_roi_heads)
