import torch
from torch.functional import F

class SoftmaxLoss(object):
    def __init__(self, num_classes, tasks):
        self.num_classes = num_classes
        self.tasks = tasks

    def __call__(self, logits, targets):
        if "OBJCLS" in self.tasks:
            categories = targets['categories'].to(logits.device)
            loss = F.cross_entropy(logits, categories)
            return {'softmax_loss': loss}

def build_loss(cfg):
    return SoftmaxLoss(cfg.MODEL.NUM_CLASSES, cfg.TASKS) 
