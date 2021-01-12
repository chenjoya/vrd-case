import torch

# describe instances in an image
class Instances(object): 
    def __init__(self, idx, bboxes=None, bcats=None):
        self.idx = idx
        self.bboxes = bboxes if bboxes is not None else []
        self.bcats = bcats if bcats is not None else []
    
    def empty(self, ):
        return len(self.bboxes) == 0
    
    def append(self, bbox, category):
        self.bboxes.append(bbox)
        self.bcats.append(category)
    
    def tensor(self, ):
        idx = torch.tensor(self.idx)
        bboxes = torch.tensor(self.bboxes)
        bcats = torch.tensor(self.bcats)
        return Instances(idx, bboxes, bcats)
    
    def to(self, device):
        idx = self.idx.to(device)
        bboxes = self.bboxes.to(device)
        bcats = self.bcats.to(device)
        return Instances(idx, bboxes, bcats)