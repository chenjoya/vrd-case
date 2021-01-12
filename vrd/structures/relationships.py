import torch
import torch.distributed as dist

# format: x1, y1, x2, y2
def union_bbox(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

# describe relationships in an image
class Relationships(object):
    def __init__(self, idx, 
        sbboxes=None, scats=None,
        pbboxes=None, pcats=None, 
        obboxes=None, ocats=None
    ):
        self.idx = idx
        self.sbboxes = sbboxes if sbboxes is not None else []
        self.scats = scats if scats is not None else []
        self.pbboxes = pbboxes if pbboxes is not None else []
        self.pcats = pcats if pcats is not None else []
        self.obboxes = obboxes if obboxes is not None else []
        self.ocats = ocats if ocats is not None else []
        
    def append(self, sbbox, scat, pcat, obbox, ocat):
        self.sbboxes.append(sbbox)
        self.scats.append(scat)
        self.pbboxes.append(union_bbox(sbbox, obbox))
        self.pcats.append(pcat)
        self.obboxes.append(obbox)
        self.ocats.append(ocat)
    
    def not_empty(self, ):
        return len(self.pbboxes) > 0

    def empty(self, ):
        return len(self.pbboxes) == 0

    def tensor(self,):
        idx = torch.tensor([self.idx])
        sbboxes = torch.tensor(self.sbboxes)
        scats = torch.tensor(self.scats)
        pbboxes = torch.tensor(self.pbboxes)
        pcats = torch.tensor(self.pcats)
        obboxes = torch.tensor(self.obboxes)
        ocats = torch.tensor(self.ocats)
        return Relationships(idx, 
            sbboxes, scats, pbboxes, pcats, obboxes, ocats)
    
    def to(self, device):
        idx =  self.idx.to(device)
        sbboxes = self.sbboxes.to(device)
        scats = self.scats.to(device)
        pbboxes = self.pbboxes.to(device)
        pcats = self.pcats.to(device)
        obboxes = self.obboxes.to(device)
        ocats = self.ocats.to(device)
        return Relationships(idx, 
            sbboxes, scats, pbboxes, pcats, obboxes, ocats)
    
    def return_gt(self, t=None):
        if t is None:
            return self.sbboxes, self.scats, \
                self.pbboxes, self.pcats, \
                self.obboxes, self.ocats
        
        #relations, sbboxes, pbboxes, obboxes = \
        #    [], [], [], []
        #for relation, sbbox, ubbox, obbox in \
        #    zip(self.relations, self.sbboxes, self.pbboxes, self.obboxes):
        #    if VR.MASKS[t, relation[1]]:
        #        relations.append(relation) 
        #        sbboxes.append(sbbox) 
        #        pbboxes.append(ubbox) 
        #        obboxes.append(obbox)
        #return relations, sbboxes, pbboxes, obboxes
