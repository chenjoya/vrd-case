import os
from os.path import join
import json
import cv2
import numpy as np

import torch
from torchvision import transforms

# VR bounding box format is [ymin, ymax, xmin, xmax]
def union_bbox(a, b):
    return [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])]

from transformers import AutoTokenizer, AutoModel
class Embedder(object):
    def __init__(self, arch):
        self.tokenizer = AutoTokenizer.from_pretrained(arch)
        self.model = AutoModel.from_pretrained(arch).cuda()
        self.model.eval()
    def __call__(self, string):
        with torch.no_grad():
            inputs = self.tokenizer(string, return_tensors="pt")
            for k, v in inputs.items():
                inputs[k] = v.cuda()
            return self.model(**inputs).last_hidden_state.mean(dim=1)
    def cosine_similarity(self, a, b):
        return torch.cosine_similarity(
            self.__call__(a),
            self.__call__(b)
        )

# describe instances in an image
class Instances(object): 
    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        self.bboxes = []
        self.bcats = [] 
    
    def append(self, bbox, category):
        self.bboxes.append(bbox)
        self.bcats.append(category)
    
    def tensor(self, idx):
        instances = Instances()
        instances.bboxes = torch.tensor(self.bboxes)
        instances.bcats = torch.tensor(self.bcats)
        instances.idx = torch.tensor(idx)
        return instances
    
    def extend(self, iss):
        instances = Instances()
        instances.bboxes = [instances.bboxes for instances in iss]
        instances.bcats = [instances.bcats for instances in iss]
        instances.idxs = [instances.idx for instances in iss]
        return instances
    
    def to(self, device):
        instances = Instances()
        instances.bboxes = [bboxes.to(device) for bboxes in self.bboxes]
        instances.bcats = [bcats.to(device) for bcats in self.bcats]
        instances.idxs = [idx.to(device) for idx in self.idxs]
        return instances

# describe relationships in an image
class Relationships(object):
    def __init__(self, predicates, bboxes=None):
        self.bboxes = bboxes
        self.num_classes = len(predicates)
        self.predicates = predicates

        # annotations
        self.ubboxes = []
        self.upreds = []
        self.sbboxes = []
        self.obboxes = []
        self.scats = []
        self.ocats = []
        self.sidxs = []
        self.oidxs = []

        self.umasks = []
        self.relations = []
        self.triplets = []

        self.num_obj_classes = len(VR.CATEGORIES)

    def append(self, sbbox, scat, upred, obbox, ocat):
        if VR.PREDICATES_70[upred] not in self.predicates:
            return
        upred = self.predicates.index(VR.PREDICATES_70[upred])
        self.upreds.append(upred)
        
        self.ubboxes.append(union_bbox(sbbox, obbox))
        self.sbboxes.append(sbbox)
        self.sidxs.append(self.bboxes.index(sbbox))
        self.obboxes.append(obbox)
        self.oidxs.append(self.bboxes.index(obbox))
        
        self.scats.append(scat)
        self.ocats.append(ocat)
        self.relations.append((scat, upred, ocat))
    
    def not_empty(self, ):
        return len(self.ubboxes) > 0

    def empty(self, ):
        return len(self.ubboxes) == 0

    def tensor(self, idx):
        relationships = Relationships(self.predicates)
        relationships.bboxes = torch.tensor(self.bboxes)
        relationships.idx = torch.tensor(idx)
        if self.not_empty():
            relationships.sbboxes = torch.tensor(self.sbboxes)
            relationships.scats = torch.tensor(self.scats)
            relationships.sidxs = torch.tensor(self.sidxs)
            relationships.ubboxes = torch.tensor(self.ubboxes)
            relationships.upreds = torch.tensor(self.upreds)
            relationships.obboxes = torch.tensor(self.obboxes)
            relationships.ocats = torch.tensor(self.ocats)
            relationships.oidxs = torch.tensor(self.oidxs)
        return relationships
    
    def extend(self, rss):
        relationships = Relationships(self.predicates)
        relationships.bboxes = [rs.bboxes for rs in rss]
        relationships.idxs = [rs.idx for rs in rss]
        relationships.sbboxes = [rs.sbboxes for rs in rss]
        relationships.scats = [rs.scats for rs in rss]
        relationships.sidxs = [rs.sidxs for rs in rss]
        relationships.ubboxes = [rs.ubboxes for rs in rss]
        relationships.upreds = [rs.upreds for rs in rss]
        relationships.obboxes = [rs.obboxes for rs in rss]
        relationships.ocats = [rs.ocats for rs in rss]
        relationships.oidxs = [rs.oidxs for rs in rss]
        return relationships
    
    def to(self, device):
        relationships = Relationships(self.predicates)
        relationships.bboxes = [b.to(device) for b in self.bboxes]
        relationships.idxs = [b.to(device) for b in self.idxs]
        relationships.sbboxes = [b.to(device) if len(b) > 0 else [] for b in self.sbboxes]
        relationships.scats = [b.to(device) if len(b) > 0 else [] for b in self.scats]
        relationships.sidxs = [b.to(device) if len(b) > 0 else [] for b in self.sidxs]
        relationships.ubboxes = [b.to(device) if len(b) > 0 else [] for b in self.ubboxes]
        relationships.upreds = [b.to(device) if len(b) > 0 else [] for b in self.upreds]
        relationships.obboxes = [b.to(device) if len(b) > 0 else [] for b in self.obboxes]
        relationships.ocats = [b.to(device) if len(b) > 0 else [] for b in self.ocats]
        relationships.oidxs = [b.to(device) if len(b) > 0 else [] for b in self.oidxs]
        return relationships
    
    def create_eval(self, relations, sbboxes, ubboxes, obboxes, idx):
        relationships = Relationships(self.predicates)
        assert len(relations) == len(sbboxes) == len(ubboxes) == len(obboxes)
        relationships.relations = relations
        relationships.sbboxes = sbboxes
        relationships.ubboxes = ubboxes
        relationships.obboxes = obboxes
        relationships.idx = idx
        return relationships

    def gt_to_eval(self, ):
        relationships = Relationships(self.predicates)
        relationships.idxs = self.idxs
        sbboxes, ubboxes, obboxes = [], [], []
        for bboxes, ubboxes_, triplets in zip(self.bboxes, self.ubboxes, self.triplets):
            if len(triplets) > 0:
                sbbox_idxs, ubbox_idxs, obbox_idxs = triplets[:,0], triplets[:,1], triplets[:,2]
                _sbboxes = bboxes[sbbox_idxs]
                _ubboxes = ubboxes_[ubbox_idxs]
                _obboxes = bboxes[obbox_idxs]
            else:
                _sbboxes = _ubboxes = _obboxes = []
            sbboxes.append(_sbboxes)
            ubboxes.append(_ubboxes)
            obboxes.append(_obboxes)
        relationships.sbboxes = sbboxes
        relationships.ubboxes = ubboxes
        relationships.obboxes = obboxes
        relationships.relations = self.relations
        return relationships

    def extend_eval(self, rss):
        relationships = Relationships(self.predicates)
        relationships.idxs = [rs.idx for rs in rss]
        relationships.relations = [rs.relations for rs in rss]
        relationships.sbboxes = [rs.sbboxes for rs in rss]
        relationships.ubboxes = [rs.ubboxes for rs in rss]
        relationships.obboxes = [rs.obboxes for rs in rss]
        return relationships
    
    def cat(self, ds):
        cat_ds = []
        for d in ds:
            if len(d) > 0:
                cat_ds += d
            else:
                cat_ds.append([])
        return cat_ds

    def extend_batch_eval(self, rss):
        relationships = Relationships(self.predicates)
        relationships.idxs = self.cat([rs.idxs for rs in rss])
        relationships.relations = self.cat([rs.relations for rs in rss])
        relationships.sbboxes = self.cat([rs.sbboxes for rs in rss])
        relationships.ubboxes = self.cat([rs.ubboxes for rs in rss])
        relationships.obboxes = self.cat([rs.obboxes for rs in rss])
        return relationships
    
    def shrink(self, ):
        _idxs, _relations, _sbboxes, _ubboxes, _obboxes = [], [], [], [], []
        for idxs, relations, sbboxes, ubboxes, obboxes in \
            zip(self.idxs, self.relations, self.sbboxes, self.ubboxes, self.obboxes):
            if len(relations) > 0:
                _idxs.append(idxs)
                _relations.append(relations)
                _sbboxes.append(sbboxes)
                _ubboxes.append(ubboxes)
                _obboxes.append(obboxes)
        self.idxs = torch.stack(_idxs)
        self.lens = torch.tensor([len(r) for r in _relations], device=self.idxs.device)
        self.relations = torch.cat(_relations)
        self.sbboxes = torch.cat(_sbboxes)
        self.ubboxes = torch.cat(_ubboxes)
        self.obboxes = torch.cat(_obboxes)

    def all_gather(self, func):
        self.shrink()
        relationships = Relationships(self.predicates)
        relationships.idxs = func(self.idxs).cpu()
        relationships.lens = func(self.lens).cpu()
        relationships.relations = func(self.relations).cpu()
        relationships.sbboxes = func(self.sbboxes).cpu()
        relationships.ubboxes = func(self.ubboxes).cpu()
        relationships.obboxes = func(self.obboxes).cpu()
        return relationships
    
    def return_gt(self, t=None):
        if t is None:
            return self.relations, self.sbboxes, self.ubboxes, self.obboxes
        relations, sbboxes, ubboxes, obboxes = \
            [], [], [], []
        for relation, sbbox, ubbox, obbox in \
            zip(self.relations, self.sbboxes, self.ubboxes, self.obboxes):
            if VR.MASKS[t, relation[1]]:
                relations.append(relation) 
                sbboxes.append(sbbox) 
                ubboxes.append(ubbox) 
                obboxes.append(obbox)
        return relations, sbboxes, ubboxes, obboxes

class VR(torch.utils.data.Dataset):
    CATEGORIES = [
        "person", "sky", "building", "truck", "bus", "table", "shirt", "chair", "car", "train", "glasses", 
        "tree", "boat", "hat", "trees", "grass", "pants", "road", "motorcycle", "jacket", "monitor", "wheel", 
        "umbrella", "plate", "bike", "clock", "bag", "shoe", "laptop", "desk", "cabinet", "counter", "bench", 
        "shoes", "tower", "bottle", "helmet", "stove", "lamp", "coat", "bed", "dog", "mountain", "horse", 
        "plane", "roof", "skateboard", "traffic light", "bush", "phone", "airplane", "sofa", "cup", "sink", 
        "shelf", "box", "van", "hand", "shorts", "post", "jeans", "cat", "sunglasses", "bowl", "computer", 
        "pillow", "pizza", "basket", "elephant", "kite", "sand", "keyboard", "plant", "can", "vase", "refrigerator", 
        "cart", "skis", "pot", "surfboard", "paper", "mouse", "trash can", "cone", "camera", "ball", "bear", "giraffe", 
        "tie", "luggage", "faucet", "hydrant", "snowboard", "oven", "engine", "watch", "face", "street", "ramp", "suitcase"
    ]
    PREDICATES_70 = [
        "on", "wear", "has", "next to", "sleep next to", "sit next to", "stand next to", 
        "park next", "walk next to", "above", "behind", "stand behind", "sit behind", "park behind", 
        "in the front of", "under", "stand under", "sit under", "near", "walk to", "walk", "walk past", 
        "in", "below", "beside", "walk beside", "over", "hold", "by", "beneath", "with", "on the top of", 
        "on the left of", "on the right of", "sit on", "ride", "carry", "look", "stand on", "use", "at", 
        "attach to", "cover", "touch", "watch", "against", "inside", "adjacent to", "across", "contain", 
        "drive", "drive on", "taller than", "eat", "park on", "lying on", "pull", "talk", "lean on", "fly", 
        "face", "play with", "sleep on", "outside of", "rest on", "follow", "hit", "feed", "kick", "skate on"
    ]
    PREDICATES_69 = [
        "on", "wear", "has", "next to", "sleep next to", "sit next to", "stand next to", 
        "park next", "walk next to", "above", "behind", "stand behind", "sit behind", "park behind", 
        "in the front of", "under", "stand under", "sit under", "near", "walk to", "walk", "walk past", 
        "in", "below", "beside", "walk beside", "over", "hold", "by", "beneath", "with", "on the top of", 
        "on the left of", "on the right of", "sit on", "ride", "carry", "look", "stand on", "use", "at", 
        "attach to", "cover", "touch", "watch", "against", "inside", "adjacent to", "across", "contain", 
        "drive", "drive on", "eat", "park on", "lying on", "pull", "talk", "lean on", "fly", 
        "face", "play with", "sleep on", "outside of", "rest on", "follow", "hit", "feed", "kick", "skate on"
    ]
    SELECTED = [
        "walk to", "walk", "walk past", "walk beside", "hold", "ride", "touch", "drive", "drive on", 
        "eat", "pull", "talk", "fly", "play with", "follow", "hit", "feed", "kick", "skate on", "wear", "has", 
        "sleep next to", "sit next to", "stand next to", "park next", "walk next to", "stand behind", "sit behind", 
        "park behind", "stand under", "sit under", "sit on", "carry", "look", "stand on", "use", "attach to", 
        "cover", "watch", "contain", "park on", "lying on", "lean on", "face", "sleep on", "rest on"
    ]

    mask = torch.zeros(len(PREDICATES_69), dtype=torch.bool)
    for s in SELECTED:
        mask[PREDICATES_69.index(s)] = True
    MASKS = torch.stack([mask, ~mask])
    TYPES = ["Action & Verb", 'Spatial & Preposition']

    def __init__(self, image_dir, anno_file, is_selected):
        annos = json.load(open(anno_file))

        num_categories = len(self.CATEGORIES)
        
        if is_selected:
            predicates = self.SELECTED
        else:
            predicates = self.PREDICATES_69

        image_names, instances, relationships = [], [], []
        for image_name, annos_per_image in annos.items():
            if not annos_per_image:
                continue
            
            instances_per_image = Instances(num_categories)
            for anno in annos_per_image:
                instances_per_image.append(anno['subject']['bbox'], anno['subject']['category'])
                instances_per_image.append(anno['object']['bbox'], anno['object']['category'])

            relationships_per_image = Relationships(predicates, instances_per_image.bboxes)
            for anno in annos_per_image:
                relationships_per_image.append(anno['subject']['bbox'], anno['subject']['category'],
                    anno['predicate'], anno['object']['bbox'], anno['object']['category'])

            image_names.append(image_name)
            instances.append(instances_per_image)
            relationships.append(relationships_per_image)
        
        self.image_names = image_names
        self.image_files = [join(image_dir, n) for n in image_names]
        self.instances = instances
        self.relationships = relationships
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def __getitem__(self, idx):
        image = cv2.imread(self.image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        instances = self.instances[idx]
        relationships = self.relationships[idx]
        
        image = self.transform(image)
        return image, instances.tensor(idx), relationships.tensor(idx)

    def __len__(self):
        return len(self.image_files)
    
    def visualize(self, sbboxes, scats, ubboxes, upreds, obboxes, ocats, idx):
        colors = {'red': (0,0,255), 'blue': (255,0,0), 'green': (0,255,0)}

        def draw(dimage, bbox, string, color, pos='down'):
            ymin, ymax, xmin, xmax = bbox
            dimage = cv2.rectangle(dimage, (xmin, ymin), (xmax, ymax), colors[color], 2)
            
            # draw background
            if pos == 'down':
                dimage = cv2.rectangle(dimage, (xmax-100, ymin+2), (xmax-2, ymin+48), (255,255,255), thickness=-1)
                cv2.putText(dimage, string, (xmax-100, ymin+25),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)
            else:
                dimage = cv2.rectangle(dimage, (xmin+2, ymin+2), (xmin + 270, ymin+50), (255,255,255), thickness=-1)
                cv2.putText(dimage, string, (xmin+2, ymin+25),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)
            return dimage

        for i, (sbbox, scat, ubbox, upred, obbox, ocat) in enumerate(zip(sbboxes, scats, ubboxes, upreds, obboxes, ocats)): 
            image = cv2.imread(self.image_files[idx])
            dimage = draw(image, sbbox, VR.CATEGORIES[scat], 'red')
            dimage = draw(dimage, ubbox, VR.CATEGORIES[scat] + '-' + VR.PREDICATES_69[upred] + '-' + VR.CATEGORIES[ocat], 'blue', pos='up')
            dimage = draw(dimage, obbox, VR.CATEGORIES[ocat], 'green')
            cv2.imwrite(f'{self.image_names[idx]}_{i}.jpg', dimage)
        