import os
from os.path import join
import json
import cv2
import numpy as np

import torch
from torchvision import transforms

from vrd.structures import Instances
from vrd.structures import Relationships

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
        
        if is_selected:
            predicates = self.SELECTED
        else:
            predicates = self.PREDICATES_69

        image_names, all_instances, all_relationships = [], [], []
        
        idx = 0
        for image_name, annos_per_image in annos.items():
            if not annos_per_image:
                continue
            
            instances = Instances(idx)
            for anno in annos_per_image:
                sbbox = anno['subject']['bbox']
                ymin, ymax, xmin, xmax = sbbox
                sbbox = (xmin, ymin, xmax, ymax)
                instances.append(sbbox, anno['subject']['category'])

                obbox = anno['object']['bbox']
                ymin, ymax, xmin, xmax = obbox
                obbox = (xmin, ymin, xmax, ymax)
                instances.append(obbox, anno['object']['category'])

            relationships = Relationships(idx)
            for anno in annos_per_image:
                pcat = anno['predicate']
                if self.PREDICATES_70[pcat] in predicates:
                    # ymin, ymax, xmin, xmax -> xmin, ymin, xmax, ymax
                    sbbox, scat = anno['subject']['bbox'], anno['subject']['category']
                    ymin, ymax, xmin, xmax = sbbox
                    sbbox = (xmin, ymin, xmax, ymax)
                    pcat = predicates.index(self.PREDICATES_70[pcat]) # transfer it
                    obbox, ocat = anno['object']['bbox'], anno['object']['category']
                    ymin, ymax, xmin, xmax = obbox
                    obbox = (xmin, ymin, xmax, ymax)
                    relationships.append(sbbox, scat, pcat, obbox, ocat)
        
            image_names.append(image_name)
            all_instances.append(instances)
            all_relationships.append(relationships)
            idx += 1

        self.image_names = image_names
        self.image_files = [join(image_dir, n) for n in image_names]
        self.all_instances = all_instances
        self.all_relationships = all_relationships
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def __getitem__(self, idx):
        image = cv2.imread(self.image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        instances = self.all_instances[idx]
        relationships = self.all_relationships[idx]
        
        image = self.transform(image)
        return image, instances.tensor(), relationships.tensor()

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
        