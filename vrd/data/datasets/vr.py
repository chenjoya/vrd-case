import os
from os.path import join
import json
import cv2
import numpy as np

import torch
from torchvision import transforms

# VR bounding box format is [ymin, ymax, xmin, xmax]
def union_bboxes(a, b):
    return [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])]

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
    PREDICATES = [
        "on", "wear", "has", "next to", "sleep next to", "sit next to", "stand next to", 
        "park next", "walk next to", "above", "behind", "stand behind", "sit behind", "park behind", 
        "in the front of", "under", "stand under", "sit under", "near", "walk to", "walk", "walk past", 
        "in", "below", "beside", "walk beside", "over", "hold", "by", "beneath", "with", "on the top of", 
        "on the left of", "on the right of", "sit on", "ride", "carry", "look", "stand on", "use", "at", 
        "attach to", "cover", "touch", "watch", "against", "inside", "adjacent to", "across", "contain", 
        "drive", "drive on", "taller than", "eat", "park on", "lying on", "pull", "talk", "lean on", "fly", 
        "face", "play with", "sleep on", "outside of", "rest on", "follow", "hit", "feed", "kick", "skate on"
    ]
    SELECTED = [
        "walk to", "walk", "walk past", "walk beside", "hold", "ride", "touch", "drive", "drive on", 
        "eat", "pull", "talk", "fly", "play with", "follow", "hit", "feed", "kick", "skate on", "wear", "has", 
        "sleep next to", "sit next to", "stand next to", "park next", "walk next to", "stand behind", "sit behind", 
        "park behind", "stand under", "sit under", "sit on", "carry", "look", "stand on", "use", "attach to", 
        "cover", "watch", "contain", "park on", "lying on", "lean on", "face", "sleep on", "rest on"
    ]

    def __init__(self, image_dir, anno_file, input_size, is_selected, tasks):
        annos = json.load(open(anno_file))

        image_files, all_bboxes, all_categories, all_ubboxes, all_triplets = [], [], [], [], []
        for image_name, relations in annos.items():
            if not relations:
                continue
            bboxes, categories, ubboxes, triplets = [], [], [], []
            for relation in relations:
                subject_box = relation['subject']['bbox']
                subject_category = relation['subject']['category']
                if subject_box in bboxes:
                    subject_index = bboxes.index(subject_box)
                else:
                    subject_index = len(bboxes)
                    bboxes.append(subject_box)
                    categories.append(subject_category)

                object_bbox = relation['object']['bbox']
                object_category = relation['object']['category']
                if object_bbox in bboxes:
                    object_index = bboxes.index(object_bbox)
                else:
                    object_index = len(bboxes)
                    bboxes.append(object_bbox)
                    categories.append(object_category)
                
                predicate = relation['predicate']
                if is_selected:
                    if self.PREDICATES[predicate] in self.SELECTED:
                        predicate = self.SELECTED.index(self.PREDICATES[predicate])
                    else:
                        continue
                ubboxes.append(union_bboxes(subject_box, object_bbox))
                triplets.append([subject_index, predicate, object_index])

            all_bboxes.append(torch.tensor(bboxes)) 
            all_categories.append(torch.tensor(categories))
            all_ubboxes.append(torch.tensor(ubboxes)) 
            all_triplets.append(torch.tensor(triplets))
            image_files.append(join(image_dir, image_name))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(input_size),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self.image_files = image_files
        self.all_bboxes = all_bboxes
        self.all_categories = all_categories
        self.all_ubboxes = all_ubboxes
        self.all_triplets = all_triplets
        self.tasks = tasks
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
        if "OBJCLS" in self.tasks:
            bboxes, categories = self.all_bboxes[idx], self.all_categories[idx]
            bbox_images = torch.stack([self.transform(image[ymin:ymax, xmin:xmax]) for ymin, ymax, xmin, xmax in bboxes])
            return bbox_images, {"categories": categories}, torch.tensor(idx)

        if "RELCLS" in self.tasks:
            ubboxes, triplets = self.all_ubboxes[idx], self.all_triplets[idx]
            ubbox_images = torch.stack([self.transform(image[ymin:ymax, xmin:xmax]) for ymin, ymax, xmin, xmax in ubboxes])
            return ubbox_images, {"predicates": triplets[:, 1]}, torch.tensor(idx)

    def __len__(self):
        return len(self.image_files)
    
    def visualize(self, image, bboxes, ubboxes, categories, triplets):
        bbox_images = [image[ymin:ymax, xmin:xmax] for ymin, ymax, xmin, xmax in bboxes]
        ubbox_images = [image[ymin:ymax, xmin:xmax] for ymin, ymax, xmin, xmax in ubboxes]

        for i, (ubbox_image, triplet) in enumerate(zip(ubbox_images, triplets)): 
            #H, W, _ = bbox_image.shape
            print(ubbox_image.shape)
            subject_index, predicate, object_index = triplet
            subject_category, object_category = categories[subject_index], categories[object_index]
            # see = cv2.putText(bbox_image, self.CATEGORIES[category], (W // 2, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            print(self.CATEGORIES[subject_category], self.SELECTED[predicate], self.CATEGORIES[object_category])
            cv2.imwrite(str(i) + ".jpg", ubbox_image)

