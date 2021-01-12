# Visual Relationship Detection

This repository supports visual relationship detection (VRD) on VR dataset with state-of-the-art performance.

## Install
```
# install pytorch (other version is okay)
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# install some python libs
pip install yacs tqdm opencv-python

# download the VR dataset (https://cs.stanford.edu/people/ranjaykrishna/vrd/), and create a soft link to here
ln -s ~/data/vr/ datasets/

# please organize the dataset files according to the following structure
└── vr
    ├── sg_train_images
    ├── sg_test_images
    └── annotations
        ├── annotations_test.json
        ├── annotations_train.json

# download the bert relationship feature in google cloud
mkdir bert/
# link: 
```

## Training and Inference

We provide training/inference scripts in ``scripts/`` folder. By using them you can easily train and evaluate the model.

## Performance

Task | PhrDet R@50 | PhrDet R@100 | RelDet R@50 | RelDet R@100 |
--- |:---:|:---:|:---:|:---:|
Lu et. al (ECCV 2016) | 17.03 | 16.17 | 13.86 | 14.70 |
Zhang et. al (CVPR 2017) | 19.42 | 22.42 | 14.07 | 15.20 |
Liang et. al (CVPR 2017) | 21.37 | 22.60 | 18.19 | 20.79 |
Dai et. al (CVPR 2017) | 19.93 | 23.45 | 17.73 | 20.88 |
BLOCK (AAAI 2019) | 26.32 | 28.96 | 19.06 | 20.96 |
Yu et. al (ICCV 2017) | 26.32 | 29.43 | 22.68 | **31.89** |
Faster R-CNN + C@VRD (Ours)  | **30.42** | **37.01** | **24.55** | 29.72 |

(Note: The results of previous works are brought from https://paperswithcode.com/sota/visual-relationship-detection-on-vrd-phrase, https://paperswithcode.com/sota/visual-relationship-detection-on-vrd-1)


