conda install pytorch torchvision cudatoolkit=10.1 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
pip install yacs tqdm opencv-python
ln -s ~/data/freihand/ datasets/

wget http://cs.stanford.edu/people/ranjaykrishna/vrd/json_dataset.zip
wget http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip
