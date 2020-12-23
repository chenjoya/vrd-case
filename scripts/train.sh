config_file="configs/manogcnx1_hrnet+resnet50_1x_freihand.yaml"

gpus=0,1,2,3
gpun=4
master_port=29501

# ------------------------ need not change -----------------------------------
CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch --nproc_per_node=$gpun --master_port $master_port \
    train_net.py --config-file $config_file
