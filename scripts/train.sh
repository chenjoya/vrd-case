config_file="configs/c@reldet.yaml"

gpus=2,3,4,5,6,7
gpun=6
master_port=29501

# ------------------------ need not change -----------------------------------
CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch --nproc_per_node=$gpun --master_port $master_port \
    train_net.py --config-file $config_file
