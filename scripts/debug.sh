config_file="configs/debug.yaml"

gpus=3,4
gpun=2
master_port=29501

# ------------------------ need not change -----------------------------------
CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch --nproc_per_node=$gpun --master_port $master_port \
    train_net.py --config-file $config_file
