config_file="configs/c@reldet.yaml"

gpus=4
gpun=1
master_port=29502

# ------------------------ need not change -----------------------------------
CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch --nproc_per_node=$gpun --master_port $master_port \
    train_net.py --config-file $config_file
