config_file="configs/c@reldet.yaml"

gpus=4,5
gpun=2
master_port=29503

# ------------------------ need not change -----------------------------------
CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch --nproc_per_node=$gpun --master_port $master_port \
    test_net.py --config-file $config_file
