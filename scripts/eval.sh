config_file="configs/c@reldet.yaml"

gpus=6,7
gpun=2
master_port=29502

# ------------------------ need not change -----------------------------------
CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch --nproc_per_node=$gpun --master_port $master_port \
    test_net.py --config-file $config_file MODEL.WEIGHT "outputs/c@reldet_best/model_10e.pth" TEST.BATCH_SIZE 16
