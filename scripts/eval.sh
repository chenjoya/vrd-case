model=manogcnx1_hrnet+resnet50_1x_freihand_notaligned

gpus=0,1
gpun=2
master_port=29502

for epoch in "7e"
do
    weight=outputs/$model/model_$epoch\.pth
    CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch --nproc_per_node=$gpun --master_port $master_port \
    test_net.py --config-file outputs/$model/config.yml MODEL.WEIGHT $weight TEST.SAVE False TEST.VISUALIZE True
done 
