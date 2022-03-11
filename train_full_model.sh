# Notice
echo "Make sure you have initialized the memory banks in ./datasets/mem_bank_buffer."
# You may also choose to download our randomly initialized memory banks from here:
# https://pan.baidu.com/s/1x9Tlil6jP-ZPpm-YuC3dBA?pwd=u5bg
# https://pan.baidu.com/s/1nURFJ1xSnNrAWqT3lgVLsg?pwd=bdqx

# Set visible GPUs
gpu=0

CUDA_VISIBLE_DEVICES=$gpu python train.py --model_type acl_spatial --train_test train --initialize ./Backbone/STVG_backbone.pkl --train_mode ACL_Spatial
CUDA_VISIBLE_DEVICES=$gpu python train.py --model_type acl_temporal --train_test train --initialize ./Backbone/STVG_backbone.pkl --train_mode ACL_Temporal
CUDA_VISIBLE_DEVICES=$gpu python train.py --model_type deconfounded --train_test train --train_mode Deconfounded
python ./tools/generator.py --model_type full --model_dir ./checkpoint/