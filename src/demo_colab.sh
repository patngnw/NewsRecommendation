#!/bin/bash

mode=$1
nGPU=1
model_dir='../model/NAML'
model='NAML'
use_category=True
use_subcategory=False
enable_gpu=False
prepare=False

if [ ${mode} == train ]
then
    epochs=5
    batch_size=32
    lr=0.0003
    user_log_mask=False
    prepare=True
    python -u main.py --mode train --model_dir ${model_dir} --batch_size ${batch_size} --epochs ${epochs} --model ${model} \
    --lr ${lr} --user_log_mask ${user_log_mask} --prepare ${prepare} --nGPU ${nGPU} --enable_gpu ${enable_gpu} \
    --use_category ${use_category} --use_subcategory ${use_subcategory} --prepare ${prepare}
elif [ ${mode} == test ]
then
    user_log_mask=True
    batch_size=128
    load_ckpt_name=$2
    prepare=True
    python -u main.py --mode test --model_dir ${model_dir} --batch_size ${batch_size} --user_log_mask ${user_log_mask} \
    --load_ckpt_name ${load_ckpt_name} --model ${model} --prepare ${prepare} --nGPU ${nGPU} --enable_gpu ${enable_gpu} \
    --use_category ${use_category} --use_subcategory ${use_subcategory} --prepare ${prepare}
else
    echo "Please select train or test mode."
fi