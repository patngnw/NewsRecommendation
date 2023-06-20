#!/bin/bash

mode=$1
nGPU=1
model_dir='../model/NAML'
model='NAML'
use_category=True
use_authorid=True
use_entity=False
enable_gpu=True
conv1d_kernel_size=3
skip_title=True
jitao_score_method=True
category_emb_dim=200 #100


if [ ${mode} == train ]
then
    epochs=1
    batch_size=32
    lr=0.0003
    user_log_mask=False
    prepare=False
    skip_count_sample=True
    python -u main.py --mode train --model_dir ${model_dir} --batch_size ${batch_size} --epochs ${epochs} --model ${model} \
    --lr ${lr} --user_log_mask ${user_log_mask} --prepare ${prepare} --nGPU ${nGPU} --enable_gpu ${enable_gpu} \
    --use_category ${use_category} --use_authorid ${use_authorid} --skip_count_sample ${skip_count_sample} \
    --conv1d_kernel_size ${conv1d_kernel_size} \
    --skip_title ${skip_title} --use_entity ${use_entity} \
    --category_emb_dim ${category_emb_dim}
elif [ ${mode} == test ]
then
    user_log_mask=True
    batch_size=128
    load_ckpt_name=$2
    prepare=False
    skip_count_sample=True
    test_users=all  # seen unseen all
    python -u main.py --mode test --model_dir ${model_dir} --batch_size ${batch_size} --user_log_mask ${user_log_mask} \
    --load_ckpt_name ${load_ckpt_name} --model ${model} --prepare ${prepare} --nGPU ${nGPU} --enable_gpu ${enable_gpu} \
    --use_category ${use_category} --use_authorid ${use_authorid} --skip_count_sample ${skip_count_sample} \
    --jitao_score_method ${jitao_score_method} --jitao_topn 5 --jitao_boost 9.5 \
    --test_users ${test_users} --conv1d_kernel_size ${conv1d_kernel_size} \
    --skip_title ${skip_title} --use_entity ${use_entity} \
    --category_emb_dim ${category_emb_dim}
else
    echo "Please select train or test mode."
fi