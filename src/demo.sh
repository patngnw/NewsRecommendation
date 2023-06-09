#!/bin/bash

mode=$1
nGPU=1
model_dir='../model/NAML'
model='NAML'
use_category=True
use_subcategory=False
enable_gpu=False

if [ ${mode} == train ]
then
    epochs=5
    batch_size=32
    lr=0.0003
    user_log_mask=False
    prepare=False
    skip_count_sample=True
    use_authorid=True
    python -u main.py --mode train --model_dir ${model_dir} --batch_size ${batch_size} --epochs ${epochs} --model ${model} \
    --lr ${lr} --user_log_mask ${user_log_mask} --prepare ${prepare} --nGPU ${nGPU} --enable_gpu ${enable_gpu} \
    --use_category ${use_category} --use_subcategory ${use_subcategory} --skip_count_sample ${skip_count_sample} \
    --use_authorid ${use_authorid}
elif [ ${mode} == test ]
then
    user_log_mask=True
    batch_size=128
    load_ckpt_name=$2
    prepare=False
    skip_count_sample=True
    jitao_score_method=True
    test_users=all  # seen unseen all
    use_authorid=True
    python -u main.py --mode test --model_dir ${model_dir} --batch_size ${batch_size} --user_log_mask ${user_log_mask} \
    --load_ckpt_name ${load_ckpt_name} --model ${model} --prepare ${prepare} --nGPU ${nGPU} --enable_gpu ${enable_gpu} \
    --use_category ${use_category} --use_subcategory ${use_subcategory} --skip_count_sample ${skip_count_sample} \
    --jitao_score_method ${jitao_score_method} --jitao_topn 5 --jitao_boost 9.5 \
    --test_users ${test_users} --use_authorid ${use_authorid}
else
    echo "Please select train or test mode."
fi