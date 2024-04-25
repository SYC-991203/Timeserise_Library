#!/bin/bash

# Assign the first passed-in argument to 'target'
targets=( "nn" )
# 循环执行脚本


for target in "${targets[@]}"
do
    echo "Running  $target"
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id jnh_cut_off_S \
    --model Transformer \
    --data DYG_base \
    --data_path DYG_2_data.csv \
    --features S \
    --target $target \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 


    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id jnh_cut_off_S \
    --model Transformer \
    --data DYG_Oneshot \
    --data_path DYG_zhn_Oneshot.csv \
    --features S \
    --target ${target}_trend \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 

    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id jnh_cut_off_S \
    --model Transformer \
    --data DYG_Oneshot \
    --data_path DYG_zhn_Oneshot.csv \
    --features S \
    --target ${target}_seasonal \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 

    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id jnh_cut_off_S \
    --model Transformer \
    --data DYG_Oneshot \
    --data_path DYG_zhn_Oneshot.csv \
    --features S \
    --target ${target}_residual \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 
done
echo "All targets processed."