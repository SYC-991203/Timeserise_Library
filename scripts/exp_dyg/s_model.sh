#!/bin/bash

# 模型列表
models=("Transformer" "iTransformer" "Autoformer" "Crossformer" "DLinear" "FEDformer" "Informer" "LightTS" "PatchTST" "Pyraformer" "Reformer")

# 循环遍历模型列表
for model in "${models[@]}"
do
    echo "Running Python script with model: $model"
    
 python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/DYG/ \
  --data_path DYG_u.csv \
  --model_id dyg_test_simple \
  --model $model \
  --data DYG_u \
  --features S \
  --target our\
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --batch_size 32 \
  --itr 1\
  --devices '0,1,2,3,4,5,6,7'\
  --use_multi_gpu 1
    echo "Python script with model $model finished"
done