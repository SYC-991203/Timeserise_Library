#!/bin/bash

# 模型列表
models=("Transformer" "iTransformer" "Autoformer" "Crossformer" "DLinear" "FEDformer" "Informer" "LightTS" \
"PatchTST" "Pyraformer" "Reformer""FiLM" "MICN" "Koopa")
target=$1 # target 取值只有 our cer kla 和 all
model_id=our_exp_1211
if [ "$target" = "all" ]; then
    feature_dim=15
else
    feature_dim=5
fi

# 循环遍历模型列表
for model in "${models[@]}"
do
    echo "Running Python script with model: $model"
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./data/DYG/ \
        --data_path DYG_u.csv \
        --model_id $model_id \
        --model $model \
        --data DYG_u \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in $feature_dim\
        --dec_in $feature_dim \
        --c_out  $feature_dim \
        --d_model 256 \
        --d_ff 512 \
        --top_k 5 \
        --des 'Exp' \
        --batch_size 32 \
        --itr 1 \
        --devices '0,1,2,3,4,5,6,7' \
        --target "$target"\
        --use_multi_gpu 1
    
    exit_code=$?  # 获取Python脚本的退出码
    
    if [ $exit_code -ne 0 ]; then
        echo "Python script with model $model encountered an error. Exit code: $exit_code"
        echo "Model ID: $model_id" >> error_log.txt  # 记录模型ID到error_log.txt文件
        echo "Date: $(date)" >> error_log.txt  # 记录当前日期到error_log.txt文件
        echo "Model :$model" >> error_log.txt  # 记录出错的模型名称到error_log.txt文件
    else
        echo "Python script with model $model finished successfully"
    fi
done