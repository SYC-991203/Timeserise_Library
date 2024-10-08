#!/bin/bash
# 要调整pred-l 做短中长测试 之前全部的实验的pred-l都是96 要补充192 336 720
# 模型列表

# feature 取s 下单变量预测，多种模型的bash脚本

models=("Transformer" "iTransformer" "Autoformer" "Crossformer" "DLinear" "FEDformer" "Informer" "LightTS" \
"PatchTST" "Pyraformer" "Reformer""HalfRouterformer" )
target=$1 # target 取值只有 our cer kla 和 all
model_id=our_exp_1211 #model id 统一命名为变量名_实验（exp）or 测试代码（test）_DATE_M（多变量预测多变量）S（单-单）MS（多-单）
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
        --data_path DYG_2-1_data.csv \
        --model_id $model_id \
        --model $model \
        --data DYG_u \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 192 \
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
        --devices '0,1,2,3' \
        --target "$target"\
        --use_multi_gpu
    
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