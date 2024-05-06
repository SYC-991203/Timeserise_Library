#!/bin/bash
## 基座检查实验，确定基座模型对分解变量的预测是否有效
#model id 统一命名为论文index_变量名_实验（exp）or 测试代码（test）_分解数量_M（多变量预测多变量）S（单-单）MS（多-单）

# 模型列表
model="Crossformer"
targets=("jn" "nd" "zt" "ht" "nn" )

compoents=("trend" "seasonal" "residual") 

# 循环遍历模型列表
for target in "${targets[@]}"
do
    echo "Running Python script with model: $model for $target"
    for component in "${compoents[@]}"
    do
        model_id=Third_${target}_${component}_exp_single_S
        python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./data/DYG/ \
            --data_path DYG_2-1_data.csv \
            --model_id $model_id \
            --model $model \
            --data DYG_Oneshot \
            --features S \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 96 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 1\
            --dec_in 1 \
            --c_out  1 \
            --d_model 256 \
            --d_ff 512 \
            --top_k 5 \
            --des 'Exp' \
            --batch_size 32 \
            --itr 1 \
            --devices '0,1,2,3' \
            --target "${target}_${component}"\
            --use_multi_gpu
    
        exit_code=$?  # 获取Python脚本的退出码
        
        if [ $exit_code -ne 0 ]; then
            echo "Python script with model $model encountered an error. Exit code: $exit_code"
            echo "Target ID: $target" >> error_log.txt # 记录出错的target 内容
            echo "Date: $(date)" >> error_log.txt  # 记录当前日期到error_log.txt文件
        else
            echo "Python script with model $model  and  ${target}_${component} finished successfully"
        fi
    done
done