#!/bin/bash

# 模型列表
models=("Transformer" "iTransformer" "Autoformer" "Crossformer" "DLinear" "FEDformer" "Informer" "LightTS" \
"PatchTST" "Pyraformer" "Reformer" "HalfRouterformer" )

target=$1 # 目标变量
model_id=kdd_sub2-2

if [ "$target" = "all" ]; then
    feature_dim=15
else
    feature_dim=5
fi

MAX_JOBS=6
job_count=0
pids=()  # 用于保存所有后台进程的PID

# 循环遍历模型列表
for model in "${models[@]:5}"
do
    echo "Running Python script with model: $model"
    
    # 启动 Python 脚本并在后台运行
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./data/DYG/ \
        --data_path DYG_data_3_sub2-2.csv \
        --model_id $model_id \
        --model $model \
        --data DYG_base \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in $feature_dim \
        --dec_in $feature_dim \
        --c_out  $feature_dim \
        --d_model 256 \
        --d_ff 512 \
        --top_k 5 \
        --des 'Exp' \
        --batch_size 1024 \
        --itr 1 \
        --devices '0,6,7' \
        --target "$target" \
        --use_multi_gpu &  # 在后台运行
    
    pid=$!  # 获取后台运行的PID
    pids+=($pid)  # 将PID添加到数组中
    echo "Started Python script with model $model, PID: $pid"
    
    job_count=$((job_count+1))

    if [ "$job_count" -ge "$MAX_JOBS" ]; then
        echo "Reached maximum concurrent jobs ($MAX_JOBS), waiting for jobs to finish..."
        wait  # 等待所有后台任务完成
        job_count=0  # 复位后台进程计数
    fi
done

# 等待所有任务完成
wait
echo "All jobs completed."

# 如果需要中断所有后台进程，可以使用以下命令
function terminate_all() {
    echo "Terminating all background processes..."
    for pid in "${pids[@]}"; do
        echo "Killing process with PID: $pid"
        kill -9 $pid  # 强制终止后台进程
    done
}

# 调用 terminate_all 函数来终止所有进程
terminate_all
