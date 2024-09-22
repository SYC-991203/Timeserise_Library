#!/bin/bash

# 模型列表
models=("Transformer" "iTransformer" "Autoformer" "Crossformer" "DLinear" "FEDformer" "Informer" "LightTS" \
"PatchTST" "Pyraformer" "Reformer" "HalfRouterformer" "Directionformer")

# 数据集列表
datasets=("sub1" "sub2" "sub3" "sub4")

# 数据集对应的 direction 参数
declare -A directions
directions=(
    ["sub1"]="0,0,1,1,1"
    ["sub2"]="0,0,1,1,1"
    ["sub3"]="0,0,0,1,1"
    ["sub4"]="0,0,1,1,1"
)

target=$1 # 目标变量

if [ "$target" = "all" ]; then
    feature_dim=15
else
    feature_dim=5
fi

MAX_JOBS=6
job_count=0
pids=()  # 用于保存所有后台进程的PID

# 外层循环，遍历数据集列表
for dataset in "${datasets[@]}"
do
    # 设置对应的数据集参数
    data_path="DYG_data_3_${dataset}.csv"
    direction="${directions[$dataset]}"
    model_id="kdd_${dataset}"

    echo "Running experiments for dataset: $dataset"
    echo "Data path: $data_path"
    echo "Direction: $direction"
    echo "Model ID: $model_id"

    # 内层循环，遍历模型列表
    for model in "${models[@]}"
    do
        echo "Running Python script with model: $model"

        # 启动 Python 脚本并在后台运行
        python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./data/DYG/KDD \
            --data_path "$data_path" \
            --model_id "$model_id" \
            --model "$model" \
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
            --c_out $feature_dim \
            --d_model 256 \
            --d_ff 512 \
            --top_k 5 \
            --des 'Exp' \
            --batch_size 1024 \
            --itr 1 \
            --devices '0,1,2,3,4,5,6,7' \
            --target "$target" \
            --direction "$direction" \
            --use_multi_gpu &  # 在后台运行

        pid=$!  # 获取后台运行的PID
        pids+=($pid)  # 将PID添加到数组中
        echo "Started Python script with model $model, PID: $pid"

        job_count=$((job_count+1))

        if [ "$job_count" -ge "$MAX_JOBS" ]; then
            echo "Reached maximum concurrent jobs ($MAX_JOBS), waiting for jobs to finish..."
            wait  # 等待所有后台任务完成
            job_count=0  # 重置后台进程计数
        fi
    done
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

# 调用 terminate_all 函数来终止所有进程（如果需要）
# terminate_all
