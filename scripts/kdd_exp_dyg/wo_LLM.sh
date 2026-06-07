#!/bin/bash

# 模型列表
models=("Directionformer")
# 数据集列表
datasets=("sub1" "sub2" "sub3")

# 固定的 direction 组合（不区分 sub）
directions=(1,0,1,0,1)

# 横轴 L 的取值列表（seq_len）
seq_lens=(96 192)

# 全局错误日志路径
global_error_log="/home/home_new/syc/code/Timeserise_Library/logs/Wo_LLM.log"
> "$global_error_log"  # 清空旧日志

# 目标变量处理
target=$1
if [ "$target" = "all" ]; then
    feature_dim=15
else
    feature_dim=5
fi

MAX_JOBS=3
job_count=0
pids=()

for dataset in "${datasets[@]}"; do
    data_path="DYG_data_3_${dataset}.csv"
    model_id_base="${dataset}"

    echo "Running experiments for dataset: $dataset"

    for seq_len in "${seq_lens[@]}"; do
        for model in "${models[@]}"; do
            for direction in "${directions[@]}"; do  # 遍历所有 direction 组合
                model_id="dir${direction//,/}_${dataset}"

                echo "Launching $model with seq_len=$seq_len for $dataset, direction=$direction"

                {
                    python -u run.py \
                        --task_name long_term_forecast \
                        --is_training 1 \
                        --root_path ./data/DYG/KDD \
                        --data_path "$data_path" \
                        --model_id "$model_id" \
                        --model "$model" \
                        --data DYG_base \
                        --features M \
                        --seq_len 48 \
                        --label_len 48 \
                        --pred_len  "$seq_len" \
                        --e_layers 2 \
                        --d_layers 1 \
                        --factor 3 \
                        --enc_in $feature_dim \
                        --dec_in $feature_dim \
                        --c_out $feature_dim \
                        --d_model 256 \
                        --d_ff 512 \
                        --top_k 5 \
                        --des 'Wo_LLM' \
                        --batch_size 64 \
                        --itr 1 \
                        --devices '4,5,6' \
                        --target "$target" \
                        --direction "1,0,1,0,1" \
                        --use_multi_gpu
                } 2>>"$global_error_log" &

                pid=$!
                pids+=($pid)
                echo "Started PID $pid → $model_id"

                job_count=$((job_count + 1))

                if [ "$job_count" -ge "$MAX_JOBS" ]; then
                    echo "Waiting for $MAX_JOBS concurrent jobs to finish..."
                    wait
                    job_count=0
                fi
            done
        done
    done
done

wait
echo "✅ All jobs completed. Check errors in $global_error_log (if any)."

# 终止所有后台任务的函数
function terminate_all() {
    echo "Terminating all background jobs..."
    for pid in "${pids[@]}"; do
        echo "Killing PID: $pid"
        kill -9 $pid 2>/dev/null
    done
}