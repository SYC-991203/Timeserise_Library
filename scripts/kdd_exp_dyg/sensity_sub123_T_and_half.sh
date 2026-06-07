#!/bin/bash

# 模型列表
# models=("Transformer" "iTransformer" "Autoformer" "Crossformer" "DLinear" "FEDformer" "Informer" "LightTS" \
# "PatchTST" "Pyraformer" "Reformer" "HalfRouterformer" "Directionformer")
models=(Directionformer)
# 数据集列表
datasets=("sub1" "sub2" "sub3" )
# datasets=("sub1")

# 数据集对应的 direction 参数
declare -A directions
directions=(
    ["sub1"]="0,0,1,1,1"
    ["sub2"]="0,0,1,1,1"
    ["sub3"]="0,0,0,1,1"
)

# 横轴 L 的取值列表（seq_len）
# seq_lens=(48 96 192 336 720)
seq_lens=(7 9 11 14 21 42)

# 全局错误日志路径（统一记录错误信息）
global_error_log="/home/home_new/syc/code/Timeserise_Library/logs/sensity.log"
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
    direction="${directions[$dataset]}"
    model_id_base="kdd_${dataset}"

    echo "Running experiments for dataset: $dataset"

    for seq_len in "${seq_lens[@]}"; do
        for model in "${models[@]}"; do

            model_id="${model_id_base}_${model}_seg_L${seq_len}"

            echo "Launching $model with seq_len=$seq_len for $dataset"

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
                    --seq_len 336\
                    --seg_len "$seq_len"\
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
                    --des 'Sens_Exp' \
                    --batch_size 1024 \
                    --itr 1 \
                    --devices '3,4,5,6' \
                    --target "$target" \
                    --direction "$direction" \
                    --use_multi_gpu
            } 2>>"$global_error_log" &  # 仅错误输出重定向到统一日志

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

wait
echo "✅ All jobs completed. Check errors in $global_error_log (if any)."

# 如需手动终止：
function terminate_all() {
    echo "Terminating all background jobs..."
    for pid in "${pids[@]}"; do
        echo "Killing PID: $pid"
        kill -9 $pid 2>/dev/null
    done
}
