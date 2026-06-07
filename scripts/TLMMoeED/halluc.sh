#!/bin/bash

# ---------------- 配置区域 ----------------

# 创建日志文件夹
mkdir -p logs/hallucination

# 模型列表
models=("TLMMoeED") 

# 数据集列表 (循环 sub1 - sub3)
datasets=("sub1" "sub2" "sub3")

# 实验组名称列表 (不包含 baseline)
exp_groups=("atomization_fix" "global_mixing_fix")

# === 定义通用幻觉参数 (假设所有 sub 数据集都是 5 通道) ===
## 是分数越高融合程度越高
# 1. Atomization: 极端原子化 - "该合没合"
# 方向: 每个通道一组 (0,1,2,3,4)
# Logits: 全部强独立 [0,0, 0.0]
atom_direction="0,1,2,3,4"
atom_moe='{"0": [0.0, 0.0], "1": [0.0, 0.0], "2": [0.0, 0.0], "3": [0.0, 0.0], "4": [0.0, 0.0]}'

# 2. Global Mixing: 极端大锅饭 - "该分没分"
# 方向: 所有通道一组 (0,0,0,0,0)
# Logits: 不确定/全开 [0, 0]
mix_direction="0,0,0,0,0"
mix_moe='{"0": [5.0,-5.0]}'

loss_func="MSE"
target=$1 

if [ -z "$target" ]; then
    echo "Error: Please specify target variable (e.g., ./run_hallucination_loop.sh zs)"
    exit 1
fi

if [ "$target" = "all" ]; then
    feature_dim=15
    # 注意：如果是 15 通道，上面的 hardcode 参数需要调整。
    # 这里默认假设您跑的是 feature_dim=5 的情况。
    echo "Warning: Running with 15 features but hallucination params are hardcoded for 5."
else
    feature_dim=5
fi

# 预测长度固定为 96
pred_lens=(96 192 336 720)

MAX_JOBS=2
job_count=0
pids=()

# ---------------- 循环开始 ----------------

for dataset in "${datasets[@]}"
do
    data_path="DYG_data_3_${dataset}.csv"
    
    # 遍历实验组
    for exp_group in "${exp_groups[@]}"
    do
        # 根据实验组选择参数
        if [ "$exp_group" == "atomization_fix" ]; then
            direction="$atom_direction"
            moe_json="$atom_moe"
        elif [ "$exp_group" == "global_mixing_fix" ]; then
            direction="$mix_direction"
            moe_json="$mix_moe"
        fi

        model_id="kdd_${dataset}_${exp_group}"

        echo "------------------------------------------------"
        echo "Dataset:    $dataset"
        echo "Experiment: $exp_group"
        echo "Direction:  $direction"
        echo "MoE Params: $moe_json"
        echo "------------------------------------------------"

        for model in "${models[@]}"
        do
            for pred_len in "${pred_lens[@]}"
            do
                # 日志文件名包含 dataset 和 exp_group
                log_file="./logs/hallucination/${model}_${dataset}_${exp_group}_len${pred_len}_${target}.log"
                
                # Model ID 加上后缀
                current_model_id="${model_id}_${pred_len}"

                python -u run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path ./data/DYG/KDD \
                    --data_path "$data_path" \
                    --model_id "$current_model_id" \
                    --model "$model" \
                    --data DYG_base \
                    --features M \
                    --seq_len 96 \
                    --label_len 48 \
                    --pred_len "$pred_len" \
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
                    --batch_size 256 \
                    --itr 1 \
                    --devices '2,3,4,5' \
                    --target "$target" \
                    --direction "$direction" \
                    --moe_logits_init "$moe_json" \
                    --loss "$loss_func" \
                    --use_multi_gpu > "$log_file" 2>&1 &

                pid=$!
                pids+=($pid)
                job_count=$((job_count+1))
                echo "Started PID: $pid (Log: $log_file)"

                if [ "$job_count" -ge "$MAX_JOBS" ]; then
                    wait
                    job_count=0
                fi
            done
        done
    done
done

wait
echo "Hallucination Loop Experiments Completed."