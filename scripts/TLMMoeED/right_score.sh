#!/bin/bash

# ---------------- 配置区域 ----------------

mkdir -p logs/routing_right

models=("TLMMoeED") 
datasets=("sub1" "sub2" "sub3")

# 只跑这一组：Routing right (路由正确)
# 也就是您定义的“最坏情况”
exp_group="routing_right"

# === 1. 正确的分组 (Direction) - 保持不变 ===
# 既然是测试路由幻觉，我们假设分组是对的，只改分数
declare -A directions
directions["sub1"]="1,1,2,0,0"
directions["sub2"]="1,1,2,0,2"
directions["sub3"]="0,0,0,1,1"

# === 2. 幻觉参数设置 (完全反转/最坏情况) ===
declare -A adv_moe_params

# Dataset: sub1
# 原本: G0=[5,-5](独), G1=[0,0](中), G2=[5,-5](独)
# 反转: G0=[-5,5](强迫共), G1=[5,-5](强迫独), G2=[-5,5](强迫共)
adv_moe_params["sub1"]='{"0": [5.0, -5.0], "1": [5.0, -5.0], "2": [0.0, 0.0]}'

# Dataset: sub2
# 原本: G0=[0,0](中), G1=[5,-5](独), G2=[5,-5](独)
# 反转: G0=[5,-5](强迫独), G1=[-5,5](强迫共), G2=[-5,5](强迫共)
adv_moe_params["sub2"]='{"0": [0.0, 0.0], "1": [5.0, -5.0], "2": [5.0, -5.0]}'

# Dataset: sub3
# 原本: G0=[3,-3](独), G1=[0,0](中)
# 反转: G0=[-3,3](强迫共), G1=[3,-3](强迫独)
adv_moe_params["sub3"]='{"0": [3.0, -3.0], "1": [0.0, 0.0]}'


loss_func="MSE"
target=$1 

if [ -z "$target" ]; then
    echo "Error: Please specify target variable (e.g., ./run_routing_right.sh zs)"
    exit 1
fi

if [ "$target" = "all" ]; then
    feature_dim=15
else
    feature_dim=5
fi

# pred_lens=(336 720)
pred_lens=(720)

MAX_JOBS=2
job_count=0
pids=()

# ---------------- 循环开始 ----------------

for dataset in "${datasets[@]}"
do
    data_path="DYG_data_3_${dataset}.csv"
    
    # 获取正确的 Direction 和 错误的 MoE Logits
    direction="${directions[$dataset]}"
    moe_json="${adv_moe_params[$dataset]}"
    
    model_id="kdd_${dataset}_${exp_group}"

    echo "------------------------------------------------"
    echo "Dataset:    $dataset"
    echo "Experiment: $exp_group (Worst Case)"
    echo "Direction:  $direction (Fixed Correct)"
    echo "right Logits: $moe_json"
    echo "------------------------------------------------"

    for model in "${models[@]}"
    do
        for pred_len in "${pred_lens[@]}"
        do
            log_file="./logs/routing_right/${model}_${dataset}_${exp_group}_len${pred_len}_${target}.log"
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
                --d_model 512 \
                --d_ff 512 \
                --top_k 5 \
                --des 'Exp' \
                --batch_size 128 \
                --itr 1 \
                --devices '4,5,6,7' \
                --target "$target" \
                --direction "$direction" \
                --moe_logits_init "$moe_json" \
                --loss "$loss_func" \
                --use_multi_gpu > "$log_file" 2>&1 &

            pid=$!
            pids+=($pid)
            job_count=$((job_count+1))
            echo "Started PID: $pid"

            if [ "$job_count" -ge "$MAX_JOBS" ]; then
                wait
                job_count=0
            fi
        done
    done
done

wait
echo "Routing right Experiments Completed."