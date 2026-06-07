#!/bin/bash

# ---------------- 配置区域 ----------------

mkdir -p logs/sensitivity

models=("TLMMoeED") 
# datasets=("sub1" "sub2" "sub3")
datasets=("sub2" "sub3")

# === 关键修改：使用整数配置 ===
# 1 -> 0.1
# 5 -> 0.5
# 10 -> 1.0 (Base)
# 20 -> 2.0
# 4 -> 0.4, 8 -> 0.8, 16 -> 1.6
# scales=(1 2 4 8 16 20)
scales=(1 2)

declare -A directions
directions=(
    ["sub1"]="1,1,2,0,0"
    ["sub2"]="1,1,2,0,2"
    ["sub3"]="0,0,0,1,1"
)

# Base Logits (对应 scale=10)
declare -A base_moe_params
base_moe_params=(
    ["sub1"]='{"0": [5.0, -5.0], "1": [0.0, 0.0], "2": [5.0, -5.0]}'
    ["sub2"]='{"0": [0.0, 0.0], "1": [5.0, -5.0], "2": [5.0, -5.0]}'
    ["sub3"]='{"0": [3.0, -3.0], "1": [0.0, 0.0]}'
)

loss_func="MSE"
target=$1 

if [ -z "$target" ]; then
    echo "Error: Please specify target variable (e.g., ./run_sensitivity.sh zs)"
    exit 1
fi

if [ "$target" = "all" ]; then
    feature_dim=15
else
    feature_dim=5
fi

pred_lens=(96)

MAX_JOBS=2
job_count=0
pids=()

# ---------------- 循环开始 ----------------

for dataset in "${datasets[@]}"
do
    data_path="DYG_data_3_${dataset}.csv"
    direction="${directions[$dataset]}"
    base_json="${base_moe_params[$dataset]}"
    model_id="kdd_${dataset}"

    # 遍历整数 Scale
    for scale_int in "${scales[@]}"
    do
        # 这一步只是为了打印好看，Python 内部会重新算
        real_scale_display=$(echo "scale=1; $scale_int / 10" | bc)
        
        echo "------------------------------------------------"
        echo "Dataset: $dataset | Integer Scale: $scale_int (Real: $real_scale_display)"
        
        # === 核心魔法：在 Python 内部做除法 ===
        # scale_int: 传入整数 (如 20)
        # scale = float(sys.argv[2]) / 10.0 : 在这里除以 10
        
        current_moe_json=$(python3 -c "import sys, json; \
data=json.loads(sys.argv[1]); \
raw_scale=float(sys.argv[2]); \
real_scale=raw_scale / 10.0; \
scaled_data={k: [x * real_scale for x in v] for k, v in data.items()}; \
print(json.dumps(scaled_data))" "$base_json" "$scale_int")
        
        echo "Scaled JSON:  $current_moe_json"
        echo "------------------------------------------------"

        for model in "${models[@]}"
        do
            for pred_len in "${pred_lens[@]}"
            do
                # === 文件名使用整数 scale_int，避免小数点 ===
                log_file="./logs/sensitivity/${model}_${dataset}_len${pred_len}_scale${scale_int}_${target}.log"
                
                # Model ID 也建议用整数，避免 wandb 或 tensorboard 解析错误
                current_model_id="${model_id}_${pred_len}_scale${scale_int}"

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
                    --batch_size 512 \
                    --itr 1 \
                    --devices '4,5' \
                    --target "$target" \
                    --direction "$direction" \
                    --moe_logits_init "$current_moe_json" \
                    --loss "$loss_func" \
                    --use_multi_gpu > "$log_file" 2>&1 &

                pid=$!
                pids+=($pid)
                job_count=$((job_count+1))

                if [ "$job_count" -ge "$MAX_JOBS" ]; then
                    wait
                    job_count=0
                fi
            done
        done
    done
done

wait
echo "Sensitivity Analysis Completed."