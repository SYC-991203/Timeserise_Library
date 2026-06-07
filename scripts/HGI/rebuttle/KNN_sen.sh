#!/bin/bash

# ---------------- 配置区域 ----------------

# 创建日志文件夹
mkdir -p logs/HGI/rebuttal

# 模型列表
models=("HGI_PerHKGNN")

# 数据集列表
datasets=("sub2" "sub3")
# datasets=("sub1")

# 温度参数列表 (新增)
# temps=(2.5 7.5 10)
temps=(2 3 4 5)

# 数据集对应的 direction 参数
declare -A directions
directions=(
    ["sub1"]="1,1,2,0,0"
    ["sub2"]="1,1,2,0,2"
    ["sub3"]="0,0,0,1,1"
)

# MoE 参数配置 (JSON 字符串格式)
declare -A moe_params
moe_params=(
    ["sub1"]='{
    "0": {0: 5.0, 1: 1.0, 2: -5.0},  
    "1": {0: 1.0, 1: 5.0, 2: -5.0},  
    "2": {0: -5.0, 1: 1.0, 2: 5.0},  
    "3": {0: -5.0, 1: 4.0, 2: 5.0},  
    "4": {0: -5.0, 1: 4.0, 2: 5.0}
}'
    ["sub2"]='{
    "0": {0: 3.0, 1: 2.0, 2: 5.0},
    "1": {0: 5.0, 1: 3.0, 2: 1.0},
    "2": {0: 2.0, 1: 5.0, 2: 3.0}, 
    "3": {0: -5.0, 1: 3.0, 2: 5.0},
    "4": {0: 5.0, 1: 4.0, 2: -5.0}
}'
    ["sub3"]='{
    "0": {0: 3.0, 1: 5.0, 2: -5.0},
    "1": {0: 5.0, 1: 1.0, 2: 1.0},  
    "2": {0: 2.0, 1: 1.0, 2: 5.0},
    "3": {0: -5.0, 1: 2.0, 2: 5.0},
    "4": {0: -5.0, 1: 5.0, 2: 2.0}
}'
)

# 损失函数参数
loss_func="MSE"

# ----------------------------------------

target=$1 # 目标变量 (例如: zs)

if [ -z "$target" ]; then
    echo "Error: Please specify target variable (e.g., ./run.sh zs)"
    exit 1
fi

if [ "$target" = "all" ]; then
    feature_dim=15
else
    feature_dim=5
fi

# 预测长度的列表
pred_lens=(720 96 192 336)

MAX_JOBS=2
job_count=0
pids=()  # 用于保存所有后台进程的PID

# 外层循环，遍历数据集列表
for dataset in "${datasets[@]}"
do
    # 获取对应的数据集参数
    data_path="DYG_data_3_${dataset}.csv"
    direction="${directions[$dataset]}"
    moe_json="${moe_params[$dataset]}"
    
    # 基础 model_id
    base_model_id="kdd_${dataset}"

    echo "------------------------------------------------"
    echo "Preparing experiments for dataset: $dataset"
    echo "Data path: $data_path"
    echo "------------------------------------------------"

    # 内层循环，遍历模型列表
    for model in "${models[@]}"
    do
        # 【新增循环层】：遍历温度参数 temp
        for temp in "${temps[@]}"
        do
            # 最内层循环，遍历不同的预测长度（pred_len）
            for pred_len in "${pred_lens[@]}"
            do
                # 1. 修改日志文件名：增加 _t${temp} 标识
                log_file="./logs/HGI/rebuttal/${model}_${dataset}_${pred_len}_${target}_g${temp}.log"
                
                # 2. 修改 model_id：必须包含 temp，否则 Checkpoint 会被覆盖！
                # 格式示例: kdd_sub2_96_t5
                current_model_id="${base_model_id}_${pred_len}_g${temp}"

                echo "Running: Model=$model, Dataset=$dataset, Pred_len=$pred_len, Group=$temp"
                echo "Logging to: $log_file"

                # 启动 Python 脚本
                # 修改点：
                # --model_id 使用 current_model_id
                # --temp 使用 $temp
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
                    --d_model 1024 \
                    --d_ff 512 \
                    --top_k 5 \
                    --des "$target" \
                    --batch_size 256 \
                    --itr 1 \
                    --devices '2,3,4,5' \
                    --target "$target" \
                    --direction "$direction" \
                    --moe_logits_init "$moe_json" \
                    --loss "$loss_func" \
                    --temp "$temp" \
                    --use_multi_gpu > "$log_file" 2>&1 &

                pid=$!
                pids+=($pid)
                echo "Process started with PID: $pid"

                job_count=$((job_count+1))

                # 并发控制
                if [ "$job_count" -ge "$MAX_JOBS" ]; then
                    echo "Reached maximum concurrent jobs ($MAX_JOBS), waiting for jobs to finish..."
                    wait
                    job_count=0
                fi
            done
        done
    done
done

# 等待所有任务完成
wait
echo "All jobs completed. Logs are saved in ./logs/HGI/rebuttal directory."