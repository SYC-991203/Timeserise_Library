#!/bin/bash

# ---------------- 配置区域 ----------------

# 创建日志文件夹
mkdir -p logs/HGI/ablation

# 模型列表 (根据你的新代码，使用了 TLMMoeED，你可以根据需要改回 TimesNet 或保留这个)
# models=("HGI_woCros" "HGI_woEnc" "HGI_woDec") 
# models=("HGI_GNN" "HGI_PerGNN")
models=("HGI_woDec") 

# 数据集列表
# datasets=("sub3" "sub3" "sub1")
datasets=("sub1")

# 数据集对应的 direction 参数 (根据新代码备注更新)
declare -A directions
directions=(
    ["sub1"]="1,1,2,0,0"
    ["sub2"]="1,1,2,0,2"
    ["sub3"]="0,0,0,1,1"
)

# MoE 参数配置 (JSON 字符串格式)
# 注意：Shell中传递JSON字符串需要小心引号，这里外层用单引号，内层用双引号
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
# pred_lens=(96 192 336 720)
pred_lens=(96 336)

MAX_JOBS=2
job_count=0
pids=()  # 用于保存所有后台进程的PID

# 外层循环，遍历数据集列表
for dataset in "${datasets[@]}"
do
    # 获取对应的数据集参数
    data_path="DYG_data_3_${dataset}.csv"
    direction="${directions[$dataset]}"
    moe_json="${moe_params[$dataset]}" # 获取 MoE JSON 字符串
    model_id="kdd_${dataset}"

    echo "------------------------------------------------"
    echo "Preparing experiments for dataset: $dataset"
    echo "Data path: $data_path"
    echo "Direction: $direction"
    echo "MoE Params: $moe_json"
    echo "------------------------------------------------"

    # 内层循环，遍历模型列表
    for model in "${models[@]}"
    do
        # 新的内层循环，遍历不同的预测长度（pred_len）
        for pred_len in "${pred_lens[@]}"
        do
            # 定义日志文件路径
            log_file="./logs/HGI/ablation/${model}_${dataset}_${pred_len}_${target}.log"
            
            echo "Running: Model=$model, Pred_len=$pred_len"
            echo "Logging to: $log_file"

            # 启动 Python 脚本并在后台运行
            # 注意：$moe_json 需要用单引号包裹或者确保其作为整体字符串传递
            python -u run.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path ./data/DYG/KDD \
                --data_path "$data_path" \
                --model_id "${model_id}_${pred_len}" \
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
                --des 'Exp' \
                --batch_size 512 \
                --itr 1 \
                --devices '1,2,3,4' \
                --target "$target" \
                --direction "$direction" \
                --moe_logits_init "$moe_json" \
                --loss "$loss_func" \
                 --temp 5.0 \
                --use_multi_gpu > "$log_file" 2>&1 &  # 重定向标准输出和错误到日志文件

            pid=$!  # 获取后台运行的PID
            pids+=($pid)  # 将PID添加到数组中
            echo "Process started with PID: $pid"

            job_count=$((job_count+1))

            if [ "$job_count" -ge "$MAX_JOBS" ]; then
                echo "Reached maximum concurrent jobs ($MAX_JOBS), waiting for jobs to finish..."
                wait  # 等待所有后台任务完成
                job_count=0  # 重置后台进程计数
            fi
        done
    done
done

# 等待所有任务完成
wait
echo "All jobs completed. Logs are saved in ./logs directory."