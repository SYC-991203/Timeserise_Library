#!/bin/bash

MAX_JOBS=1  # 最大并行任务数
mkdir -p logs  # 创建 logs 目录
TARGETS=("kla_s_pred" "kla_true" "error")
GPUS=("5" "6" "7")  # 分配 GPU
declare -a PIDS=()

for i in "${!TARGETS[@]}"; do
    TARGET=${TARGETS[$i]}
    GPU=${GPUS[$i]}  # 分配 GPU

    # 启动 Python 任务
    nohup python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./data/DYG/ \
        --data_path kla_case.csv \
        --model_id WoPiplineTest \
        --model Transformer \
        --data DYG_vmd \
        --features S \
        --seq_len 250 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 1 \
        --enc_out 5 \
        --dec_in 1 \
        --c_out 1 \
        --d_model 256 \
        --d_ff 512 \
        --top_k 5 \
        --des test \
        --batch_size 64 \
        --itr 1 \
        --devices "$GPU" \
        --target "$TARGET" \
        --direction 1,1,0,0,0 \
        --output_attention \
        > logs/${TARGET}.log 2>&1 &

    sleep 2  # 等待进程稳定启动
    PID=$!
    PIDS+=($PID)

    echo "Started $TARGET on GPU $GPU with PID: $PID"
    echo "$TARGET PID: $PID" >> logs/${TARGET}.log

    # 控制最大并行任务数
    while (( $(jobs -r -p | wc -l) >= MAX_JOBS )); do
        sleep 5  # 每 5 秒检查一次，不会直接卡住
    done
done

# 等待所有任务完成
wait

echo "All tasks completed."
