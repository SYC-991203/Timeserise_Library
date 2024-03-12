N=$1
model_name=PatchTST
# 循环执行脚本
for (( i=1; i<=N; i++ ))
do
    echo "Running iteration $i"
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/DYG/ \
    --data_path DYG_vmd_3.csv \
    --model_id our_exp_single_S_repeat_$i \
    --model $model_name \
    --data DYG_u \
    --features S \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp_$N' \
    --n_heads 16 \
    --batch_size 128 \
    --itr 1\
    --target "our"

    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/DYG/ \
    --data_path DYG_vmd_3.csv \
    --model_id cer_exp_single_S_repeat_$i \
    --model $model_name \
    --data DYG_u \
    --features S \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp_$N' \
    --n_heads 16 \
    --batch_size 128 \
    --itr 1\
    --target "cer"

    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/DYG/ \
    --data_path DYG_vmd_3.csv \
    --model_id kla_exp_single_S_repeat_$i \
    --model $model_name \
    --data DYG_u \
    --features S \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp_$N' \
    --n_heads 16 \
    --batch_size 128 \
    --itr 1\
    --target "kla"
done