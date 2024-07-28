feature_dim=5

for i in $(seq 1 24)
do
    target="order$i"
    echo "Running Python script with order: $target"
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./data/DYG/ \
        --data_path DYG_2-1_data.csv \
        --model_id Third_M_${target} \
        --model Crossformer \
        --data DYG_Oneshot \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 2 \
        --enc_in $feature_dim\
        --dec_in $feature_dim \
        --c_out  $feature_dim \
        --d_model 512 \
        --d_ff 2048 \
        --top_k 5 \
        --des 'Exp' \
        --itr 1 \
        --devices '0,1,2,3' \
        --target "$target"\
        --use_multi_gpu
done