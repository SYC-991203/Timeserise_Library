model_name=PatchTST
vmd_dim=5
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/DYG/ \
  --data_path DYG_vmd_${vmd_dim}.csv \
  --model_id our_imf_exp_${vmd_dim}_M \
  --model $model_name \
  --data DYG_u \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in ${vmd_dim+1} \
  --dec_in ${vmd_dim+1} \
  --c_out ${vmd_dim+1} \
  --des 'Exp' \
  --n_heads 16 \
  --batch_size 32 \
  --itr 1\
  --target "our_imf"

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/DYG/ \
  --data_path DYG_vmd_${vmd_dim}.csv \
  --model_id cer_imf_exp_${vmd_dim}_M \
  --model $model_name \
  --data DYG_u \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in ${vmd_dim+1}  \
  --dec_in ${vmd_dim+1}  \
  --c_out ${vmd_dim+1}  \
  --des 'Exp' \
  --n_heads 16 \
  --batch_size 32 \
  --itr 1\
  --target "cer_imf"

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/DYG/ \
  --data_path DYG_vmd_${vmd_dim}.csv \
  --model_id kla_imf_exp_${vmd_dim}_M \
  --model $model_name \
  --data DYG_u \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in ${vmd_dim+1}  \
  --dec_in ${vmd_dim+1}  \
  --c_out ${vmd_dim+1}  \
  --des 'Exp' \
  --n_heads 16 \
  --batch_size 32 \
  --itr 1\
  --target "kla_imf"

