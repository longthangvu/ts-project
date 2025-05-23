#!/bin/bash
model_name=PAttn
dataset='../Time-Series-Library/dataset/m4'
PATTERNS=("Monthly" "Yearly" "Quarterly" "Weekly" "Daily" "Hourly")
# PATTERNS=("Yearly")

for pattern in "${PATTERNS[@]}"; do
  for i in 1 0; do
    python -u run.py \
      --task_name short_term_forecast \
      --is_training $i \
      --root_path $dataset \
      --seasonal_patterns $pattern \
      --model_id "m4_$pattern" \
      --model $model_name \
      --data m4 \
      --features M \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --batch_size 16 \
      --d_model 512 \
      --des 'Exp' \
      --itr 1 \
      --learning_rate 0.001 \
      --loss 'SMAPE' \
      # --top_k 5 
      # --p_hidden_dims 256 256 \
      # --p_hidden_layers 2
  done
done