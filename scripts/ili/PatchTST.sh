#!/bin/bash
model_name=PatchTST
dataset='../Time-Series-Library/dataset/illness/'

for i in 1 0; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training $i \
    --root_path $dataset \
    --data_path national_illness.csv \
    --model_id ili_36_24 \
    --model $model_name \
    --data custom \
    --features S \
    --seq_len 36 \
    --label_len 18 \
    --pred_len 24 \
    --e_layers 4 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --n_heads 4 \
    --d_model 1024\
    --itr 1
done