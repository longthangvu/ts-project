#!/bin/bash

model_name=FEDformer
dataset='../Time-Series-Library/dataset/ETT-small/'

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $dataset \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1