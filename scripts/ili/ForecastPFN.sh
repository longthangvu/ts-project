#!/bin/bash
model_name=ForecastPFN
dataset='../Time-Series-Library/dataset/illness/'

python -u run.py \
--task_name forecastpfn \
--is_training 0 \
--root_path $dataset \
--data_path national_illness.csv \
--model_id ili_36_24 \
--model $model_name \
--data ili \
--features S \
--seq_len 36 \
--label_len 18 \
--pred_len 24 \
--e_layers 2 \
--d_layers 1 \
--factor 3 \
--enc_in 7 \
--dec_in 7 \
--c_out 7 \
--des 'Exp' \
--itr 1 