#!/bin/bash

# Default values
d_model=512
n_heads=8
e_layers=16
d_ff=3072
seq_len=512
model=LinearPFN
data_path="/home/lvu/playground/Time-Series-Library/dataset"
c_min=32
c_max=1536

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --d_model) d_model="$2"; shift ;;
        --seq_len) seq_len="$2"; shift ;;
        --n_heads) n_heads="$2"; shift ;;
        --e_layers) e_layers="$2"; shift ;;
        --d_ff) d_ff="$2"; shift ;;
        --model) model="$2"; shift ;;
        --data_version) data_version="$2"; shift ;;
        --c_min) c_min="$2"; shift ;;
        --c_max) c_max="$2"; shift ;;
        # --data_path) data_path="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

for train_budget in 0.05 0.075 0.1 0.25 0.5 0.75 1.0
do

for pred_len in 96
do

python run.py \
  --task_name meta_learning_pfn \
  --is_training 0 \
  --data electricity \
  --root_path $data_path/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$model \
  --model $model \
  --features S \
  --seq_len $seq_len --label_len 36 \
  --pred_len $pred_len \
  --d_model $d_model \
  --n_heads $n_heads \
  --e_layers $e_layers \
  --d_ff $d_ff \
  --train_budget $train_budget \
  --data_version $data_version \
  --c_min $c_min \
  --c_max $c_max 

done;
done