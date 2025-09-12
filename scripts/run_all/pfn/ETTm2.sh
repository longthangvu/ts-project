model=LinearPFN
data_path="/home/lvu/playground/Time-Series-Library/dataset"

for train_budget in 0.1 0.5 1.0
do

for pred_len in 96
do

python run.py \
  --task_name meta_learning_pfn \
  --is_training 0 \
  --data ETTm2 \
  --root_path $data_path/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96_$model \
  --model $model \
  --features S \
  --seq_len 96 --label_len 36 \
  --pred_len $pred_len \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 16 \
  --d_ff 3072 \
  --train_budget $train_budget

done;
done
